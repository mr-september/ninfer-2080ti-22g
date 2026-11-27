#pragma once

#include "core/device_launch.h"
#include "core/device_wait.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace ninfer {

// Stages an over-aligned (alignas(128)) CUtensorMap descriptor struct into device global memory
// for a kernel launch on Windows, where MSVC cannot pass the struct by value as a
// __grid_constant__ parameter (C2719). Builds with NINFER_TMA_STAGED_DESCRIPTORS take this path;
// Windows always does, and NINFER_TMA_STAGING=ON builds it elsewhere to test it. The kernel reads the descriptors from the device pointer
// returned by stage() and makes each tensor map visible to the TMA (tensormap) proxy with a
// fence.proxy.tensormap acquire before its first cp.async.bulk.tensor.
//
// An eager launch: MSVC does pass the same bytes as an 8-byte-aligned word array, so stage()
// launches a one-block kernel that takes the words by value and stores them into one persistent
// device buffer. Kernel parameters are copied when the launch is issued, so the launch never
// reads host memory the host may since have rewritten, and the host never waits for the GPU. The
// staging kernel and the TMA kernel that reads the buffer are ordinary stream-ordered launches,
// so a launch's descriptors are stored only after the previous TMA kernel on the stream has
// finished reading them.
//
// A launch being captured into a CUDA Graph: its descriptors never change across replays, so
// they are copied once, while capturing, into a device copy of their own that is never
// rewritten, and the graph carries no staging kernel. (A staging kernel in a decode graph runs
// on every replay: on an RTX 5090 the FP8 TMA decode route's 72 launches per round cost about
// 65 us of each 18-23 ms DFlash2 round; Ian Ranson, W47 d0abd0cbf.) Copies are keyed by device
// and descriptor bytes, so a graph captured again over the same workspace reuses them.
//
// Invariants, documented here because this is the single staging site:
//   - one stream per device: every eager staged launch on a device runs on that device's compute
//     stream, so its persistent buffer orders against itself in-stream; each device of a
//     pipeline has its own buffer and its own captured copies;
//   - allocation and copies outside the graph: buffers are allocated, and captured descriptors
//     copied, in relaxed capture mode by synchronous calls, so a route first reached while a
//     decode graph is being captured still works and the copy has landed before any replay;
//   - nothing is freed: the device buffers are reclaimed at process exit.
namespace tma_staging_detail {

template <std::size_t Words>
struct DescriptorWords {
    std::uint64_t words[Words];
};

template <std::size_t Words>
__global__ void store_descriptor_words(const DescriptorWords<Words> source,
                                       std::uint64_t* __restrict__ destination) {
    for (std::size_t word = threadIdx.x; word < Words; word += blockDim.x) {
        destination[word] = source.words[word];
    }
    // The consumer's tensor-map acquire pairs with this release. An image without TMA (a build of
    // several architectures compiles this kernel into every image) is never launched here.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
    __trap();
#else
    asm volatile("fence.proxy.tensormap::generic.release.gpu;" : : : "memory");
#endif
}

} // namespace tma_staging_detail

// Makes the tensor map at `tensor_map` (128 bytes in global memory, stored by the staging
// kernel through the generic proxy) visible to the TMA proxy. Call once per tensor map, in the
// thread that issues cp.async.bulk.tensor with it, before its first use.
__device__ __forceinline__ void acquire_staged_tensor_map(const void* tensor_map) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
    (void)tensor_map;
    __trap();
#else
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"
                 :
                 : "l"(tensor_map)
                 : "memory");
#endif
}

template <class Descriptor>
class TmaDescriptorStaging {
  public:
    static_assert(sizeof(Descriptor) % sizeof(std::uint64_t) == 0);

    // Stages the descriptors for the next launch on `stream` and returns the device pointer
    // the kernel must read them from.
    Descriptor* stage(const Descriptor& descriptors, cudaStream_t stream) {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        if (capture == cudaStreamCaptureStatusActive) { return captured(device, descriptors); }
        Descriptor* const buffer = eager_buffer(device);
        tma_staging_detail::DescriptorWords<kWords> words;
        std::memcpy(words.words, &descriptors, sizeof(Descriptor));
        tma_staging_detail::store_descriptor_words<kWords>
            <<<1, 64, 0, stream>>>(words, reinterpret_cast<std::uint64_t*>(buffer));
        CUDA_CHECK(cudaGetLastError());
        return buffer;
    }

  private:
    static constexpr std::size_t kWords         = sizeof(Descriptor) / sizeof(std::uint64_t);
    static constexpr std::size_t kCapturedChunk = 256;
    using Key = std::pair<int, std::array<std::uint64_t, kWords>>;

    struct DeviceChunks {
        int device = 0;
        std::vector<Descriptor*> chunks;
        std::size_t used = kCapturedChunk;
    };

    // cudaMalloc is not a stream operation; relaxed mode lets it run while this thread captures,
    // so it never lands in (or aborts) a graph.
    static Descriptor* allocate(std::size_t count) {
        cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
        CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        void* allocation         = nullptr;
        const cudaError_t status = cudaMalloc(&allocation, count * sizeof(Descriptor));
        CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        CUDA_CHECK(status);
        return static_cast<Descriptor*>(allocation);
    }

    Descriptor* eager_buffer(int device) {
        const std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [owner, buffer] : buffers_) {
            if (owner == device) { return buffer; }
        }
        buffers_.emplace_back(device, allocate(1));
        return buffers_.back().second;
    }

    // The device copy a captured launch reads, written now by a synchronous copy (which the
    // relaxed capture mode admits) and never again.
    Descriptor* captured(int device, const Descriptor& descriptors) {
        Key key{device, {}};
        std::memcpy(key.second.data(), &descriptors, sizeof(Descriptor));
        const std::lock_guard<std::mutex> lock(mutex_);
        if (const auto found = captured_.find(key); found != captured_.end()) {
            return found->second;
        }
        DeviceChunks* pool = nullptr;
        for (DeviceChunks& candidate : pools_) {
            if (candidate.device == device) { pool = &candidate; }
        }
        if (pool == nullptr) { pool = &pools_.emplace_back(DeviceChunks{device, {}, kCapturedChunk}); }
        if (pool->used == kCapturedChunk) {
            pool->chunks.push_back(allocate(kCapturedChunk));
            pool->used = 0;
        }
        Descriptor* const slot     = pool->chunks.back() + pool->used++;
        cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
        CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        const cudaError_t status =
            ninfer::copy_and_wait(slot, &descriptors, sizeof(Descriptor), cudaMemcpyHostToDevice);
        CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));
        CUDA_CHECK(status);
        captured_.emplace(key, slot);
        return slot;
    }

    std::mutex mutex_;
    std::vector<std::pair<int, Descriptor*>> buffers_;
    std::vector<DeviceChunks> pools_;
    std::map<Key, Descriptor*> captured_;
};

} // namespace ninfer
