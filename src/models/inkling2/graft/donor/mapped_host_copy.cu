#include "core/mapped_host_copy.h"

#include "core/device_launch.h"
#include "core/device_wait.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace ninfer {
namespace {

constexpr unsigned kThreads = 256;
constexpr std::size_t kMaximumBlocks = 4096;

__global__ void copy_mapped_bytes(unsigned char* destination, const unsigned char* source,
                                  std::size_t bytes) {
    const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
    for (std::size_t at = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; at < bytes;
         at += stride) {
        destination[at] = source[at];
    }
}

} // namespace

bool serialized_copy_queue() noexcept {
#ifdef _WIN32
    return true;
#else
    // WSL2 reaches the GPU through the dxgkrnl device; a native Linux driver has no such node.
    static const bool wsl = ::access("/dev/dxg", F_OK) == 0;
    return wsl;
#endif
}

MappedHostBuffer::MappedHostBuffer(std::size_t bytes) : size_(bytes) {
    if (bytes == 0) { throw std::invalid_argument("MappedHostBuffer size must be nonzero"); }
    CUDA_CHECK(cudaHostAlloc(&data_, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
}

MappedHostBuffer::~MappedHostBuffer() {
    if (data_ != nullptr) { ninfer::free_host_memory(data_); }
}

void MappedHostBuffer::copy_to_device(void* destination, std::size_t bytes,
                                      cudaStream_t stream) const {
    if (bytes > size_) { throw std::out_of_range("mapped host copy exceeds its source"); }
    if (bytes == 0) { return; }
    if (destination == nullptr) { throw std::invalid_argument("mapped host copy has no target"); }
    void* source = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&source, data_, 0));
    const auto blocks =
        static_cast<unsigned>(std::min((bytes - 1) / kThreads + 1, kMaximumBlocks));
    copy_mapped_bytes<<<blocks, kThreads, 0, stream>>>(
        static_cast<unsigned char*>(destination), static_cast<const unsigned char*>(source), bytes);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer
