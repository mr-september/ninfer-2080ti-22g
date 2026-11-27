#pragma once

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cstddef>

namespace ninfer::ops::detail {

// Streaming-multiprocessor count of the calling thread's current device, for launch policies that
// size grids in waves. Cached per device index, because a model split over several GPUs launches
// each stage on its own device and the parts need not match. Returns `fallback` when the device
// cannot be queried.
inline int current_device_multiprocessors(int fallback) noexcept {
    constexpr int kCachedDevices = 64;
    static std::array<std::atomic<int>, kCachedDevices> cache{};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kCachedDevices) {
        return fallback;
    }
    auto& slot = cache[static_cast<std::size_t>(device)];
    if (const int known = slot.load(std::memory_order_relaxed); known > 0) { return known; }
    int count = 0;
    if (cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess ||
        count <= 0) {
        return fallback;
    }
    slot.store(count, std::memory_order_relaxed);
    return count;
}

// Compute capability of the calling thread's current device as 10 * major + minor (80 for an
// A100), for the few routes that follow a part's arithmetic rates rather than its architecture
// tier. Cached per device index; returns `fallback` when the device cannot be queried.
inline int current_device_compute_capability(int fallback) noexcept {
    constexpr int kCachedDevices = 64;
    static std::array<std::atomic<int>, kCachedDevices> cache{};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kCachedDevices) {
        return fallback;
    }
    auto& slot = cache[static_cast<std::size_t>(device)];
    if (const int known = slot.load(std::memory_order_relaxed); known > 0) { return known; }
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess ||
        major <= 0) {
        return fallback;
    }
    slot.store(10 * major + minor, std::memory_order_relaxed);
    return 10 * major + minor;
}

} // namespace ninfer::ops::detail
