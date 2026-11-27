#pragma once

#include "core/arena.h"

#include <cuda.h>

#include <cstddef>
#include <span>
#include <vector>

namespace ninfer {

// CUDA virtual memory management, shared by every owner of fixed-address device memory: the
// suspendable regions below and the eviction pools a Vision window borrows from. A reservation
// fixes the addresses that tensors, block tables and CUDA Graphs capture; the physical pieces
// mapped behind it can be unmapped, moved or released and recreated without any of those
// addresses changing.
namespace vmm {

void check(CUresult result, const char* expression);

void ensure_driver_initialized();

// Whether `device` supports cuMemCreate/cuMemMap. Pools and suspendable regions refuse to start on
// a device without it rather than silently falling back to an allocation that cannot be released.
[[nodiscard]] bool supported(int device);

// Minimum physical allocation granularity of `device`, or zero without virtual memory management.
[[nodiscard]] std::size_t granularity(int device);

[[nodiscard]] CUmemGenericAllocationHandle create(std::size_t bytes, int device);

// Grants read/write access to `[va, va + bytes)` for each device in `devices`. The first device is
// the one that owns the physical memory; the rest are peers that address it directly.
void set_access(CUdeviceptr va, std::size_t bytes, std::span<const int> devices);

void map(CUdeviceptr va, std::size_t bytes, CUmemGenericAllocationHandle handle,
         std::span<const int> devices);

// cuMemUnmap, which CUDA documents as synchronous for most uses: a device wait of the calling
// thread (core/device_wait.h), as is the cuMemSetAccess of set_access().
CUresult unmap(CUdeviceptr va, std::size_t bytes) noexcept;

} // namespace vmm

#define NINFER_CU_CHECK(expr) ::ninfer::vmm::check((expr), #expr)

// Device memory at an address fixed for the region's lifetime whose physical backing can be released
// and recreated: suspending a model gives its memory back to the device without invalidating
// anything that captured an address inside it. Contents do not survive a release; an owner that
// needs them copies them out first.
class VmmRegion {
public:
    struct Options {
        int device = 0;
        // Further devices granted direct access, for memory a peer reads or writes across the bus.
        std::vector<int> peers;
        // Physical allocation unit, a multiple of the device granularity. Larger pieces map and
        // release faster; the last piece is shortened to the reservation. Zero picks 256 MiB.
        std::size_t piece_bytes = 0;
    };

    VmmRegion() noexcept = default;
    VmmRegion(std::size_t bytes, Options options);
    ~VmmRegion();

    VmmRegion(const VmmRegion&)            = delete;
    VmmRegion& operator=(const VmmRegion&) = delete;
    VmmRegion(VmmRegion&& other) noexcept;
    VmmRegion& operator=(VmmRegion&& other) noexcept;

    // The usable extent, `bytes` from the constructor; the reservation is rounded up past it.
    [[nodiscard]] DeviceSpan span() const noexcept;
    [[nodiscard]] std::size_t reserved_bytes() const noexcept { return reserved_; }
    [[nodiscard]] int device() const noexcept { return options_.device; }
    [[nodiscard]] bool backed() const noexcept { return backed_; }
    [[nodiscard]] explicit operator bool() const noexcept { return base_ != 0; }

    // Unmaps and frees every physical piece. The caller has drained all work that addresses the
    // region. Idempotent.
    void release_backing();

    // Maps fresh physical pieces at the same addresses; their contents are undefined. When the
    // device cannot supply them, every piece this call created is released again before it throws,
    // so the region stays unbacked and the caller may retry once memory is free. Idempotent.
    void restore_backing();

private:
    void reset() noexcept;

    CUdeviceptr base_      = 0;
    std::size_t bytes_     = 0;
    std::size_t reserved_  = 0;
    std::size_t piece_     = 0;
    Options options_{};
    std::vector<int> access_; // owner first, then peers
    std::vector<CUmemGenericAllocationHandle> handles_;
    bool backed_ = false;
};

} // namespace ninfer
