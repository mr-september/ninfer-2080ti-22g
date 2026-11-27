#pragma once

#include "core/vmm.h"

#include <cuda.h>
#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <vector>

namespace ninfer {

// Device memory at addresses fixed for the region's lifetime whose physical backing follows what
// its owner uses: whole granules map and unmap anywhere inside the reservation when the owner asks,
// so a cache sized for a whole context costs only the granules its positions have reached, and an
// elastic cache gives granules back. Tensors, tables and captured CUDA Graphs keep addresses inside
// it; work may touch mapped bytes only, and the owner drains every stream that addresses granules
// before it unmaps them. Freshly mapped bytes are undefined.
class GrowableRegion {
public:
    struct Options {
        int device = 0;
        std::vector<int> peers; // further devices granted direct access
    };

    GrowableRegion() noexcept = default;
    GrowableRegion(std::size_t bytes, Options options);
    ~GrowableRegion();

    GrowableRegion(const GrowableRegion&)            = delete;
    GrowableRegion& operator=(const GrowableRegion&) = delete;
    GrowableRegion(GrowableRegion&& other) noexcept;
    GrowableRegion& operator=(GrowableRegion&& other) noexcept;

    [[nodiscard]] void* data() const noexcept { return reinterpret_cast<void*>(base_); }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] std::size_t granule() const noexcept { return granule_; }
    [[nodiscard]] std::size_t mapped_bytes() const noexcept { return mapped_ * granule_; }
    [[nodiscard]] explicit operator bool() const noexcept { return base_ != 0; }

    // Maps every granule [offset, offset + bytes) touches that is not mapped yet, and with a stream
    // zeroes them on it. False when the device cannot supply them: the granules this call mapped
    // are unmapped again, so the region is as it was. Other driver failures throw.
    [[nodiscard]] bool map(std::size_t offset, std::size_t bytes, cudaStream_t zero = nullptr);
    // Unmaps the mapped granules that lie wholly inside [offset, offset + bytes).
    void unmap(std::size_t offset, std::size_t bytes);
    // Whether every granule [offset, offset + bytes) touches is mapped.
    [[nodiscard]] bool mapped(std::size_t offset, std::size_t bytes) const noexcept;
    // Bytes of the granules [offset, offset + bytes) touches that are not mapped.
    [[nodiscard]] std::size_t unmapped_bytes(std::size_t offset, std::size_t bytes) const noexcept;

    // Model suspend: every granule goes back to the device, then the same granules map again.
    // restore_backing throws (leaving nothing mapped) when the device cannot supply them.
    void release_backing();
    void restore_backing();

private:
    void reset() noexcept;
    bool map_granule(std::size_t index);
    void unmap_granule(std::size_t index) noexcept;

    CUdeviceptr base_    = 0;
    std::size_t bytes_   = 0;
    std::size_t granule_ = 0;
    std::size_t mapped_  = 0;
    Options options_{};
    std::vector<int> access_; // owner first, then peers
    std::vector<CUmemGenericAllocationHandle> handles_;
    std::vector<bool> wanted_; // the granules restore_backing maps again
    bool released_ = false;
};

} // namespace ninfer
