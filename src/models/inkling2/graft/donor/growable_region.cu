#include "core/growable_region.h"

#include "core/device_launch.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer {

GrowableRegion::GrowableRegion(std::size_t bytes, Options options) : options_(std::move(options)) {
    if (bytes == 0) { throw std::invalid_argument("a growable region must not be empty"); }
    granule_ = vmm::granularity(options_.device);
    if (granule_ == 0) {
        throw std::invalid_argument("a growable region requires CUDA virtual memory management "
                                    "on device " + std::to_string(options_.device));
    }
    bytes_ = (bytes + granule_ - 1) / granule_ * granule_;
    access_.push_back(options_.device);
    for (const int peer : options_.peers) {
        if (std::find(access_.begin(), access_.end(), peer) == access_.end()) {
            access_.push_back(peer);
        }
    }
    vmm::ensure_driver_initialized();
    NINFER_CU_CHECK(cuMemAddressReserve(&base_, bytes_, granule_, 0, 0));
    handles_.assign(bytes_ / granule_, 0);
    wanted_.assign(bytes_ / granule_, false);
}

GrowableRegion::~GrowableRegion() { reset(); }

GrowableRegion::GrowableRegion(GrowableRegion&& other) noexcept
    : base_(std::exchange(other.base_, 0)), bytes_(std::exchange(other.bytes_, 0)),
      granule_(std::exchange(other.granule_, 0)), mapped_(std::exchange(other.mapped_, 0)),
      options_(std::move(other.options_)), access_(std::move(other.access_)),
      handles_(std::move(other.handles_)), wanted_(std::move(other.wanted_)),
      released_(std::exchange(other.released_, false)) {}

GrowableRegion& GrowableRegion::operator=(GrowableRegion&& other) noexcept {
    if (this != &other) {
        reset();
        base_     = std::exchange(other.base_, 0);
        bytes_    = std::exchange(other.bytes_, 0);
        granule_  = std::exchange(other.granule_, 0);
        mapped_   = std::exchange(other.mapped_, 0);
        options_  = std::move(other.options_);
        access_   = std::move(other.access_);
        handles_  = std::move(other.handles_);
        wanted_   = std::move(other.wanted_);
        released_ = std::exchange(other.released_, false);
    }
    return *this;
}

void GrowableRegion::reset() noexcept {
    if (base_ == 0) { return; }
    for (std::size_t index = 0; index < handles_.size(); ++index) { unmap_granule(index); }
    (void)cuMemAddressFree(base_, bytes_);
    base_ = 0;
}

bool GrowableRegion::map_granule(std::size_t index) {
    if (handles_[index] != 0) { return true; }
    CUmemAllocationProp prop{};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = options_.device;
    CUmemGenericAllocationHandle handle{};
    const CUresult created = cuMemCreate(&handle, granule_, &prop, 0);
    if (created == CUDA_ERROR_OUT_OF_MEMORY) { return false; }
    NINFER_CU_CHECK(created);
    try {
        vmm::map(base_ + index * granule_, granule_, handle, access_);
    } catch (...) {
        (void)cuMemRelease(handle);
        throw;
    }
    handles_[index] = handle;
    ++mapped_;
    return true;
}

void GrowableRegion::unmap_granule(std::size_t index) noexcept {
    if (handles_[index] == 0) { return; }
    (void)vmm::unmap(base_ + index * granule_, granule_);
    (void)cuMemRelease(handles_[index]);
    handles_[index] = 0;
    --mapped_;
}

bool GrowableRegion::map(std::size_t offset, std::size_t bytes, cudaStream_t zero) {
    if (bytes == 0) { return true; }
    if (offset > bytes_ || bytes > bytes_ - offset) {
        throw std::out_of_range("growable region: a range past the reservation");
    }
    if (released_) { throw std::logic_error("growable region: backing is released"); }
    const std::size_t first = offset / granule_;
    const std::size_t end   = (offset + bytes + granule_ - 1) / granule_;
    std::vector<std::size_t> created;
    for (std::size_t index = first; index < end; ++index) {
        if (handles_[index] != 0) { continue; }
        bool ok = false;
        try {
            ok = map_granule(index);
        } catch (...) {
            for (const std::size_t undo : created) { unmap_granule(undo); }
            throw;
        }
        if (!ok) {
            for (const std::size_t undo : created) { unmap_granule(undo); }
            return false;
        }
        created.push_back(index);
    }
    for (const std::size_t index : created) { wanted_[index] = true; }
    if (zero != nullptr) {
        // One memset per run of consecutive granules.
        for (std::size_t i = 0; i < created.size();) {
            std::size_t j = i + 1;
            while (j < created.size() && created[j] == created[j - 1] + 1) { ++j; }
            CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(base_ + created[i] * granule_), 0,
                                       (j - i) * granule_, zero));
            i = j;
        }
    }
    return true;
}

void GrowableRegion::unmap(std::size_t offset, std::size_t bytes) {
    if (bytes == 0 || offset >= bytes_) { return; }
    bytes                   = std::min(bytes, bytes_ - offset);
    const std::size_t first = (offset + granule_ - 1) / granule_;
    const std::size_t end   = (offset + bytes) / granule_;
    for (std::size_t index = first; index < end; ++index) {
        unmap_granule(index);
        wanted_[index] = false;
    }
}

bool GrowableRegion::mapped(std::size_t offset, std::size_t bytes) const noexcept {
    if (bytes == 0) { return true; }
    if (offset > bytes_ || bytes > bytes_ - offset || released_) { return false; }
    const std::size_t end = (offset + bytes + granule_ - 1) / granule_;
    for (std::size_t index = offset / granule_; index < end; ++index) {
        if (handles_[index] == 0) { return false; }
    }
    return true;
}

std::size_t GrowableRegion::unmapped_bytes(std::size_t offset, std::size_t bytes) const noexcept {
    if (bytes == 0 || offset >= bytes_) { return 0; }
    bytes                 = std::min(bytes, bytes_ - offset);
    const std::size_t end = (offset + bytes + granule_ - 1) / granule_;
    std::size_t missing   = 0;
    for (std::size_t index = offset / granule_; index < end; ++index) {
        missing += handles_[index] == 0 ? granule_ : 0;
    }
    return missing;
}

void GrowableRegion::release_backing() {
    if (released_) { return; }
    for (std::size_t index = 0; index < handles_.size(); ++index) { unmap_granule(index); }
    released_ = true;
}

void GrowableRegion::restore_backing() {
    if (!released_) { return; }
    try {
        for (std::size_t index = 0; index < wanted_.size(); ++index) {
            if (wanted_[index] && !map_granule(index)) {
                throw std::runtime_error("growable region: the device cannot restore " +
                                         std::to_string(granule_ >> 20) + " MiB pieces");
            }
        }
    } catch (...) {
        for (std::size_t index = 0; index < handles_.size(); ++index) { unmap_granule(index); }
        throw;
    }
    released_ = false;
}

} // namespace ninfer
