#include "core/vmm.h"

#include "core/device_wait.h"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer {
namespace vmm {

void check(CUresult result, const char* expression) {
    if (result == CUDA_SUCCESS) { return; }
    const char* name = nullptr;
    (void)cuGetErrorName(result, &name);
    throw std::runtime_error(std::string(expression) + " failed: " +
                             (name != nullptr ? name : "unknown CUresult"));
}

void ensure_driver_initialized() {
    static std::once_flag once;
    std::call_once(once, [] { NINFER_CU_CHECK(cuInit(0)); });
}

bool supported(int device) {
    ensure_driver_initialized();
    CUdevice handle = 0;
    if (cuDeviceGet(&handle, device) != CUDA_SUCCESS) { return false; }
    int value = 0;
    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                             handle) != CUDA_SUCCESS) {
        return false;
    }
    return value != 0;
}

namespace {

CUmemAllocationProp allocation_properties(int device) {
    CUmemAllocationProp prop{};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = device;
    return prop;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

} // namespace

std::size_t granularity(int device) {
    if (!supported(device)) { return 0; }
    const CUmemAllocationProp prop = allocation_properties(device);
    std::size_t value              = 0;
    if (cuMemGetAllocationGranularity(&value, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) !=
        CUDA_SUCCESS) {
        return 0;
    }
    return value;
}

CUmemGenericAllocationHandle create(std::size_t bytes, int device) {
    const CUmemAllocationProp prop = allocation_properties(device);
    CUmemGenericAllocationHandle handle{};
    NINFER_CU_CHECK(cuMemCreate(&handle, bytes, &prop, 0));
    return handle;
}

void set_access(CUdeviceptr va, std::size_t bytes, std::span<const int> devices) {
    std::vector<CUmemAccessDesc> access(devices.size());
    for (std::size_t index = 0; index < devices.size(); ++index) {
        access[index].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access[index].location.id   = devices[index];
        access[index].flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    const DeviceWaitScope waiting("a change of memory access");
    NINFER_CU_CHECK(cuMemSetAccess(va, bytes, access.data(), access.size()));
}

CUresult unmap(CUdeviceptr va, std::size_t bytes) noexcept {
    const DeviceWaitScope waiting("an unmap");
    return cuMemUnmap(va, bytes);
}

void map(CUdeviceptr va, std::size_t bytes, CUmemGenericAllocationHandle handle,
         std::span<const int> devices) {
    NINFER_CU_CHECK(cuMemMap(va, bytes, 0, handle, 0));
    try {
        set_access(va, bytes, devices);
    } catch (...) {
        (void)vmm::unmap(va, bytes);
        throw;
    }
}

} // namespace vmm

namespace {

constexpr std::size_t kDefaultPieceBytes = 256ULL * 1024ULL * 1024ULL;

} // namespace

VmmRegion::VmmRegion(std::size_t bytes, Options options) : options_(std::move(options)) {
    if (bytes == 0) { throw std::invalid_argument("a suspendable region must not be empty"); }
    const std::size_t granule = vmm::granularity(options_.device);
    if (granule == 0) {
        throw std::invalid_argument(
            "model suspend requires CUDA virtual memory management on device " +
            std::to_string(options_.device));
    }
    piece_ = options_.piece_bytes == 0 ? kDefaultPieceBytes : options_.piece_bytes;
    piece_ = (piece_ + granule - 1) / granule * granule;
    bytes_ = bytes;
    // The reservation is rounded to whole granules only; the last piece is shortened to it, so a
    // small region does not hold a full piece of physical memory.
    reserved_ = (bytes + granule - 1) / granule * granule;
    access_.push_back(options_.device);
    for (const int peer : options_.peers) {
        if (std::find(access_.begin(), access_.end(), peer) == access_.end()) {
            access_.push_back(peer);
        }
    }
    vmm::ensure_driver_initialized();
    NINFER_CU_CHECK(cuMemAddressReserve(&base_, reserved_, granule, 0, 0));
    try {
        restore_backing();
    } catch (...) {
        (void)cuMemAddressFree(base_, reserved_);
        base_ = 0;
        throw;
    }
}

VmmRegion::~VmmRegion() { reset(); }

VmmRegion::VmmRegion(VmmRegion&& other) noexcept
    : base_(std::exchange(other.base_, 0)), bytes_(std::exchange(other.bytes_, 0)),
      reserved_(std::exchange(other.reserved_, 0)), piece_(std::exchange(other.piece_, 0)),
      options_(std::move(other.options_)), access_(std::move(other.access_)),
      handles_(std::move(other.handles_)), backed_(std::exchange(other.backed_, false)) {}

VmmRegion& VmmRegion::operator=(VmmRegion&& other) noexcept {
    if (this != &other) {
        reset();
        base_     = std::exchange(other.base_, 0);
        bytes_    = std::exchange(other.bytes_, 0);
        reserved_ = std::exchange(other.reserved_, 0);
        piece_    = std::exchange(other.piece_, 0);
        options_  = std::move(other.options_);
        access_   = std::move(other.access_);
        handles_  = std::move(other.handles_);
        backed_   = std::exchange(other.backed_, false);
    }
    return *this;
}

void VmmRegion::reset() noexcept {
    if (base_ == 0) { return; }
    try {
        release_backing();
    } catch (...) {}
    (void)cuMemAddressFree(base_, reserved_);
    base_   = 0;
    backed_ = false;
}

DeviceSpan VmmRegion::span() const noexcept {
    return DeviceSpan{reinterpret_cast<void*>(base_), bytes_};
}

void VmmRegion::release_backing() {
    if (!backed_) { return; }
    std::size_t offset = 0;
    for (const CUmemGenericAllocationHandle handle : handles_) {
        const std::size_t size = std::min(piece_, reserved_ - offset);
        (void)vmm::unmap(base_ + offset, size);
        (void)cuMemRelease(handle);
        offset += size;
    }
    handles_.clear();
    backed_ = false;
}

void VmmRegion::restore_backing() {
    if (backed_) { return; }
    handles_.clear();
    std::size_t offset = 0;
    try {
        while (offset < reserved_) {
            const std::size_t size = std::min(piece_, reserved_ - offset);
            const CUmemGenericAllocationHandle handle = vmm::create(size, options_.device);
            try {
                vmm::map(base_ + offset, size, handle, access_);
            } catch (...) {
                (void)cuMemRelease(handle);
                throw;
            }
            handles_.push_back(handle);
            offset += size;
        }
    } catch (...) {
        std::size_t undo = 0;
        for (const CUmemGenericAllocationHandle handle : handles_) {
            const std::size_t size = std::min(piece_, reserved_ - undo);
            (void)vmm::unmap(base_ + undo, size);
            (void)cuMemRelease(handle);
            undo += size;
        }
        handles_.clear();
        throw;
    }
    backed_ = true;
}

} // namespace ninfer
