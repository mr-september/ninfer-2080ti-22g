#include "core/arena.h"

#include "core/device_buffer.h"
#include "core/device_wait.h"
#include "core/pinned_memory.h"
#include "core/wddm_budget.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <string>

namespace ninfer {
namespace {

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

bool is_power_of_two(std::size_t value) { return value != 0 && (value & (value - 1)) == 0; }

std::uintptr_t checked_add_uintptr(std::uintptr_t a, std::size_t b) {
    if (b > std::numeric_limits<std::uintptr_t>::max() - a) {
        throw std::overflow_error("arena address arithmetic overflow");
    }
    return a + b;
}

std::uintptr_t align_up_addr(std::uintptr_t addr, std::size_t align) {
    const std::size_t mask         = align - 1;
    const std::uintptr_t with_mask = checked_add_uintptr(addr, mask);
    return with_mask & ~static_cast<std::uintptr_t>(mask);
}

void free_device(void*& ptr) noexcept {
    if (ptr != nullptr) {
        log_cuda_error("cudaFree", ninfer::free_device_memory(ptr));
        ptr = nullptr;
    }
}

void free_pinned(void*& ptr) noexcept {
    if (ptr != nullptr) {
        log_cuda_error("cudaFreeHost", ninfer::free_host_memory(ptr));
        ptr = nullptr;
    }
}

std::atomic<bool> g_wddm_evictable_budget{false};

} // namespace

namespace core {

void set_wddm_evictable_budget(bool enabled) {
#if defined(_WIN32)
    g_wddm_evictable_budget.store(enabled, std::memory_order_relaxed);
#else
    if (enabled) {
        throw std::invalid_argument("--wddm-evictable-budget needs a Windows build");
    }
#endif
}

bool wddm_evictable_budget() noexcept {
    return g_wddm_evictable_budget.load(std::memory_order_relaxed);
}

} // namespace core

namespace graft {
DeviceBuffer::DeviceBuffer(std::size_t size_bytes) : bytes(size_bytes) {
    if (bytes == 0) { return; }

    void* ptr             = nullptr;
    const cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMalloc failed", err));
    }
    p = ptr;
}

DeviceBuffer::~DeviceBuffer() { free_device(p); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept : p(other.p), bytes(other.bytes) {
    other.p     = nullptr;
    other.bytes = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) { return *this; }

    free_device(p);
    p     = other.p;
    bytes = other.bytes;

    other.p     = nullptr;
    other.bytes = 0;
    return *this;
}

// `fill_and_wait` waits for the legacy stream that `cudaMemset` runs on, so the zeros are visible
// to the non-blocking streams the caller uses next.
void DeviceBuffer::fill(int byte_value) {
    if (bytes == 0) { return; }
    const cudaError_t err = ninfer::fill_and_wait(p, byte_value, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMemset failed", err));
    }
}

// `cudaMemcpy` does not finish the transfer before it returns. CUDA's documented
// synchronization behaviour for a host-to-device copy out of *pageable* memory is that the call
// returns once the source has been staged for DMA, "but the DMA to final destination may not have
// completed". That trailing DMA rides the legacy stream, and a stream created with
// `cudaStreamNonBlocking` -- which is every stream `DeviceContext` owns -- is exempt from the
// legacy stream's implicit ordering. So work enqueued on one of those streams immediately after
// a bare `cudaMemcpy` can, and does, read the destination before the copy lands.
//
// Measured on this RTX 3090 with a faithful replica of the sequence
// tests/ops/test_attn_input_proj.cpp runs at graph replay phase 1 -- a 1,146,880-byte pageable
// H2D, three `cudaMemsetAsync` on a non-blocking stream, then a captured graph launched on it --
// the reader saw stale bytes in 146 of 200 iterations, up to 17.8% of the buffer, on an
// *otherwise idle* GPU. Copies at or below ~256 KiB never raced; every size above it did.
// tests/test_device_buffer_visibility.cu is that experiment, reduced.
//
// `copy_and_wait` therefore synchronizes the legacy stream after the copy: on return, the bytes
// are visible to any stream. Callers wanting an overlapped copy should use an explicitly
// stream-ordered one (`cudaMemcpy2DAsync`, as the paged-KV and state-image pools already do)
// rather than this.
void DeviceBuffer::copy_from_host(const void* source, std::size_t count, std::size_t byte_offset) {
    require_range(byte_offset, count, "host-to-device copy");
    if (count == 0) { return; }
    void* destination     = static_cast<std::uint8_t*>(p) + byte_offset;
    const cudaError_t err =
        ninfer::copy_and_wait(destination, source, count, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMemcpy host-to-device failed", err));
    }
}

void DeviceBuffer::copy_to_host(void* destination, std::size_t count,
                                std::size_t byte_offset) const {
    require_range(byte_offset, count, "device-to-host copy");
    if (count == 0) { return; }
    const void* source    = static_cast<const std::uint8_t*>(p) + byte_offset;
    const cudaError_t err =
        ninfer::copy_and_wait(destination, source, count, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMemcpy device-to-host failed", err));
    }
}

void DeviceBuffer::require_range(std::size_t byte_offset, std::size_t count,
                                 const char* operation) const {
    if (byte_offset > bytes || count > bytes - byte_offset) {
        throw std::out_of_range(std::string(operation) + " exceeds device buffer");
    }
}

} // namespace graft
DeviceArena::Scope::Scope(DeviceArena& arena) noexcept
    : arena_(&arena), saved_offset_(arena.off_), saved_rank_(arena.active_rank_) {}

DeviceArena::Scope::~Scope() noexcept {
    if (arena_ == nullptr) { return; }
    if (saved_rank_ == arena_->active_rank_) {
        if (saved_offset_ <= arena_->off_) { arena_->off_ = saved_offset_; }
        return;
    }
    // The arena moved to another rank while this scope was open. Roll back the rank the scope was
    // actually taken on, not whichever one happens to be active now -- restoring a foreign bump
    // pointer would hand out overlapping workspace on the next allocation.
    if (saved_rank_ < arena_->ranks_.size()) {
        RankBacking& backing = arena_->ranks_[saved_rank_];
        if (saved_offset_ <= backing.off) { backing.off = saved_offset_; }
    }
}

DeviceArena::Scope::Scope(Scope&& other) noexcept
    : arena_(other.arena_), saved_offset_(other.saved_offset_), saved_rank_(other.saved_rank_) {
    other.arena_ = nullptr;
}

DeviceArena::DeviceArena(std::size_t capacity_bytes) {
    if (capacity_bytes == 0) {
        throw std::invalid_argument("DeviceArena capacity must be nonzero");
    }

    void* ptr             = nullptr;
    const cudaError_t err = cudaMalloc(&ptr, capacity_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaMalloc failed", err));
    }
    // Under the evictable budget, commit the pages now so WDDM makes room at startup; the fill
    // completes before uploads on the non-blocking load stream.
    if (core::wddm_evictable_budget()) {
        const cudaError_t done = ninfer::fill_and_wait(ptr, 0, capacity_bytes);
        if (done != cudaSuccess) {
            log_cuda_error("cudaFree", ninfer::free_device_memory(ptr));
            throw std::runtime_error(cuda_error_message("cudaMemset failed", done));
        }
    }

    base_       = ptr;
    owned_base_ = ptr;
    cap_        = capacity_bytes;
    off_        = 0;
}

DeviceArena::DeviceArena(DeviceSpan storage)
    : base_(storage.data), cap_(storage.bytes), owns_(false) {
    if (base_ == nullptr || cap_ == 0) {
        throw std::invalid_argument("borrowed DeviceArena storage must be non-empty");
    }
}

DeviceArena::~DeviceArena() {
    if (owns_) { free_device(owned_base_); }
}

DeviceArena::DeviceArena(DeviceArena&& other) noexcept
    : base_(other.base_), cap_(other.cap_), off_(other.off_), peak_(other.peak_),
      owns_(other.owns_), owned_base_(other.owned_base_), ranks_(std::move(other.ranks_)),
      active_rank_(other.active_rank_) {
    other.base_        = nullptr;
    other.cap_         = 0;
    other.off_         = 0;
    other.peak_        = 0;
    other.owns_        = true;
    other.owned_base_  = nullptr;
    other.ranks_.clear();
    other.active_rank_ = 0;
}

DeviceArena& DeviceArena::operator=(DeviceArena&& other) noexcept {
    if (this == &other) { return *this; }

    if (owns_) { free_device(owned_base_); }
    base_       = other.base_;
    cap_        = other.cap_;
    off_        = other.off_;
    peak_       = other.peak_;
    owns_       = other.owns_;
    owned_base_ = other.owned_base_;

    ranks_       = std::move(other.ranks_);
    active_rank_ = other.active_rank_;

    other.base_        = nullptr;
    other.cap_         = 0;
    other.off_         = 0;
    other.peak_        = 0;
    other.owns_        = true;
    other.owned_base_  = nullptr;
    other.ranks_.clear();
    other.active_rank_ = 0;
    return *this;
}

DeviceSpan DeviceArena::alloc_bytes(std::size_t bytes, std::size_t align) {
    if (base_ == nullptr) { throw std::runtime_error("DeviceArena has no backing allocation"); }
    if (!is_power_of_two(align)) {
        throw std::invalid_argument("arena alignment must be a nonzero power of two");
    }
    if (bytes == 0) { throw std::invalid_argument("arena allocation must be nonzero"); }

    const std::uintptr_t base_addr           = reinterpret_cast<std::uintptr_t>(base_);
    const std::uintptr_t current_addr        = checked_add_uintptr(base_addr, off_);
    const std::uintptr_t aligned_addr        = align_up_addr(current_addr, align);
    const std::uintptr_t aligned_offset_addr = aligned_addr - base_addr;
    if (aligned_offset_addr > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("arena aligned offset overflows size_t");
    }
    const auto aligned_offset = static_cast<std::size_t>(aligned_offset_addr);
    if (bytes > std::numeric_limits<std::size_t>::max() - aligned_offset) {
        throw std::overflow_error("arena allocation end offset overflows size_t");
    }
    const std::size_t end = aligned_offset + bytes;
    if (end > cap_) { throw std::bad_alloc(); }

    auto* ptr = static_cast<unsigned char*>(base_) + aligned_offset;
    off_      = end;
    if (off_ > peak_) { peak_ = off_; }
    return DeviceSpan{ptr, bytes};
}

Tensor DeviceArena::alloc(DType dtype, std::initializer_list<std::int32_t> shape,
                          std::size_t align) {
    Tensor view(nullptr, dtype, shape);
    const DeviceSpan storage = alloc_bytes(view.bytes(), align);
    return Tensor(storage.data, dtype, shape);
}

DeviceArena::Scope DeviceArena::scope() noexcept { return Scope(*this); }

void DeviceArena::reset() noexcept {
    off_ = 0;
    for (RankBacking& backing : ranks_) { backing.off = 0; }
}

void* DeviceArena::base() const noexcept { return base_; }

std::size_t DeviceArena::used() const noexcept { return off_; }

std::size_t DeviceArena::capacity() const noexcept { return cap_; }

std::size_t DeviceArena::peak_used() const noexcept { return peak_; }

void DeviceArena::reset_peak() noexcept { peak_ = off_; }

void DeviceArena::store_active_rank() noexcept {
    if (active_rank_ < ranks_.size()) {
        RankBacking& backing = ranks_[active_rank_];
        backing.base         = base_;
        backing.cap          = cap_;
        backing.off          = off_;
        backing.peak         = peak_;
    }
}

void DeviceArena::attach_rank_storage(DeviceSpan storage) {
    if (storage.data == nullptr || storage.bytes == 0) {
        throw std::invalid_argument("attached rank storage must be a nonempty device span");
    }
    if (ranks_.empty()) {
        // First attach materializes rank 0 from the arena's own storage, so indices line up with
        // pipeline ranks from here on.
        ranks_.push_back(RankBacking{base_, cap_, off_, peak_});
        active_rank_ = 0;
    }
    ranks_.push_back(RankBacking{storage.data, storage.bytes, 0, 0});
}

void DeviceArena::activate_rank(std::size_t rank) {
    if (ranks_.empty()) {
        // Single-rank arenas only ever have rank 0; anything else is a planning bug worth hearing
        // about rather than silently ignoring.
        if (rank != 0) { throw std::out_of_range("arena has no storage for that pipeline rank"); }
        return;
    }
    if (rank >= ranks_.size()) {
        throw std::out_of_range("arena has no storage for that pipeline rank");
    }
    if (rank == active_rank_) { return; }

    store_active_rank();
    const RankBacking& next = ranks_[rank];
    base_                   = next.base;
    cap_                    = next.cap;
    off_                    = next.off;
    peak_                   = next.peak;
    active_rank_            = rank;
}

std::size_t DeviceArena::active_rank() const noexcept { return active_rank_; }

std::size_t DeviceArena::rank_count() const noexcept {
    return ranks_.empty() ? 1U : ranks_.size();
}

std::size_t DeviceArena::peak_used_for_rank(std::size_t rank) const {
    if (ranks_.empty()) {
        if (rank != 0) { throw std::out_of_range("arena has no storage for that pipeline rank"); }
        return peak_;
    }
    if (rank >= ranks_.size()) {
        throw std::out_of_range("arena has no storage for that pipeline rank");
    }
    // The active rank's live counters have not been written back yet.
    return rank == active_rank_ ? peak_ : ranks_[rank].peak;
}

ScopedArenaRank::ScopedArenaRank(DeviceArena& arena, std::size_t rank)
    : arena_(arena), previous_rank_(arena.active_rank()) {
    arena_.activate_rank(rank);
}

ScopedArenaRank::~ScopedArenaRank() noexcept {
    if (arena_.active_rank() != previous_rank_) { arena_.activate_rank(previous_rank_); }
}

namespace graft {
PinnedHostBuffer::PinnedHostBuffer(std::size_t size_bytes) {
    if (size_bytes == 0) { throw std::invalid_argument("PinnedHostBuffer size must be nonzero"); }

    // A pending error from an earlier call is returned by whatever runs next, so read and clear it
    // first. Without this, a failure here can be somebody else's error wearing this message.
    const cudaError_t pending = cudaGetLastError();

    // Portable, so the buffer counts as pinned in every device's context rather than only the one
    // that was current when it was allocated. With several ranks, ingress frames are copied to each
    // stage's device from one host buffer, and a copy that treats it as pageable is neither fast nor
    // capturable. (The error text below keeps its `cudaMallocHost` wording: launcher docs and the
    // test guard match on it.)
    void* ptr             = nullptr;
    const cudaError_t err = cudaHostAlloc(&ptr, size_bytes, cudaHostAllocPortable);
    if (err != cudaSuccess) {
        // Say the size, and say which memory. `cudaErrorMemoryAllocation: out of memory` reads as
        // a VRAM shortfall and sends people to nvidia-smi when the shortfall is pinned system RAM.
        // But the converse also happens and used to be asserted away here: on Windows/WDDM this
        // allocation is mapped into the GPU's address space and charged against the card, so a
        // large pin fails once the device is nearly full even with tens of GiB of system RAM free
        // -- measured on a 24 GiB RTX 3090, where resident-device plus pinned-host lands within a
        // few hundred MiB of the card's capacity every time. `cudaErrorAlreadyMapped` is the
        // failure that shape produces. Report free VRAM so the two are distinguishable rather than
        // asserting which one it is.
        std::size_t free_device = 0, total_device = 0;
        const cudaError_t info = cudaMemGetInfo(&free_device, &total_device);
        std::string prefix     = "cudaMallocHost failed to pin " +
                             std::to_string((size_bytes + (1ULL << 20) - 1) >> 20) +
                             " MiB of host memory";
        if (info == cudaSuccess) {
            prefix += " (device has " + std::to_string(free_device >> 20) + " MiB of " +
                      std::to_string(total_device >> 20) + " MiB free; on Windows a pinned host "
                      "allocation is mapped into the GPU address space and competes with it)";
        } else {
            prefix += " (system RAM, pinned)";
        }
        if (pending != cudaSuccess) {
            prefix += ", after an unretrieved earlier error (";
            prefix += cudaGetErrorName(pending);
            prefix += ")";
        }
        throw std::runtime_error(cuda_error_message(prefix.c_str(), err));
    }

    // The pending error read above is now gone for good -- `cudaGetLastError()` clears it, and
    // nothing else has seen it since. A successful pin does not mean it never mattered: it means
    // an earlier, unrelated CUDA call failed and nobody checked. Report it rather than let it
    // disappear silently.
    if (pending != cudaSuccess) {
        std::fprintf(stderr,
                      "ninfer: pinning %zu bytes of host memory succeeded, but cleared an "
                      "unretrieved earlier CUDA error: %s: %s\n",
                      size_bytes, cudaGetErrorName(pending), cudaGetErrorString(pending));
    }

    data_ = ptr;
    size_ = size_bytes;
}

PinnedHostBuffer::~PinnedHostBuffer() { free_pinned(data_); }

HostMemoryRegistration::HostMemoryRegistration(void* data, std::size_t bytes, bool mapped) {
    if (!data || !bytes) { throw std::invalid_argument("host registration needs a nonempty allocation"); }
    const unsigned flags = cudaHostRegisterPortable | (mapped ? cudaHostRegisterMapped : 0U);
    const cudaError_t error = cudaHostRegister(data, bytes, flags);
    if (error != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaHostRegister failed", error));
    }
    data_ = data;
}
HostMemoryRegistration::~HostMemoryRegistration() {
    if (data_) { log_cuda_error("cudaHostUnregister", ninfer::unregister_host_memory(data_)); }
}

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
    if (this == &other) { return *this; }

    free_pinned(data_);
    data_ = other.data_;
    size_ = other.size_;

    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
}

void* PinnedHostBuffer::data() const noexcept { return data_; }

std::size_t PinnedHostBuffer::size() const noexcept { return size_; }

} // namespace graft
} // namespace ninfer
