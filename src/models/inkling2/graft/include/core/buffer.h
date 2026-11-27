#pragma once

// graft shim: one owning-buffer surface for grafted translation units. ninfer's DeviceBuffer
// and PinnedHostBuffer (core/buffer.h in the destination tree) are DIFFERENT classes from the
// donor's (port: private data_/bytes_/device_ and a (Device,size) ctor; donor: public p/bytes,
// bytes-only ctor) — same mangled names, incompatible layouts. Sharing one object file between
// them corrupted silently, and 10x LNK2005 made it loud (link round 17). Resolution, per
// "the destination owns the name": the donor-shaped classes live in ninfer::graft and are
// aliased back into ninfer for grafted TUs only. A TU that includes BOTH headers gets an
// ambiguity error — the guardrail is intentional: un-namespace one of them for real.
// HostBuffer and CudaHostRegistration stay the port's, as the destination tree's headers name
// them; their definitions are copied from the port's core/buffer.h and must keep that layout.

#include "core/memory.h"

#include <cstddef>

namespace ninfer {

class HostBuffer {
public:
    HostBuffer() noexcept = default;
    explicit HostBuffer(std::size_t bytes, std::size_t alignment = 256);
    ~HostBuffer();
    HostBuffer(const HostBuffer&)            = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
    HostBuffer(HostBuffer&& other) noexcept;
    HostBuffer& operator=(HostBuffer&& other) noexcept;

    [[nodiscard]] void* data() noexcept { return data_; }

    [[nodiscard]] const void* data() const noexcept { return data_; }

    [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

    [[nodiscard]] Device device() const noexcept { return Device::cpu(); }

    [[nodiscard]] MemorySpan span() noexcept { return {data_, bytes_, device()}; }

    [[nodiscard]] ConstMemorySpan span() const noexcept { return {data_, bytes_, device()}; }

private:
    void* data_            = nullptr;
    std::size_t bytes_     = 0;
    std::size_t alignment_ = 256;
};

// Owns CUDA registration, not the Host allocation. The caller finishes all uses
// before unregistering and keeps the original Host backing alive until then.
class CudaHostRegistration {
public:
    CudaHostRegistration() noexcept = default;
    CudaHostRegistration(ConstMemorySpan host, Device cuda_device);
    ~CudaHostRegistration();
    CudaHostRegistration(const CudaHostRegistration&)            = delete;
    CudaHostRegistration& operator=(const CudaHostRegistration&) = delete;
    CudaHostRegistration(CudaHostRegistration&& other) noexcept;
    CudaHostRegistration& operator=(CudaHostRegistration&& other) noexcept;

    [[nodiscard]] CudaHostView view() const noexcept { return view_; }

private:
    CudaHostView view_;
};

// --- Donor-shaped owning buffers: ninfer::graft, aliased back for grafted TUs (see top) ---------
namespace graft {

class DeviceBuffer {
public:
    DeviceBuffer() noexcept = default;
    explicit DeviceBuffer(std::size_t size_bytes);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    void fill(int byte_value = 0);
    // Completes the upload before returning. Callers must first order any prior device
    // accesses to the destination range.
    void copy_from_host(const void* source, std::size_t count, std::size_t byte_offset = 0);
    void copy_to_host(void* destination, std::size_t count, std::size_t byte_offset = 0) const;

    // Raw access is intentional: Tensor and Weight are non-owning views.
    void* p           = nullptr;
    std::size_t bytes = 0;

    // The destination tree's accessor names, aliases of the public members above.
    [[nodiscard]] void* data() noexcept { return p; }
    [[nodiscard]] const void* data() const noexcept { return p; }
    [[nodiscard]] std::size_t size() const noexcept { return bytes; }

private:
    void require_range(std::size_t byte_offset, std::size_t count, const char* operation) const;
};

class PinnedHostBuffer {
public:
    explicit PinnedHostBuffer(std::size_t size_bytes);
    ~PinnedHostBuffer();

    PinnedHostBuffer(const PinnedHostBuffer&)            = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;

    void* data() const noexcept;
    std::size_t size() const noexcept;

private:
    void* data_       = nullptr;
    std::size_t size_ = 0;
};

// Registers an existing host allocation without moving its bytes. The allocation must outlive
// the registration and every queued transfer; portable registration serves every device rank.
// `mapped` also maps it into the device address space, for kernels that read it in place
// (cudaHostGetDevicePointer gives the address they use).
class HostMemoryRegistration {
public:
    HostMemoryRegistration(void* data, std::size_t bytes, bool mapped = false);
    ~HostMemoryRegistration();
    HostMemoryRegistration(const HostMemoryRegistration&) = delete;
    HostMemoryRegistration& operator=(const HostMemoryRegistration&) = delete;

private:
    void* data_ = nullptr;
};

} // namespace graft

using graft::DeviceBuffer;
using graft::PinnedHostBuffer;
using graft::HostMemoryRegistration;

} // namespace ninfer
