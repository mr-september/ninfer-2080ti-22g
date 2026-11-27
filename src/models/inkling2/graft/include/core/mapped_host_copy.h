#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace ninfer {

// Whether the driver model queues a device's memory copies behind one another whatever their
// streams: WDDM, which is every Windows build and, through /dev/dxg, a Linux build under WSL2.
// There a copy a stream queued behind a kernel that waits for a host worker also holds back that
// worker's own copies on another stream, so the two wait for each other. Work that puts a
// host-to-device copy behind such a wait uses MappedHostBuffer instead where this is true.
[[nodiscard]] bool serialized_copy_queue() noexcept;

// Pinned host memory that every device also maps into its address space, so a kernel reads it in
// place: its copy to device memory is an exact, capturable kernel launch on the compute queue.
class MappedHostBuffer {
public:
    explicit MappedHostBuffer(std::size_t bytes);
    ~MappedHostBuffer();
    MappedHostBuffer(const MappedHostBuffer&)            = delete;
    MappedHostBuffer& operator=(const MappedHostBuffer&) = delete;

    [[nodiscard]] void* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    // Copies the first `bytes` to `destination`, memory of the current device, with a kernel on
    // `stream`. The kernel reads the host bytes when it runs, as a queued cudaMemcpyAsync does.
    void copy_to_device(void* destination, std::size_t bytes, cudaStream_t stream) const;

private:
    void* data_       = nullptr;
    std::size_t size_ = 0;
};

} // namespace ninfer
