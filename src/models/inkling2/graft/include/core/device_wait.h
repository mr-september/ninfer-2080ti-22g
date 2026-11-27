#pragma once

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>

namespace ninfer {

struct DeviceContext;
struct DeviceWaitState;

// CUDA reports a device that stops finishing work as no error: a thread that waits for it stays
// inside the driver for good, while every other thread of the process keeps answering (on
// 2026-11-07 an Engine worker waited 46 minutes in cudaEventSynchronize with the GPU idle and
// /health answering ok). Two watches end such a process, the way CUDA_CHECK ends it on a CUDA
// error, so that its supervisor starts a new one, which recovers the device:
//
// - Every thread's device waits. Code that waits for the device calls the functions below in place
//   of the CUDA calls they name. A wait that lasts the stall limit while no host service of its
//   devices answers them ends the process with a message that names the thread, the devices and
//   the kind of wait. A thread that holds a DeviceWaitWatch is named by it; any other one by its
//   operating-system thread id.
// - Every device of a DeviceContext. One thread of the process submits a probe to each such device
//   every few seconds on a stream of its own: a copy to the device, a kernel and a copy back. A
//   probe the device has not finished within the stall limit while no host service answered the
//   device ends the process, whatever its other threads wait in: a kernel launch on a full queue,
//   a copy from pageable memory, or nothing at all on an idle server.
//
// NINFER_DEVICE_STALL_SECONDS sets the stall limit (kStallLimit unless set) for diagnosis under
// tools that slow the device many times over (Nsight Compute, compute-sanitizer); 0 turns both
// watches off. A watch constructed with a limit of its own keeps it.

// Waits for device work. Each returns what the CUDA call it names returns, or the error of the
// wait that follows it.
cudaError_t wait_for_stream(cudaStream_t stream) noexcept;    // cudaStreamSynchronize
cudaError_t wait_for_event(cudaEvent_t event) noexcept;       // cudaEventSynchronize
cudaError_t wait_for_device() noexcept;                       // cudaDeviceSynchronize
// cudaMemcpy, cudaMemcpyPeer and cudaMemset return before a copy out of pageable memory, a copy
// between devices or a fill of device memory lands, and that work runs on the legacy stream, which
// DeviceContext's non-blocking streams do not wait for. These three also wait for the legacy
// stream, so on return the bytes are visible to work on any stream.
cudaError_t copy_and_wait(void* destination, const void* source, std::size_t bytes,
                          cudaMemcpyKind kind) noexcept;      // cudaMemcpy
cudaError_t copy_between_devices_and_wait(void* destination, int destination_device,
                                          const void* source, int source_device,
                                          std::size_t bytes) noexcept; // cudaMemcpyPeer
cudaError_t fill_and_wait(void* destination, int value, std::size_t bytes) noexcept; // cudaMemset

// Calls that may wait for device work without saying so in their names.
cudaError_t free_device_memory(void* memory) noexcept;     // cudaFree
cudaError_t free_host_memory(void* memory) noexcept;       // cudaFreeHost
cudaError_t unregister_host_memory(void* memory) noexcept; // cudaHostUnregister
// cudaGraphLaunch, which waits while the stream's launch queue is full.
cudaError_t launch_graph(cudaGraphExec_t graph, cudaStream_t stream) noexcept;

// Marks the scope of any other call that may wait for device work (cuMemUnmap, cuMemSetAccess) as
// a device wait of the calling thread. `kind` names the call in a stall message and outlives the
// scope.
class DeviceWaitScope {
public:
    explicit DeviceWaitScope(const char* kind) noexcept;
    ~DeviceWaitScope();
    DeviceWaitScope(const DeviceWaitScope&)            = delete;
    DeviceWaitScope& operator=(const DeviceWaitScope&) = delete;

private:
    DeviceWaitState* state_;
};

// A host service answered a request of CUDA device `device` (CudaHostHandshake: host or disk
// experts, a CPU share of expert products). The device may wait for such answers inside a watched
// wait, so the wait is not a stall while they continue.
void note_host_service_progress(int device) noexcept;

// The stall limit of every watch that names none: NINFER_DEVICE_STALL_SECONDS, or kStallLimit
// while it is unset; zero turns the watches off. Throws std::invalid_argument when the variable is
// not a whole number of seconds.
[[nodiscard]] std::chrono::milliseconds device_stall_limit();

// Keep CUDA device `device` probed while one or more holders hold it. `holder` is any address
// unique to the holder while it holds (a DeviceContext rank holds under its stream);
// release_device_probe() of an address that holds nothing does nothing. The first hold of a device
// creates its probe stream, event, words and kernel on the calling thread and runs one probe there,
// so nothing of the probe is allocated or loaded later; a probe the stall limit turned off creates
// nothing and holds nothing. Throws on a CUDA error.
void hold_device_probe(int device, const void* holder);
void release_device_probe(const void* holder) noexcept;

// While alive, names the thread that constructed it in stall messages and sets the limit and the
// devices of its waits: a wait is not a stall while a host service of the context's devices
// answers. Watches nest on one thread; the innermost one governs until it ends.
class DeviceWaitWatch {
public:
    // No device wait of an Engine lasts this long without a host service answering: the longest
    // device work between two host waits is one prompt step, seconds on the slowest card.
    static constexpr std::chrono::seconds kStallLimit{120};

    // Watches with device_stall_limit().
    DeviceWaitWatch(const DeviceContext& context, const char* thread);
    DeviceWaitWatch(const DeviceContext& context, const char* thread,
                    std::chrono::milliseconds limit);
    ~DeviceWaitWatch();
    DeviceWaitWatch(const DeviceWaitWatch&)            = delete;
    DeviceWaitWatch& operator=(const DeviceWaitWatch&) = delete;

private:
    DeviceWaitState* state_ = nullptr;
    DeviceWaitState* outer_ = nullptr;
    bool bound_             = false;
};

} // namespace ninfer
