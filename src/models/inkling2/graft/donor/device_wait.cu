#include "core/device_wait.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <sys/syscall.h>
#    include <unistd.h>
#endif

namespace ninfer {

struct DeviceWaitState {
    // "the Gemma 4 worker (thread 4940)", or "thread 4940" for a thread without a watch.
    std::string label;
    // Devices whose host-service answers keep the waits alive; empty for a thread without a watch,
    // whose waits any device's answers keep alive (a pipeline stage waits for its predecessors).
    std::vector<int> devices;
    std::int64_t limit_ns = 0;
    // Steady-clock nanoseconds when the thread's outermost watched wait began; 0 between waits.
    std::atomic<std::int64_t> waiting_since{0};
    std::atomic<const char*> kind{""};
    // The device current when the outermost wait began, named for a thread without a watch.
    std::atomic<int> current_device{-1};
    // Watched waits the thread is inside (a wait may call another); read by that thread only.
    int depth = 0;
};

namespace {

using Clock = std::chrono::steady_clock;

// The monitoring thread checks every wait and probe this often; a stall is reported this much
// past its limit.
constexpr auto kPoll = std::chrono::milliseconds(250);
// A pause of the monitoring thread this long means that the process or the host stood still (a
// stop signal, a debugger, a paused container, a suspended machine): every quiet period starts
// again when it ends, so the pause does not count as a stall.
constexpr std::int64_t kStandstillNs = 2'000'000'000;
// A finished probe is followed by the next one this much later, or after a quarter of the limit
// when that is shorter.
constexpr std::int64_t kProbeIntervalNs = 5'000'000'000;
// CUDA device ordinals whose host-service answers are counted and that are probed.
constexpr std::size_t kTrackedDevices = 64;

std::int64_t now_ns() noexcept {
    const auto ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch());
    return std::max<std::int64_t>(1, ns.count());
}

long long os_thread_id() noexcept {
#if defined(_WIN32)
    return static_cast<long long>(GetCurrentThreadId());
#elif defined(__linux__)
    return static_cast<long long>(syscall(SYS_gettid));
#else
    return 0;
#endif
}

std::chrono::milliseconds stall_limit_from_environment() {
    const char* value = std::getenv("NINFER_DEVICE_STALL_SECONDS");
    if (value == nullptr) { return DeviceWaitWatch::kStallLimit; }
    const char* const end = value + std::strlen(value);
    std::uint32_t seconds = 0;
    const auto [last, error] = std::from_chars(value, end, seconds);
    if (value == end || error != std::errc{} || last != end) {
        throw std::invalid_argument("NINFER_DEVICE_STALL_SECONDS must be a whole number of "
                                    "seconds; 0 turns the device stall watches off");
    }
    return std::chrono::seconds(seconds);
}

std::chrono::milliseconds stall_limit_or_default() noexcept {
    try {
        return device_stall_limit();
    } catch (...) {
        // The DeviceContext that starts the probes reports the bad value.
        return DeviceWaitWatch::kStallLimit;
    }
}

std::int64_t to_ns(std::chrono::milliseconds limit) noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(limit).count();
}

// One device's probe: its own stream, so that it depends on no other work of the process.
struct Probe {
    // Guarded by Registry::probe_mutex.
    int holders     = 0;
    bool ready      = false;
    bool failed     = false;
    std::int64_t next_ns = 0;
    cudaStream_t stream  = nullptr;
    cudaEvent_t finished = nullptr;
    unsigned int* word   = nullptr; // device memory
    unsigned int* host   = nullptr; // two pinned words
    // When the unfinished probe began to be submitted; 0 while none is. Read by the monitor.
    std::atomic<std::int64_t> pending_since{0};
};

struct Registry {
    // When a host service last answered each device; 0 before its first answer.
    std::array<std::atomic<std::int64_t>, kTrackedDevices> answered{};
    // Guards states and monitoring.
    std::mutex mutex;
    std::vector<const DeviceWaitState*> states;
    bool monitoring = false;
    // Guards the probes' holders and resources, probing and exiting; taken before `mutex`.
    std::mutex probe_mutex;
    std::array<Probe, kTrackedDevices> probes{};
    // Each hold: its holder's address and the device it holds.
    std::vector<std::pair<const void*, int>> holds;
    bool probing = false;
    bool exiting = false;
    std::atomic<std::int64_t> probe_limit_ns{0};
};

// Never destroyed: the monitoring and probing threads run until the process ends, past static
// destruction.
Registry& registry() {
    static auto* const instance = new Registry();
    return *instance;
}

std::string device_list(const DeviceWaitState& state) {
    if (state.devices.empty()) {
        const int device = state.current_device.load(std::memory_order_relaxed);
        return device < 0 ? std::string("?") : std::to_string(device);
    }
    std::string devices;
    for (const int device : state.devices) {
        devices += (devices.empty() ? "" : ", ") + std::to_string(device);
    }
    return devices;
}

[[noreturn]] void end_waiting_process(const DeviceWaitState& state, std::int64_t waited_ns) {
    std::fprintf(stderr,
                 "ninfer: %s has waited %lld s for CUDA device %s (%s) while no host service "
                 "answered it: the device stopped finishing work, which CUDA reports as no error. "
                 "Ending the process; a new one recovers the device.\n",
                 state.label.c_str(), static_cast<long long>(waited_ns / 1'000'000'000),
                 device_list(state).c_str(), state.kind.load(std::memory_order_relaxed));
    std::fflush(stderr);
    std::abort();
}

[[noreturn]] void end_probed_process(std::size_t device, std::int64_t waited_ns) {
    std::fprintf(stderr,
                 "ninfer: CUDA device %zu has not finished a probe (a copy to the device, a kernel "
                 "and a copy back, on a stream of their own) in %lld s while no host service "
                 "answered it: the device stopped finishing work, which CUDA reports as no error. "
                 "Ending the process; a new one recovers the device.\n",
                 device, static_cast<long long>(waited_ns / 1'000'000'000));
    std::fflush(stderr);
    std::abort();
}

void monitor(Registry& watched) {
    std::int64_t last    = now_ns();
    std::int64_t resumed = 0;
    for (;;) {
        std::this_thread::sleep_for(kPoll);
        const std::lock_guard lock(watched.mutex);
        const std::int64_t now = now_ns();
        if (now - last >= kStandstillNs) { resumed = now; }
        last = now;
        for (const DeviceWaitState* state : watched.states) {
            const std::int64_t since = state->waiting_since.load(std::memory_order_acquire);
            if (since == 0) { continue; }
            // An answer the device received after the wait began restarts the quiet period.
            std::int64_t quiet = std::max(since, resumed);
            if (state->devices.empty()) {
                for (const auto& answered : watched.answered) {
                    quiet = std::max(quiet, answered.load(std::memory_order_acquire));
                }
            }
            for (const int device : state->devices) {
                if (device >= 0 && std::size_t(device) < kTrackedDevices) {
                    quiet =
                        std::max(quiet, watched.answered[device].load(std::memory_order_acquire));
                }
            }
            if (now - quiet >= state->limit_ns) { end_waiting_process(*state, now - since); }
        }
        const std::int64_t probe_limit = watched.probe_limit_ns.load(std::memory_order_acquire);
        if (probe_limit == 0) { continue; }
        for (std::size_t device = 0; device < kTrackedDevices; ++device) {
            const std::int64_t since =
                watched.probes[device].pending_since.load(std::memory_order_acquire);
            if (since == 0) { continue; }
            const std::int64_t quiet = std::max(
                {since, resumed, watched.answered[device].load(std::memory_order_acquire)});
            if (now - quiet >= probe_limit) { end_probed_process(device, now - since); }
        }
    }
}

// Starts the monitoring thread once; `watched.mutex` is held.
void start_monitor(Registry& watched) {
    if (watched.monitoring) { return; }
    std::thread([&watched] { monitor(watched); }).detach();
    watched.monitoring = true;
}

void bind_state(DeviceWaitState* state) {
    Registry& watched = registry();
    const std::lock_guard lock(watched.mutex);
    start_monitor(watched);
    watched.states.push_back(state);
}

void unbind_state(const DeviceWaitState* state) noexcept {
    Registry& watched = registry();
    const std::lock_guard lock(watched.mutex);
    std::erase(watched.states, state);
}

thread_local DeviceWaitState* t_watched = nullptr;

// The watch of a thread that holds no DeviceWaitWatch, made at its first wait. Both pointers are
// trivially destructible, so a wait in a later thread-exit destructor finds them still valid.
thread_local DeviceWaitState* t_unnamed = nullptr;
thread_local bool t_unnamed_made        = false;

struct UnnamedWatchRelease {
    // User-provided, so the first use on a thread constructs it and registers its destructor.
    UnnamedWatchRelease() noexcept {}
    ~UnnamedWatchRelease() {
        if (t_unnamed != nullptr) {
            unbind_state(t_unnamed);
            delete t_unnamed;
            t_unnamed = nullptr;
        }
    }
};
thread_local UnnamedWatchRelease t_unnamed_release;

DeviceWaitState* watched_state() noexcept {
    if (t_watched != nullptr) { return t_watched; }
    if (!t_unnamed_made) {
        t_unnamed_made                        = true;
        const std::chrono::milliseconds limit = stall_limit_or_default();
        if (limit.count() == 0) { return nullptr; }
        try {
            auto state      = std::make_unique<DeviceWaitState>();
            state->label    = "thread " + std::to_string(os_thread_id());
            state->limit_ns = to_ns(limit);
            (void)&t_unnamed_release; // constructs the release before the state exists
            bind_state(state.get());
            t_unnamed = state.release();
        } catch (...) {
            return nullptr;
        }
    }
    return t_unnamed;
}

__global__ void probe_kernel(unsigned int* word) { *word += 1U; }

// Submits one probe: a copy to the device, a kernel and a copy back, then the event that marks
// them finished. False on a CUDA error.
bool submit(Probe& probe) noexcept {
    if (cudaMemcpyAsync(probe.word, probe.host, sizeof(unsigned int), cudaMemcpyHostToDevice,
                        probe.stream) != cudaSuccess) {
        return false;
    }
    probe_kernel<<<1, 1, 0, probe.stream>>>(probe.word);
    return cudaGetLastError() == cudaSuccess &&
           cudaMemcpyAsync(probe.host + 1, probe.word, sizeof(unsigned int),
                           cudaMemcpyDeviceToHost, probe.stream) == cudaSuccess &&
           cudaEventRecord(probe.finished, probe.stream) == cudaSuccess;
}

void check_probe(cudaError_t error, int device, const char* call) {
    if (error == cudaSuccess) { return; }
    throw std::runtime_error("the stall probe of CUDA device " + std::to_string(device) +
                             " could not be set up: " + call + ": " + cudaGetErrorName(error) +
                             ": " + cudaGetErrorString(error));
}

void destroy(Probe& probe) noexcept {
    if (probe.host != nullptr) { (void)cudaFreeHost(probe.host); }
    if (probe.word != nullptr) { (void)cudaFree(probe.word); }
    if (probe.finished != nullptr) { (void)cudaEventDestroy(probe.finished); }
    if (probe.stream != nullptr) { (void)cudaStreamDestroy(probe.stream); }
    probe.host     = nullptr;
    probe.word     = nullptr;
    probe.finished = nullptr;
    probe.stream   = nullptr;
}

// Creates device `device`'s probe on the calling thread and runs it once, so that the probing
// thread later allocates and loads nothing.
void create(Probe& probe, int device) {
    const DeviceBinding bound(device);
    try {
        int least    = 0;
        int greatest = 0;
        check_probe(cudaDeviceGetStreamPriorityRange(&least, &greatest), device,
                    "cudaDeviceGetStreamPriorityRange");
        // The highest priority puts the probe kernel's block ahead of every waiting block of the
        // process's own kernels.
        check_probe(cudaStreamCreateWithPriority(&probe.stream, cudaStreamNonBlocking, greatest),
                    device, "cudaStreamCreateWithPriority");
        check_probe(cudaEventCreateWithFlags(&probe.finished, cudaEventDisableTiming), device,
                    "cudaEventCreateWithFlags");
        check_probe(cudaMalloc(&probe.word, sizeof(unsigned int)), device, "cudaMalloc");
        check_probe(cudaHostAlloc(&probe.host, 2 * sizeof(unsigned int), cudaHostAllocPortable),
                    device, "cudaHostAlloc");
        probe.host[0] = 0;
        probe.host[1] = 0;
        // Loads the kernel now: a first launch later could make lazy module loading synchronize
        // the context while a CUDA Graph waits on a host service.
        cudaFuncAttributes attributes{};
        check_probe(cudaFuncGetAttributes(&attributes, probe_kernel), device,
                    "cudaFuncGetAttributes");
        if (!submit(probe)) {
            const cudaError_t error = cudaGetLastError();
            check_probe(error == cudaSuccess ? cudaErrorUnknown : error, device, "the first probe");
        }
        check_probe(wait_for_event(probe.finished), device, "the first probe");
    } catch (...) {
        destroy(probe);
        throw;
    }
}

void fail(Probe& probe) noexcept {
    probe.failed = true;
    probe.pending_since.store(0, std::memory_order_release);
}

void probe_devices(Registry& watched, int first) {
    // A new thread has no current context, so its first CUDA call would create the current
    // device's: device 0's, on a card the process may not have been given (#279). The device whose
    // probe started this thread is made current first.
    (void)cudaSetDevice(first);
    // This thread's calls touch no capture; Relaxed keeps a global-mode capture on another thread
    // from refusing them.
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    (void)cudaThreadExchangeStreamCaptureMode(&mode);
    const std::int64_t interval =
        std::min(kProbeIntervalNs, watched.probe_limit_ns.load(std::memory_order_relaxed) / 4);
    for (;;) {
        std::this_thread::sleep_for(kPoll);
        const std::lock_guard lock(watched.probe_mutex);
        if (watched.exiting) { return; }
        for (std::size_t device = 0; device < kTrackedDevices; ++device) {
            Probe& probe = watched.probes[device];
            if (!probe.ready || probe.failed) { continue; }
            const bool pending = probe.pending_since.load(std::memory_order_relaxed) != 0;
            if (!pending && (probe.holders == 0 || now_ns() < probe.next_ns)) { continue; }
            if (cudaSetDevice(int(device)) != cudaSuccess) {
                fail(probe);
                continue;
            }
            if (!pending) {
                // From before the submission: a submission the driver never returns from is a
                // stall too.
                probe.pending_since.store(now_ns(), std::memory_order_release);
                if (!submit(probe)) {
                    // A CUDA error belongs to the code that caused it, which CUDA_CHECK reports.
                    fail(probe);
                    continue;
                }
            }
            // The query also submits the probe on a batched driver model (WDDM).
            const cudaError_t state = cudaEventQuery(probe.finished);
            if (state == cudaSuccess) {
                probe.pending_since.store(0, std::memory_order_release);
                probe.next_ns = now_ns() + interval;
            } else if (state != cudaErrorNotReady) {
                fail(probe);
            }
        }
    }
}

// Runs at exit before CUDA's own teardown, which the probing thread must not race.
void stop_probing() noexcept {
    Registry& watched = registry();
    const std::lock_guard lock(watched.probe_mutex);
    watched.exiting = true;
    for (Probe& probe : watched.probes) { probe.pending_since.store(0, std::memory_order_release); }
}

} // namespace

std::chrono::milliseconds device_stall_limit() {
    static const std::chrono::milliseconds limit = stall_limit_from_environment();
    return limit;
}

DeviceWaitScope::DeviceWaitScope(const char* kind) noexcept : state_(watched_state()) {
    if (state_ != nullptr && state_->depth++ == 0) {
        state_->kind.store(kind, std::memory_order_relaxed);
        if (state_->devices.empty()) {
            int device = -1;
            if (cudaGetDevice(&device) != cudaSuccess) { device = -1; }
            state_->current_device.store(device, std::memory_order_relaxed);
        }
        state_->waiting_since.store(now_ns(), std::memory_order_release);
    }
}

DeviceWaitScope::~DeviceWaitScope() {
    if (state_ != nullptr && --state_->depth == 0) {
        state_->waiting_since.store(0, std::memory_order_release);
    }
}

cudaError_t wait_for_stream(cudaStream_t stream) noexcept {
    const DeviceWaitScope scope("a stream");
    return cudaStreamSynchronize(stream);
}

cudaError_t wait_for_event(cudaEvent_t event) noexcept {
    const DeviceWaitScope scope("an event");
    return cudaEventSynchronize(event);
}

cudaError_t wait_for_device() noexcept {
    const DeviceWaitScope scope("the device");
    return cudaDeviceSynchronize();
}

cudaError_t copy_and_wait(void* destination, const void* source, std::size_t bytes,
                          cudaMemcpyKind kind) noexcept {
    const DeviceWaitScope scope("a copy");
    const cudaError_t copy = cudaMemcpy(destination, source, bytes, kind);
    return copy != cudaSuccess ? copy : cudaStreamSynchronize(nullptr);
}

cudaError_t copy_between_devices_and_wait(void* destination, int destination_device,
                                          const void* source, int source_device,
                                          std::size_t bytes) noexcept {
    const DeviceWaitScope scope("a copy between devices");
    const cudaError_t copy =
        cudaMemcpyPeer(destination, destination_device, source, source_device, bytes);
    return copy != cudaSuccess ? copy : cudaStreamSynchronize(nullptr);
}

cudaError_t fill_and_wait(void* destination, int value, std::size_t bytes) noexcept {
    const DeviceWaitScope scope("a fill");
    const cudaError_t fill = cudaMemset(destination, value, bytes);
    return fill != cudaSuccess ? fill : cudaStreamSynchronize(nullptr);
}

cudaError_t free_device_memory(void* memory) noexcept {
    const DeviceWaitScope scope("a release of device memory");
    return cudaFree(memory);
}

cudaError_t free_host_memory(void* memory) noexcept {
    const DeviceWaitScope scope("a release of pinned host memory");
    return cudaFreeHost(memory);
}

cudaError_t unregister_host_memory(void* memory) noexcept {
    const DeviceWaitScope scope("an unregistration of host memory");
    return cudaHostUnregister(memory);
}

cudaError_t launch_graph(cudaGraphExec_t graph, cudaStream_t stream) noexcept {
    const DeviceWaitScope scope("a CUDA Graph launch");
    return cudaGraphLaunch(graph, stream);
}

void note_host_service_progress(int device) noexcept {
    if (device < 0 || std::size_t(device) >= kTrackedDevices) { return; }
    registry().answered[device].store(now_ns(), std::memory_order_release);
}

void hold_device_probe(int device, const void* holder) {
    const std::chrono::milliseconds limit = device_stall_limit();
    if (limit.count() == 0 || device < 0 || std::size_t(device) >= kTrackedDevices) { return; }
    Registry& watched = registry();
    const std::lock_guard lock(watched.probe_mutex);
    Probe& probe = watched.probes[device];
    if (!probe.ready && !probe.failed) {
        create(probe, device);
        probe.ready = true;
    }
    watched.holds.emplace_back(holder, device);
    ++probe.holders;
    if (!watched.probing) {
        watched.probe_limit_ns.store(to_ns(limit), std::memory_order_release);
        std::thread([&watched, device] { probe_devices(watched, device); }).detach();
        // Registered after CUDA initialized, so it runs before CUDA's own exit teardown.
        (void)std::atexit(stop_probing);
        watched.probing = true;
        const std::lock_guard monitoring(watched.mutex);
        start_monitor(watched);
    }
}

void release_device_probe(const void* holder) noexcept {
    Registry& watched = registry();
    const std::lock_guard lock(watched.probe_mutex);
    const auto hold = std::find_if(watched.holds.begin(), watched.holds.end(),
                                   [holder](const auto& entry) { return entry.first == holder; });
    if (hold == watched.holds.end()) { return; }
    --watched.probes[std::size_t(hold->second)].holders;
    watched.holds.erase(hold);
}

DeviceWaitWatch::DeviceWaitWatch(const DeviceContext& context, const char* thread)
    : DeviceWaitWatch(context, thread, device_stall_limit()) {}

DeviceWaitWatch::DeviceWaitWatch(const DeviceContext& context, const char* thread,
                                 std::chrono::milliseconds limit) {
    if (limit.count() <= 0) { return; }
    auto state = std::make_unique<DeviceWaitState>();
    state->label =
        std::string("the ") + thread + " (thread " + std::to_string(os_thread_id()) + ")";
    state->limit_ns = to_ns(limit);
    for (std::size_t rank = 0; rank < context.size(); ++rank) {
        state->devices.push_back(context.rank(rank).device);
    }
    bind_state(state.get());
    state_    = state.release();
    outer_    = t_watched;
    t_watched = state_;
    bound_    = true;
}

DeviceWaitWatch::~DeviceWaitWatch() {
    if (!bound_) { return; }
    t_watched = outer_;
    unbind_state(state_);
    delete state_;
}

} // namespace ninfer
