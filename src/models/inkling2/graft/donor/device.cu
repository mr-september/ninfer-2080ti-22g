#include "core/device.h"

#include "core/device_wait.h"
#include "core/peer_preflight.h"

#include <cuda.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer {
namespace {

struct SyncSchedule {
    std::string_view name;
    unsigned int flags;
};

constexpr SyncSchedule kSyncSchedules[] = {
    {"spin", cudaDeviceScheduleSpin},
    {"blocking", cudaDeviceScheduleBlockingSync},
    {"yield", cudaDeviceScheduleYield},
    {"auto", cudaDeviceScheduleAuto},
};

// Unset keeps CUDA's own default (auto), which spins while the host has more cores than active
// contexts.
std::optional<unsigned int> sync_schedule_from_environment() {
    const char* value = std::getenv("NINFER_CUDA_SYNC");
    if (value == nullptr) { return std::nullopt; }
    for (const auto& schedule : kSyncSchedules) {
        if (schedule.name == value) { return schedule.flags; }
    }
    throw std::invalid_argument("NINFER_CUDA_SYNC must be spin, blocking, yield, or auto");
}

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

// Loaded on a device only where this build has code that runs there: its own architecture's
// machine code, which also runs on later minor versions of that major (sm_86 code on 8.9, sm_80
// code on 8.6 and 8.9), or its PTX compiled for a newer device.
__global__ void kernel_image_probe() {}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_stream(cudaStream_t& stream) noexcept {
    if (stream != nullptr) {
        log_cuda_error("cudaStreamDestroy", cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        log_cuda_error("cudaEventDestroy", cudaEventDestroy(event));
        event = nullptr;
    }
}

void check_probe(cudaError_t err, const char* op) {
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message(op, err));
    }
}

// One store into and one load from a peer's memory, the two accesses a kernel makes through a peer
// mapping (the tensor-parallel exchange reads its peers' partials in place).
__global__ void peer_canary_kernel(unsigned long long* remote_store,
                                   const unsigned long long* remote_load,
                                   unsigned long long pattern, unsigned long long* loaded) {
    *reinterpret_cast<volatile unsigned long long*>(remote_store) = pattern;
    __threadfence_system();
    *loaded = *reinterpret_cast<const volatile unsigned long long*>(remote_load);
}

PeerDeviceFacts peer_facts(const RankContext& rank) {
    char bus_id[32] = {};
    check_probe(cudaDeviceGetPCIBusId(bus_id, sizeof bus_id, rank.device),
                "cudaDeviceGetPCIBusId failed");
    return read_peer_device_facts("/sys", rank.props.name, bus_id, rank.props.totalGlobalMem);
}

// A probe allocation freed on its own device whatever the probe throws.
struct ProbeWords {
    int device                = 0;
    unsigned long long* words = nullptr;

    ProbeWords(int owner, std::size_t count) : device(owner) {
        DeviceBinding bind(device);
        check_probe(cudaMalloc(&words, count * sizeof(unsigned long long)),
                    "peer probe cudaMalloc");
    }
    ~ProbeWords() {
        int previous = 0;
        if (cudaGetDevice(&previous) != cudaSuccess) { return; }
        log_cuda_error("cudaSetDevice", cudaSetDevice(device));
        log_cuda_error("cudaFree", ninfer::free_device_memory(words));
        log_cuda_error("cudaSetDevice", cudaSetDevice(previous));
    }
    ProbeWords(const ProbeWords&)            = delete;
    ProbeWords& operator=(const ProbeWords&) = delete;
};

// Empty when `source`, with peer access to `destination` enabled, stores into and loads from its
// memory through a kernel and copies into it with the copy engine; otherwise what went wrong. A
// mapping that is reported but not backed loses stores and returns garbage without an error.
std::string probe_peer(int source, int destination) {
    constexpr unsigned long long kStored   = 0x6E696E666572A5C3ULL;
    constexpr unsigned long long kResident = 0x3C5A7065657201FFULL;
    constexpr unsigned long long kCopied   = 0xA5C3706565720777ULL;
    ProbeWords remote(destination, 3);  // store target, load source, copy target
    ProbeWords local(source, 2);        // loaded value, copy source
    {
        DeviceBinding bind(destination);
        const unsigned long long initial[3] = {0, kResident, 0};
        check_probe(
            ninfer::copy_and_wait(remote.words, initial, sizeof initial, cudaMemcpyHostToDevice),
            "peer probe upload");
    }
    unsigned long long loaded = 0;
    {
        DeviceBinding bind(source);
        const unsigned long long initial[2] = {0, kCopied};
        check_probe(
            ninfer::copy_and_wait(local.words, initial, sizeof initial, cudaMemcpyHostToDevice),
            "peer probe upload");
        peer_canary_kernel<<<1, 1>>>(remote.words, remote.words + 1, kStored, local.words);
        check_probe(cudaGetLastError(), "peer probe launch");
        check_probe(ninfer::wait_for_device(), "peer probe kernel");
        check_probe(ninfer::copy_between_devices_and_wait(remote.words + 2, destination,
                                                          local.words + 1, source,
                                                          sizeof(unsigned long long)),
                    "peer probe copy");
        check_probe(
            ninfer::copy_and_wait(&loaded, local.words, sizeof loaded, cudaMemcpyDeviceToHost),
            "peer probe readback");
    }
    unsigned long long seen[3] = {};
    {
        DeviceBinding bind(destination);
        check_probe(ninfer::copy_and_wait(seen, remote.words, sizeof seen, cudaMemcpyDeviceToHost),
                    "peer probe readback");
    }
    if (seen[0] != kStored) { return "a kernel's store into the peer's memory did not arrive"; }
    if (loaded != kResident) {
        return "a kernel's load from the peer's memory returned the wrong value";
    }
    if (seen[2] != kCopied) { return "a peer copy delivered the wrong value"; }
    return {};
}

} // namespace

int memory_device(const void* pointer) {
    if (pointer == nullptr) { return -1; }
    cudaPointerAttributes attributes{};
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, pointer));
    return attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged
               ? attributes.device
               : -1;
}

DeviceBinding::DeviceBinding(int device) {
    const cudaError_t get = cudaGetDevice(&previous_);
    if (get != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDevice failed", get));
    }
    if (previous_ == device) { return; }
    // A thread with no current context, such as a new service thread, reads device 0 without
    // having used it. Returning there would create device 0's primary context: a few hundred MiB
    // on a card the process was not given, and cudaErrorMemoryAllocation when that card is full,
    // which the destructor can only log and which the next cudaGetLastError() then reports (#279).
    CUcontext current = nullptr;
    restore_          = cuCtxGetCurrent(&current) == CUDA_SUCCESS && current != nullptr;
    const cudaError_t set = cudaSetDevice(device);
    if (set != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice failed", set));
    }
}

DeviceBinding::~DeviceBinding() noexcept {
    if (restore_) { log_cuda_error("cudaSetDevice", cudaSetDevice(previous_)); }
}

DeviceContext::DeviceContext(int device_id)
    : DeviceContext(std::span<const int>(&device_id, 1)) {}

DeviceContext::DeviceContext(std::span<const int> device_ids) {
    const std::optional<unsigned int> sync_flags = sync_schedule_from_environment();
    int count       = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceCount failed", err));
    }
    if (count <= 0) { throw std::runtime_error("no CUDA devices available"); }
    if (device_ids.empty() || device_ids.size() > kMaxRanks) {
        throw std::invalid_argument("DeviceContext requires between one and " +
                                    std::to_string(kMaxRanks) + " CUDA devices");
    }
    device_ids_.assign(device_ids.begin(), device_ids.end());
    for (std::size_t i = 0; i < device_ids_.size(); ++i) {
        const int id = device_ids_[i];
        if (id < 0 || id >= count) {
            throw std::runtime_error("CUDA device " + std::to_string(id) + " does not exist: " +
                                     std::to_string(count) +
                                     (count == 1 ? " device is visible" : " devices are visible"));
        }
        // A repeated id is allowed on purpose. Several ranks on one card is not a useful
        // deployment -- it frees no memory -- but it exercises the entire multi-rank path
        // (per-rank binding and materialization, per-rank workspaces, and the cross-rank copies)
        // on a single-GPU machine. Every bug caught that way is one not caught by renting a
        // second card.
    }

    endpoints_.resize(device_ids_.size());
    try {
        for (std::size_t rank = 0; rank < device_ids_.size(); ++rank) {
            RankContext& endpoint = endpoints_[rank];
            endpoint.index        = rank;
            endpoint.device       = device_ids_[rank];
            err                = cudaSetDevice(endpoint.device);
            if (err != cudaSuccess) {
                throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
            }
            if (sync_flags.has_value()) {
                err = cudaSetDeviceFlags(*sync_flags);
                if (err != cudaSuccess) {
                    throw std::runtime_error(cuda_error_message("cudaSetDeviceFlags failed", err));
                }
            }
            err = cudaGetDeviceProperties(&endpoint.props, endpoint.device);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    cuda_error_message("cudaGetDeviceProperties failed", err));
            }
            err = cudaStreamCreateWithFlags(&endpoint.stream, cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    cuda_error_message("cudaStreamCreateWithFlags(stream) failed", err));
            }
            err = cudaStreamCreateWithFlags(&endpoint.transfer_stream, cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    cuda_error_message("cudaStreamCreateWithFlags(transfer_stream) failed", err));
            }
            err = cudaStreamCreateWithFlags(&endpoint.vision_stream, cudaStreamNonBlocking);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    cuda_error_message("cudaStreamCreateWithFlags(vision_stream) failed", err));
            }
            err = cudaEventCreateWithFlags(&endpoint.fence, cudaEventDisableTiming);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    cuda_error_message("cudaEventCreateWithFlags(fence) failed", err));
            }
        }

        // Every rank must run this build's code. Ranks may differ in compute capability as long as
        // each does (an sm_86 build: RTX 30 and RTX 40 cards; sm_80: A100 with either): every
        // rank then runs the code it would run alone, with its own route profile.
        for (RankContext& endpoint : endpoints_) {
            err = cudaSetDevice(endpoint.device);
            if (err != cudaSuccess) {
                throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
            }
            cudaFuncAttributes attributes{};
            err = cudaFuncGetAttributes(&attributes, kernel_image_probe);
            if (err != cudaSuccess) {
                (void)cudaGetLastError();
                throw std::invalid_argument(
                    "CUDA device " + std::to_string(endpoint.device) + " (" + endpoint.props.name +
                    ", compute capability " + std::to_string(endpoint.props.major) + "." +
                    std::to_string(endpoint.props.minor) + ") cannot run this build, compiled for "
                    "CUDA architectures " NINFER_CUDA_ARCHITECTURES ": " + cudaGetErrorName(err));
            }
        }

        // A device that stops finishing work ends the process while any context holds it. Each
        // rank holds under its stream, which is its own and moves with it.
        for (const RankContext& endpoint : endpoints_) {
            hold_device_probe(endpoint.device, endpoint.stream);
        }

        // peer_matrix_[from][to]: can `from` DMA into `to`. Ranks on one card need no peer setup:
        // copies between them are ordinary device-to-device, and enabling peer access to self is an
        // error, so they are recorded as reachable.
        //
        // Peer access is a performance capability, not a requirement. NVIDIA disables P2P over PCIe
        // on GeForce, so a pair of 3090s without an NVLink bridge reports can_access == 0 -- but
        // transfers still work there, staged through host memory. Refusing to start would rule out
        // the most common consumer multi-GPU box for no reason, so record the capability and let
        // callers choose a transport that suits it. (Neither a 2x 3090 nor a 2x A4000 PCIe box
        // measured so far reported any.)
        const std::size_t ranks = endpoints_.size();
        peer_matrix_.assign(ranks * ranks, 0);
        peer_access_ = true;
        // A pair of devices is set up and probed once, however many ranks they hold.
        std::vector<int> pair_state(static_cast<std::size_t>(count) * count, -1);
        for (std::size_t source = 0; source < ranks; ++source) {
            for (std::size_t destination = 0; destination < ranks; ++destination) {
                char& reachable = peer_matrix_[source * ranks + destination];
                const RankContext& from = endpoints_[source];
                const RankContext& to   = endpoints_[destination];
                if (from.device == to.device) {
                    reachable = 1;
                    continue;
                }
                int& state = pair_state[static_cast<std::size_t>(from.device) * count + to.device];
                if (state < 0) { state = enable_peer(from, to) ? 1 : 0; }
                reachable = static_cast<char>(state);
                if (reachable == 0) { peer_access_ = false; }
            }
        }
    } catch (...) {
        release();
        throw;
    }

    active_rank_ = 0;
    err          = cudaSetDevice(endpoints_[0].device);
    if (err != cudaSuccess) {
        release();
        throw std::runtime_error(cuda_error_message("cudaSetDevice(primary) failed", err));
    }
    refresh_active_aliases();
}

DeviceContext::~DeviceContext() { release(); }

void DeviceContext::release() noexcept {
    for (RankContext& endpoint : endpoints_) {
        release_device_probe(endpoint.stream);
        if (endpoint.stream != nullptr || endpoint.transfer_stream != nullptr ||
            endpoint.vision_stream != nullptr || endpoint.fence != nullptr) {
            log_cuda_error("cudaSetDevice", cudaSetDevice(endpoint.device));
        }
        destroy_event(endpoint.fence);
        destroy_stream(endpoint.vision_stream);
        destroy_stream(endpoint.transfer_stream);
        destroy_stream(endpoint.stream);
    }
    endpoints_.clear();
    device_ids_.clear();
    peer_matrix_.clear();
    peer_notes_.clear();
    device          = 0;
    stream          = nullptr;
    transfer_stream = nullptr;
    vision_stream   = nullptr;
    props           = {};
    active_rank_    = 0;
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : endpoints_(std::move(other.endpoints_)), device_ids_(std::move(other.device_ids_)),
      peer_matrix_(std::move(other.peer_matrix_)), peer_notes_(std::move(other.peer_notes_)),
      active_rank_(other.active_rank_),
      peer_access_(other.peer_access_) {
    refresh_active_aliases();
    other.device                  = 0;
    other.stream                  = nullptr;
    other.transfer_stream         = nullptr;
    other.vision_stream           = nullptr;
    other.props                   = {};
    other.active_rank_            = 0;
    other.peer_access_            = false;
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
    if (this == &other) { return *this; }

    release();
    endpoints_              = std::move(other.endpoints_);
    device_ids_             = std::move(other.device_ids_);
    peer_matrix_            = std::move(other.peer_matrix_);
    peer_notes_             = std::move(other.peer_notes_);
    active_rank_            = other.active_rank_;
    peer_access_            = other.peer_access_;
    refresh_active_aliases();
    other.device                  = 0;
    other.stream                  = nullptr;
    other.transfer_stream         = nullptr;
    other.vision_stream           = nullptr;
    other.props                   = {};
    other.active_rank_            = 0;
    other.peer_access_            = false;
    return *this;
}

void DeviceContext::refresh_active_aliases() noexcept {
    if (endpoints_.empty()) {
        device          = 0;
        stream          = nullptr;
        transfer_stream = nullptr;
        vision_stream   = nullptr;
        props           = {};
        return;
    }
    const RankContext& endpoint = endpoints_[active_rank_];
    device                   = endpoint.device;
    stream                   = endpoint.stream;
    transfer_stream          = endpoint.transfer_stream;
    vision_stream            = endpoint.vision_stream;
    props                    = endpoint.props;
}

void DeviceContext::bind_to_current_thread() const {
    const cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
    }
}

void DeviceContext::bind_to_current_thread_noexcept() const noexcept {
    log_cuda_error("cudaSetDevice", cudaSetDevice(device));
}

int DeviceContext::sm() const noexcept { return props.major * 10 + props.minor; }

int DeviceContext::compute_capability() const noexcept { return props.major * 10 + props.minor; }

// Streaming-multiprocessor count of the active device. Distinct from compute_capability(): every
// sm_86 part shares capability 86 but not this count, so residency budgets must read this.
int DeviceContext::multiprocessor_count() const noexcept { return props.multiProcessorCount; }

DeviceExecutionView DeviceContext::execution_view() const noexcept {
    return {.stream = stream, .multiprocessor_count = multiprocessor_count()};
}

std::size_t DeviceContext::total_vram() const noexcept { return props.totalGlobalMem; }

std::size_t DeviceContext::free_bytes() const {
    std::size_t free  = 0;
    std::size_t total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return free;
}

const char* DeviceContext::sync_mode() const {
    bind_to_current_thread();
    unsigned int flags    = 0;
    const cudaError_t err = cudaGetDeviceFlags(&flags);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceFlags failed", err));
    }
    for (const auto& schedule : kSyncSchedules) {
        if ((flags & cudaDeviceScheduleMask) == schedule.flags) { return schedule.name.data(); }
    }
    throw std::runtime_error("unknown CUDA synchronization schedule");
}

std::size_t DeviceContext::size() const noexcept { return endpoints_.size(); }

bool DeviceContext::model_parallel() const noexcept { return endpoints_.size() > 1; }

const RankContext& DeviceContext::rank(std::size_t index) const {
    if (index >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    return endpoints_[index];
}

bool DeviceContext::same_physical_device(std::size_t a, std::size_t b) const {
    return rank(a).device == rank(b).device;
}

bool DeviceContext::enable_peer(const RankContext& from, const RankContext& to) {
    int can_access  = 0;
    cudaError_t err = cudaDeviceCanAccessPeer(&can_access, from.device, to.device);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaDeviceCanAccessPeer failed", err));
    }
    if (can_access == 0) { return false; }
    const auto refuse = [&](const std::string& reason) {
        peer_notes_.push_back("CUDA device " + std::to_string(from.device) + " -> " +
                              std::to_string(to.device) + ": " + reason +
                              "; their transfers stage through host memory");
        return false;
    };
    // A configuration that reports a mapping it cannot back is refused before any probe touches
    // it: probing a broken mailbox mapping writes to arbitrary memory.
    const std::string refusal = peer_preflight_refusal(peer_facts(from), peer_facts(to));
    if (!refusal.empty()) { return refuse(refusal); }
    DeviceBinding bind(from.device);
    err = cudaDeviceEnablePeerAccess(to.device, 0);
    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
        // Capability without a usable mapping: fall back rather than fail outright.
        (void)cudaGetLastError();
        return refuse(std::string("enabling peer access failed: ") + cudaGetErrorName(err));
    }
    (void)cudaGetLastError();
    const std::string failure = probe_peer(from.device, to.device);
    if (failure.empty()) { return true; }
    log_cuda_error("cudaDeviceDisablePeerAccess", cudaDeviceDisablePeerAccess(to.device));
    return refuse(failure);
}

const std::vector<std::string>& DeviceContext::peer_notes() const noexcept { return peer_notes_; }

bool DeviceContext::peer_access() const noexcept { return peer_access_; }

bool DeviceContext::peer_access(std::size_t from, std::size_t to) const {
    if (from >= endpoints_.size() || to >= endpoints_.size()) {
        throw std::out_of_range("CUDA device rank is out of range");
    }
    return peer_matrix_[from * endpoints_.size() + to] != 0;
}

std::size_t DeviceContext::active_rank() const noexcept { return active_rank_; }

const std::vector<int>& DeviceContext::device_ids() const noexcept { return device_ids_; }

cudaStream_t DeviceContext::stream_for_rank(std::size_t rank) const {
    if (rank >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    return endpoints_[rank].stream;
}

cudaStream_t DeviceContext::transfer_stream_for_rank(std::size_t rank) const {
    if (rank >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    return endpoints_[rank].transfer_stream;
}

cudaEvent_t DeviceContext::fence_for_rank(std::size_t rank) const {
    if (rank >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    return endpoints_[rank].fence;
}

void DeviceContext::activate_rank(std::size_t rank) {
    if (rank >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    CUDA_CHECK(cudaSetDevice(endpoints_[rank].device));
    active_rank_ = rank;
    refresh_active_aliases();
}

void DeviceContext::synchronize_rank(std::size_t rank) const {
    if (rank >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    CUDA_CHECK(cudaSetDevice(endpoints_[rank].device));
    CUDA_CHECK(ninfer::wait_for_stream(endpoints_[rank].stream));
    CUDA_CHECK(cudaSetDevice(endpoints_[active_rank_].device));
}

void DeviceContext::synchronize() const {
    for (const RankContext& endpoint : endpoints_) {
        CUDA_CHECK(cudaSetDevice(endpoint.device));
        CUDA_CHECK(ninfer::wait_for_stream(endpoint.stream));
    }
    CUDA_CHECK(cudaSetDevice(endpoints_[active_rank_].device));
}

void DeviceContext::flush() const {
    for (const RankContext& endpoint : endpoints_) {
        CUDA_CHECK(cudaSetDevice(endpoint.device));
        const cudaError_t status = cudaStreamQuery(endpoint.stream);
        if (status == cudaErrorNotReady) {
            // Pending work is the expected answer; do not leave it as the thread's last error.
            (void)cudaGetLastError();
            continue;
        }
        CUDA_CHECK(status);
    }
    CUDA_CHECK(cudaSetDevice(endpoints_[active_rank_].device));
}

ScopedDeviceRank::ScopedDeviceRank(DeviceContext& context, std::size_t rank)
    : context_(context), previous_rank_(context.active_rank()) {
    context_.activate_rank(rank);
}

ScopedDeviceRank::~ScopedDeviceRank() noexcept {
    if (context_.active_rank() != previous_rank_) { context_.activate_rank(previous_rank_); }
}

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx)
    : CudaEventTimer(ctx, ctx.stream) {}

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream) : stream_(stream) {
    cudaError_t err = cudaSetDevice(ctx.device);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice(timer) failed", err));
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    err               = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreate(start) failed", err));
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        destroy_event(start);
        throw std::runtime_error(cuda_error_message("cudaEventCreate(stop) failed", err));
    }

    start_ = start;
    stop_  = stop;
}

CudaEventTimer::~CudaEventTimer() {
    destroy_event(stop_);
    destroy_event(start_);
}

CudaEventTimer::CudaEventTimer(CudaEventTimer&& other) noexcept
    : stream_(other.stream_), start_(other.start_), stop_(other.stop_) {
    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
}

CudaEventTimer& CudaEventTimer::operator=(CudaEventTimer&& other) noexcept {
    if (this == &other) { return *this; }

    destroy_event(stop_);
    destroy_event(start_);

    stream_ = other.stream_;
    start_  = other.start_;
    stop_   = other.stop_;

    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
    return *this;
}

void CudaEventTimer::start() { CUDA_CHECK(cudaEventRecord(start_, stream_)); }

void CudaEventTimer::record_stop() { CUDA_CHECK(cudaEventRecord(stop_, stream_)); }

float CudaEventTimer::elapsed_ms() const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
}

float CudaEventTimer::stop_ms() {
    record_stop();
    CUDA_CHECK(ninfer::wait_for_event(stop_));
    return elapsed_ms();
}

CudaCompletionEvent::CudaCompletionEvent(const DeviceContext& ctx)
    : CudaCompletionEvent(ctx.rank(ctx.active_rank())) {}

CudaCompletionEvent::CudaCompletionEvent(const RankContext& rank) : device_(rank.device) {
    // The event belongs to this rank's device, which is the one that will record it.
    DeviceBinding bind(rank.device);
    const cudaError_t err = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreateWithFlags failed", err));
    }
}

CudaCompletionEvent::~CudaCompletionEvent() { destroy_event(event_); }

CudaCompletionEvent::CudaCompletionEvent(CudaCompletionEvent&& other) noexcept
    : device_(other.device_), event_(std::exchange(other.event_, nullptr)) {}

CudaCompletionEvent& CudaCompletionEvent::operator=(CudaCompletionEvent&& other) noexcept {
    if (this == &other) { return *this; }
    destroy_event(event_);
    device_ = other.device_;
    event_  = std::exchange(other.event_, nullptr);
    return *this;
}

void CudaCompletionEvent::record(cudaStream_t stream) {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not recordable");
    }
    CUDA_CHECK(cudaEventRecord(event_, stream));
}

void CudaCompletionEvent::wait(cudaStream_t stream) const {
    if (event_ == nullptr || stream == nullptr) {
        throw std::logic_error("CUDA completion event is not waitable");
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0));
}

bool CudaCompletionEvent::ready() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    const cudaError_t status = cudaEventQuery(event_);
    if (status == cudaSuccess) { return true; }
    if (status == cudaErrorNotReady) { return false; }
    CUDA_CHECK(status);
    return false;
}

void CudaCompletionEvent::synchronize() const {
    if (event_ == nullptr) { throw std::logic_error("CUDA completion event is empty"); }
    CUDA_CHECK(ninfer::wait_for_event(event_));
}

RankFenceSet::RankFenceSet(const DeviceContext& context) {
    events_.reserve(context.size());
    for (std::size_t rank = 0; rank < context.size(); ++rank) {
        events_.emplace_back(context.rank(rank));
    }
}

void RankFenceSet::record(RankStreams streams) {
    for (std::size_t rank = 0; rank < events_.size(); ++rank) { events_[rank].record(streams[rank]); }
}

void RankFenceSet::wait(RankStreams streams) const {
    for (std::size_t rank = 0; rank < events_.size(); ++rank) { events_[rank].wait(streams[rank]); }
}

bool RankFenceSet::ready() const {
    for (const CudaCompletionEvent& event : events_) {
        if (!event.ready()) { return false; }
    }
    return true;
}

void RankFenceSet::synchronize() const {
    for (const CudaCompletionEvent& event : events_) { event.synchronize(); }
}

} // namespace ninfer
