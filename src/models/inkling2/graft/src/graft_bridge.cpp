// graft bridge: destination-side definitions for donor symbols the grafted executor reads.
//
// Two kinds, deliberately separated:
//  1. REAL bridges — names the destination engine implements differently (Tensor's device-less
//     construction; ops::joins over the port's WeightView). First-class: these ARE the adaptation.
//  2. DISABLED-FEATURE stubs — constructor/method bodies for machinery the port's mandate drops
//     (expert delivery/parallel exchange, divided head, dspine runtime, cpu experts, vision).
//     A single-rank text server never calls them; the memory-plan queries return 0 and any live
//     call throws with the feature name, so a wrong path fails loudly, never silently.
// Promoting any of these to a real implementation = delete its stub, wire the port path, run the
// registered battery (graft-checklist §8 ledger tracks promotions).

#include "core/tensor.h"
#include "core/expert_delivery.h"
#include "core/expert_parallel.h"
#include "core/stage_link.h"
#include "models/divided_head.h"
#include "models/dspine/runtime.h"
#include "models/inkling2/cpu_experts.h"
#include "models/inkling2/model.h"
#include "models/inkling2/vision.h"
#include "ninfer/ops/weight_input.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace {

[[noreturn]] void graft_disabled(const char* what) {
    throw std::logic_error(std::string("graft v0: disabled feature touched: ") + what);
}

ninfer::Device current_device() {
    int id = 0;
    return cudaGetDevice(&id) == cudaSuccess ? ninfer::Device::cuda(id) : ninfer::Device::cuda(0);
}

} // namespace

namespace ninfer {

// --- Tensor: the donor's device-less construction over the port's device-carrying ctor --------
Tensor::Tensor(void* data, DType dtype, std::initializer_list<std::int32_t> shape)
    : Tensor(data, current_device(), dtype, shape) {}

// --- ops::joins: the donor predicate over the port's WeightView --------------------------------
namespace ops {

bool joins(std::span<const WeightInput> inputs) {
    if (inputs.empty()) { return false; }
    const WeightParent* parent = nullptr;
    std::uint64_t       end    = 0;
    for (const auto& input : inputs) {
        for (const auto& part : input.weight.parts) {
            if (!part.parent || (parent && (part.parent != parent || part.begin != end))) {
                return false;
            }
            parent = part.parent;
            end    = part.end;
        }
    }
    return parent != nullptr;
}

} // namespace ops

// --- ExpertDelivery / ExpertParallelExchange (expert parallelism: dropped, single rank) --------
// The pimpl: the stub link never constructs a real Impl; give unique_ptr cleanup a complete type.
// (Real definition lives in the donor's expert_delivery.cpp, which this link does not include.)
struct ExpertDelivery::Impl {};

ExpertDelivery::ExpertDelivery(const RankContext&, std::vector<std::filesystem::path>,
                               std::vector<ExpertDeliveryLayer>, ExpertDeliveryOptions) {
    graft_disabled("ExpertDelivery");
}
ExpertDelivery::~ExpertDelivery() noexcept {}
const void* const* ExpertDelivery::tables(std::size_t) const { graft_disabled("ExpertDelivery::tables"); }
bool ExpertDelivery::alias_aligned(std::size_t, std::size_t, std::size_t) const {
    graft_disabled("ExpertDelivery::alias_aligned");
}
std::string ExpertDelivery::cpu_description() const { graft_disabled("ExpertDelivery::cpu_description"); }
std::vector<std::vector<std::uint64_t>> ExpertDelivery::totals() const { graft_disabled("ExpertDelivery::totals"); }
ExpertDelivery::Call ExpertDelivery::begin(std::size_t, const int*, unsigned, cudaStream_t,
                                           std::span<const Input>) {
    graft_disabled("ExpertDelivery::begin");
}
ExpertDeliveryMemory ExpertDelivery::memory() const noexcept { return {}; }
ExpertDeliveryStats ExpertDelivery::stats() const { graft_disabled("ExpertDelivery::stats"); }
std::uint64_t ExpertDelivery::grow_cache() { graft_disabled("ExpertDelivery::grow_cache"); }
std::uint64_t ExpertDelivery::shrink_cache(std::uint64_t) { graft_disabled("ExpertDelivery::shrink_cache"); }
void ExpertDelivery::await(cudaStream_t) { graft_disabled("ExpertDelivery::await"); }
void ExpertDelivery::check() const { graft_disabled("ExpertDelivery::check"); }
void ExpertDelivery::drain() { graft_disabled("ExpertDelivery::drain"); }
void ExpertDelivery::prime(const std::vector<std::vector<std::uint64_t>>*) {
    graft_disabled("ExpertDelivery::prime");
}

ExpertParallelExchange::ExpertParallelExchange(const DeviceContext&, std::uint64_t, std::uint64_t) {
    graft_disabled("ExpertParallelExchange");
}
ExpertParallelExchange::~ExpertParallelExchange() noexcept {}
std::vector<void*> ExpertParallelExchange::collect(std::uint64_t, std::uint64_t, cudaStream_t) {
    graft_disabled("ExpertParallelExchange::collect");
}
std::uint64_t ExpertParallelExchange::send(const void*, std::uint64_t, cudaStream_t) {
    graft_disabled("ExpertParallelExchange::send");
}
void* ExpertParallelExchange::part(std::uint64_t) const { graft_disabled("ExpertParallelExchange::part"); }
void* ExpertParallelExchange::receive(std::uint64_t, std::uint64_t, std::uint64_t, cudaStream_t) {
    graft_disabled("ExpertParallelExchange::receive");
}
void ExpertParallelExchange::reply(std::uint64_t, std::uint64_t, std::uint64_t, cudaStream_t) {
    graft_disabled("ExpertParallelExchange::reply");
}

} // namespace ninfer

namespace ninfer::models {

// --- DividedHead (multi-rank logits split: dropped, single rank) --------------------------------
DividedHead::DividedHead(const DeviceContext&, const Weight&, std::int64_t, std::span<const double>,
                         std::uint64_t, int) {
    graft_disabled("DividedHead");
}
DividedHead::~DividedHead() noexcept {}
bool DividedHead::divides(const Weight&, std::uint64_t) { return false; }
void DividedHead::project(const void*, std::size_t, int, __nv_bfloat16*,
                          std::span<const cudaStream_t>, const Project&) {
    graft_disabled("DividedHead::project");
}

namespace dspine {

std::uint32_t ring_capacity(const Config&, std::uint32_t) { graft_disabled("dspine::ring_capacity"); }

Runtime::Runtime(const Model& model, Target, RuntimeOptions) : model_(model) {
    graft_disabled("dspine::Runtime");
}
Runtime::~Runtime() noexcept {}
std::uint64_t Runtime::planned_bytes(const Model&, DType, const RuntimeOptions&) {
    graft_disabled("dspine::Runtime::planned_bytes");
}
std::uint64_t Runtime::device_bytes() const noexcept { return 0; }
std::uint64_t Runtime::image_bytes(std::uint32_t) const noexcept { return 0; }
std::uint32_t Runtime::lane(std::uint32_t, std::uint32_t) const { graft_disabled("dspine::Runtime::lane"); }
int Runtime::tap_of(std::uint32_t) const noexcept { return -1; } // -1: drafter reads no layer
void Runtime::prepare(cudaStream_t) { graft_disabled("dspine::Runtime::prepare"); }
void Runtime::store_tap(std::uint32_t, const Tensor&, std::uint32_t, cudaStream_t) {
    graft_disabled("dspine::Runtime::store_tap");
}
void Runtime::commit(std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, cudaStream_t) {
    graft_disabled("dspine::Runtime::commit");
}
void Runtime::commit_rows(const Tensor&, const Tensor&, std::uint32_t, cudaStream_t) {
    graft_disabled("dspine::Runtime::commit_rows");
}
void Runtime::draft(std::span<const std::uint32_t>, std::span<const std::int32_t>,
                    std::span<const std::uint32_t>, std::uint32_t, std::span<std::int32_t>,
                    std::span<float>, cudaStream_t) {
    graft_disabled("dspine::Runtime::draft");
}
void Runtime::copy_lane(std::uint32_t, std::uint32_t, cudaStream_t) {
    graft_disabled("dspine::Runtime::copy_lane");
}

} // namespace dspine
} // namespace ninfer::models

namespace ninfer::models::inkling2 {

// --- VisionTower (text-only v0) ------------------------------------------------------------------
VisionTower::VisionTower(const Model&, unsigned) { graft_disabled("VisionTower"); }
VisionTower::~VisionTower() noexcept {}
std::uint64_t VisionTower::workspace_bytes() const { return 0; }
void VisionTower::encode(const __nv_bfloat16*, int, int, __nv_bfloat16*, WorkspaceArena&,
                         cudaStream_t) const {
    graft_disabled("VisionTower::encode");
}

// --- CpuExperts (CPU expert offload: dropped, device residency only) ----------------------------
CpuExperts::CpuExperts(std::vector<CpuExpertLayer>, std::vector<std::vector<float>>, int, int, int,
                       float, float) {
    graft_disabled("CpuExperts");
}
bool CpuExperts::computes(const CpuExpertLayer&, int, int) noexcept { return false; }
bool CpuExperts::supports(std::size_t) const { return false; } // CPU expert offload: disabled
ExpertCpuProducts::Done CpuExperts::compute(const ExpertCpuProducts::Work&) {
    graft_disabled("CpuExperts::compute");
}
std::uint64_t CpuExperts::activation_bytes(int) const { return 0; }
std::uint64_t CpuExperts::input_bytes() const { return 0; }
std::uint64_t CpuExperts::product_bytes() const { return 0; }
std::uint64_t CpuExperts::weight_bytes(int) const { return 0; }

// --- Model: this-rank expert range (single rank owns everything) ---------------------------------
std::pair<std::uint32_t, std::uint32_t> Model::expert_range(std::size_t) const {
    graft_disabled("Model::expert_range");
}

// Model::input is defined by the donor load layer (model.cpp), which this link excludes — the
// graft Model is populated by the port loader at serve integration (M4 step 2). Touching input()
// before then is a wiring bug. Ledger: §8 model.cpp promotion.
ops::WeightInput Model::input(models::qwen3_5::WeightId) const {
    graft_disabled("Model::input (load layer not linked; M4 step 2)");
}

} // namespace ninfer::models::inkling2

// --- Launch layer: raise_dynamic_shared_memory (donor core/device_launch.cpp) --------------------
// The port engine already OWNS cuda_check (src/core/device.cu) — destination owns the name, so
// only the per-kernel dynamic-shared-memory attribute cache is grafted here. sm_75's 64 KiB cap
// makes it rarely exercised, but the ggml mma kernels call it on every large-tile route.
#include "core/device_launch.h"
#include <cuda.h> // cuCtxGetCurrent (driver API): DeviceBinding donor transplant
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>

namespace ninfer {

void raise_dynamic_shared_memory(const void* kernel, int bytes) {
    static std::mutex mutex;
    static std::map<std::pair<int, const void*>, int> raised;
    int device = -1;
    CUDA_CHECK(cudaGetDevice(&device));
    const std::scoped_lock lock(mutex);
    int& limit = raised[{device, kernel}];
    if (limit >= bytes) { return; }
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    limit = bytes;
}

// --- Multi-rank queries (donor tensor_parallel.cu) — single rank: touching them is a bug --------
// Executor::divide_head() calls these; DividedHead is a disabled feature (throws on construction),
// so reaching a rank query means a disabled path was wired. Ledger: §8 disabled features.
std::vector<std::uint64_t> rank_free_bytes(const DeviceContext&) {
    graft_disabled("rank_free_bytes (single rank)");
}
std::vector<double> rank_copy_bandwidth(const DeviceContext&) {
    graft_disabled("rank_copy_bandwidth (single rank)");
}

// StageLink (cudaIpc stage pipes between ranks): never constructed single-rank; the dtor exists
// because ExpertParallelExchange::Peer's vector dtor instantiates it (donor stage_link.cu).
StageLink::~StageLink() noexcept {}

// --- DeviceBinding / DeviceContext rank queries (donor core/device.cu transplant) ----------------
// The graft does not compile donor device.cu (38x LNK2005 with the engine's own device.cu over
// the same mangled ctor names). These donor member functions have no port counterpart, so the
// ones the grafted code calls are transplanted here verbatim in behavior.
namespace {
[[noreturn]] void graft_cuda_throw(const char* prefix, cudaError_t err) {
    throw std::runtime_error(std::string(prefix) + ": " + cudaGetErrorName(err) + ": " +
                             cudaGetErrorString(err));
}
} // namespace

DeviceBinding::DeviceBinding(int device) {
    const cudaError_t get = cudaGetDevice(&previous_);
    if (get != cudaSuccess) { graft_cuda_throw("cudaGetDevice failed", get); }
    if (previous_ == device) { return; }
    // Donor #279: a thread with no current context must not create device 0's primary context as
    // a side effect; only restore a binding if this thread actually had one.
    CUcontext current = nullptr;
    restore_          = cuCtxGetCurrent(&current) == CUDA_SUCCESS && current != nullptr;
    const cudaError_t set = cudaSetDevice(device);
    if (set != cudaSuccess) { graft_cuda_throw("cudaSetDevice failed", set); }
}

DeviceBinding::~DeviceBinding() noexcept {
    if (restore_) {
        const cudaError_t err = cudaSetDevice(previous_);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "CUDA cleanup failed during cudaSetDevice: %s: %s\n",
                         cudaGetErrorName(err), cudaGetErrorString(err));
        }
    }
}

std::size_t DeviceContext::size() const noexcept { return endpoints_.size(); }

const RankContext& DeviceContext::rank(std::size_t index) const {
    if (index >= endpoints_.size()) { throw std::out_of_range("CUDA device rank is out of range"); }
    return endpoints_[index];
}

} // namespace ninfer
