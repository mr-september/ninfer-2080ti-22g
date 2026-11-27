#include "models/inkling2/executor.h"

#include "core/decode_graph.h"
#include "core/device_buffer.h"
#include "core/device_wait.h"
#include "core/expert_parallel.h"
#include "core/growable_region.h"
#include "core/mapped_host_copy.h"
#include "core/pinned_memory.h"
#include "core/qtype.h"
#include "core/tensor_parallel.h"
#include "core/vmm.h"
#include "core/weight_view.h"
#include "models/divided_head.h"
#include "models/dspine/runtime.h"
#include "models/inkling2/cpu_experts.h"
#include "models/inkling2/kernels.h"
#include "models/inkling2/kv_planes.h"
#include "models/inkling2/vision.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_policy.h"
#include "ninfer/ops/weight_input.h"
#include "ops/linear/gguf/ggml_bridge.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::inkling2 {
namespace {

namespace gguf = ops::gguf;

constexpr std::size_t kAlign = 256;
constexpr int kD             = 128; // head width
constexpr int kTaps          = 4;
constexpr int kPool          = 8; // residual states one call keeps at once

std::size_t round_up(std::size_t value, std::size_t to) { return (value + to - 1) / to * to; }

const float* fp32(const Model& model, WeightId id, std::int32_t count) {
    const Tensor tensor = weight_tensor(model.weight(id).view, {count});
    if (tensor.dtype != DType::FP32) {
        throw std::invalid_argument(model.weight(id).name + " must be stored as fp32");
    }
    return static_cast<const float*>(tensor.data);
}

Weight matrix(const Model& model, std::initializer_list<WeightId> ids) {
    std::vector<ops::WeightInput> inputs;
    for (const WeightId id : ids) { inputs.push_back(model.input(id)); }
    if (inputs.size() == 1) { return ops::prepare_linear_weight(inputs.front()).weight; }
    if (!ops::joins(inputs)) {
        throw std::invalid_argument("inkling2: projection parts are not one stored parent");
    }
    return ops::prepare_linear_weight(std::span<const ops::WeightInput>(inputs)).weight;
}

// The matrices of `ids`: one when they are rows of one stored parent, else each on its own (a
// mixed-format GGUF stores Q, the gate and the relative feature in different formats).
std::vector<Weight> matrices(const Model& model, std::initializer_list<WeightId> ids) {
    std::vector<ops::WeightInput> inputs;
    for (const WeightId id : ids) { inputs.push_back(model.input(id)); }
    if (inputs.size() > 1 && ops::joins(inputs)) {
        return {ops::prepare_linear_weight(std::span<const ops::WeightInput>(inputs)).weight};
    }
    std::vector<Weight> out;
    for (const auto& input : inputs) { out.push_back(ops::prepare_linear_weight(input).weight); }
    return out;
}

void require_gguf(const Model& model, WeightId id, const Weight& w) {
    if (!is_gguf(w.qtype) && w.qtype != QType::BF16) {
        throw std::invalid_argument(model.weight(id).name + ": Inkling 2 runs GGUF block or BF16 "
                                    "matrices (" + std::to_string(int(w.qtype)) + ")");
    }
}

std::int64_t row_bytes(const Weight& w) {
    if (w.qtype == QType::BF16) { return std::int64_t(w.k) * 2; }
    const auto block = gguf_block_shape(w.qtype);
    return std::int64_t(w.k / block.elements) * block.bytes;
}

// A context-sized cache plane: rows of `row_bytes`, `rows` of them for each owner (a sequence's
// KV heads), `position_bytes` a position. Owned whole, or, beside an elastic expert cache, a
// growable region whose rows are backed only as far as their owner has reached (Impl::back).
struct KvArea {
    DeviceBuffer owned;
    GrowableRegion region;
    void* p = nullptr;
    std::size_t bytes = 0, row_bytes = 0, position_bytes = 0;
    int rows = 0;
    std::vector<std::uint32_t> backed; // positions backed, by owner

    KvArea() = default;
    KvArea(std::size_t owners, int rows_per_owner, std::size_t row, std::size_t position,
           bool growing, int device)
        : bytes(owners * std::size_t(rows_per_owner) * row), row_bytes(row), position_bytes(position),
          rows(rows_per_owner), backed(owners, 0) {
        if (growing) {
            region = GrowableRegion(bytes, {.device = device});
            p      = region.data();
        } else {
            owned = DeviceBuffer(bytes);
            owned.fill(0);
            p = owned.p;
        }
    }
};

// A stored layer's matrices and vectors.
struct LayerPlan {
    bool full = false;
    std::vector<Weight> query; // one [Q; gate; relative] parent, or its parts
    int query_rows = 0;
    Weight key, value, output;
    const float *query_norm = nullptr, *key_norm = nullptr, *value_norm = nullptr,
                *input_norm = nullptr;
    const float *sinks = nullptr, *query_clip = nullptr, *relative_clip = nullptr,
                *distances = nullptr, *key_conv = nullptr, *value_conv = nullptr,
                *output_conv = nullptr, *key_taps = nullptr, *value_taps = nullptr;
    const float *attention_mix = nullptr, *ffn_mix = nullptr, *ffn_norm = nullptr;
    int mix_queries = 0;
    // MoE.
    const float *router = nullptr, *router_bias = nullptr, *moe_conv = nullptr,
                *gate_up_scale = nullptr, *down_scale = nullptr;
    int delivery_layer = -1;
    gguf::GgmlType gate_up_type{}, down_type{};
    bool bf16 = false; // BF16 experts (both banks), multiplied by kernels::bf16_expert_rows
    std::int64_t gate_up_row_bytes = 0, down_row_bytes = 0;
    DeviceBuffer tables; // pointers: [E] gate, [E] up, [E] gate_up, [E] down
    bool gate_up_aligned16 = true, down_aligned16 = true;
    std::vector<DeviceBuffer> peer_tables; // expert parallelism: rank r's (r > 0)
    // A core layer's message queries [4][H] and its controllers of Q, K, V and the MoE input.
    const float* loop_queries = nullptr;
    std::array<const float*, 4> control_down{}, control_gain{}, control_forget{};
};

// One visit of a stored layer: its caches.
struct VisitPlan {
    std::uint32_t layer = 0, pass = 0;
    bool full = false;
    bool save_prelude = false; // the core's first visit of the first pass
    bool restart      = false; // the core's first visit of a later pass
    bool control      = false; // a later pass's core visit
    bool start        = false; // the layer starts a residual block
    // The F16 K and V caches: [sequences][kv heads][slots][D]. A quantized full visit has none;
    // its rows live in its code and scale planes.
    KvArea k, v;
    // The quantized planes' codes and scales (a full visit of a quantized run): the same
    // [sequences][kv heads][slots] addressing, one row of codes and one of scales per slot.
    KvArea k_codes, k_scales, v_codes, v_scales;
    int slots = 0, ring = 0;
    std::int64_t seq_stride = 0, head_stride = 0;
    std::vector<DeviceBuffer> checkpoint; // sliding: every sequence's rings, K then V
};

struct GraphEntry {
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    bool warmed = false;
};

// A DSpark drafter layer: its matrices (rows of the stored parents), vectors and caches.
struct SparkLayer {
    Weight query, key, value, output, context_key, context_value, gate, up, down;
    const float *context_norm = nullptr, *attention_norm = nullptr, *ffn_norm = nullptr,
                *query_norm = nullptr, *key_norm = nullptr, *context_key_norm = nullptr;
    VisitPlan cache; // full, k, v, slots, ring, strides, checkpoint
};

// The DSpark drafter (Model::dspark()): its layers, the taps it reads, every sequence's ring of
// recurrent-head contexts ([sequences][kR][hidden] FP32, position p in slot p % kR) and its
// checkpoint, the head's table of the recurrent correction (M BF16 [vocab][256], c FP32 [vocab],
// prepared once), and a round's device state.
struct Spark {
    const DSparkConfig* config = nullptr;
    std::vector<SparkLayer> layers;
    const float *fuse = nullptr, *context_norm = nullptr, *final_norm = nullptr,
                *mask_embedding = nullptr, *joint = nullptr, *joint_bias = nullptr,
                *hidden_out = nullptr, *hidden_out_bias = nullptr;
    const __nv_bfloat16* markov_embed = nullptr;
    // tap_after[v]: the stored tap (0..taps - 2) visit v's output is, or -1; the last tap is the
    // final hidden.
    std::vector<int> tap_after;
    int taps = 0, block = 0, window = 0;
    DeviceBuffer contexts, contexts_kept, table_m, table_c, state, prev, drafts, confidence;
    std::unique_ptr<PinnedHostBuffer> host;
    std::unique_ptr<CudaCompletionEvent> copied;
    bool prepared = false;
    std::uint64_t bytes = 0; // beside the caches (in kv_bytes)
};

// The residual states of a call: buffers of [T][H] FP32 rows, and which of them the blocks, the
// current state, the prelude's output and the previous pass's states are.
struct States {
    std::array<float*, kPool> buffers{};
    std::vector<int> blocks, prelude_blocks, bank;
    int cur = -1, prelude_cur = -1, bank_last = -1;

    [[nodiscard]] bool used(int i) const {
        const auto in = [i](const std::vector<int>& list) {
            return std::find(list.begin(), list.end(), i) != list.end();
        };
        return i == cur || i == prelude_cur || i == bank_last || in(blocks) ||
               in(prelude_blocks) || in(bank);
    }
    [[nodiscard]] int acquire() const {
        for (int i = 0; i < kPool; ++i) {
            if (!used(i)) { return i; }
        }
        throw std::logic_error("inkling2: residual state pool exhausted");
    }
};

} // namespace

struct Executor::Impl {
    const Model& model;
    DeviceContext& device;
    ExecutorOptions options;
    const TextConfig& config;
    cudaStream_t stream = nullptr;
    int hidden = 0, query_heads = 0, kv_heads = 0, vocab = 0, window = 0, ring = 0,
        global_slots = 0, prefill_ring = 0;
    int kvd = 0, slot_floats = 0; // K's (and V's) width, a history slot's floats
    float eps = 1e-6f;
    KvFormat kvfmt; // the caches' storage: F16, or the quantized full-attention visits of rk8v4/rk4v4
    DeviceBuffer staging_k, staging_v;
    bool growing = false;
    std::vector<KvArea*> sequence_areas;
    std::atomic<std::uint64_t> kv_backed{0};
    std::uint64_t kv_fixed = 0;
    std::vector<LayerPlan> layers;
    std::vector<VisitPlan> visits;
    // The convolutions' histories: [sequences][visits][kHistory][slot_floats], each slot K's,
    // V's, the attention output's and the MoE output's raw inputs; and their checkpoint.
    DeviceBuffer history, history_kept;
    std::uint64_t history_bytes = 0;
    std::unique_ptr<CpuExperts> cpu_experts;
    std::unique_ptr<ExpertDelivery> delivery;

    struct ExpertPeer {
        cudaStream_t stream = nullptr;
        std::unique_ptr<WorkspaceArena> workspace;
    };
    std::vector<ExpertPeer> expert_peers;
    std::unique_ptr<ExpertParallelExchange> expert_exchange;
    DeviceBuffer expert_pack;
    std::unique_ptr<models::DividedHead> divided_head;
    mutable std::mutex expert_stats_mutex;
    ExpertDeliveryStats expert_startup_stats;
    // head: the vocabulary's rows; head_tiles: the stored rows the matrix kernel reads for them,
    // whole 128-row tiles (the GGUF pads the vocabulary past them).
    Weight embedding, head, head_tiles;
    // A model with BF16 matrices runs every call on BF16 rows: the BF16 products read no q8_1 form.
    bool bf16_rows = false;
    const float *embedding_norm = nullptr, *final_norm = nullptr, *output_mix = nullptr;
    std::vector<std::uint32_t> positions;
    // The position at which each sequence's rings last came whole from a prompt, a checkpoint or
    // an image (rewind()).
    std::vector<std::uint32_t> ring_floor;
    DeviceBuffer d_tokens, d_positions, d_sequences, d_starts;
    std::unique_ptr<PinnedHostBuffer> h_staging;
    std::unique_ptr<CudaCompletionEvent> staged;
    DeviceBuffer logits_bf16;
    std::unique_ptr<WorkspaceArena> workspace;
    std::size_t workspace_bytes = 0;
    std::uint64_t kv_bytes      = 0;
    int decode_splits_full = 1, decode_splits_sliding = 1;
    std::map<std::uint32_t, GraphEntry> graphs;
    std::vector<std::uint32_t> verify_widths;
    std::unique_ptr<dspine::Runtime> spine;
    std::unique_ptr<Spark> spark;
    // The vision tower and its buffers: an image's patches, a span's rows.
    std::unique_ptr<VisionTower> tower;
    DeviceBuffer d_patches, d_image_rows;

    static constexpr int kR = Executor::kHistory;

    Impl(const Model& m, DeviceContext& d, ExecutorOptions o)
        : model(m), device(d), options(o), config(m.config()) {
        if (options.sequences == 0 || options.prefill_chunk == 0 || options.max_context == 0) {
            throw std::invalid_argument("inkling2 executor: invalid options");
        }
        if (device.size() != 1 && !model.expert_parallel()) {
            throw std::invalid_argument("Inkling 2 runs on one device, or divides its experts over "
                                        "several");
        }
        if (options.draft_tokens > std::uint32_t(gguf::kMaxVectorColumns) - 1) {
            throw std::invalid_argument("inkling2 executor: at most 7 drafts a round");
        }
        if (options.dspark &&
            (!model.dspark() || options.dspine || options.draft_tokens == 0 ||
             options.draft_tokens + 1 > model.dspark_config()->block_size)) {
            throw std::invalid_argument("inkling2 executor: DSpark drafting needs the model's "
                                        "drafter and 1 to its block less one drafts a round");
        }
        stream = device.stream;
        const int device_id = device.rank(0).device;
        kvfmt = kv_format(o.kv_cache);
        growing = !model.expert_banks().empty() && !options.expert_cache_bytes &&
                  vmm::granularity(device_id) != 0;
        hidden      = int(config.hidden_size);
        query_heads = int(config.num_attention_heads);
        kv_heads    = int(config.num_key_value_heads);
        vocab       = int(config.unpadded_vocab_size);
        window      = int(config.sliding_window);
        eps         = config.rms_norm_eps;
        kvd         = kv_heads * kD;
        slot_floats = 2 * kvd + 2 * hidden;
        if (int(config.head_dim) != kD || int(config.conv_taps) != kTaps) {
            throw std::invalid_argument("inkling2 executor: heads of 128 values and 4 taps");
        }
        if (options.draft_tokens + 3 >= std::uint32_t(kR)) {
            throw std::invalid_argument("inkling2 executor: drafts beyond the histories");
        }
        options.prefill_chunk = std::min<std::uint32_t>(options.prefill_chunk, 4096);
        // A verification writes a round's drafts past the window before keeping only some, and
        // rewind() goes back as far as the histories allow (kR less the taps and the drafts): the
        // ring holds the window and kR slots beyond it, no more, since 40 visits keep one each.
        ring         = int(round_up(config.sliding_window + kR, 16));
        prefill_ring = int(round_up(config.sliding_window + options.prefill_chunk, 256));
        global_slots = int(round_up(options.max_context + 1, 256));
        positions.assign(options.sequences, 0);
        ring_floor.assign(options.sequences, 0);
        const auto& w  = model.weights();
        embedding      = native_weight(model.weight(w.token_embedding).view);
        head           = matrix(model, {w.output_head});
        head_tiles     = head;
        head_tiles.n   = int(round_up(std::size_t(vocab), 128));
        if (head.n < vocab || head_tiles.n > head.n) {
            throw std::invalid_argument("inkling2: the head holds " + std::to_string(head.n) +
                                        " rows, not the vocabulary in whole 128-row tiles");
        }
        head.n = vocab;
        embedding_norm = fp32(model, w.embedding_norm, hidden);
        final_norm     = fp32(model, w.final_norm, hidden);
        output_mix     = fp32(model, w.output_mix, hidden);
        require_gguf(model, w.token_embedding, embedding);
        require_gguf(model, w.output_head, head);
        bf16_rows = embedding.qtype == QType::BF16 || head.qtype == QType::BF16;
        if (options.dspine && (model.dspine() == nullptr || options.draft_tokens == 0 ||
                               options.draft_tokens + 1 > model.dspine()->config().block_size)) {
            throw std::invalid_argument("inkling2 executor: DSpine drafting needs the model's "
                                        "drafter and 1 to its block less one drafts a round");
        }
        const auto schedule  = config.visits();
        const int sequences  = int(options.sequences);
        const std::size_t row_f16 = std::size_t(kD) * sizeof(__half);
        // Bytes of one full visit's K and V rows of one slot: F16 halves, or a quantized row's
        // codes and scales.
        const auto plane_bytes = [](KvCoding c) {
            return std::size_t(kv_code_bytes(c)) + std::size_t(kv_scale_bytes(c));
        };
        const std::size_t full_slot_bytes =
            kvfmt.quantized()
                ? std::size_t(kv_heads) * (plane_bytes(kvfmt.key.coding) +
                                           plane_bytes(kvfmt.value.coding)) * std::size_t(sequences)
                : 2 * std::size_t(kv_heads) * row_f16 * std::size_t(sequences);
        {
            std::size_t needed = 0;
            for (const auto& [layer, pass] : schedule) {
                needed += config.full_attention[layer]
                              ? std::size_t(global_slots) * full_slot_bytes
                              : (options.checkpoints ? 2 : 1) * std::size_t(ring) *
                                    2 * std::size_t(kv_heads) * row_f16 * std::size_t(sequences);
            }
            needed += (options.checkpoints ? 2 : 1) * std::size_t(sequences) * schedule.size() *
                      kR * std::size_t(slot_floats) * 4;
            needed += std::size_t(prefill_ring) * 2 * kv_heads * row_f16;
            if (options.dspine) {
                needed += dspine::Runtime::planned_bytes(*model.dspine(), DType::FP32,
                                                         spine_options());
            }
            if (options.dspark) { needed += spark_planned_bytes(); }
            std::size_t free_bytes = 0, total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            const std::size_t reserve = start_reserve();
            if (needed + reserve > free_bytes) {
                throw InsufficientMemory(
                    "Inkling 2: the KV caches, histories, checkpoints and drafter state of " +
                    std::to_string(sequences) + " sequence(s) of " +
                    std::to_string(options.max_context) + " tokens need " +
                    std::to_string(needed >> 20) + " MiB beside " + std::to_string(reserve >> 20) +
                    " MiB for the workspace and CUDA graphs, and " +
                    std::to_string(free_bytes >> 20) +
                    " MiB are free after the weights; lower --max-concurrency or --max-context");
            }
        }
        for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
            layers.push_back(plan_layer(i));
        }
        for (std::size_t v = 0; v < schedule.size(); ++v) {
            const auto [layer, pass] = schedule[v];
            VisitPlan p;
            p.layer        = layer;
            p.pass         = pass;
            p.full         = config.full_attention[layer];
            p.save_prelude = config.core(layer) && layer == config.core_start && pass == 0;
            p.restart      = config.core(layer) && layer == config.core_start && pass > 0;
            p.control      = config.core(layer) && pass > 0;
            p.start        = layer % config.block_size == 0;
            p.slots        = p.full ? global_slots : ring;
            p.ring         = p.full ? 0 : ring;
            p.head_stride  = std::int64_t(p.slots) * kD;
            p.seq_stride   = p.head_stride * kv_heads;
            const bool grows = growing && p.full;
            if (p.full && kvfmt.quantized()) {
                // One code and one scale area per plane; a row is one head's whole block of
                // slots (the F16 caches' row shape), position_bytes one slot's codes or scales.
                const auto area = [&](KvCoding coding) {
                    const std::size_t code  = std::size_t(kv_code_bytes(coding));
                    const std::size_t scale = std::size_t(kv_scale_bytes(coding));
                    KvArea codes(std::size_t(sequences), kv_heads,
                                 std::size_t(p.slots) * code, code, grows, device_id);
                    kv_bytes += codes.bytes;
                    KvArea scales(std::size_t(sequences), kv_heads,
                                  std::size_t(p.slots) * scale, scale, grows, device_id);
                    kv_bytes += scales.bytes;
                    return std::pair<KvArea, KvArea>(std::move(codes), std::move(scales));
                };
                auto k     = area(kvfmt.key.coding);
                p.k_codes  = std::move(k.first);
                p.k_scales = std::move(k.second);
                auto v      = area(kvfmt.value.coding);
                p.v_codes   = std::move(v.first);
                p.v_scales  = std::move(v.second);
            } else {
                for (KvArea* area : {&p.k, &p.v}) {
                    *area = KvArea(std::size_t(sequences), kv_heads,
                                   std::size_t(p.head_stride) * sizeof(__half), row_f16, grows,
                                   device_id);
                    kv_bytes += area->bytes;
                }
            }
            if (!p.full && options.checkpoints) {
                for (int c = 0; c < 2; ++c) {
                    p.checkpoint.emplace_back(std::size_t(p.seq_stride) * sizeof(__half) * sequences);
                    kv_bytes += p.checkpoint.back().bytes;
                }
            }
            visits.push_back(std::move(p));
        }
        {
            history_bytes = std::size_t(sequences) * visits.size() * kR * slot_floats * 4;
            history       = DeviceBuffer(history_bytes);
            history.fill(0);
            if (options.checkpoints) {
                history_kept = DeviceBuffer(history_bytes);
                history_bytes *= 2;
            }
            kv_bytes += history_bytes;
            const std::size_t staging = std::size_t(prefill_ring) * kv_heads * row_f16;
            staging_k = DeviceBuffer(staging);
            staging_v = DeviceBuffer(staging);
            staging_k.fill(0);
            staging_v.fill(0);
            kv_bytes += 2 * staging;
        }
        // A full visit's decode or verification row splits its keys so that one row's blocks
        // number about 32 per multiprocessor, at most 512 (the combine's capacity); the kernel
        // uses fewer while the keys are short (active_splits). Every row of every call splits
        // alike, so a verification row computes what a decode step at its position computes. On
        // an RTX 3090 after 120K tokens 32 (328 splits) decoded at 67.1 tok/s, 16 at 66.7, 64
        // (512) at 67.0 but 1.2% slower after 32K, and 128 splits at 65.5; 2 a multiprocessor
        // decoded 4-40% slower.
        const int sms         = device.multiprocessor_count();
        decode_splits_full =
            std::clamp((32 * sms + kv_heads - 1) / std::max(1, kv_heads), 1, 512);
        decode_splits_sliding = 16;
        const std::size_t rows = std::max<std::size_t>(options.prefill_chunk, sequences);
        d_tokens    = DeviceBuffer(rows * 4);
        d_positions = DeviceBuffer(rows * 4);
        d_sequences = DeviceBuffer(rows * 4);
        d_starts    = DeviceBuffer(rows * 4);
        h_staging   = std::make_unique<PinnedHostBuffer>(kStagingRegions * rows * 4);
        staged      = std::make_unique<CudaCompletionEvent>(device);
        logits_bf16 = DeviceBuffer(std::size_t(vocab) * kMaxLogitRows * 2);
        if (options.dspark) { plan_spark(device_id, sequences); }
        for (VisitPlan& p : visits) {
            if (!p.full) { continue; }
            if (p.k_codes.p != nullptr) {
                for (KvArea* area : {&p.k_codes, &p.k_scales, &p.v_codes, &p.v_scales}) {
                    sequence_areas.push_back(area);
                }
            } else {
                sequence_areas.push_back(&p.k);
                sequence_areas.push_back(&p.v);
            }
        }
        if (spark) {
            for (SparkLayer& l : spark->layers) {
                if (!l.cache.full) { continue; }
                sequence_areas.push_back(&l.cache.k);
                sequence_areas.push_back(&l.cache.v);
            }
        }
        kv_fixed = kv_bytes;
        if (growing) {
            for (KvArea* area : sequence_areas) { kv_fixed -= area->bytes; }
        }
        kv_backed       = kv_fixed;
        if (options.vision_max_patches > 0) { plan_vision(); }
        workspace_bytes = plan_workspace();
        if (tower) { workspace_bytes = std::max(workspace_bytes, tower->workspace_bytes()); }
        {
            std::size_t free_bytes = 0, total_bytes = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
            if (workspace_bytes + kGraphReserve + unbacked() > free_bytes) {
                throw InsufficientMemory(
                    "Inkling 2: the workspace of " + std::to_string(workspace_bytes >> 20) +
                    " MiB and " + std::to_string(kGraphReserve >> 20) +
                    " MiB for the CUDA graphs need more than the " +
                    std::to_string(free_bytes >> 20) +
                    " MiB free after the caches; lower --max-concurrency, --max-context or the "
                    "prompt chunk");
            }
        }
        workspace = std::make_unique<WorkspaceArena>(workspace_bytes);
        if (options.dspine) { make_spine(); }
        allocate_expert_peers();
        if (!model.expert_banks().empty()) { plan_delivery(); }
    }

    LayerPlan plan_layer(std::uint32_t i) {
        const LayerWeights& lw = model.weights().layers[i];
        const AttentionWeights& aw = lw.attention;
        const int q = query_heads * kD, rel = int(config.relative_width);
        LayerPlan p;
        p.full  = config.full_attention[i];
        p.query = matrices(model, {aw.query, aw.gate, aw.relative});
        for (const Weight& part : p.query) { p.query_rows += part.n; }
        if (p.query_rows != 2 * q + query_heads * rel) {
            throw std::invalid_argument("inkling2: the query projections' rows");
        }
        p.key           = matrix(model, {aw.key});
        p.value         = matrix(model, {aw.value});
        p.output        = matrix(model, {aw.output});
        p.sinks         = fp32(model, aw.sinks, query_heads * hidden);
        p.query_clip    = fp32(model, aw.query_clip, q);
        p.relative_clip = fp32(model, aw.relative_clip, query_heads * rel);
        p.distances = fp32(model, aw.distances,
                           int(p.full ? config.relative_extent : config.sliding_window) * rel);
        p.key_conv    = fp32(model, aw.key_conv, kvd * kTaps);
        p.value_conv  = fp32(model, aw.value_conv, kvd * kTaps);
        p.output_conv = fp32(model, aw.output_conv, hidden * kTaps);
        if (p.full) {
            p.query_norm = fp32(model, *aw.query_norm, hidden);
            p.key_norm   = fp32(model, *aw.key_norm, hidden);
            p.value_norm = fp32(model, *aw.value_norm, hidden);
            p.key_taps   = fp32(model, *aw.key_taps, kv_heads * kTaps * hidden);
            p.value_taps = fp32(model, *aw.value_taps, kv_heads * kTaps * hidden);
        } else {
            p.input_norm = fp32(model, *aw.input_norm, hidden);
        }
        p.mix_queries = p.full ? 3 : 1;
        if (lw.attention_mix) {
            p.attention_mix = fp32(model, *lw.attention_mix, p.mix_queries * hidden);
        }
        p.ffn_mix  = fp32(model, lw.ffn_mix, hidden);
        p.ffn_norm = fp32(model, lw.ffn_norm, hidden);
        require_gguf(model, aw.key, p.key);
        require_gguf(model, aw.value, p.value);
        require_gguf(model, aw.output, p.output);
        for (const Weight& m : p.query) { require_gguf(model, aw.query, m); }
        for (const Weight* m : {&p.key, &p.value, &p.output}) {
            bf16_rows = bf16_rows || m->qtype == QType::BF16;
        }
        for (const Weight& m : p.query) { bf16_rows = bf16_rows || m.qtype == QType::BF16; }
        const MoeWeights& mw = lw.moe;
        const int e = int(config.num_experts), width = int(config.moe_intermediate_size);
        p.router      = fp32(model, mw.router, e * hidden);
        p.router_bias = fp32(model, mw.router_bias, e);
        p.moe_conv    = fp32(model, mw.output_conv, hidden * kTaps);
        if (mw.gate_up_scale) { p.gate_up_scale = fp32(model, *mw.gate_up_scale, e); }
        if (mw.down_scale) { p.down_scale = fp32(model, *mw.down_scale, e); }
        if (model.expert_residency() == ExpertResidency::Device) {
            const auto bank_tables = [&](WeightId gate_up_id, WeightId down_id, std::uint32_t first,
                                         std::uint32_t last) {
                const Weight gate_up = matrix(model, {gate_up_id});
                const Weight down    = matrix(model, {down_id});
                const bool bf16      = gate_up.qtype == QType::BF16 && down.qtype == QType::BF16;
                if (!bf16 && (!is_gguf(gate_up.qtype) || !is_gguf(down.qtype))) {
                    throw std::invalid_argument("Inkling 2 experts must be GGUF block matrices or "
                                                "BF16 (both banks)");
                }
                const auto gate_up_type = bf16 ? gguf::GgmlType{} : ops::detail::gguf_type(gate_up.qtype);
                const auto down_type    = bf16 ? gguf::GgmlType{} : ops::detail::gguf_type(down.qtype);
                if (p.gate_up_row_bytes == 0) {
                    p.bf16              = bf16;
                    p.gate_up_type      = gate_up_type;
                    p.down_type         = down_type;
                    p.gate_up_row_bytes = row_bytes(gate_up);
                    p.down_row_bytes    = row_bytes(down);
                } else if (bf16 != p.bf16 || gate_up_type != p.gate_up_type ||
                           down_type != p.down_type) {
                    throw std::invalid_argument("Inkling 2: one layer's experts in different "
                                                "formats");
                }
                std::vector<const void*> table(4 * std::size_t(e), nullptr);
                for (std::uint32_t x = first; x < last; ++x) {
                    const auto* gu = static_cast<const std::byte*>(gate_up.qdata) +
                                     std::int64_t(x - first) * 2 * width * p.gate_up_row_bytes;
                    table[x]         = gu;
                    table[e + x]     = gu + std::int64_t(width) * p.gate_up_row_bytes;
                    table[2 * e + x] = gu;
                    table[3 * e + x] = static_cast<const std::byte*>(down.qdata) +
                                       std::int64_t(x - first) * hidden * p.down_row_bytes;
                }
                bool gate_up_aligned = true, down_aligned = true;
                for (std::size_t j = 0; j < table.size(); ++j) {
                    const auto address = reinterpret_cast<std::uintptr_t>(table[j]);
                    const bool aligned = table[j] == nullptr || address % 16 == 0;
                    (j < 3 * std::size_t(e) ? gate_up_aligned : down_aligned) &= aligned;
                }
                p.gate_up_aligned16 = p.gate_up_aligned16 && gate_up_aligned;
                p.down_aligned16    = p.down_aligned16 && down_aligned;
                DeviceBuffer buffer(table.size() * sizeof(void*));
                buffer.copy_from_host(table.data(), table.size() * sizeof(void*));
                return buffer;
            };
            if (!model.expert_parallel()) {
                p.tables = bank_tables(*mw.gate_up, *mw.down, 0, std::uint32_t(e));
            } else {
                p.peer_tables.resize(device.size());
                for (std::size_t r = 0; r < device.size(); ++r) {
                    const auto [first, last] = model.expert_range(r);
                    RankBinding bind(device, r);
                    (r == 0 ? p.tables : p.peer_tables[r]) = bank_tables(
                        mw.gate_up_shares.at(r), mw.down_shares.at(r), first, last);
                }
            }
        } else {
            p.delivery_layer    = int(i);
            const auto& bank    = model.expert_banks()[i];
            p.bf16 = bank.gate_up_type == QType::BF16 && bank.down_type == QType::BF16;
            if (!p.bf16) {
                p.gate_up_type = ops::detail::gguf_type(bank.gate_up_type);
                p.down_type    = ops::detail::gguf_type(bank.down_type);
            }
            p.gate_up_row_bytes = std::int64_t(bank.gate_up_row_bytes);
            p.down_row_bytes    = std::int64_t(bank.down_row_bytes);
        }
        bf16_rows = bf16_rows || p.bf16;
        if (!p.bf16 && (!gguf::moe_matrix_supported(p.gate_up_type, 2 * width, hidden, true) ||
                        !gguf::moe_matrix_supported(p.down_type, hidden, width, true))) {
            throw std::invalid_argument("Inkling 2: no integer expert kernel for these expert "
                                        "formats");
        }
        if (lw.loop) {
            const int r     = int(config.gate_rank);
            p.loop_queries  = fp32(model, lw.loop->queries, 4 * hidden);
            for (std::size_t c = 0; c < 4; ++c) {
                const ControllerWeights& cw = lw.loop->controllers[c];
                p.control_down[c]   = fp32(model, cw.down, r * 2 * hidden);
                p.control_gain[c]   = fp32(model, cw.gain, hidden * r);
                p.control_forget[c] = fp32(model, cw.forget, hidden * r);
            }
        }
        return p;
    }

    // ---- Drafters --------------------------------------------------------------------------------

    [[nodiscard]] dspine::RuntimeOptions spine_options() const {
        const dspine::Config& c    = model.dspine()->config();
        const std::uint32_t window = dspine::ring_capacity(c, options.max_context);
        dspine::RuntimeOptions o;
        o.sequences       = options.sequences;
        o.kept_states     = options.checkpoints ? 1 : 0;
        o.max_context     = options.max_context;
        o.draft_tokens    = options.draft_tokens;
        o.draft_rows      = std::min<std::uint32_t>(options.sequences, 8);
        o.commit_rows     = std::min<std::uint32_t>(options.sequences, 8);
        o.feature_columns = std::max<std::uint32_t>(std::min(options.prefill_chunk, window),
                                                    std::uint32_t(decode_rows()));
        const auto columns = static_cast<std::int32_t>(o.draft_tokens * o.draft_rows);
        if (head_tiles.qtype == QType::BF16) {
            o.head_bytes = round_up(std::size_t(columns) * head_tiles.n * 4, 256);
        } else {
            const ops::detail::GgufShape shape{head_tiles.qtype, head_tiles.n, head_tiles.k};
            o.head_bytes = ops::detail::gguf_project_workspace_bytes({&shape, 1}, 1, columns) +
                           round_up(std::size_t(columns) * head_tiles.n * 4, 256);
        }
        return o;
    }

    // The drafter reads the target's token rows (the normed embedding the first visit reads) and
    // its head as FP32 logits before the soft cap, as Gemma 4's does, on rank 0. Its vocabulary
    // is the head's stored rows (config.vocab_size): the scored rows come from the head, the
    // padding rows past them are zero. No token of a sequence reaches those, since the
    // tokenizer's domain lies within the scored rows.
    void make_spine() {
        dspine::Target target;
        target.hidden = std::uint32_t(hidden);
        target.vocab  = std::uint32_t(config.vocab_size);
        target.layers = std::uint32_t(visits.size());
        target.domain = static_cast<std::int32_t>(model.resources().public_token_count);
        target.logits = DType::FP32;
        target.embed  = [this](const Tensor& ids, Tensor& rows, cudaStream_t on) {
            const int n  = ids.ne[0];
            auto scope   = workspace->scope();
            auto* scratch = static_cast<float*>(workspace->alloc_bytes(std::size_t(n) * hidden * 4).data);
            table_rows(embedding, static_cast<const std::int32_t*>(ids.data), n, scratch, hidden, on);
            kernels::rms_norm(scratch, embedding_norm, hidden, n, eps,
                              static_cast<__nv_bfloat16*>(rows.data), on);
        };
        target.head = [this](const Tensor& x, Tensor& logits, WorkspaceArena& work,
                             cudaStream_t on) {
            // The vector kernel reads the scored rows, the matrix kernel whole 128-row tiles into
            // a plane whose scored rows are then copied out; the padding rows become zero.
            const int n             = x.ne[1];
            const std::int64_t rows = logits.ne[0];
            auto* out               = static_cast<float*>(logits.data);
            ops::detail::GgufProduct product;
            if (head.qtype == QType::BF16) {
                project_f32(x, head_tiles, out, int(rows), 0, work, on);
            } else if (n <= gguf::kMaxVectorColumns) {
                product.weight   = &head;
                product.f32      = out;
                product.f32_rows = std::int32_t(rows);
                ops::detail::gguf_project(x, {&product, 1}, work, on);
            } else {
                auto scope       = work.scope();
                auto* plane      = static_cast<float*>(
                    work.alloc_bytes(std::size_t(n) * head_tiles.n * 4).data);
                product.weight   = &head_tiles;
                product.f32      = plane;
                product.f32_rows = head_tiles.n;
                ops::detail::gguf_project(x, {&product, 1}, work, on);
                CUDA_CHECK(cudaMemcpy2DAsync(out, std::size_t(rows) * 4, plane,
                                             std::size_t(head_tiles.n) * 4,
                                             std::size_t(vocab) * 4, std::size_t(n),
                                             cudaMemcpyDeviceToDevice, on));
            }
            if (rows > vocab) {
                CUDA_CHECK(cudaMemset2DAsync(out + vocab, std::size_t(rows) * 4, 0,
                                             std::size_t(rows - vocab) * 4, std::size_t(n), on));
            }
        };
        spine = std::make_unique<dspine::Runtime>(*model.dspine(), std::move(target),
                                                  spine_options());
    }

    // Tap `v` of the DSpine drafter, when it reads that visit's output: rows [first, first + n)
    // of the state h [T][H] into feature columns [0, n).
    void store_tap(std::size_t v, const float* h, int first, int n) {
        if (!spine || n <= 0) { return; }
        const int tap = spine->tap_of(std::uint32_t(v));
        if (tap < 0) { return; }
        spine->store_tap(std::uint32_t(tap),
                         Tensor(const_cast<float*>(h) + std::int64_t(first) * hidden, DType::FP32,
                                {hidden, n}),
                         0, stream);
    }

    // ---- The DSpark drafter --------------------------------------------------------------------------

    // Rows [first, first + n) of a stored parent (or the part itself).
    static Weight rows_of(const Weight& w, int first, int n) {
        Weight out = w;
        out.qdata  = static_cast<const std::uint8_t*>(w.qdata) + std::int64_t(first) * row_bytes(w);
        out.n      = n;
        return out;
    }

    // The matrices of `ids` (rows `sizes` each), whether one stored parent holds them or each is its
    // own.
    std::vector<Weight> parts_of(std::initializer_list<WeightId> ids, std::initializer_list<int> sizes) {
        std::vector<Weight> parts = matrices(model, ids);
        std::vector<Weight> out;
        if (parts.size() == 1 && ids.size() > 1) {
            int first = 0;
            for (const int n : sizes) {
                out.push_back(rows_of(parts.front(), first, n));
                first += n;
            }
            if (first != parts.front().n) {
                throw std::invalid_argument("inkling2 DSpark: a stored parent's rows");
            }
        } else {
            out = std::move(parts);
        }
        auto id = ids.begin();
        for (const Weight& w : out) { require_gguf(model, *id++, w); }
        return out;
    }

    struct Call;

    // A drafter layer's caches: a sliding layer's ring (the window and kR slots past it, like the
    // target's) with its checkpoint, a full layer's every position.
    [[nodiscard]] std::size_t spark_planned_bytes() const {
        const DSparkConfig& dc = *model.dspark_config();
        const std::size_t kv = 2 * std::size_t(dc.num_key_value_heads) * dc.head_dim * sizeof(__half) *
                               options.sequences;
        std::size_t bytes = 0;
        for (const bool full : dc.full_attention) {
            bytes += full ? std::size_t(round_up(options.max_context + 1, 256)) * kv
                          : (options.checkpoints ? 2 : 1) * std::size_t(ring) * kv;
        }
        bytes += (options.checkpoints ? 2 : 1) * std::size_t(options.sequences) * kR *
                 config.hidden_size * 4;
        bytes += std::size_t(config.unpadded_vocab_size) * (dc.markov_rank * 2 + 4);
        return bytes;
    }

    void plan_spark(int device_id, int sequences) {
        const DSparkConfig& dc = *model.dspark_config();
        const DSparkWeights& dw = *model.dspark();
        if (int(dc.num_key_value_heads) != kv_heads || int(dc.head_dim) != kD ||
            int(dc.num_attention_heads) != query_heads || int(dc.sliding_window) != window ||
            dc.target_visits.size() > std::size_t(kernels::kMaxSparkTaps) ||
            dc.num_hidden_layers + 1 > std::uint32_t(kernels::kMaxSparkMixes)) {
            throw std::invalid_argument("inkling2 DSpark: a drafter geometry outside the target's "
                                        "caches and attention");
        }
        auto d    = std::make_unique<Spark>();
        d->config = &dc;
        d->block  = int(dc.block_size);
        d->window = int(dc.sliding_window);
        d->taps   = int(dc.target_visits.size());
        d->tap_after.assign(visits.size(), -1);
        for (int j = 0; j + 1 < d->taps; ++j) {
            const std::uint32_t v = dc.target_visits[std::size_t(j)];
            // A visit's input is the state the visit before it leaves, unless the visit restarts the
            // core from the prelude's output.
            if (v == 0 || v >= visits.size() || visits[v].restart) {
                throw std::invalid_argument("inkling2 DSpark: a tap outside the implemented visits");
            }
            d->tap_after[v - 1] = j;
        }
        d->fuse            = fp32(model, dw.fuse, int(dc.num_hidden_layers + 1) * d->taps);
        d->context_norm    = fp32(model, dw.context_norm, hidden);
        d->final_norm      = fp32(model, dw.final_norm, hidden);
        d->mask_embedding  = fp32(model, dw.mask_embedding, hidden);
        const int rank     = int(dc.markov_rank);
        d->joint           = fp32(model, dw.joint, 3 * rank * (2 * rank + 2 * hidden));
        d->joint_bias      = fp32(model, dw.joint_bias, 3 * rank);
        d->hidden_out      = fp32(model, dw.hidden_out, hidden * rank);
        d->hidden_out_bias = fp32(model, dw.hidden_out_bias, hidden);
        {
            const Tensor t = weight_tensor(model.weight(dw.markov_embed).view,
                                           {std::int32_t(rank), std::int32_t(config.vocab_size)});
            if (t.dtype != DType::BF16) {
                throw std::invalid_argument("inkling2 DSpark: the Markov embedding must be BF16");
            }
            d->markov_embed = static_cast<const __nv_bfloat16*>(t.data);
        }
        const int q = query_heads * kD, w = int(dc.intermediate_size);
        const std::size_t row_f16 = std::size_t(kD) * sizeof(__half);
        for (std::uint32_t i = 0; i < dc.num_hidden_layers; ++i) {
            const DSparkLayerWeights& lw = dw.layers[i];
            SparkLayer l;
            auto qkv = parts_of({lw.query, lw.key, lw.value}, {q, kvd, kvd});
            l.query = qkv[0];
            l.key   = qkv[1];
            l.value = qkv[2];
            l.output = parts_of({lw.output}, {hidden}).front();
            auto context = parts_of({lw.context_key, lw.context_value}, {kvd, kvd});
            l.context_key   = context[0];
            l.context_value = context[1];
            auto mlp = parts_of({lw.gate, lw.up}, {w, w});
            l.gate = mlp[0];
            l.up   = mlp[1];
            l.down = parts_of({lw.down}, {hidden}).front();
            l.context_norm     = fp32(model, lw.context_norm, hidden);
            l.attention_norm   = fp32(model, lw.attention_norm, hidden);
            l.ffn_norm         = fp32(model, lw.ffn_norm, hidden);
            l.query_norm       = fp32(model, lw.query_norm, kD);
            l.key_norm         = fp32(model, lw.key_norm, kD);
            l.context_key_norm = fp32(model, lw.context_key_norm, kD);
            VisitPlan& c   = l.cache;
            c.full         = dc.full_attention[i];
            c.slots        = c.full ? global_slots : ring;
            c.ring         = c.full ? 0 : ring;
            c.head_stride  = std::int64_t(c.slots) * kD;
            c.seq_stride   = c.head_stride * kv_heads;
            for (KvArea* area : {&c.k, &c.v}) {
                *area = KvArea(std::size_t(sequences), kv_heads,
                               std::size_t(c.head_stride) * sizeof(__half), row_f16,
                               growing && c.full, device_id);
                kv_bytes += area->bytes;
            }
            if (!c.full && options.checkpoints) {
                for (int k = 0; k < 2; ++k) {
                    c.checkpoint.emplace_back(std::size_t(c.seq_stride) * sizeof(__half) * sequences);
                    kv_bytes += c.checkpoint.back().bytes;
                }
            }
            d->layers.push_back(std::move(l));
        }
        const std::size_t contexts = std::size_t(sequences) * kR * hidden * 4;
        d->contexts = DeviceBuffer(contexts);
        d->contexts.fill(0);
        d->bytes += contexts;
        if (options.checkpoints) {
            d->contexts_kept = DeviceBuffer(contexts);
            d->bytes += contexts;
        }
        d->table_m = DeviceBuffer(std::size_t(vocab) * rank * 2);
        d->table_c = DeviceBuffer(std::size_t(vocab) * 4);
        d->state   = DeviceBuffer(std::size_t(rank) * 4);
        d->prev    = DeviceBuffer(4);
        d->drafts  = DeviceBuffer(std::size_t(d->block) * 4);
        d->confidence = DeviceBuffer(std::size_t(d->block) * 4);
        d->bytes += d->table_m.bytes + d->table_c.bytes;
        d->host   = std::make_unique<PinnedHostBuffer>(std::size_t(d->block) * 8);
        d->copied = std::make_unique<CudaCompletionEvent>(device);
        spark = std::move(d);
    }

    // The head's products of the recurrent correction (M = head hidden_out, c = head bias), from
    // the head's decoded rows in FP32, once.
    void prepare_spark() {
        Spark& d = *spark;
        if (d.prepared) { return; }
        const int rank = int(d.config->markov_rank), cols = rank + 1;
        workspace->reset();
        auto scope   = workspace->scope();
        auto* inputs = static_cast<float*>(workspace->alloc_bytes(std::size_t(cols) * hidden * 4).data);
        kernels::spark_head_inputs(d.hidden_out, d.hidden_out_bias, hidden, rank, inputs, stream);
        const std::size_t free = workspace->capacity() - workspace->used() - (2u << 20);
        int chunk = int(std::min<std::size_t>(8192, free / (std::size_t(hidden + cols) * 4 + 4)));
        chunk     = chunk / 256 * 256;
        if (chunk < 256) { throw std::runtime_error("inkling2 DSpark: no workspace for the table"); }
        auto* ids  = static_cast<std::int32_t*>(workspace->alloc_bytes(std::size_t(chunk) * 4).data);
        auto* rows = static_cast<float*>(workspace->alloc_bytes(std::size_t(chunk) * hidden * 4).data);
        auto* out  = static_cast<float*>(workspace->alloc_bytes(std::size_t(chunk) * cols * 4).data);
        std::vector<std::int32_t> host(chunk);
        for (int first = 0; first < vocab; first += chunk) {
            const int n = std::min(chunk, vocab - first);
            for (int i = 0; i < n; ++i) { host[i] = first + i; }
            CUDA_CHECK(cudaMemcpyAsync(ids, host.data(), std::size_t(n) * 4, cudaMemcpyHostToDevice,
                                       stream));
            table_rows(head, ids, n, rows, hidden, stream);
            kernels::gemm_f32(cols, n, hidden, inputs, rows, out, stream);
            kernels::spark_table(out, n,
                                 static_cast<__nv_bfloat16*>(d.table_m.p) + std::int64_t(first) * rank,
                                 static_cast<float*>(d.table_c.p) + first, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        d.prepared = true;
    }

    // The drafter's caches of rows [0, c.spark_rows) of a call, from its taps: each layer's
    // context K and V at the rows' positions (a sliding layer's within its window of the last), and
    // the recurrent head's context of the last kR rows.
    void spark_inject(const Call& c) {
        Spark& d = *spark;
        const int n = c.spark_rows, t = c.t;
        if (n <= 0) { return; }
        const DSparkConfig& dc = *d.config;
        const int contexts     = int(d.layers.size());
        auto scope             = workspace->scope();
        kernels::SparkMixArgs ma;
        ma.rows = c.rows;
        for (int j = 0; j + 1 < d.taps; ++j) { ma.taps[j] = c.spark_taps + std::int64_t(j) * c.plane; }
        ma.taps[d.taps - 1] = c.ffn_f32;
        ma.tap_count        = d.taps;
        ma.fuse             = d.fuse;
        for (int l = 0; l < contexts; ++l) { ma.norm[l] = d.layers[l].context_norm; }
        ma.norm[contexts] = d.context_norm;
        ma.context_count  = contexts;
        ma.hidden         = hidden;
        ma.begin          = 0;
        ma.end            = n;
        ma.ring_begin     = std::max(0, n - kR);
        ma.eps            = dc.rms_norm_eps;
        ma.ring           = static_cast<float*>(d.contexts.p);
        ma.seq_stride     = std::int64_t(kR) * hidden;
        ma.slots          = kR;
        Q8Act xq;
        __nv_bfloat16* xb = nullptr;
        if (c.q8) {
            xq           = alloc_q8(hidden, t, contexts);
            ma.out_q     = xq.rows;
            ma.q8_stride = std::int64_t(q8_bytes(hidden, t));
        } else {
            xb = static_cast<__nv_bfloat16*>(
                workspace->alloc_bytes(std::size_t(contexts) * t * hidden * 2).data);
            ma.out = xb;
        }
        kernels::spark_mix(ma, stream);
        auto* kv = static_cast<float*>(workspace->alloc_bytes(std::size_t(n) * 2 * kvd * 4).data);
        for (int l = 0; l < contexts; ++l) {
            const SparkLayer& layer = d.layers[l];
            if (c.q8) {
                const void* x = static_cast<const std::byte*>(xq.base) + std::size_t(l) * q8_bytes(hidden, t);
                vector_f32(x, layer.context_key, t, kv, 2 * kvd, 0);
                vector_f32(x, layer.context_value, t, kv, 2 * kvd, kvd);
            } else {
                const Tensor x(xb + std::int64_t(l) * t * hidden, DType::BF16, {hidden, n});
                project_f32(x, layer.context_key, kv, 2 * kvd, 0);
                project_f32(x, layer.context_value, kv, 2 * kvd, kvd);
            }
            kernels::SparkRotateArgs ra;
            ra.rows        = c.rows;
            ra.x           = kv;
            ra.stride      = 2 * kvd;
            ra.heads       = kv_heads;
            ra.begin       = layer.cache.full ? 0 : std::max(0, n - d.window);
            ra.end         = n;
            ra.norm        = layer.context_key_norm;
            ra.eps         = dc.rms_norm_eps;
            ra.theta       = dc.rope_theta;
            ra.k_cache     = static_cast<__half*>(layer.cache.k.p);
            ra.v_cache     = static_cast<__half*>(layer.cache.v.p);
            ra.seq_stride  = layer.cache.seq_stride;
            ra.head_stride = layer.cache.head_stride;
            ra.ring        = layer.cache.ring;
            kernels::spark_rotate(ra, stream);
        }
    }

    // The drafter's block over staged rows (the anchor, then masks, at the sequence's next
    // positions): its layers over the context and the block (every row seeing the anchor's window
    // and the whole block), the final norm, the head's products, then `k` steps of the recurrent
    // head, their drafts and probabilities copied to the pinned buffer.
    void spark_pass(int k) {
        Spark& d = *spark;
        const DSparkConfig& dc = *d.config;
        const int b = d.block, q = query_heads * kD, w = int(dc.intermediate_size);
        const float deps = dc.rms_norm_eps;
        workspace->reset();
        auto scope = workspace->scope();
        auto* h    = static_cast<float*>(workspace->alloc_bytes(std::size_t(b) * hidden * 4).data);
        table_rows(embedding, static_cast<const std::int32_t*>(d_tokens.p), 1, h, hidden, stream);
        kernels::rms_norm(h, embedding_norm, hidden, 1, eps, nullptr, stream, {}, h);
        for (int i = 1; i < b; ++i) {
            CUDA_CHECK(cudaMemcpyAsync(h + std::int64_t(i) * hidden, d.mask_embedding,
                                       std::size_t(hidden) * 4, cudaMemcpyDeviceToDevice, stream));
        }
        kernels::Rows rows;
        rows.positions = static_cast<const std::int32_t*>(d_positions.p);
        rows.sequences = static_cast<const std::int32_t*>(d_sequences.p);
        rows.tokens    = b;
        Q8Act x        = alloc_q8(hidden, b);
        auto* qv       = static_cast<float*>(workspace->alloc_bytes(std::size_t(b) * q * 4).data);
        auto* kv       = static_cast<float*>(workspace->alloc_bytes(std::size_t(b) * 2 * kvd * 4).data);
        Q8Act attn     = alloc_q8(q, b);
        auto* y        = static_cast<float*>(workspace->alloc_bytes(std::size_t(b) * hidden * 4).data);
        auto* gu       = static_cast<float*>(workspace->alloc_bytes(std::size_t(b) * 2 * w * 4).data);
        auto* mid      = static_cast<__nv_bfloat16*>(workspace->alloc_bytes(std::size_t(b) * w * 2).data);
        void* act      = workspace->alloc_bytes(gguf::vector_activation_bytes(w, b)).data;
        const int splits_sliding = 4;
        auto* partial = static_cast<float*>(workspace->alloc_bytes(
            kernels::attention_workspace_floats(b, query_heads,
                                                std::max(decode_splits_full, splits_sliding)) * 4).data);
        for (const SparkLayer& l : d.layers) {
            kernels::rms_norm(h, l.attention_norm, hidden, b, deps, nullptr, stream, x.rows);
            vector_f32(x.base, l.query, b, qv, q);
            vector_f32(x.base, l.key, b, kv, 2 * kvd, 0);
            vector_f32(x.base, l.value, b, kv, 2 * kvd, kvd);
            kernels::SparkRotateArgs ra;
            ra.rows   = rows;
            ra.x      = qv;
            ra.stride = q;
            ra.heads  = query_heads;
            ra.end    = b;
            ra.norm   = l.query_norm;
            ra.eps    = deps;
            ra.theta  = dc.rope_theta;
            kernels::spark_rotate(ra, stream);
            ra.x           = kv;
            ra.stride      = 2 * kvd;
            ra.heads       = kv_heads;
            ra.norm        = l.key_norm;
            ra.k_cache     = static_cast<__half*>(l.cache.k.p);
            ra.v_cache     = static_cast<__half*>(l.cache.v.p);
            ra.seq_stride  = l.cache.seq_stride;
            ra.head_stride = l.cache.head_stride;
            ra.ring        = l.cache.ring;
            kernels::spark_rotate(ra, stream);
            kernels::AttentionArgs aa;
            aa.rows        = rows;
            aa.q           = qv;
            aa.k_cache     = static_cast<const __half*>(l.cache.k.p);
            aa.v_cache     = static_cast<const __half*>(l.cache.v.p);
            aa.seq_stride  = l.cache.seq_stride;
            aa.head_stride = l.cache.head_stride;
            aa.ring        = l.cache.ring;
            aa.window      = l.cache.full ? 0 : d.window + b;
            aa.query_heads = query_heads;
            aa.kv_heads    = kv_heads;
            aa.scale       = 1.0f / std::sqrt(float(kD));
            aa.splits      = l.cache.full ? decode_splits_full : splits_sliding;
            aa.uniform     = true;
            aa.partial     = partial;
            aa.out_q       = attn.rows;
            kernels::attention(aa, stream);
            vector_f32(attn.base, l.output, b, y, hidden);
            kernels::add_rows(h, y, hidden, hidden, b, stream);
            kernels::rms_norm(h, l.ffn_norm, hidden, b, deps, nullptr, stream, x.rows);
            vector_f32(x.base, l.gate, b, gu, 2 * w, 0);
            vector_f32(x.base, l.up, b, gu, 2 * w, w);
            kernels::silu_mul(gu, w, b, mid, stream);
            gguf::quantize_vector_activation(mid, w, b, nullptr, act, stream);
            vector_f32(act, l.down, b, y, hidden);
            kernels::add_rows(h, y, hidden, hidden, b, stream);
        }
        auto* final_rows = static_cast<float*>(workspace->alloc_bytes(std::size_t(b) * hidden * 4).data);
        kernels::rms_norm(h, d.final_norm, hidden, b, deps, nullptr, stream, {}, final_rows);
        // The head's products of the rows the k steps read (1..k), in their rows of `base`.
        auto* base = static_cast<float*>(workspace->alloc_bytes(std::size_t(b) * vocab * 4).data);
        if (head.qtype == QType::BF16) {
            auto* head_in = static_cast<__nv_bfloat16*>(
                workspace->alloc_bytes(std::size_t(k) * hidden * 2).data);
            kernels::rms_norm(h + hidden, d.final_norm, hidden, k, deps, head_in, stream);
            project_f32(Tensor(head_in, DType::BF16, {hidden, k}), head_tiles, base + vocab, vocab);
        } else {
            const Q8Act head_in = alloc_q8(hidden, k);
            kernels::rms_norm(h + hidden, d.final_norm, hidden, k, deps, nullptr, stream,
                              head_in.rows);
            vector_f32(head_in.base, head, k, base + vocab, vocab);
        }
        auto* scratch = static_cast<float*>(
            workspace->alloc_bytes(kernels::spark_step_floats(vocab) * 4).data);
        CUDA_CHECK(cudaMemsetAsync(d.state.p, 0, d.state.bytes, stream));
        CUDA_CHECK(cudaMemcpyAsync(d.prev.p, d_tokens.p, 4, cudaMemcpyDeviceToDevice, stream));
        kernels::SparkStepArgs sa;
        sa.weight       = d.joint;
        sa.bias         = d.joint_bias;
        sa.embed        = d.markov_embed;
        sa.hidden       = final_rows;
        sa.ring         = static_cast<const float*>(d.contexts.p);
        sa.seq_stride   = std::int64_t(kR) * hidden;
        sa.slots        = kR;
        sa.positions    = rows.positions;
        sa.sequences    = rows.sequences;
        sa.base         = base;
        sa.vocab_stride = vocab;
        sa.m            = static_cast<const __nv_bfloat16*>(d.table_m.p);
        sa.c            = static_cast<const float*>(d.table_c.p);
        sa.vocab        = vocab;
        sa.hidden_size  = hidden;
        sa.cap          = config.logit_softcap;
        sa.state        = static_cast<float*>(d.state.p);
        sa.prev         = static_cast<std::int32_t*>(d.prev.p);
        sa.drafts       = static_cast<std::int32_t*>(d.drafts.p);
        sa.confidence   = static_cast<float*>(d.confidence.p);
        sa.joint        = scratch;
        sa.logits       = scratch + 3 * dc.markov_rank;
        sa.partials     = sa.logits + vocab;
        for (int i = 1; i <= k; ++i) {
            sa.row = i;
            kernels::spark_step(sa, stream);
        }
        auto* host = static_cast<std::uint8_t*>(d.host->data());
        CUDA_CHECK(cudaMemcpyAsync(host, d.drafts.p, std::size_t(k) * 4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(host + std::size_t(d.block) * 4, d.confidence.p, std::size_t(k) * 4,
                                   cudaMemcpyDeviceToHost, stream));
    }

    void spark_draft(std::uint32_t sequence, std::int32_t anchor, std::span<std::int32_t> drafts,
                     std::span<float> confidence) {
        if (!spark) { throw std::logic_error("inkling2: an executor without the DSpark drafter"); }
        Spark& d    = *spark;
        const int b = d.block;
        const int k = int(drafts.size());
        if (sequence >= options.sequences || k < 1 || k > b - 1 ||
            confidence.size() != drafts.size() || anchor < 0 || anchor >= vocab) {
            throw std::invalid_argument("inkling2 draft: invalid call");
        }
        const std::uint32_t first = positions[sequence];
        if (first == 0 || first + std::uint32_t(b) > options.max_context) {
            throw std::invalid_argument("inkling2 draft: a block needs a committed token before it "
                                        "and room in the context");
        }
        prepare_spark();
        back(sequence, first + std::uint32_t(b));
        std::vector<std::int32_t> tokens(b, 0), pos(b), seq(b, std::int32_t(sequence));
        tokens[0] = anchor;
        for (int i = 0; i < b; ++i) { pos[i] = std::int32_t(first) + i; }
        stage(tokens, pos, seq);
        run_graph(48 + std::uint32_t(k), [&] { spark_pass(k); });
        d.copied->record(stream);
        d.copied->synchronize();
        const auto* host = static_cast<const std::uint8_t*>(d.host->data());
        std::memcpy(drafts.data(), host, std::size_t(k) * 4);
        std::memcpy(confidence.data(), host + std::size_t(b) * 4, std::size_t(k) * 4);
    }

    // ---- Expert delivery -------------------------------------------------------------------------

    void plan_delivery() {
        std::vector<ExpertDeliveryLayer> sources;
        std::vector<std::uint64_t> sizes;
        for (const auto& bank : model.expert_banks()) {
            sources.push_back(bank.delivery);
            for (const auto& expert : bank.delivery.experts) {
                sizes.resize(std::max(sizes.size(), expert.size()), 0);
                for (std::size_t b = 0; b < expert.size(); ++b) {
                    sizes[b] = std::max(sizes[b], std::uint64_t(round_up(
                        expert[b].bytes + expert[b].zero_tail, kAlign)));
                }
            }
        }
        std::uint64_t slot = 0;
        for (auto bytes : sizes) { slot += bytes; }
        const auto bank     = slot * config.num_experts;
        const auto metadata = std::uint64_t(options.prefill_chunk) * config.top_k_experts * 8 +
                              std::uint64_t(config.num_hidden_layers) * config.num_experts * 4 *
                                  sizeof(void*) +
                              config.num_experts * 4;
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        const auto cache = options.expert_cache_bytes.value_or(0);
        if (bank + metadata + cache + unbacked() + kGraphReserve > free) {
            throw InsufficientMemory("Inkling 2: expert delivery needs " +
                std::to_string((bank + metadata + cache) >> 20) + " MiB beside " +
                std::to_string(unbacked() >> 20) +
                " MiB for the growing caches and the CUDA Graph reserve after workspace allocation");
        }
        ExpertDeliveryOptions transport;
        transport.pair_capacity   = options.prefill_chunk * config.top_k_experts;
        transport.cache_bytes     = options.expert_cache_bytes;
        transport.keep_free_bytes = kGraphReserve + (growing ? kKvHeadroom : 0);
        transport.elastic         = growing;
        transport.disk            = options.expert_disk;
        const float dma_share     = options.expert_dma_share.value_or(0.0F);
        if (!(dma_share >= 0 && dma_share <= 1) || options.expert_cpu_threads > 256) {
            throw std::invalid_argument("Inkling 2: the DMA share must be 0..1 and CPU threads "
                                        "0..256");
        }
        std::vector<CpuExpertLayer> cpu_layers;
        std::vector<std::vector<float>> scales;
        bool computes = false;
        for (std::size_t l = 0; l < model.expert_banks().size(); ++l) {
            const auto& bank = model.expert_banks()[l];
            cpu_layers.push_back({bank.gate_up_type, bank.down_type,
                                  std::int64_t(bank.gate_up_row_bytes),
                                  std::int64_t(bank.down_row_bytes)});
            computes = computes || CpuExperts::computes(cpu_layers.back(), hidden,
                                                        int(config.moe_intermediate_size));
            std::vector<float> s;
            if (layers[l].gate_up_scale != nullptr) {
                s.resize(config.num_experts);
                CUDA_CHECK(cudaMemcpy(s.data(), layers[l].gate_up_scale, s.size() * 4,
                                      cudaMemcpyDeviceToHost));
            }
            scales.push_back(std::move(s));
        }
        // A model with BF16 matrices has no q8_1 decode calls for the CPU to read.
        if (computes && dma_share < 1 && !bf16_rows) {
            cpu_experts = std::make_unique<CpuExperts>(
                std::move(cpu_layers), std::move(scales), hidden, int(config.moe_intermediate_size),
                int(config.top_k_experts), config.gate_cap, config.up_cap);
            transport.cpu.products      = cpu_experts.get();
            transport.cpu.input_bytes   = cpu_experts->input_bytes();
            transport.cpu.product_bytes = cpu_experts->product_bytes();
            transport.cpu.share         = 1.0 - double(dma_share);
            transport.cpu.balance       = !options.expert_dma_share;
            transport.cpu.threads       = options.expert_cpu_threads;
        }
        delivery = std::make_unique<ExpertDelivery>(device.rank(0), model.expert_files(),
                                                    std::move(sources), transport);
        // Where a device's copies queue behind one another whatever their streams
        // (serialized_copy_queue: Windows, WSL2), a replayed graph's device-side wait for the
        // misses can hold the copies the graph queued after it ahead of the delivery's own, which
        // end that wait. Eager passes wait on the host and queue nothing behind the wait.
        if (serialized_copy_queue()) { options.cuda_graphs = false; }
        for (auto& p : layers) {
            if (p.delivery_layer < 0) { continue; }
            const auto l       = std::size_t(p.delivery_layer);
            const auto aligned = [&](std::size_t alias) {
                return delivery->alias_aligned(l, alias, 16);
            };
            p.gate_up_aligned16 = aligned(0) && aligned(1) && aligned(2);
            p.down_aligned16    = aligned(3);
        }
    }

    static constexpr std::uint32_t kFullChunk  = 2048;
    static constexpr std::size_t kGraphReserve = std::size_t(320) << 20;
    static constexpr std::size_t kKvHeadroom   = std::size_t(256) << 20;

    // ---- Growing caches --------------------------------------------------------------------------

    [[nodiscard]] std::size_t unbacked() const {
        std::size_t bytes = 0;
        for (const KvArea* area : sequence_areas) {
            if (area->region) { bytes += area->region.bytes() - area->region.mapped_bytes(); }
        }
        return bytes;
    }

    void count_backed() {
        std::uint64_t bytes = kv_fixed;
        for (const KvArea* area : sequence_areas) { bytes += area->region.mapped_bytes(); }
        kv_backed.store(bytes, std::memory_order_relaxed);
    }

    [[nodiscard]] std::size_t missing(const KvArea& a, std::size_t owner, std::uint32_t positions) const {
        if (!a.region || positions <= a.backed[owner]) { return 0; }
        std::size_t bytes = 0;
        for (int r = 0; r < a.rows; ++r) {
            bytes += a.region.unmapped_bytes((owner * a.rows + r) * a.row_bytes,
                                             std::size_t(positions) * a.position_bytes);
        }
        return bytes;
    }
    [[nodiscard]] bool extend(KvArea& a, std::size_t owner, std::uint32_t positions) {
        if (!a.region || positions <= a.backed[owner]) { return true; }
        for (int r = 0; r < a.rows; ++r) {
            if (!a.region.map((owner * a.rows + r) * a.row_bytes,
                              std::size_t(positions) * a.position_bytes, stream)) {
                return false;
            }
        }
        a.backed[owner] = positions;
        return true;
    }

    void trim(KvArea& a, std::size_t owner, std::uint32_t positions) {
        if (!a.region || a.backed[owner] <= positions) { return; }
        const std::size_t g     = a.region.granule();
        const std::uint32_t was = a.backed[owner];
        a.backed[owner]         = positions;
        const auto reached = [&](std::size_t lo, std::size_t hi) {
            for (std::size_t o = 0; o < a.backed.size(); ++o) {
                for (int r = 0; r < a.rows; ++r) {
                    const std::size_t row = (o * a.rows + r) * a.row_bytes;
                    if (row < hi && lo < row + std::size_t(a.backed[o]) * a.position_bytes) { return true; }
                }
            }
            return false;
        };
        for (int r = 0; r < a.rows; ++r) {
            const std::size_t row = (owner * a.rows + r) * a.row_bytes;
            const std::size_t hi  = row + std::size_t(was) * a.position_bytes;
            for (std::size_t at = (row + std::size_t(positions) * a.position_bytes) / g * g; at < hi; at += g) {
                if (!reached(at, at + g)) { a.region.unmap(at, g); }
            }
        }
    }

    // Before a call that writes or reads `sequence`'s positions below `end`: its full-attention
    // rows are backed up to the 256-slot tile that holds `end`. Outside capture.
    void back(std::uint32_t sequence, std::uint32_t end) {
        if (!growing) { return; }
        const auto slots = std::uint32_t(round_up(end, 256));
        std::size_t need = 0;
        for (const KvArea* area : sequence_areas) { need += missing(*area, sequence, slots); }
        const auto apply = [&] {
            bool ok = true;
            for (KvArea* area : sequence_areas) { ok = ok && extend(*area, sequence, slots); }
            return ok;
        };
        if (need == 0) {
            (void)apply();
            return;
        }
        std::size_t free = 0, total = 0;
        CUDA_CHECK(cudaMemGetInfo(&free, &total));
        if (free < need + kGraphReserve) { delivery->shrink_cache(need + kGraphReserve - free); }
        while (!apply()) {
            if (delivery->shrink_cache(std::max(need, std::size_t(64) << 20)) == 0) {
                count_backed();
                throw InsufficientMemory("Inkling 2: no device memory left to grow the KV cache "
                                         "to " + std::to_string(slots) + " positions");
            }
        }
        count_backed();
    }

    void release(std::uint32_t sequence) {
        if (!growing || sequence_areas.empty() || sequence_areas.front()->backed[sequence] == 0) { return; }
        CUDA_CHECK(ninfer::wait_for_stream(stream));
        for (KvArea* area : sequence_areas) { trim(*area, sequence, 0); }
        count_backed();
        delivery->grow_cache();
    }

    std::size_t start_reserve() const {
        return options.prefill_chunk >= kFullChunk ? std::size_t(1536) << 20
                                                   : std::size_t(512) << 20;
    }

    ~Impl() {
        if (delivery || growing) { ninfer::wait_for_stream(stream); }
        for (std::size_t r = 1; r < expert_peers.size(); ++r) {
            if (expert_peers[r].stream == nullptr) { continue; }
            RankBinding bind(device, r);
            ninfer::wait_for_stream(expert_peers[r].stream);
            expert_peers[r].workspace.reset();
            cudaStreamDestroy(expert_peers[r].stream);
        }
        divided_head.reset();
    }
    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;

    // ---- Workspace -------------------------------------------------------------------------------

    static constexpr int kHeadRows = 64; // LM-head columns per FP32 plane pass

    [[nodiscard]] int decode_rows() const {
        return std::min<int>(gguf::kMaxVectorColumns,
                             std::max<int>(int(options.sequences), int(options.draft_tokens) + 1));
    }

    // Bytes of one q8_1 vector activation of rows of width k, 256-aligned.
    static std::size_t q8_bytes(int k, int t) {
        return round_up(gguf::vector_activation_bytes(k, t), kAlign);
    }

    std::size_t plan_workspace() const {
        const std::size_t t = options.prefill_chunk;
        const std::size_t h = hidden;
        const std::size_t q_floats = std::size_t(query_heads) * kD;
        const std::size_t width    = config.moe_intermediate_size;
        const std::size_t rank     = config.gate_rank;
        std::size_t query_rows = 0, gguf_scratch = 0, bf16_plane = 0;
        for (const LayerPlan& p : layers) {
            query_rows = std::max<std::size_t>(query_rows, p.query_rows);
            // A BF16 matrix writing columns wider than its rows goes through a plane (project_f32).
            for (const Weight& part : p.query) {
                if (part.qtype == QType::BF16 && part.n != p.query_rows) {
                    bf16_plane = std::max<std::size_t>(bf16_plane, part.n);
                }
            }
            for (const Weight* m : {&p.key, &p.value}) {
                if (m->qtype == QType::BF16) { bf16_plane = std::max<std::size_t>(bf16_plane, m->n); }
            }
            std::vector<const Weight*> products{&p.key, &p.value, &p.output};
            for (const Weight& part : p.query) { products.push_back(&part); }
            for (const Weight* m : products) {
                if (!is_gguf(m->qtype)) { continue; }
                const ops::detail::GgufShape shape{m->qtype, m->n, m->k};
                gguf_scratch = std::max(gguf_scratch, ops::detail::gguf_project_workspace_bytes(
                                                          {&shape, 1}, 1, int(t)));
            }
        }
        if (is_gguf(head_tiles.qtype)) {
            const ops::detail::GgufShape hs{head_tiles.qtype, head_tiles.n, head_tiles.k};
            gguf_scratch = std::max(
                gguf_scratch, ops::detail::gguf_project_workspace_bytes({&hs, 1}, 1, kHeadRows));
        }
        gguf_scratch += round_up(t * bf16_plane * 4, kAlign);
        std::size_t bytes = 0;
        const auto add    = [&](std::size_t n) { bytes += round_up(n, kAlign); };
        // Call scope.
        add(kPool * t * h * 4);           // residual states
        add(3 * t * h * 2);               // attention inputs, BF16
        add(3 * t * h * 4);               // attention inputs, FP32
        add(3 * t * h * 4);               // raw mixes
        add(4 * t * h * 4);               // messages
        add(t * h * 2);                   // MoE input BF16 (and the final rows)
        add(t * h * 4);                   // MoE input FP32
        add(t * h * 4);                   // MoE raw mix
        if (options.dspark) {             // the drafter's stored taps
            add((model.dspark_config()->target_visits.size() - 1) * t * h * 4);
        }
        // Visit scope.
        add(t * query_rows * 4);          // [Q; gate; relative]
        add(t * 2 * kvd * 4);             // raw K, V
        add(t * 2 * kvd * 4 / kD * kTaps); // dynamic taps
        add(t * query_heads * 4);         // sinks
        add(t * q_floats * 4);            // q
        add(t * query_heads * 16 * 4);    // r
        add(kernels::attention_workspace_floats(int(t), query_heads, 1) * 4); // far field states
        add(t * q_floats * 2);            // attention out BF16
        add(t * h * 4);                   // output product
        if (kvfmt.staged()) { add(t * 2 * kvd * 2); } // a quantized visit's F16 staging (kv_store reads it)
        // Controllers of three inputs: [rms x; rms m], the rank's hidden state, forget and gain.
        add(3 * t * 2 * h * 4);
        add(3 * t * rank * 4);
        add(2 * 3 * t * h * 4);
        add(std::size_t(kHeadRows) * head_tiles.n * 4); // head FP32 plane
        add(kHeadRows * h * 2);
        std::size_t moe = 0;
        {
            const std::size_t pairs = t * config.top_k_experts;
            moe += round_up(pairs * 4, kAlign) * 3;                   // ids, weights, pair scales
            moe += round_up(t * config.num_experts * 4, kAlign);      // logits
            moe += round_up(gguf::moe_routing_bytes(int(config.num_experts), int(pairs)), kAlign);
            moe += round_up(gguf::moe_matrix_activation_bytes(int(h), int(pairs)), kAlign);
            moe += round_up(pairs * 2 * width * 4, kAlign);
            moe += round_up(pairs * width * 2, kAlign);
            moe += round_up(gguf::moe_matrix_activation_bytes(int(width), int(pairs)), kAlign);
            moe += round_up(pairs * h * 4, kAlign);
            if (model.expert_parallel()) {
                moe += round_up(pairs * 4, kAlign) +
                       round_up(gguf::moe_routing_bytes(int(config.num_experts), int(pairs)), kAlign) +
                       round_up(t * h * 8, kAlign);
            }
        }
        const std::size_t decode = kernels::attention_workspace_floats(
            decode_rows(), query_heads, std::max(decode_splits_full, decode_splits_sliding)) * 4;
        return bytes + moe + round_up(decode, kAlign) + gguf_scratch + (16u << 20);
    }

    // ---- Expert parallelism ----------------------------------------------------------------------

    [[nodiscard]] int max_moe_tokens() const {
        return std::max<int>(int(options.prefill_chunk), gguf::kMaxVectorColumns);
    }

    // A MoE call's packed input: the activation the experts read (a call of up to 8 tokens with a
    // q8_1 form: that form; others: BF16 [T][H]), then the ids, weights and pair scales,
    // 256-aligned.
    struct ExpertInput {
        std::size_t activation = 0, ids = 0, weights = 0, scales = 0, bytes = 0;
    };

    // `q8`: the call carries its q8_1 vector form (up to 8 tokens), which it then sends instead
    // of its BF16 rows.
    [[nodiscard]] ExpertInput expert_input(int t, bool q8) const {
        const std::size_t pairs = std::size_t(t) * config.top_k_experts;
        ExpertInput out;
        out.activation = q8 ? gguf::vector_activation_bytes(hidden, t) : std::size_t(t) * hidden * 2;
        out.ids     = round_up(out.activation, kAlign);
        out.weights = out.ids + round_up(pairs * 4, kAlign);
        out.scales  = out.weights + round_up(pairs * 4, kAlign);
        out.bytes   = out.scales + pairs * 4;
        return out;
    }

    [[nodiscard]] std::size_t expert_part_bytes(int t) const {
        const std::size_t pairs = std::size_t(t) * config.top_k_experts;
        const std::size_t vector_pairs =
            std::size_t(gguf::kMaxVectorColumns) * config.top_k_experts;
        const std::size_t h = hidden, w = config.moe_intermediate_size;
        std::size_t bytes =
            round_up(pairs * 4, kAlign) +
            round_up(gguf::moe_routing_bytes(int(config.num_experts), int(pairs)), kAlign) +
            round_up(pairs * w * 2, kAlign) + round_up(pairs * 2 * w * 4, kAlign);
        const std::size_t vector =
            round_up(gguf::vector_activation_bytes(int(w), int(vector_pairs)), kAlign);
        const std::size_t matrix =
            round_up(gguf::moe_matrix_activation_bytes(int(h), int(pairs)), kAlign) +
            round_up(gguf::moe_matrix_activation_bytes(int(w), int(pairs)), kAlign) +
            round_up(pairs * h * 4, kAlign);
        return bytes + std::max(vector, matrix) + (std::size_t(1) << 20);
    }

    void allocate_expert_peers() {
        if (!model.expert_parallel()) { return; }
        const int t = max_moe_tokens();
        const std::size_t input =
            std::max(expert_input(t, false).bytes, expert_input(gguf::kMaxVectorColumns, true).bytes);
        const std::size_t part = std::size_t(t) * hidden * 8;
        expert_pack            = DeviceBuffer(input);
        expert_exchange        = std::make_unique<ExpertParallelExchange>(device, input, part);
        expert_peers.resize(device.size());
        for (std::size_t r = 1; r < device.size(); ++r) {
            RankBinding bind(device, r);
            CUDA_CHECK(cudaStreamCreateWithFlags(&expert_peers[r].stream, cudaStreamNonBlocking));
            expert_peers[r].workspace = std::make_unique<WorkspaceArena>(expert_part_bytes(t));
        }
        divide_head();
    }

    void divide_head() {
        const int columns = gguf::kMaxVectorColumns;
        // Only the q8_1 decode head divides; a model with BF16 matrices has no such calls.
        if (bf16_rows) { return; }
        if (!models::DividedHead::divides(head, device.size())) { return; }
        const auto speed        = rank_copy_bandwidth(device);
        const auto free         = rank_free_bytes(device);
        const std::size_t input = gguf::vector_activation_bytes(hidden, columns);
        double speeds           = 0;
        for (const double v : speed) { speeds += v; }
        for (std::size_t r = 1; r < device.size(); ++r) {
            const double rows = double(head.n) * speed[r] / speeds;
            const double need = rows * double(row_bytes(head) + 8 * columns) + double(input) +
                                double(kHeadRoom);
            if (double(free[r]) < need) { return; }
        }
        divided_head = std::make_unique<models::DividedHead>(device, head, row_bytes(head), speed,
                                                             input, columns);
    }
    static constexpr std::size_t kHeadRoom = std::size_t(512) << 20;

    // The SiTU middles of a call's pairs from their FP32 [gate; up] rows.
    void situ(const float* gu, int pairs, const float* pair_scale, __nv_bfloat16* middle,
              cudaStream_t on) const {
        const int width = int(config.moe_intermediate_size);
        kernels::situ_rows(gu, 2 * std::int64_t(width), pairs, width, pair_scale, config.gate_cap,
                           config.up_cap, middle, on);
    }

    // A BF16 layer's routed experts over the pairs of `routing`: FP32 [gate; up] rows `gu`, their
    // SiTU middles, then each pair's FP32 product row in `rows` ([pairs][hidden]).
    void bf16_experts(const void* const* tables, const gguf::MoeRouting& routing, int max_active,
                      const __nv_bfloat16* x, float* gu, __nv_bfloat16* middle,
                      const float* pair_scale, int pairs, float* rows, cudaStream_t on) const {
        const int e = int(config.num_experts), k = int(config.top_k_experts);
        const int width = int(config.moe_intermediate_size);
        kernels::bf16_expert_rows(tables + 2 * e, 2 * width, hidden, routing.bounds, routing.sorted,
                                  routing.active, routing.active_count, max_active, x, k, gu, on);
        situ(gu, pairs, pair_scale, middle, on);
        kernels::bf16_expert_rows(tables + 3 * e, hidden, width, routing.bounds, routing.sorted,
                                  routing.active, routing.active_count, max_active, middle, 1, rows,
                                  on);
    }

    // One rank's part of a MoE call of layer `p`: the call's pairs routed to the experts the rank
    // holds (`range`), through its `tables`, weighted into the 2^-32 fixed-point sum `fixed`
    // ([T][H]). Up to 8 tokens read the call's q8_1 activation `act` when it has one, other calls
    // its BF16 rows `x`.
    void expert_part(const LayerPlan& p, const void* const* tables,
                     std::pair<std::uint32_t, std::uint32_t> range, const void* act,
                     const __nv_bfloat16* x, const std::int32_t* ids, const float* weights,
                     const float* pair_scale, int t, unsigned long long* fixed,
                     WorkspaceArena& space, cudaStream_t on) {
        auto scope  = space.scope();
        const int e = int(config.num_experts), k = int(config.top_k_experts);
        const int width = int(config.moe_intermediate_size);
        const int pairs = t * k;
        auto* mine = static_cast<std::int32_t*>(space.alloc_bytes(std::size_t(pairs) * 4).data);
        gguf::moe_keep_experts(ids, pairs, int(range.first), int(range.second), mine, on);
        const gguf::MoeRouting routing = gguf::moe_sort_routes(
            mine, pairs, e, space.alloc_bytes(gguf::moe_routing_bytes(e, pairs)).data, on);
        const gguf::MoeTable gate_up{tables + 2 * e, p.gate_up_row_bytes, 2 * width, hidden,
                                     p.gate_up_aligned16};
        const gguf::MoeTable down{tables + 3 * e, p.down_row_bytes, hidden, width,
                                  p.down_aligned16};
        const int max_active = std::max(1, std::min(pairs, int(range.second - range.first)));
        auto* middle =
            static_cast<__nv_bfloat16*>(space.alloc_bytes(std::size_t(pairs) * width * 2).data);
        auto* gu =
            static_cast<float*>(space.alloc_bytes(std::size_t(pairs) * 2 * width * 4).data);
        if (p.bf16) {
            auto* rows = static_cast<float*>(space.alloc_bytes(std::size_t(pairs) * hidden * 4).data);
            bf16_experts(tables, routing, max_active, x, gu, middle, pair_scale, pairs, rows, on);
            gguf::moe_weighted_rows(rows, mine, weights, t, k, hidden, fixed, on);
            return;
        }
        if (act != nullptr && t <= gguf::kMaxVectorColumns) {
            const int chunk = t <= 1 ? 1 : t <= 2 ? 2 : t <= 4 ? 4 : 8;
            void* act2      = space.alloc_bytes(gguf::vector_activation_bytes(width, pairs)).data;
            CUDA_CHECK(cudaMemsetAsync(fixed, 0, std::size_t(t) * hidden * 8, on));
            // Another rank's pairs keep zero rows; their middles are never read.
            CUDA_CHECK(cudaMemsetAsync(gu, 0, std::size_t(pairs) * 2 * width * 4, on));
            gguf::MoeOutput rows;
            rows.f32 = gu;
            gguf::moe_vector_product(p.gate_up_type, gate_up, routing, max_active, k, k, act, t,
                                     rows, chunk, on);
            situ(gu, pairs, pair_scale, middle, on);
            gguf::quantize_vector_activation(middle, width, pairs, nullptr, act2, on);
            gguf::MoeOutput out;
            out.weighted = fixed;
            out.weights  = weights;
            gguf::moe_vector_product(p.down_type, down, routing, max_active, 1, k, act2, pairs, out,
                                     chunk, on);
            return;
        }
        void* rows_act = space.alloc_bytes(gguf::moe_matrix_activation_bytes(hidden, pairs)).data;
        gguf::quantize_moe_matrix_activation(p.gate_up_type, x, hidden, routing, pairs, k, rows_act,
                                             on);
        gguf::moe_matrix_product(p.gate_up_type, gate_up, routing, max_active, pairs, true,
                                 rows_act, gu, on);
        situ(gu, pairs, pair_scale, middle, on);
        void* act2 = space.alloc_bytes(gguf::moe_matrix_activation_bytes(width, pairs)).data;
        gguf::quantize_moe_matrix_activation(p.down_type, middle, width, routing, pairs, 1, act2,
                                             on);
        auto* rows = static_cast<float*>(space.alloc_bytes(std::size_t(pairs) * hidden * 4).data);
        gguf::moe_matrix_product(p.down_type, down, routing, max_active, pairs, true, act2, rows,
                                 on);
        gguf::moe_weighted_rows(rows, mine, weights, t, k, hidden, fixed, on);
    }

    // Expert parallelism's MoE call after rank 0 has routed it: the input goes to every other
    // rank, each runs its part while rank 0 runs its own, and the parts' exact fixed-point sum
    // becomes the residual's product.
    void experts_parallel(const LayerPlan& p, const __nv_bfloat16* x, const void* act,
                          const std::int32_t* ids, const float* weights, const float* pair_scale,
                          int t, kernels::ResidualArgs& residual) {
        const std::size_t pairs  = std::size_t(t) * config.top_k_experts;
        const std::size_t part   = std::size_t(t) * hidden * 8;
        const ExpertInput layout = expert_input(t, act != nullptr);
        auto* pack               = static_cast<std::byte*>(expert_pack.p);
        CUDA_CHECK(cudaMemcpyAsync(pack, act != nullptr ? act : x, layout.activation,
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(
            cudaMemcpyAsync(pack + layout.ids, ids, pairs * 4, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(pack + layout.weights, weights, pairs * 4,
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(pack + layout.scales, pair_scale, pairs * 4,
                                   cudaMemcpyDeviceToDevice, stream));
        const std::size_t slot = expert_exchange->send(pack, layout.bytes, stream);
        for (std::size_t r = 1; r < device.size(); ++r) {
            RankBinding bind(device, r);
            ExpertPeer& peer = expert_peers[r];
            auto* input      = static_cast<std::byte*>(
                expert_exchange->receive(r, layout.bytes, slot, peer.stream));
            expert_part(p, static_cast<const void* const*>(p.peer_tables[r].p),
                        model.expert_range(r), act != nullptr ? input : nullptr,
                        reinterpret_cast<const __nv_bfloat16*>(input),
                        reinterpret_cast<const std::int32_t*>(input + layout.ids),
                        reinterpret_cast<const float*>(input + layout.weights),
                        reinterpret_cast<const float*>(input + layout.scales), t,
                        static_cast<unsigned long long*>(expert_exchange->part(r)), *peer.workspace,
                        peer.stream);
            expert_exchange->reply(r, part, slot, peer.stream);
        }
        auto* fixed = static_cast<unsigned long long*>(workspace->alloc_bytes(part).data);
        expert_part(p, static_cast<const void* const*>(p.tables.p), model.expert_range(0), act, x,
                    ids, weights, pair_scale, t, fixed, *workspace, stream);
        std::vector<const unsigned long long*> parts;
        for (void* received : expert_exchange->collect(part, slot, stream)) {
            parts.push_back(static_cast<const unsigned long long*>(received));
        }
        gguf::add_fixed_parts(fixed, parts.data(), int(parts.size()), std::int64_t(t) * hidden,
                              stream);
        residual.fixed = fixed;
    }

    // ---- Shared pieces ---------------------------------------------------------------------------

    struct Q8Act {
        void* base = nullptr;
        kernels::Q8Rows rows;
    };
    static Q8Act q8_view(void* base, int k, int t) {
        Q8Act q;
        q.base    = base;
        q.rows.qs = static_cast<std::int8_t*>(base);
        q.rows.ds = reinterpret_cast<half2*>(static_cast<std::uint8_t*>(base) +
                                             gguf::vector_activation_scales_offset(k, t));
        return q;
    }
    Q8Act alloc_q8(int k, int t, int count = 1) {
        return q8_view(workspace->alloc_bytes(std::size_t(count) * q8_bytes(k, t)).data, k, t);
    }

    // out[:, c] = W x_c for the q8_1 rows x: FP32 columns `column_stride` apart, from row `row`.
    void vector_f32(const void* x, const Weight& w, int t, float* out, int column_stride,
                    int row = 0) {
        gguf::VectorOutput o;
        o.f32           = out + row;
        o.column_stride = column_stride;
        gguf::vector_product(ops::detail::gguf_type(w.qtype), w.qdata, row_bytes(w), w.n, w.k, x,
                             t, o, stream);
    }

    // out[:, c] = W x_c for BF16 rows x [K, T]: FP32 columns `column_stride` apart, from `row`.
    // A BF16 W (rows a multiple of 4) multiplies through Linear's A16 route, which writes 16-byte
    // aligned [N, T] planes: straight into `out` when it is one, else into a scratch plane whose
    // first min(N, column_stride - row) rows of each column are then copied out.
    void project_f32(const Tensor& x, const Weight& w, float* out, int column_stride, int row = 0) {
        project_f32(x, w, out, column_stride, row, *workspace, stream);
    }
    void project_f32(const Tensor& x, const Weight& w, float* out, int column_stride, int row,
                     WorkspaceArena& work, cudaStream_t on) const {
        if (w.qtype == QType::BF16) {
            if (w.n % 4 != 0) {
                throw std::invalid_argument("inkling2: a BF16 matrix of rows not a multiple of 4");
            }
            const int t = x.ne[1];
            if (column_stride == w.n && reinterpret_cast<std::uintptr_t>(out + row) % 16 == 0) {
                Tensor y(out + row, DType::FP32, {w.n, t});
                ops::linear(x, w, y, on);
                return;
            }
            auto scope  = work.scope();
            auto* plane = static_cast<float*>(work.alloc_bytes(std::size_t(w.n) * t * 4).data);
            Tensor y(plane, DType::FP32, {w.n, t});
            ops::linear(x, w, y, on);
            const int kept = std::min(w.n, column_stride - row);
            CUDA_CHECK(cudaMemcpy2DAsync(out + row, std::size_t(column_stride) * 4, plane,
                                         std::size_t(w.n) * 4, std::size_t(kept) * 4,
                                         std::size_t(t), cudaMemcpyDeviceToDevice, on));
            return;
        }
        ops::detail::GgufProduct product;
        product.weight   = &w;
        product.f32      = out;
        product.f32_rows = column_stride;
        product.row      = row;
        ops::detail::gguf_project(x, {&product, 1}, work, on);
    }

    // Rows ids[0, n) of the token table or the head as FP32 rows `ld` apart, every value exact.
    void table_rows(const Weight& w, const std::int32_t* ids, int n, float* out, std::int64_t ld,
                    cudaStream_t on) const {
        if (w.qtype == QType::BF16) {
            kernels::bf16_rows_f32(static_cast<const __nv_bfloat16*>(w.qdata), w.k, ids, n, out, ld,
                                   on);
            return;
        }
        gguf::dequantize_rows(ops::detail::gguf_type(w.qtype), w.qdata, row_bytes(w), w.k, ids, n,
                              out, ld, on);
    }

    // W x into FP32 [T][column_stride] from row `row`, x in the call's form: q8_1 rows (up to 8)
    // or BF16 rows.
    void product(const Weight& w, const void* xq, const __nv_bfloat16* x, int t, float* out,
                 int column_stride, int row = 0) {
        if (xq != nullptr) {
            vector_f32(xq, w, t, out, column_stride, row);
            return;
        }
        project_f32(Tensor(const_cast<__nv_bfloat16*>(x), DType::BF16, {w.k, t}), w, out,
                    column_stride, row);
    }

    static constexpr std::size_t kStagingRegions = 4;

    // Tokens, positions, and when given sequences and branch starts, to the device.
    void stage(std::span<const std::int32_t> a, std::span<const std::int32_t> b,
               std::span<const std::int32_t> c, std::span<const std::int32_t> g = {}) {
        staged->synchronize();
        auto* host = static_cast<std::int32_t*>(h_staging->data());
        const std::size_t rows = std::max<std::size_t>(options.prefill_chunk, options.sequences);
        if (a.size() > rows || b.size() > rows || c.size() > rows || g.size() > rows) {
            throw std::invalid_argument("inkling2 executor: staging overflow");
        }
        std::copy(a.begin(), a.end(), host);
        std::copy(b.begin(), b.end(), host + rows);
        std::copy(c.begin(), c.end(), host + 2 * rows);
        std::copy(g.begin(), g.end(), host + 3 * rows);
        CUDA_CHECK(cudaMemcpyAsync(d_tokens.p, host, a.size() * 4, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_positions.p, host + rows, b.size() * 4,
                                   cudaMemcpyHostToDevice, stream));
        if (!c.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(d_sequences.p, host + 2 * rows, c.size() * 4,
                                       cudaMemcpyHostToDevice, stream));
        }
        if (!g.empty()) {
            CUDA_CHECK(cudaMemcpyAsync(d_starts.p, host + 3 * rows, g.size() * 4,
                                       cudaMemcpyHostToDevice, stream));
        }
        staged->record(stream);
    }

    // ---- The forward pass ------------------------------------------------------------------------

    // A call's rows and the buffers that live across its visits.
    struct Call {
        int t     = 0;
        bool q8   = false; // up to 8 rows: the vector kernels' q8_1 activations
        bool chunk = false;
        std::uint32_t sequence = 0; // a prompt chunk's
        int first = 0;              // a prompt chunk's first position
        int kept  = 0;              // a prompt chunk's rows the sequence keeps
        bool keep = false;          // a decode or verification call keeps its rows' history
        int row_group = 1;          // a verification's rows a full visit's attention block takes
        kernels::Rows rows;
        States states;
        // The attention's three normed inputs (input 0 alone when `distinct` is 1): BF16 rows,
        // their q8_1 activations and FP32 rows; the raw mixes (a later pass's core visits); the
        // messages; the MoE input in the same forms.
        int distinct = 1;
        bool inputs_ready = false;
        __nv_bfloat16* in_bf16 = nullptr;
        Q8Act in_q;
        std::size_t in_q_stride = 0;
        float* in_f32 = nullptr;
        float* raw    = nullptr;
        float* messages = nullptr;
        __nv_bfloat16* ffn_bf16 = nullptr;
        Q8Act ffn_q;
        float* ffn_f32 = nullptr;
        float* ffn_raw = nullptr;
        std::int64_t plane = 0; // elements of one [T][H] plane
        // DSpark: the stored taps ([taps - 1] planes; the last tap is the final hidden, in ffn_f32)
        // and the rows [0, spark_rows) whose drafter caches the call writes.
        float* spark_taps = nullptr;
        int spark_rows    = 0;
        // A call of at most kernels::kSplitRows rows: the split depth mixes' scratch.
        float* split = nullptr;
    };

    [[nodiscard]] const void* input_q(const Call& c, int i) const {
        if (!c.q8) { return nullptr; }
        return static_cast<const std::byte*>(c.in_q.base) + std::size_t(i) * c.in_q_stride;
    }

    // The depth mix over the states (the blocks so far and the current state).
    kernels::MixArgs mix_args(const Call& c, const float* queries, int count) const {
        kernels::MixArgs m;
        const States& s = c.states;
        int n           = 0;
        for (const int b : s.blocks) { m.states[n++] = s.buffers[b]; }
        m.states[n++]  = s.buffers[s.cur];
        m.count        = n;
        m.queries      = queries;
        m.query_count  = count;
        m.tokens       = c.t;
        m.hidden       = hidden;
        m.eps          = eps;
        m.query_stride = c.plane;
        m.split        = c.split;
        return m;
    }

    // The attention inputs of visit v as the visit's mix gives them: normed into the input buffers,
    // or raw for its controllers. Layer 0 reads the current state alone.
    kernels::MixArgs attention_mix(Call& c, std::size_t v) {
        const VisitPlan& vp = visits[v];
        const LayerPlan& lp = layers[vp.layer];
        kernels::MixArgs m  = mix_args(c, lp.attention_mix, lp.mix_queries);
        if (vp.control) {
            m.raw = c.raw;
            return m;
        }
        if (lp.full) {
            m.norm[0] = lp.query_norm;
            m.norm[1] = lp.key_norm;
            m.norm[2] = lp.value_norm;
        } else {
            m.norm[0] = lp.input_norm;
        }
        set_outputs(c, m.out, m.out_q, m.f32, m.q8_stride, c.in_bf16, c.in_q, c.in_q_stride,
                    c.in_f32);
        return m;
    }

    void set_outputs(const Call& c, __nv_bfloat16*& out, kernels::Q8Rows& out_q, float*& f32,
                     std::int64_t& q8_stride, __nv_bfloat16* bf16, const Q8Act& q,
                     std::size_t q_stride, float* rows) const {
        if (c.q8) {
            out_q     = q.rows;
            q8_stride = std::int64_t(q_stride);
        } else {
            out = bf16;
        }
        f32 = rows;
    }

    // The FFN input of visit v's MoE: normed, or raw for its controller.
    kernels::MixArgs ffn_mix(Call& c, std::size_t v) {
        const VisitPlan& vp = visits[v];
        const LayerPlan& lp = layers[vp.layer];
        kernels::MixArgs m  = mix_args(c, lp.ffn_mix, 1);
        if (vp.control) {
            m.raw = c.ffn_raw;
            return m;
        }
        m.norm[0] = lp.ffn_norm;
        std::int64_t unused = 0;
        set_outputs(c, m.out, m.out_q, m.f32, unused, c.ffn_bf16, c.ffn_q, 0, c.ffn_f32);
        return m;
    }

    // The output's mix: the final norm into the MoE input's BF16 or q8_1 rows.
    // With the DSpark drafter its FP32 rows too (the drafter's last tap), in the MoE input's.
    kernels::MixArgs output_mix_args(Call& c) {
        kernels::MixArgs m = mix_args(c, output_mix, 1);
        m.norm[0]          = final_norm;
        std::int64_t unused = 0;
        float* none         = nullptr;
        set_outputs(c, m.out, m.out_q, none, unused, c.ffn_bf16, c.ffn_q, 0,
                    spark ? c.ffn_f32 : nullptr);
        return m;
    }

    // Whether visit v + 1's attention mix can run inside visit v's last residual update: the
    // states it reads are the ones visit v leaves.
    [[nodiscard]] bool fuses_next(std::size_t v) const {
        return v + 1 < visits.size() && !visits[v + 1].restart && visits[v + 1].layer > 0;
    }

    // The controllers of `count` inputs: x[i] + its message, through the layer's controllers
    // `which[i]`, normed with norm[i] into the outputs (query_stride and q8_stride apart).
    void control(const Call& c, const LayerPlan& lp, int count, const std::array<const float*, 3>& x,
                 const std::array<const float*, 3>& message, const std::array<int, 3>& which,
                 const std::array<const float*, 3>& norm, __nv_bfloat16* out_bf16,
                 const Q8Act& out_q, std::size_t q_stride, float* out_f32) {
        auto scope    = workspace->scope();
        const int t   = c.t, r = int(config.gate_rank);
        auto* concat  = static_cast<float*>(
            workspace->alloc_bytes(std::size_t(count) * t * 2 * hidden * 4).data);
        auto* state   = static_cast<float*>(workspace->alloc_bytes(std::size_t(count) * t * r * 4).data);
        auto* forget  = static_cast<float*>(
            workspace->alloc_bytes(std::size_t(count) * c.plane * 4).data);
        auto* gain    = static_cast<float*>(
            workspace->alloc_bytes(std::size_t(count) * c.plane * 4).data);
        kernels::ControlInputs ci;
        kernels::ControlApply ca;
        for (int i = 0; i < count; ++i) {
            ci.x[i] = ca.x[i] = x[i];
            ci.message[i] = ca.message[i] = message[i];
            ca.forget[i] = forget + i * c.plane;
            ca.gain[i]   = gain + i * c.plane;
            ca.norm[i]   = norm[i];
        }
        ci.count = ca.count = count;
        ci.tokens = ca.tokens = t;
        ci.hidden = ca.hidden = hidden;
        ci.concat = concat;
        kernels::control_inputs(ci, stream);
        // Every input's down product in one launch, then their forget and gain products in one.
        kernels::GemvBatch down, up;
        for (int i = 0; i < count; ++i) {
            const int w = which[i];
            float* h    = state + std::int64_t(i) * t * r;
            down.w[i]   = lp.control_down[w];
            down.x[i]   = concat + std::int64_t(i) * t * 2 * hidden;
            down.out[i] = h;
            up.w[2 * i]       = lp.control_forget[w];
            up.x[2 * i]       = h;
            up.out[2 * i]     = forget + i * c.plane;
            up.w[2 * i + 1]   = lp.control_gain[w];
            up.x[2 * i + 1]   = h;
            up.out[2 * i + 1] = gain + i * c.plane;
        }
        down.count = count;
        up.count   = 2 * count;
        kernels::gemm_f32_batch(r, t, 2 * hidden, down, stream);
        kernels::gemm_f32_batch(hidden, t, r, up, stream);
        ca.eps          = eps;
        ca.query_stride = c.plane;
        set_outputs(c, ca.out, ca.out_q, ca.f32, ca.q8_stride, out_bf16, out_q, q_stride, out_f32);
        kernels::control_apply(ca, stream);
    }

    // The embedding rows of the staged tokens, normed, into the call's first state. An image's
    // rows then take the place of their tokens' rows as the tower made them, without the norm.
    void embed(Call& c, std::span<const ImageRows> images) {
        States& s = c.states;
        s.cur     = 0;
        float* h  = s.buffers[0];
        table_rows(embedding, static_cast<const std::int32_t*>(d_tokens.p), c.t, h, hidden, stream);
        kernels::rms_norm(h, embedding_norm, hidden, c.t, eps, nullptr, stream, {}, h);
        // A piece at a time through the device rows, each copy ordered after the placement before.
        for (const ImageRows& piece : images) {
            const int count = int(piece.embeddings.size() / std::size_t(hidden));
            CUDA_CHECK(cudaMemcpyAsync(d_image_rows.p, piece.embeddings.data(),
                                       piece.embeddings.size() * 2, cudaMemcpyHostToDevice, stream));
            kernels::place_rows(static_cast<const __nv_bfloat16*>(d_image_rows.p), count, hidden,
                                h + std::int64_t(piece.row) * hidden, stream);
        }
    }

    // ---- The vision tower -----------------------------------------------------------------------

    void plan_vision() {
        if (!model.vision()) {
            throw std::invalid_argument("inkling2 executor: images need the model's vision tower");
        }
        tower = std::make_unique<VisionTower>(model, options.vision_max_patches);
        const VisionConfig& vc = tower->config();
        d_image_rows = DeviceBuffer(std::size_t(vc.max_image_tokens) * hidden * 2);
        d_patches    = DeviceBuffer(std::size_t(options.vision_max_patches) * vc.patch_features() * 2);
    }

    [[nodiscard]] std::uint64_t vision_owned_bytes() const noexcept {
        if (!tower) { return 0; }
        return tower->owned_bytes() + d_image_rows.bytes + d_patches.bytes;
    }

    std::uint32_t encode_image(std::span<const std::uint16_t> patches, int grid_w, int grid_h,
                               std::span<std::uint16_t> rows) {
        if (!tower) { throw std::invalid_argument("inkling2 encode_image: no vision tower"); }
        const VisionConfig& vc  = tower->config();
        const std::uint32_t span = VisionTower::span_rows(grid_w, grid_h);
        if (grid_w <= 0 || grid_h <= 0 || grid_w % 2 != 0 || grid_h % 2 != 0 ||
            std::uint64_t(grid_w) * std::uint64_t(grid_h) > options.vision_max_patches ||
            patches.size() != std::size_t(grid_w) * grid_h * vc.patch_features() ||
            span > vc.max_image_tokens || rows.size() < std::size_t(span) * hidden) {
            throw std::invalid_argument("inkling2 encode_image: invalid image");
        }
        CUDA_CHECK(cudaMemcpyAsync(d_patches.p, patches.data(), patches.size() * 2,
                                   cudaMemcpyHostToDevice, stream));
        workspace->reset();
        tower->encode(static_cast<const __nv_bfloat16*>(d_patches.p), grid_w, grid_h,
                      static_cast<__nv_bfloat16*>(d_image_rows.p), *workspace, stream);
        CUDA_CHECK(cudaMemcpyAsync(rows.data(), d_image_rows.p, std::size_t(span) * hidden * 2,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(ninfer::wait_for_stream(stream));
        return span;
    }

    // Checks a call's images against its t rows.
    void check_images(std::span<const ImageRows> images, int t) const {
        if (images.empty()) { return; }
        if (!tower) { throw std::invalid_argument("inkling2 forward: images need the vision tower"); }
        std::size_t next = 0;
        for (const ImageRows& piece : images) {
            const std::size_t count = piece.embeddings.size() / std::size_t(hidden);
            if (count == 0 || piece.embeddings.size() % std::size_t(hidden) != 0 || piece.row < next ||
                piece.row + count > std::size_t(t) || count > tower->config().max_image_tokens) {
                throw std::invalid_argument("inkling2 forward: image rows outside the call or out "
                                            "of order");
            }
            next = piece.row + count;
        }
    }

    // Rows of the call whose history a prompt chunk keeps: the last kHistory the sequence keeps.
    [[nodiscard]] std::pair<int, int> kept_history(const Call& c) const {
        return {std::max(0, c.kept - kR), c.kept};
    }

    kernels::History site(std::size_t v, int offset) const {
        kernels::History h;
        h.ring        = static_cast<float*>(history.p) + std::int64_t(v) * kR * slot_floats + offset;
        h.seq_stride  = std::int64_t(visits.size()) * kR * slot_floats;
        h.slot_stride = slot_floats;
        h.slots       = kR;
        return h;
    }

    void visit(Call& c, std::size_t v) {
        const VisitPlan& vp = visits[v];
        const LayerPlan& lp = layers[vp.layer];
        States& s           = c.states;
        const int t         = c.t;
        auto scope          = workspace->scope();
        if (vp.save_prelude) {
            s.prelude_blocks = s.blocks;
            s.prelude_cur    = s.cur;
        } else if (vp.restart) {
            s.bank.assign(s.blocks.begin() + std::ptrdiff_t(s.prelude_blocks.size() + 1),
                          s.blocks.end());
            s.bank_last     = s.cur;
            s.blocks        = s.prelude_blocks;
            s.cur           = s.prelude_cur;
            c.inputs_ready  = false;
        }
        if (vp.control) {
            // The messages of the previous pass's states, one per input.
            kernels::MixArgs m;
            int n = 0;
            for (const int b : s.bank) { m.states[n++] = s.buffers[b]; }
            m.states[n++]  = s.buffers[s.bank_last];
            m.count        = n;
            m.queries      = lp.loop_queries;
            m.query_count  = 4;
            m.tokens       = t;
            m.hidden       = hidden;
            m.query_stride = c.plane;
            m.raw          = c.messages;
            m.split        = c.split;
            kernels::mix(m, stream);
        }
        // The attention's inputs.
        if (!c.inputs_ready) {
            if (vp.layer == 0) {
                kernels::rms_norm(s.buffers[s.cur], lp.input_norm, hidden, t, eps, c.in_bf16,
                                  stream, c.q8 ? c.in_q.rows : kernels::Q8Rows{}, c.in_f32);
            } else {
                kernels::mix(attention_mix(c, v), stream);
            }
        }
        c.inputs_ready = false;
        if (vp.control) {
            const float* x0 = c.raw;
            const float* x1 = lp.full ? c.raw + c.plane : c.raw;
            const float* x2 = lp.full ? c.raw + 2 * c.plane : c.raw;
            const std::array<const float*, 3> norms =
                lp.full ? std::array<const float*, 3>{lp.query_norm, lp.key_norm, lp.value_norm}
                        : std::array<const float*, 3>{lp.input_norm, lp.input_norm, lp.input_norm};
            control(c, lp, 3, {x0, x1, x2},
                    {c.messages, c.messages + c.plane, c.messages + 2 * c.plane}, {0, 1, 2}, norms,
                    c.in_bf16, c.in_q, c.in_q_stride, c.in_f32);
            c.distinct = 3;
        } else {
            c.distinct = lp.full ? 3 : 1;
        }
        const int ki = c.distinct == 3 ? 1 : 0, vi = c.distinct == 3 ? 2 : 0;
        const auto bf16_in = [&](int i) {
            return c.in_bf16 != nullptr ? c.in_bf16 + i * c.plane : nullptr;
        };
        const auto f32_in = [&](int i) { return c.in_f32 + i * c.plane; };

        // Projections.
        const int q_floats = query_heads * kD, rel = int(config.relative_width);
        auto* qgr = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * lp.query_rows * 4).data);
        auto* kv  = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * 2 * kvd * 4).data);
        auto* sinks = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * query_heads * 4).data);
        for (std::size_t i = 0, row = 0; i < lp.query.size(); row += lp.query[i].n, ++i) {
            product(lp.query[i], input_q(c, 0), bf16_in(0), t, qgr, lp.query_rows, int(row));
        }
        product(lp.key, input_q(c, ki), bf16_in(ki), t, kv, 2 * kvd, 0);
        product(lp.value, input_q(c, vi), bf16_in(vi), t, kv, 2 * kvd, kvd);
        kernels::gemm_f32(query_heads, t, hidden, lp.sinks, f32_in(0), sinks, stream);
        float* taps = nullptr;
        const int tap_columns = 2 * kv_heads * kTaps;
        if (lp.full) {
            taps = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * tap_columns * 4).data);
            kernels::gemm_f32(kv_heads * kTaps, t, hidden, lp.key_taps, f32_in(ki), taps, stream, 0,
                              tap_columns);
            kernels::gemm_f32(kv_heads * kTaps, t, hidden, lp.value_taps, f32_in(vi),
                              taps + kv_heads * kTaps, stream, 0, tap_columns);
        }
        auto* q = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * q_floats * 4).data);
        auto* r = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * query_heads * rel * 4).data);
        {
            kernels::QueryArgs qa;
            qa.qgr             = qgr;
            qa.stride          = lp.query_rows;
            qa.relative_column = 2 * q_floats;
            qa.tokens          = t;
            qa.heads           = query_heads;
            qa.query_clip      = lp.query_clip;
            qa.relative_clip   = lp.relative_clip;
            qa.qk_clip         = config.qk_clip;
            qa.rel_clip        = config.relative_clip;
            qa.log_scale       = lp.full;
            qa.alpha           = config.log_scale_alpha;
            qa.floor           = float(config.log_scale_floor);
            qa.positions       = c.rows.positions;
            qa.q_out           = q;
            qa.r_out           = r;
            kernels::query_post(qa, stream);
        }
        // K and V into the cache: a prompt chunk's sliding visit attends over the staging ring
        // (the window before the chunk comes over from the sequence's ring, the chunk's keys land
        // there, and the window that ends the chunk's kept rows goes back). A full visit of an
        // mxfp8 run has kv_post encode its rows into its planes; one of rk8v4 or rk4v4 stages
        // them as F16 (one slot per row) for kv_store to encode.
        const std::int64_t stage_stride = std::int64_t(prefill_ring) * kD;
        const std::int64_t seq_offset   = std::int64_t(c.sequence) * vp.seq_stride;
        const bool staging   = c.chunk && !vp.full;
        const bool quantized = vp.full && kvfmt.quantized();
        KvPlaneView k_view, v_view; // a quantized full visit's planes, else null codings
        if (quantized) {
            const auto view = [&](const KvArea& codes, const KvArea& scales, KvCoding coding) {
                KvPlaneView plane;
                plane.codes   = static_cast<std::uint8_t*>(codes.p);
                plane.scales  = static_cast<std::uint8_t*>(scales.p);
                plane.code_row  = kv_code_bytes(coding);
                plane.scale_row = kv_scale_bytes(coding);
                plane.code_head  = std::int64_t(vp.slots) * plane.code_row;
                plane.scale_head = std::int64_t(vp.slots) * plane.scale_row;
                plane.code_seq  = plane.code_head * kv_heads;
                plane.scale_seq = plane.scale_head * kv_heads;
                plane.coding    = coding;
                return plane;
            };
            k_view = view(vp.k_codes, vp.k_scales, kvfmt.key.coding);
            v_view = view(vp.v_codes, vp.v_scales, kvfmt.value.coding);
        }
        kernels::RingCopyArgs keep;
        keep.kv_heads = kv_heads;
        kernels::KvArgs ka;
        ka.rows        = c.rows;
        ka.kv          = kv;
        ka.kv_stride   = 2 * kvd;
        ka.k_taps      = taps;
        ka.v_taps      = taps != nullptr ? taps + kv_heads * kTaps : nullptr;
        ka.taps_stride = tap_columns;
        ka.k_conv      = lp.key_conv;
        ka.v_conv      = lp.value_conv;
        ka.k_history   = site(v, 0);
        ka.v_history   = site(v, kvd);
        ka.keep        = c.keep;
        ka.kv_heads    = kv_heads;
        ka.qk_clip     = config.qk_clip;
        ka.eps         = eps;
        if (quantized && kvfmt.mxfp8()) {
            ka.k_plane = k_view;
            ka.v_plane = v_view;
        } else if (quantized) {
            auto* stage = static_cast<__half*>(workspace->alloc_bytes(std::size_t(t) * 2 * kvd * 2).data);
            ka.k_cache     = stage;
            ka.v_cache     = stage + std::size_t(t) * kvd;
            ka.seq_stride  = 0;
            ka.head_stride = std::int64_t(t) * kD;
            ka.ring        = 0;
            ka.row_slots   = true;
        } else if (staging) {
            keep.src_k           = static_cast<const __half*>(vp.k.p) + seq_offset;
            keep.src_v           = static_cast<const __half*>(vp.v.p) + seq_offset;
            keep.dst_k           = static_cast<__half*>(staging_k.p);
            keep.dst_v           = static_cast<__half*>(staging_v.p);
            keep.src_head_stride = vp.head_stride;
            keep.dst_head_stride = stage_stride;
            keep.src_ring        = vp.ring;
            keep.dst_ring        = prefill_ring;
            keep.begin           = std::max(0, c.first - window);
            keep.end             = c.first;
            kernels::ring_copy(keep, stream);
            ka.k_cache     = static_cast<__half*>(staging_k.p);
            ka.v_cache     = static_cast<__half*>(staging_v.p);
            ka.seq_stride  = 0;
            ka.head_stride = stage_stride;
            ka.ring        = prefill_ring;
        } else {
            ka.k_cache     = static_cast<__half*>(vp.k.p);
            ka.v_cache     = static_cast<__half*>(vp.v.p);
            ka.seq_stride  = vp.seq_stride;
            ka.head_stride = vp.head_stride;
            ka.ring        = vp.ring;
        }
        kernels::kv_post(ka, stream);
        if (quantized && kvfmt.staged()) {
            for (const auto& [src, plane] :
                 {std::pair{ka.k_cache, k_view}, std::pair{ka.v_cache, v_view}}) {
                kernels::KvStoreArgs store;
                store.rows            = c.rows;
                store.src             = src;
                store.src_head_stride = std::int64_t(t) * kD;
                store.rows_count      = t;
                store.heads           = kv_heads;
                store.dst             = plane;
                kernels::kv_store(store, stream);
            }
        }
        if (c.chunk) {
            const auto [begin, end] = kept_history(c);
            kernels::keep_history(kv, 2 * kvd, 2 * kvd, c.rows.positions, begin, end, site(v, 0),
                                  int(c.sequence), stream);
        }
        // Attention.
        __nv_bfloat16* attn = nullptr;
        Q8Act attn_q;
        if (c.q8) {
            attn_q = alloc_q8(q_floats, t);
        } else {
            attn = static_cast<__nv_bfloat16*>(workspace->alloc_bytes(std::size_t(t) * q_floats * 2).data);
        }
        {
            kernels::AttentionArgs aa;
            aa.rows        = c.rows;
            aa.q           = q;
            aa.r           = r;
            aa.sinks       = sinks;
            aa.sink_stride = query_heads;
            aa.table       = lp.distances;
            aa.extent      = int(lp.full ? config.relative_extent : config.sliding_window);
            aa.k_cache     = ka.k_cache;
            aa.v_cache     = ka.v_cache;
            aa.k_plane     = k_view;
            aa.v_plane     = v_view;
            aa.seq_stride  = ka.seq_stride;
            aa.head_stride = ka.head_stride;
            aa.ring        = ka.ring;
            aa.window      = lp.full ? 0 : window;
            aa.query_heads = query_heads;
            aa.kv_heads    = kv_heads;
            aa.scale       = 1.0f / std::sqrt(float(kD));
            aa.splits      = c.chunk ? 1 : (lp.full ? decode_splits_full : decode_splits_sliding);
            // A prompt chunk far enough into a full visit runs its far field on tensor cores.
            if (c.chunk && lp.full && ka.ring == 0 &&
                c.first > int(config.relative_extent)) {
                aa.far = static_cast<float*>(workspace->alloc_bytes(
                    kernels::attention_workspace_floats(t, query_heads, 1) * 4).data);
            }
            if (!c.chunk && lp.full && ka.ring == 0 && c.row_group > 1) {
                // The groups: at most ceil(t / row_group), one more for each further sequence.
                aa.row_group  = c.row_group;
                aa.row_groups = std::min(t, (t + c.row_group - 1) / c.row_group +
                                                std::min(t, int(options.sequences)) - 1);
            }
            if (!c.chunk && aa.splits > 1) {
                aa.partial = static_cast<float*>(workspace->alloc_bytes(
                    kernels::attention_workspace_floats(t, query_heads, aa.splits) * 4).data);
            }
            aa.gate        = qgr + q_floats;
            aa.gate_stride = lp.query_rows;
            aa.out         = attn;
            aa.out_q       = attn_q.rows;
            kernels::attention(aa, stream);
        }
        if (staging && c.kept > 0) {
            kernels::RingCopyArgs back = keep;
            std::swap(back.src_head_stride, back.dst_head_stride);
            std::swap(back.src_ring, back.dst_ring);
            back.src_k = static_cast<const __half*>(staging_k.p);
            back.src_v = static_cast<const __half*>(staging_v.p);
            back.dst_k = static_cast<__half*>(vp.k.p) + seq_offset;
            back.dst_v = static_cast<__half*>(vp.v.p) + seq_offset;
            back.begin = std::max(0, c.first + c.kept - window);
            back.end   = c.first + c.kept;
            kernels::ring_copy(back, stream);
        }
        auto* y = static_cast<float*>(workspace->alloc_bytes(std::size_t(c.plane) * 4).data);
        product(lp.output, c.q8 ? attn_q.base : nullptr, attn, t, y, hidden);
        // The attention's residual update, then the MoE input's mix.
        if (vp.start) {
            s.blocks.push_back(s.cur);
            s.cur = s.acquire();
        }
        {
            kernels::ResidualArgs ra;
            ra.rows    = c.rows;
            ra.y       = y;
            ra.conv    = lp.output_conv;
            ra.history = site(v, 2 * kvd);
            ra.keep    = c.keep;
            ra.start   = vp.start;
            ra.cur     = s.buffers[s.cur];
            ra.hidden  = hidden;
            ra.mix     = ffn_mix(c, v);
            kernels::residual(ra, stream);
            if (c.chunk) {
                const auto [begin, end] = kept_history(c);
                kernels::keep_products(ra, begin, end, stream);
            }
        }
        if (vp.control) {
            control(c, lp, 1, {c.ffn_raw, nullptr, nullptr}, {c.messages + 3 * c.plane, nullptr, nullptr},
                    {3, 0, 0}, {lp.ffn_norm, nullptr, nullptr}, c.ffn_bf16, c.ffn_q, 0, c.ffn_f32);
        }
        // The routed experts, then their residual update and the next visit's mix (or the
        // output's after the last visit).
        kernels::ResidualArgs ra;
        ra.rows    = c.rows;
        ra.conv    = lp.moe_conv;
        ra.history = site(v, 2 * kvd + hidden);
        ra.keep    = c.keep;
        ra.cur     = s.buffers[s.cur];
        ra.hidden  = hidden;
        ra.mix.split = c.split;
        experts(lp, c, ra);
        const bool last = v + 1 == visits.size();
        if (last) {
            ra.mix = output_mix_args(c);
        } else if (fuses_next(v)) {
            ra.mix         = attention_mix(c, v + 1);
            c.inputs_ready = true;
        }
        kernels::residual(ra, stream);
        if (c.chunk) {
            const auto [begin, end] = kept_history(c);
            kernels::keep_products(ra, begin, end, stream);
        }
    }

    // The routed experts of layer `p` for the call's MoE input; their product lands where the
    // residual update `ra` reads it.
    void experts(const LayerPlan& p, Call& c, kernels::ResidualArgs& ra) {
        const int t = c.t;
        const int e = int(config.num_experts), k = int(config.top_k_experts);
        const int width = int(config.moe_intermediate_size);
        const int pairs = t * k;
        auto* ids        = static_cast<std::int32_t*>(workspace->alloc_bytes(std::size_t(pairs) * 4).data);
        auto* weights    = static_cast<float*>(workspace->alloc_bytes(std::size_t(pairs) * 4).data);
        auto* pair_scale = static_cast<float*>(workspace->alloc_bytes(std::size_t(pairs) * 4).data);
        auto* logits     = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * e * 4).data);
        kernels::gemm_f32(e, t, hidden, p.router, c.ffn_f32, logits, stream);
        kernels::RouteArgs route;
        route.hidden        = hidden;
        route.tokens        = t;
        route.bias          = p.router_bias;
        route.experts       = e;
        route.top_k         = k;
        route.scale         = config.route_scale;
        route.gate_up_scale = p.gate_up_scale;
        route.down_scale    = p.down_scale;
        route.logits        = logits;
        route.ids           = ids;
        route.weights       = weights;
        route.pair_scale    = pair_scale;
        // A short call's route kernel also groups its pairs by expert.
        void* routing_space = workspace->alloc_bytes(gguf::moe_routing_bytes(e, pairs)).data;
        const bool sorted = t <= kernels::kSortedRouteRows && e <= kernels::kSortedRouteExperts;
        gguf::MoeRouting routing;
        if (sorted) {
            routing = gguf::moe_sorted_routes(routing_space, pairs, e);
            route.sorted = kernels::RouteSort{const_cast<std::int32_t*>(routing.bounds),
                                              const_cast<std::int32_t*>(routing.sorted),
                                              const_cast<std::int32_t*>(routing.active),
                                              const_cast<std::int32_t*>(routing.active_count),
                                              const_cast<std::int32_t*>(routing.tile_count),
                                              const_cast<std::int32_t*>(routing.tiles),
                                              routing.tile_capacity};
        }
        kernels::route(route, stream);
        const bool vector = c.q8;
        const void* act   = vector ? c.ffn_q.base : nullptr;
        if (model.expert_parallel()) {
            experts_parallel(p, c.ffn_bf16, act, ids, weights, pair_scale, t, ra);
            return;
        }
        if (!sorted) { routing = gguf::moe_sort_routes(ids, pairs, e, routing_space, stream); }
        const bool cpu = cpu_experts && t <= CpuExperts::kMaxTokens && vector;
        std::optional<ExpertDelivery::Call> call;
        if (delivery) {
            const std::array<ExpertDelivery::Input, 2> inputs{
                ExpertDelivery::Input{act, cpu ? cpu_experts->activation_bytes(t) : 0},
                ExpertDelivery::Input{weights, cpu ? cpu_experts->weight_bytes(t) : 0}};
            call = delivery->begin(std::size_t(p.delivery_layer), ids, std::uint32_t(pairs), stream,
                                   cpu ? std::span<const ExpertDelivery::Input>(inputs)
                                       : std::span<const ExpertDelivery::Input>());
        }
        const auto* tables = delivery ? delivery->tables(std::size_t(p.delivery_layer))
                                      : static_cast<const void* const*>(p.tables.p);
        const gguf::MoeTable gate_up{tables + 2 * e, p.gate_up_row_bytes, 2 * width, hidden,
                                     p.gate_up_aligned16};
        const gguf::MoeTable down{tables + 3 * e, p.down_row_bytes, hidden, width,
                                  p.down_aligned16};
        const int max_active = std::min(pairs, e);
        auto* middle = static_cast<__nv_bfloat16*>(
            workspace->alloc_bytes(std::size_t(pairs) * width * 2).data);
        auto* gu = static_cast<float*>(
            workspace->alloc_bytes(std::size_t(pairs) * 2 * width * 4).data);
        if (vector) {
            const int chunk = t <= 1 ? 1 : t <= 2 ? 2 : t <= 4 ? 4 : 8;
            void* act2  = workspace->alloc_bytes(gguf::vector_activation_bytes(width, pairs)).data;
            auto* fixed = static_cast<unsigned long long*>(
                workspace->alloc_bytes(std::size_t(t) * hidden * 8).data);
            CUDA_CHECK(cudaMemsetAsync(fixed, 0, std::size_t(t) * hidden * 8, stream));
            gguf::MoeOutput rows;
            rows.f32 = gu;
            gguf::MoeOutput out;
            out.weighted = fixed;
            out.weights  = weights;
            const auto compute = [&](const gguf::MoeRouting& part) {
                gguf::moe_vector_product(p.gate_up_type, gate_up, part, max_active, k, k, act, t,
                                         rows, chunk, stream);
                situ(gu, pairs, pair_scale, middle, stream);
                gguf::quantize_vector_activation(middle, width, pairs, nullptr, act2, stream);
                gguf::moe_vector_product(p.down_type, down, part, max_active, 1, k, act2, pairs,
                                         out, chunk, stream);
            };
            if (call) {
                // Both stages add to the same exact fixed-point sum, each over its experts of the
                // call's sorted routes. The middles of the other stage's pairs come from zero rows
                // and are never read.
                const auto stage = [&](const std::int32_t* set) {
                    gguf::MoeRouting part = routing;
                    part.active_count     = set;
                    part.active           = set + 1;
                    return part;
                };
                CUDA_CHECK(cudaMemsetAsync(gu, 0, std::size_t(pairs) * 2 * width * 4, stream));
                compute(stage(call->cached_experts));
                delivery->await(stream);
                compute(stage(call->missing_experts));
                if (call->cpu != nullptr) {
                    kernels::add_fixed(fixed, call->products, call->cpu, std::int64_t(t) * hidden,
                                       stream);
                }
            } else {
                compute(routing);
            }
            ra.fixed = fixed;
            return;
        }
        if (delivery) { delivery->await(stream); }
        if (p.bf16) {
            auto* rows = static_cast<float*>(
                workspace->alloc_bytes(std::size_t(pairs) * hidden * 4).data);
            bf16_experts(tables, routing, max_active, c.ffn_bf16, gu, middle, pair_scale, pairs,
                         rows, stream);
            ra.moe_rows    = rows;
            ra.moe_weights = weights;
            ra.top_k       = k;
            return;
        }
        void* rows_act = workspace->alloc_bytes(gguf::moe_matrix_activation_bytes(hidden, pairs)).data;
        gguf::quantize_moe_matrix_activation(p.gate_up_type, c.ffn_bf16, hidden, routing, pairs, k,
                                             rows_act, stream);
        gguf::moe_matrix_product(p.gate_up_type, gate_up, routing, max_active, pairs, true,
                                 rows_act, gu, stream);
        situ(gu, pairs, pair_scale, middle, stream);
        void* act2 = workspace->alloc_bytes(gguf::moe_matrix_activation_bytes(width, pairs)).data;
        gguf::quantize_moe_matrix_activation(p.down_type, middle, width, routing, pairs, 1, act2,
                                             stream);
        auto* rows = static_cast<float*>(workspace->alloc_bytes(std::size_t(pairs) * hidden * 4).data);
        gguf::moe_matrix_product(p.down_type, down, routing, max_active, pairs, true, act2, rows,
                                 stream);
        ra.moe_rows    = rows;
        ra.moe_weights = weights;
        ra.top_k       = k;
    }

    // The head of `rows` final-normed BF16 rows x into logits_bf16 columns [first, first + rows).
    void head_logits(const __nv_bfloat16* rows_x, int rows, int first) {
        for (int at = 0; at < rows; at += kHeadRows) {
            const int n = std::min(kHeadRows, rows - at);
            auto scope  = workspace->scope();
            Tensor x(const_cast<__nv_bfloat16*>(rows_x) + std::int64_t(at) * hidden, DType::BF16,
                     {hidden, n});
            const int stride = head_tiles.n;
            auto* plane = static_cast<float*>(workspace->alloc_bytes(std::size_t(n) * stride * 4).data);
            project_f32(x, head_tiles, plane, stride);
            kernels::softcap_bf16(plane, stride,
                                  static_cast<__nv_bfloat16*>(logits_bf16.p) +
                                      std::int64_t(first + at) * vocab,
                                  vocab, vocab, n, config.logit_softcap, stream);
        }
    }

    // The head of the last `rows` of t <= kMaxVectorColumns final-normed q8_1 rows into
    // logits_bf16 columns [0, rows).
    void head_logits_q8(const Q8Act& x, int t, int rows) {
        const float cap = config.logit_softcap;
        if (divided_head) {
            std::vector<cudaStream_t> streams{stream};
            for (std::size_t r = 1; r < expert_peers.size(); ++r) {
                streams.push_back(expert_peers[r].stream);
            }
            divided_head->project(
                x.base, gguf::vector_activation_bytes(hidden, t), rows,
                static_cast<__nv_bfloat16*>(logits_bf16.p), streams,
                [&](std::size_t, const void* input, const Weight& piece, int, float* plane,
                    __nv_bfloat16* out, cudaStream_t on) {
                    gguf::VectorOutput o;
                    o.f32           = plane;
                    o.column_stride = piece.n;
                    gguf::vector_product(ops::detail::gguf_type(piece.qtype), piece.qdata,
                                         row_bytes(piece), piece.n, piece.k, input, t, o, on);
                    kernels::softcap_bf16(plane + std::int64_t(t - rows) * piece.n, out,
                                          std::int64_t(rows) * piece.n, cap, on);
                });
            return;
        }
        auto scope  = workspace->scope();
        auto* plane = static_cast<float*>(workspace->alloc_bytes(std::size_t(t) * vocab * 4).data);
        vector_f32(x.base, head, t, plane, vocab);
        kernels::softcap_bf16(plane + std::int64_t(t - rows) * vocab,
                              static_cast<__nv_bfloat16*>(logits_bf16.p),
                              std::int64_t(rows) * vocab, cap, stream);
    }

    // ---- Prompt chunks ---------------------------------------------------------------------------

    void forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                 std::uint32_t logit_rows, std::span<const std::uint32_t> branches = {},
                 std::uint32_t shared = 0, std::span<const ImageRows> images = {}) {
        const int t = int(tokens.size());
        if (sequence >= options.sequences || t <= 0 || std::uint32_t(t) > options.prefill_chunk ||
            logit_rows > std::min<std::uint32_t>(t, kMaxLogitRows) ||
            (logit_rows == 0 && !branches.empty())) {
            throw std::invalid_argument("inkling2 forward: invalid call");
        }
        const bool branching = !branches.empty();
        if (branching && (logit_rows != branches.size() || shared >= std::uint32_t(t) ||
                          std::accumulate(branches.begin(), branches.end(), std::uint64_t{0}) !=
                              std::uint64_t(t) - shared ||
                          std::find(branches.begin(), branches.end(), 0U) != branches.end())) {
            throw std::invalid_argument("inkling2 forward_branches: invalid branches");
        }
        if (!images.empty() && branching) {
            throw std::invalid_argument("inkling2 forward_branches: images in a branch pass");
        }
        check_images(images, t);
        const std::uint32_t first = positions[sequence];
        if (first + std::uint32_t(t) > options.max_context) {
            throw std::invalid_argument("inkling2 forward: context capacity exceeded");
        }
        back(sequence, first + std::uint32_t(t));
        std::vector<std::int32_t> pos(t), starts;
        for (int i = 0; i < t; ++i) { pos[i] = std::int32_t(first) + i; }
        if (branching) {
            starts.resize(t);
            for (std::size_t b = 0, row = shared; b < branches.size(); row += branches[b++]) {
                for (std::uint32_t j = 0; j < branches[b]; ++j) {
                    pos[row + j]    = std::int32_t(first + shared + j);
                    starts[row + j] = std::int32_t(row);
                }
            }
        }
        stage(tokens, pos, {}, starts);
        workspace->reset();
        auto scope = workspace->scope();
        Call c;
        c.t        = t;
        // A branch pass gathers its branches' last rows from the BF16 final rows.
        c.q8       = t <= gguf::kMaxVectorColumns && !branching && !bf16_rows;
        c.chunk    = true;
        c.sequence = sequence;
        c.first    = int(first);
        c.kept     = branching ? int(shared) : t;
        c.rows.positions = static_cast<const std::int32_t*>(d_positions.p);
        c.rows.sequence  = int(sequence);
        c.rows.tokens    = t;
        c.rows.chunk     = true;
        c.rows.first     = int(first);
        c.rows.shared    = int(shared);
        c.rows.starts    = branching ? static_cast<const std::int32_t*>(d_starts.p) : nullptr;
        // The drafter's taps of the rows the sequence keeps, the last window of them.
        const int tap_rows  = spine ? std::min(c.kept, int(spine->window())) : 0;
        const int tap_first = c.kept - tap_rows;
        c.plane = std::int64_t(t) * hidden;
        c.spark_rows = c.kept;
        run_with_taps(c, images, tap_first, tap_rows);
        if (tap_rows > 0) {
            spine->commit(spine->lane(sequence), first + std::uint32_t(tap_first), 0,
                          std::uint32_t(tap_rows), stream);
        }
        if (branching) {
            // Each branch's last row to the front, in branch order.
            auto* rows      = c.ffn_bf16;
            std::size_t row = shared;
            for (std::size_t b = 0; b < branches.size(); ++b) {
                row += branches[b];
                if (row - 1 == b) { continue; }
                CUDA_CHECK(cudaMemcpyAsync(
                    rows + std::int64_t(b) * hidden, rows + std::int64_t(row - 1) * hidden,
                    std::size_t(hidden) * 2, cudaMemcpyDeviceToDevice, stream));
            }
            head_logits(rows, int(logit_rows), 0);
            positions[sequence]  = first + shared;
            ring_floor[sequence] = first + shared;
            return;
        }
        if (logit_rows > 0) {
            if (c.q8) {
                head_logits_q8(c.ffn_q, t, int(logit_rows));
            } else {
                head_logits(c.ffn_bf16 + std::int64_t(t - int(logit_rows)) * hidden,
                            int(logit_rows), 0);
            }
        }
        positions[sequence]  = first + std::uint32_t(t);
        ring_floor[sequence] = first + std::uint32_t(t);
    }

    // run() with the DSpine drafter's taps of rows [tap_first, tap_first + tap_rows): each
    // visit's output state.
    void run_with_taps(Call& c, std::span<const ImageRows> images, int tap_first, int tap_rows) {
        const int t = c.t;
        for (int i = 0; i < kPool; ++i) {
            c.states.buffers[i] =
                static_cast<float*>(workspace->alloc_bytes(std::size_t(c.plane) * 4).data);
        }
        if (c.q8) {
            c.in_q_stride = q8_bytes(hidden, t);
            c.in_q        = alloc_q8(hidden, t, 3);
            c.ffn_q       = alloc_q8(hidden, t);
        } else {
            c.in_bf16  = static_cast<__nv_bfloat16*>(
                workspace->alloc_bytes(std::size_t(3) * c.plane * 2).data);
            c.ffn_bf16 = static_cast<__nv_bfloat16*>(workspace->alloc_bytes(std::size_t(c.plane) * 2).data);
        }
        c.in_f32   = static_cast<float*>(workspace->alloc_bytes(std::size_t(3) * c.plane * 4).data);
        c.raw      = static_cast<float*>(workspace->alloc_bytes(std::size_t(3) * c.plane * 4).data);
        c.messages = static_cast<float*>(workspace->alloc_bytes(std::size_t(4) * c.plane * 4).data);
        c.ffn_f32  = static_cast<float*>(workspace->alloc_bytes(std::size_t(c.plane) * 4).data);
        c.ffn_raw  = static_cast<float*>(workspace->alloc_bytes(std::size_t(c.plane) * 4).data);
        if (spark) {
            c.spark_taps = static_cast<float*>(workspace->alloc_bytes(
                std::size_t(spark->taps - 1) * c.plane * 4).data);
        }
        if (t <= kernels::kSplitRows) {
            c.split = static_cast<float*>(
                workspace->alloc_bytes(kernels::mix_split_floats(t, hidden) * 4).data);
        }
        embed(c, images);
        for (std::size_t v = 0; v < visits.size(); ++v) {
            visit(c, v);
            store_tap(v, c.states.buffers[c.states.cur], tap_first, tap_rows);
            if (spark && spark->tap_after[v] >= 0) {
                CUDA_CHECK(cudaMemcpyAsync(c.spark_taps + std::int64_t(spark->tap_after[v]) * c.plane,
                                           c.states.buffers[c.states.cur],
                                           std::size_t(c.plane) * 4, cudaMemcpyDeviceToDevice,
                                           stream));
            }
        }
        if (spark) { spark_inject(c); }
    }

    // ---- Decode ----------------------------------------------------------------------------------

    // One token per row; tokens, positions and sequences already staged on the device. With
    // `keep` (a decode step) each row's drafter taps go into its sequence's lane; a verification
    // keeps them for commit(). Both keep every row's convolution history (positions a later call
    // rewrites when commit() keeps fewer).
    void decode_body(int b, bool keep, int row_group = 1) {
        workspace->reset();
        auto scope = workspace->scope();
        Call c;
        c.t              = b;
        c.row_group      = row_group;
        c.q8             = !bf16_rows;
        c.keep           = true;
        c.rows.positions = static_cast<const std::int32_t*>(d_positions.p);
        c.rows.sequences = static_cast<const std::int32_t*>(d_sequences.p);
        c.rows.tokens    = b;
        c.plane          = std::int64_t(b) * hidden;
        c.spark_rows     = b;
        run_with_taps(c, {}, 0, spine ? b : 0);
        if (c.q8) {
            head_logits_q8(c.ffn_q, b, b);
        } else {
            head_logits(c.ffn_bf16, b, 0);
        }
        if (spine && keep) {
            spine->commit_rows(Tensor(d_sequences.p, DType::I32, {b}),
                               Tensor(d_positions.p, DType::I32, {b}), 0, stream);
        }
    }

    template <typename Body>
    void run_graph(std::uint32_t key, Body&& body) {
        if (!options.cuda_graphs) {
            body();
            return;
        }
        GraphEntry& entry = graphs[key];
        if (!entry.warmed) {
            CUDA_CHECK(ninfer::wait_for_stream(stream));
            body();
            entry.warmed = true;
            return;
        }
        if (!entry.executable.ready()) {
            entry.definition.capture(stream, [&] { body(); });
            entry.executable.instantiate(entry.definition);
        }
        entry.executable.launch(stream);
    }

    void decode(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens) {
        const int b = int(sequences.size());
        if (b <= 0 || std::uint32_t(b) > options.sequences || tokens.size() != sequences.size() ||
            b > gguf::kMaxVectorColumns) {
            throw std::invalid_argument("inkling2 decode: invalid batch");
        }
        std::vector<std::int32_t> pos(b), seq(b);
        for (int i = 0; i < b; ++i) {
            if (sequences[i] >= options.sequences ||
                std::count(sequences.begin(), sequences.begin() + i, sequences[i]) != 0) {
                throw std::invalid_argument("inkling2 decode: invalid sequence");
            }
            if (positions[sequences[i]] + 1 > options.max_context) {
                throw std::invalid_argument("inkling2 decode: context capacity exceeded");
            }
            pos[i] = std::int32_t(positions[sequences[i]]);
            seq[i] = std::int32_t(sequences[i]);
        }
        for (int i = 0; i < b; ++i) { back(sequences[i], positions[sequences[i]] + 1); }
        stage(tokens, pos, seq);
        run_graph(std::uint32_t(b), [&] { decode_body(b, true); });
        for (int i = 0; i < b; ++i) { positions[sequences[i]] += 1; }
    }

    // ---- Speculative rounds ----------------------------------------------------------------------

    void verify(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> tokens,
                std::span<const std::uint32_t> widths = {}) {
        const int n = int(sequences.size());
        if (options.draft_tokens == 0 || n <= 0 ||
            (widths.empty() && tokens.size() % sequences.size() != 0) ||
            (!widths.empty() && widths.size() != sequences.size())) {
            throw std::invalid_argument("inkling2 verify: invalid call");
        }
        std::vector<std::uint32_t> row_widths(widths.begin(), widths.end());
        if (row_widths.empty()) { row_widths.assign(n, std::uint32_t(tokens.size()) / n); }
        int rows = 0;
        for (const std::uint32_t w : row_widths) {
            if (w < 1 || w > options.draft_tokens + 1) {
                throw std::invalid_argument("inkling2 verify: invalid width");
            }
            rows += int(w);
        }
        if (std::size_t(rows) != tokens.size() || rows > gguf::kMaxVectorColumns) {
            throw std::invalid_argument("inkling2 verify: invalid width");
        }
        std::vector<std::int32_t> pos(rows), seq(rows);
        for (int j = 0, at = 0; j < n; ++j) {
            const std::uint32_t s = sequences[j];
            if (s >= options.sequences ||
                std::count(sequences.begin(), sequences.begin() + j, s) != 0) {
                throw std::invalid_argument("inkling2 verify: invalid sequence");
            }
            if (positions[s] + row_widths[j] > options.max_context) {
                throw std::invalid_argument("inkling2 verify: context capacity exceeded");
            }
            for (std::uint32_t i = 0; i < row_widths[j]; ++i, ++at) {
                pos[at] = std::int32_t(positions[s] + i);
                seq[at] = std::int32_t(s);
            }
        }
        for (int j = 0; j < n; ++j) { back(sequences[j], positions[sequences[j]] + row_widths[j]); }
        stage(tokens, pos, seq);
        // A full visit's attention reads each key tile once for up to four rows of a sequence.
        run_graph(16 + std::uint32_t(rows),
                  [&] { decode_body(rows, false, std::min(rows, kernels::kMaxRowGroup)); });
        verify_widths = std::move(row_widths);
    }

    void commit(std::span<const std::uint32_t> sequences, std::span<const std::uint32_t> kept) {
        if (kept.size() != sequences.size() || verify_widths.size() != sequences.size()) {
            throw std::invalid_argument("inkling2 commit: invalid call");
        }
        for (std::size_t j = 0; j < sequences.size(); ++j) {
            const std::uint32_t s = sequences[j];
            if (s >= options.sequences || kept[j] < 1 || kept[j] > verify_widths[j]) {
                throw std::invalid_argument("inkling2 commit: invalid row");
            }
        }
        for (std::size_t j = 0, column = 0; j < sequences.size(); column += verify_widths[j++]) {
            const std::uint32_t s = sequences[j];
            if (spine) {
                spine->commit(spine->lane(s), positions[s], std::uint32_t(column), kept[j], stream);
            }
            positions[s] += kept[j];
        }
        verify_widths.clear();
    }

    void draft(std::span<const std::uint32_t> sequences, std::span<const std::int32_t> anchors,
               std::uint32_t k, std::span<std::int32_t> drafts, std::span<float> confidence) {
        if (spark) {
            if (anchors.size() != sequences.size() || drafts.size() != sequences.size() * k ||
                confidence.size() != drafts.size()) {
                throw std::invalid_argument("inkling2 draft: invalid call");
            }
            for (std::size_t j = 0; j < sequences.size(); ++j) {
                spark_draft(sequences[j], anchors[j], drafts.subspan(j * k, k),
                            confidence.subspan(j * k, k));
            }
            return;
        }
        if (!spine) { throw std::logic_error("inkling2: an executor without a drafter"); }
        std::vector<std::uint32_t> lanes, at;
        for (std::size_t j = 0; j < sequences.size(); ++j) {
            const std::uint32_t s = sequences[j];
            if (s >= options.sequences ||
                std::count(sequences.begin(), sequences.begin() + std::ptrdiff_t(j), s) != 0) {
                throw std::invalid_argument("inkling2 draft: invalid sequence");
            }
            lanes.push_back(spine->lane(s));
            at.push_back(positions[s]);
        }
        spine->draft(lanes, anchors, at, k, drafts, confidence, stream);
    }

    void warm_up() {
        if (spine) { spine->prepare(stream); }
        if (spark) { prepare_spark(); }
        const std::vector<std::int32_t> prompt(std::min<std::uint32_t>(options.prefill_chunk, 16), 2);
        forward(0, prompt, 1);
        if (spine) {
            const std::uint32_t s     = 0;
            const std::int32_t anchor = 2;
            for (std::uint32_t k = 1; k <= options.draft_tokens; ++k) {
                std::vector<std::int32_t> drafts(k);
                std::vector<float> confidence(k);
                draft({&s, 1}, {&anchor, 1}, k, drafts, confidence);
            }
        }
        if (spark) {
            // The round's graph: its first run loads the kernels, the second captures it.
            std::vector<std::int32_t> drafts(options.draft_tokens);
            std::vector<float> confidence(options.draft_tokens);
            for (int pass = 0; pass < 2; ++pass) { spark_draft(0, 2, drafts, confidence); }
        }
        positions[0] = 0;
        std::vector<std::uint32_t> sequences;
        std::vector<std::int32_t> tokens;
        for (std::uint32_t b = 1; b <= std::min<std::uint32_t>(options.sequences, 8); ++b) {
            sequences.push_back(b - 1);
            tokens.push_back(2);
            for (int pass = 0; pass < 2; ++pass) {
                decode(sequences, tokens);
                for (const std::uint32_t s : sequences) { positions[s] = 0; }
            }
        }
        if (options.draft_tokens > 0) {
            const std::uint32_t s = 0;
            std::vector<std::int32_t> row(options.draft_tokens + 1, 2);
            for (int pass = 0; pass < 2; ++pass) {
                verify({&s, 1}, row);
                const std::uint32_t one = 1;
                commit({&s, 1}, {&one, 1});
                positions[0] = 0;
            }
        }
        CUDA_CHECK(ninfer::wait_for_stream(stream));
        if (delivery) {
            delivery->drain();
            delivery->prime(options.expert_profile ? &*options.expert_profile : nullptr);
            std::lock_guard lock(expert_stats_mutex);
            expert_startup_stats = delivery->stats();
        }
    }

    // ---- Checkpoints and images ------------------------------------------------------------------

    // A sliding visit's ring buffers of one sequence, (start, bytes): K and V.
    std::array<std::pair<std::uint8_t*, std::size_t>, 2> ring_buffers(const VisitPlan& p,
                                                                      std::uint32_t sequence) const {
        const std::size_t bytes = std::size_t(p.seq_stride) * sizeof(__half);
        return {{{static_cast<std::uint8_t*>(p.k.p) + sequence * bytes, bytes},
                 {static_cast<std::uint8_t*>(p.v.p) + sequence * bytes, bytes}}};
    }

    [[nodiscard]] std::size_t sequence_history_bytes() const {
        return visits.size() * kR * std::size_t(slot_floats) * 4;
    }

    // Every cache a sequence's state holds: the visits', then the DSpark drafter's layers'.
    [[nodiscard]] std::vector<const VisitPlan*> cache_plans() const {
        std::vector<const VisitPlan*> out;
        for (const VisitPlan& p : visits) { out.push_back(&p); }
        if (spark) {
            for (const SparkLayer& l : spark->layers) { out.push_back(&l.cache); }
        }
        return out;
    }

    // One sequence's ring of the recurrent head's contexts.
    [[nodiscard]] std::size_t spark_context_bytes() const {
        return std::size_t(kR) * hidden * 4;
    }

    void copy_rings(std::uint32_t sequence, bool save) {
        if (!options.checkpoints) { throw std::logic_error("inkling2: an executor without checkpoints"); }
        if (spine) {
            const std::uint32_t live = spine->lane(sequence), kept = spine->lane(sequence, 1);
            spine->copy_lane(save ? live : kept, save ? kept : live, stream);
        }
        for (const VisitPlan* p : cache_plans()) {
            if (p->full) { continue; }
            const auto rings = ring_buffers(*p, sequence);
            for (std::size_t i = 0; i < rings.size(); ++i) {
                auto* kept = static_cast<std::uint8_t*>(p->checkpoint[i].p) + sequence * rings[i].second;
                CUDA_CHECK(cudaMemcpyAsync(save ? kept : rings[i].first, save ? rings[i].first : kept,
                                           rings[i].second, cudaMemcpyDeviceToDevice, stream));
            }
        }
        if (spark) {
            const std::size_t ring_bytes = spark_context_bytes();
            auto* live = static_cast<std::uint8_t*>(spark->contexts.p) + sequence * ring_bytes;
            auto* kept = static_cast<std::uint8_t*>(spark->contexts_kept.p) + sequence * ring_bytes;
            CUDA_CHECK(cudaMemcpyAsync(save ? kept : live, save ? live : kept, ring_bytes,
                                       cudaMemcpyDeviceToDevice, stream));
        }
        const std::size_t bytes = sequence_history_bytes();
        auto* live = static_cast<std::uint8_t*>(history.p) + sequence * bytes;
        auto* kept = static_cast<std::uint8_t*>(history_kept.p) + sequence * bytes;
        CUDA_CHECK(cudaMemcpyAsync(save ? kept : live, save ? live : kept, bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }

    [[nodiscard]] std::uint64_t image_bytes(std::uint32_t position) const {
        std::uint64_t bytes = sequence_history_bytes();
        for (const VisitPlan* p : cache_plans()) {
            if (p->full && p->k_codes.p != nullptr) {
                // A quantized visit's planes: per head, `position` slots of codes and scales.
                for (const KvArea* area : {&p->k_codes, &p->k_scales, &p->v_codes, &p->v_scales}) {
                    bytes += std::uint64_t(kv_heads) * position * area->position_bytes;
                }
            } else if (p->full) {
                bytes += 2 * std::uint64_t(kv_heads) * position * kD * sizeof(__half);
            } else {
                for (const auto& r : ring_buffers(*p, 0)) { bytes += r.second; }
            }
        }
        if (spark) { bytes += spark_context_bytes(); }
        return bytes + (spine ? spine->image_bytes(position) : 0);
    }

    void copy_image(std::uint32_t sequence, std::uint32_t position, std::span<std::byte* const> pages,
                    std::size_t page_bytes, bool save, bool checkpoint) {
        if (page_bytes == 0 || pages.size() < (image_bytes(position) + page_bytes - 1) / page_bytes) {
            throw std::invalid_argument("inkling2 image: pages do not hold the image");
        }
        std::size_t at = 0;
        const auto copy = [&](std::uint8_t* device, std::size_t bytes) {
            while (bytes > 0) {
                const std::size_t in_page = std::min(bytes, page_bytes - at % page_bytes);
                std::byte* host           = pages[at / page_bytes] + at % page_bytes;
                if (save) {
                    CUDA_CHECK(cudaMemcpyAsync(host, device, in_page, cudaMemcpyDeviceToHost, stream));
                } else {
                    CUDA_CHECK(cudaMemcpyAsync(device, host, in_page, cudaMemcpyHostToDevice, stream));
                }
                device += in_page;
                bytes -= in_page;
                at += in_page;
            }
        };
        const std::size_t head_bytes = std::size_t(global_slots) * kD * sizeof(__half);
        for (const VisitPlan* p : cache_plans()) {
            if (p->full) {
                if (p->k_codes.p != nullptr) {
                    for (const KvArea* area :
                         {&p->k_codes, &p->k_scales, &p->v_codes, &p->v_scales}) {
                        auto* base = static_cast<std::uint8_t*>(area->p) +
                                     sequence * area->row_bytes * kv_heads;
                        for (int h = 0; h < kv_heads; ++h) {
                            copy(base + h * area->row_bytes,
                                 std::size_t(position) * area->position_bytes);
                        }
                    }
                    continue;
                }
                for (const KvArea* area : {&p->k, &p->v}) {
                    auto* base = static_cast<std::uint8_t*>(area->p) + sequence * head_bytes * kv_heads;
                    for (int h = 0; h < kv_heads; ++h) {
                        copy(base + h * head_bytes, std::size_t(position) * kD * sizeof(__half));
                    }
                }
                continue;
            }
            const auto rings = ring_buffers(*p, sequence);
            for (std::size_t i = 0; i < rings.size(); ++i) {
                std::uint8_t* device =
                    checkpoint ? static_cast<std::uint8_t*>(p->checkpoint[i].p) + sequence * rings[i].second
                               : rings[i].first;
                copy(device, rings[i].second);
            }
        }
        if (spark) {
            const std::size_t ring_bytes = spark_context_bytes();
            copy(static_cast<std::uint8_t*>(checkpoint ? spark->contexts_kept.p : spark->contexts.p) +
                     sequence * ring_bytes,
                 ring_bytes);
        }
        const std::size_t bytes = sequence_history_bytes();
        copy(static_cast<std::uint8_t*>(checkpoint ? history_kept.p : history.p) + sequence * bytes,
             bytes);
        if (spine) { spine->copy_image(spine->lane(sequence, checkpoint ? 1 : 0), position, copy); }
    }

    SequenceImage save_image(std::uint32_t sequence, const SequenceSnapshot* at,
                             std::span<std::byte* const> pages, std::size_t page_bytes) {
        const std::uint32_t position = at != nullptr ? at->position : positions.at(sequence);
        copy_image(sequence, position, pages, page_bytes, true, at != nullptr);
        return {position};
    }

    void load_image(std::uint32_t sequence, const SequenceImage& image,
                    std::span<std::byte* const> pages, std::size_t page_bytes) {
        if (image.position > options.max_context) {
            throw std::invalid_argument("inkling2 load_image: invalid image");
        }
        back(sequence, image.position);
        copy_image(sequence, image.position, pages, page_bytes, false, false);
        positions.at(sequence)  = image.position;
        ring_floor.at(sequence) = image.position;
    }

    void snapshot(std::uint32_t sequence, SequenceSnapshot& out) {
        copy_rings(sequence, true);
        out.position = positions[sequence];
    }

    void restore(std::uint32_t sequence, const SequenceSnapshot& from) {
        copy_rings(sequence, false);
        positions[sequence]  = from.position;
        ring_floor[sequence] = from.position;
    }

    // A position's state lies in the rings while the sequence has written fewer positions past it
    // than the sliding rings hold beyond the window and a round's drafts, and than the histories
    // hold beyond the convolutions' three earlier inputs and a round's drafts.
    bool rewind(std::uint32_t sequence, std::uint32_t tokens) {
        const std::uint32_t at = positions.at(sequence);
        const std::int64_t drafts = std::int64_t(options.draft_tokens);
        const std::int64_t slack  = std::min<std::int64_t>(std::int64_t(ring) - window - drafts,
                                                           kR - (kTaps - 1) - drafts);
        if (tokens > at || at - tokens < ring_floor[sequence] || std::int64_t(tokens) > slack) {
            return false;
        }
        positions[sequence] = at - tokens;
        return true;
    }
};

Executor::Executor(const Model& model, DeviceContext& device, ExecutorOptions options)
    : impl_(std::make_unique<Impl>(model, device, options)) {}

Executor::~Executor() = default;

const ExecutorOptions& Executor::options() const noexcept { return impl_->options; }

ExecutorMemory Executor::memory() const noexcept {
    return ExecutorMemory{.kv_bytes        = impl_->kv_bytes,
                          .kv_backed_bytes = impl_->kv_backed.load(std::memory_order_relaxed),
                          .kv_grows        = impl_->growing,
                          .history_bytes   = impl_->history_bytes,
                          .workspace_bytes = impl_->workspace_bytes,
                          .vision_encode_bytes = impl_->tower ? impl_->tower->workspace_bytes() : 0,
                          .vision_owned_bytes  = impl_->vision_owned_bytes(),
                          .drafter_bytes   = impl_->spine   ? impl_->spine->device_bytes()
                                             : impl_->spark ? impl_->spark->bytes
                                                            : 0,
                          .experts = impl_->delivery ? impl_->delivery->memory() : ExpertDeliveryMemory{}};
}

std::uint32_t Executor::vocab() const noexcept { return std::uint32_t(impl_->vocab); }

std::uint64_t Executor::image_bytes(std::uint32_t position) const {
    return impl_->image_bytes(position);
}

SequenceImage Executor::save_image(std::uint32_t sequence, const SequenceSnapshot* at,
                                   std::span<std::byte* const> pages, std::size_t page_bytes) {
    return impl_->save_image(sequence, at, pages, page_bytes);
}

void Executor::load_image(std::uint32_t sequence, const SequenceImage& image,
                          std::span<std::byte* const> pages, std::size_t page_bytes) {
    impl_->load_image(sequence, image, pages, page_bytes);
}

void Executor::reset(std::uint32_t sequence) {
    impl_->positions.at(sequence)  = 0;
    impl_->ring_floor.at(sequence) = 0;
    impl_->release(sequence);
}

std::uint32_t Executor::position(std::uint32_t sequence) const {
    return impl_->positions.at(sequence);
}

void Executor::warm_up() {
    impl_->warm_up();
    check_experts();
}

void Executor::check_experts() const {
    if (impl_->delivery) { impl_->delivery->check(); }
}
ExpertDeliveryStats Executor::expert_stats() const {
    if (!impl_->delivery) { return {}; }
    std::lock_guard lock(impl_->expert_stats_mutex);
    return impl_->delivery->stats() - impl_->expert_startup_stats;
}
ExpertRouteCounts Executor::expert_totals() const {
    return impl_->delivery ? impl_->delivery->totals() : ExpertRouteCounts{};
}
std::string Executor::expert_cpu_description() const {
    return impl_->delivery ? impl_->delivery->cpu_description() : std::string();
}

void Executor::forward(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                       std::uint32_t logit_rows, std::span<const ImageRows> images) {
    impl_->forward(sequence, tokens, logit_rows, {}, 0, images);
}

void Executor::forward_branches(std::uint32_t sequence, std::span<const std::int32_t> tokens,
                                std::span<const std::uint32_t> lengths) {
    impl_->forward(sequence, tokens, static_cast<std::uint32_t>(lengths.size()), lengths);
}

void Executor::forward_prompt_branches(std::uint32_t sequence,
                                       std::span<const std::int32_t> prompt,
                                       std::span<const std::int32_t> tokens,
                                       std::span<const std::uint32_t> lengths) {
    std::vector<std::int32_t> chunk(prompt.begin(), prompt.end());
    chunk.insert(chunk.end(), tokens.begin(), tokens.end());
    impl_->forward(sequence, chunk, static_cast<std::uint32_t>(lengths.size()), lengths,
                   static_cast<std::uint32_t>(prompt.size()));
}

void Executor::decode(std::span<const std::uint32_t> sequences,
                      std::span<const std::int32_t> tokens) {
    impl_->decode(sequences, tokens);
}

Tensor Executor::logits(std::uint32_t rows) const {
    if (rows == 0 || rows > kMaxLogitRows) {
        throw std::invalid_argument("inkling2 logits: invalid row count");
    }
    return Tensor(impl_->logits_bf16.p, DType::BF16,
                  {impl_->vocab, static_cast<std::int32_t>(rows)});
}

cudaStream_t Executor::head_stream() const noexcept { return impl_->stream; }

std::uint32_t Executor::draft_tokens() const noexcept { return impl_->options.draft_tokens; }

void Executor::verify(std::span<const std::uint32_t> sequences,
                      std::span<const std::int32_t> tokens, std::span<const std::uint32_t> widths) {
    impl_->verify(sequences, tokens, widths);
}

void Executor::commit(std::span<const std::uint32_t> sequences,
                      std::span<const std::uint32_t> kept) {
    impl_->commit(sequences, kept);
}

void Executor::draft(std::span<const std::uint32_t> sequences,
                     std::span<const std::int32_t> anchors, std::uint32_t k,
                     std::span<std::int32_t> drafts, std::span<float> confidence) {
    impl_->draft(sequences, anchors, k, drafts, confidence);
}

std::uint32_t Executor::dspark_block() const noexcept {
    return impl_->spark ? std::uint32_t(impl_->spark->block - 1) : 0;
}

void Executor::draft(std::uint32_t sequence, std::int32_t anchor, std::span<std::int32_t> drafts,
                     std::span<float> confidence) {
    impl_->spark_draft(sequence, anchor, drafts, confidence);
}

void Executor::snapshot(std::uint32_t sequence, SequenceSnapshot& out) {
    impl_->snapshot(sequence, out);
}

void Executor::restore(std::uint32_t sequence, const SequenceSnapshot& from) {
    impl_->restore(sequence, from);
}

bool Executor::rewind(std::uint32_t sequence, std::uint32_t tokens) {
    return impl_->rewind(sequence, tokens);
}

bool Executor::vision() const noexcept { return impl_->tower != nullptr; }

std::uint32_t Executor::encode_image(std::span<const std::uint16_t> patches, int grid_w, int grid_h,
                                     std::span<std::uint16_t> out) {
    return impl_->encode_image(patches, grid_w, grid_h, out);
}

} // namespace ninfer::models::inkling2
