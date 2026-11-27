// Inkling-2 decode executor: the 48-visit loop, one token at a time.
//
// Mirrors the reference executor's visit() call-for-call (port/m3/visit-flow.md is the
// authoritative transcription); GGUF-side storage plumbing is replaced by engine linear and
// per-session planes. Prompt chunks come later; this path is T=1 decode.
#include "models/inkling2/execution/executor.h"

#include "models/inkling2/kernels.h"  // vendored (vendor/ is first on this target's include path)

#include "core/arena.h"
#include "core/buffer.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight_view.h"
#include "models/common/bindings.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/weight_input.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ninfer::models::inkling2::execution {

namespace kern = kernels;
namespace kv128 = ops::kv128;

namespace {

constexpr int kHistory = 4;  // conv-history slots per site (reference Executor::kHistory)
constexpr int kHead    = 128;
constexpr int kTaps    = 4;

[[nodiscard]] kv128::KvPlaneView f16_plane(const __half* base, std::int64_t head_stride) {
    kv128::KvPlaneView p;
    p.codes     = const_cast<std::uint8_t*>(reinterpret_cast<const std::uint8_t*>(base));  // read-only window
    p.code_row  = kHead * 2;  // one 128-element F16 slot
    p.code_head = head_stride * 2;
    p.coding    = kv128::KvCoding::F16;
    return p;
}

// One prepared matrix (engine linear admission; the binder's recorded policy).
struct Mat {
    ops::SingleProjectionWeight w;

    Mat() = default;
    explicit Mat(const ops::WeightInput& input)
        : w(ops::prepare_linear_weight(input)) {}
};

// The device-dispatch form of a prepared Mat: same policy and geometry; the weight planes are
// fetched device-side from `slot` ([qdata, qhigh, scales]). The representative weight (any
// expert of the same kind) supplies admission and validation data.
[[nodiscard]] inline Mat with_slot(const Mat& m, const void* const* slot) {
    Mat c = m;
    c.w.weight.plane_slot = slot;
    return c;
}

struct LPlan {
    bool full = false, loops = false;
    const float *input_norm = nullptr, *residual_query = nullptr, *loop_queries = nullptr,
                *ffn_residual_query = nullptr, *ffn_norm = nullptr, *query_norm = nullptr,
                *key_norm = nullptr, *value_norm = nullptr, *query_clip = nullptr,
                *relative_clip = nullptr, *distances = nullptr, *key_conv = nullptr,
                *value_conv = nullptr, *output_conv = nullptr, *ffn_conv = nullptr,
                *router = nullptr, *router_bias = nullptr, *sink = nullptr;
    const float *key_taps = nullptr, *value_taps = nullptr;
    const float* query_mix = nullptr;           // 3 distinct rows for full layers (pre-tiled in artifact)
    Mat query, gate, relative, key, value, output;
    std::array<const float*, 3> ctrl_down{}, ctrl_gain{}, ctrl_forget{};  // [q|k|v]
    const float *ffn_down = nullptr, *ffn_gain = nullptr, *ffn_forget = nullptr;
    std::vector<Mat> expert_gate, expert_up, expert_down;
    const void** expert_ptrs = nullptr;  // device table [expert][kind][plane] (route_ptrs)
    int query_rows = 0, tap_columns = 0;
};

struct VPlan {
    std::uint32_t layer = 0, pass = 0;
    bool save_prelude = false, restart = false, control = false, start = false;
};

} // namespace

// ---- Session -------------------------------------------------------------------------------

struct Session::Storage {
    DeviceBuffer kv_planes;         // all visits' K|V planes (F16)
    DeviceBuffer history;           // per visit: sites 0..3 rings
    std::vector<__half*> kp, vp;    // per visit
    std::vector<std::int64_t> head_stride{};
    std::vector<int> ring{};        // 0 full, window sliding
    std::vector<float*> site{};     // per visit base
};

std::vector<std::uint32_t> visit_layers(const Config& cfg) {
    std::vector<std::uint32_t> out;
    for (std::uint32_t pass = 0; pass < cfg.loop.count; ++pass) {
        for (std::uint32_t l = 0; l < cfg.num_hidden_layers; ++l) {
            if (pass > 0 && !cfg.loop.loops(l)) { continue; }
            out.push_back(l);
        }
    }
    return out;
}

Session::Session(const Config& cfg, Device device, std::size_t visit_count,
                 std::uint32_t max_context)
    : store(std::make_unique<Storage>()) {
    const auto layers = visit_layers(cfg);
    const int kvd = int(cfg.key_width()), window = int(cfg.sliding_window), hidden = int(cfg.hidden_size);
    std::size_t kv_elems = 0, hist_floats = 0;
    store->kp.resize(visit_count);
    store->vp.resize(visit_count);
    store->head_stride.resize(visit_count);
    store->ring.resize(visit_count);
    store->site.resize(visit_count);
    for (std::size_t v = 0; v < visit_count; ++v) {
        const bool full  = cfg.is_full_attention(layers[v]);
        const int  slots = full ? int(max_context) : window;
        kv_elems    += 2 * std::size_t(cfg.num_key_value_heads) * slots * kHead;
        hist_floats += 2 * std::size_t(kHistory) * kvd + 2 * std::size_t(kHistory) * hidden;
    }
    store->kv_planes = DeviceBuffer(device, kv_elems * sizeof(__half));
    store->history   = DeviceBuffer(device, hist_floats * sizeof(float));
    cudaMemset(store->kv_planes.data(), 0, store->kv_planes.size());
    cudaMemset(store->history.data(),   0, store->history.size());
    auto* kv = static_cast<__half*>(store->kv_planes.data());
    auto* hi = static_cast<float*>(store->history.data());
    for (std::size_t v = 0; v < visit_count; ++v) {
        const bool full  = cfg.is_full_attention(layers[v]);
        const int  slots = full ? int(max_context) : window;
        store->ring[v] = full ? 0 : window;
        store->head_stride[v] = std::int64_t(slots) * kHead;
        store->kp[v] = kv;
        store->vp[v] = kv + std::size_t(cfg.num_key_value_heads) * slots * kHead;
        kv += 2 * std::size_t(cfg.num_key_value_heads) * slots * kHead;
        store->site[v] = hi;
        hi += 2 * std::size_t(kHistory) * kvd + 2 * std::size_t(kHistory) * hidden;
    }
}

Session::~Session() = default;

// ---- Executor ------------------------------------------------------------------------------

struct Executor::Impl {
    const Config&                config;
    std::span<const BoundWeight> bound;
    cudaStream_t                 stream;
    Device                       dev;
    CudaExecutionView            view;
    int hidden, q_heads, head_dim, kv_heads, kvd, rel, window, vocab, experts, top_k, width;
    float eps, softcap, scale;
    std::uint32_t max_ctx;
    std::vector<LPlan> layers;
    std::vector<VPlan> visits;
    Mat head_mat;
    Weight embed_table;
    const float *f_embed_norm = nullptr, *f_final_query = nullptr, *f_final_norm = nullptr;

    DeviceBuffer states;                    // kMaxStates × [hidden] f32
    DeviceBuffer messages, in_f32, ffn_f32;
    DeviceBuffer in_bf16, ffn_bf16;
    DeviceBuffer stage;                     // u32 token | i32 position
    DeviceBuffer identity;                  // i32[8] 0..7 (bf16_rows_f32 row ids)
    DeviceBuffer logits_f32, logits_bf16;
    DeviceBuffer arena_backing;
    MemoryArena  arena;

    // Per-visit scratch reused across visits (T=1).
    DeviceBuffer qgr, kv_rows, taps_rows;
    DeviceBuffer q_flat, r_flat, sinks;
    DeviceBuffer attn_bf16;
    DeviceBuffer raw_attn, raw_ffn;               // controller layers' raw mix planes
    DeviceBuffer ctrl_state, ctrl_fg, ctrl_concat;
    DeviceBuffer gu_bf16, gu_f32, mid_bf16, down_bf16, down_f32;
    DeviceBuffer route_scratch;             // ids[i32 top_k] | weights[f32] | logits[f32 experts]
    DeviceBuffer route_ptr_slot;            // [top_k][kind][plane] expert weight planes (device)

    // CUDA-graph decode path (M4): capture one full 48-visit step, replay the rest. Routing is
    // device-side (route_ptrs), so the step contains no host sync.
    DecodeGraphDefinition graph_def;
    DecodeGraphExecutable graph_exec;
    bool graphs_enabled = false, graph_ready = false;
    int  step_calls     = 0;
    std::vector<DeviceBuffer> expert_ptr_tables;  // owned; LPlan::expert_ptrs are views into them

    bool inputs_ready = false;              // reference Session::inputs_ready

    struct StateBook {
        std::vector<int> blocks, prelude_blocks, bank;
        int cur = -1, prelude_cur = -1, bank_last = -1;
        [[nodiscard]] bool used(int i) const {
            const auto in = [i](const std::vector<int>& l) {
                return std::find(l.begin(), l.end(), i) != l.end();
            };
            return i == cur || i == prelude_cur || i == bank_last || in(blocks) ||
                   in(prelude_blocks) || in(bank);
        }
        [[nodiscard]] int acquire() const {
            for (int i = 0; i < kern::kMaxStates; ++i) { if (!used(i)) { return i; } }
            throw std::logic_error("inkling2: residual state pool exhausted");
        }
    } st;

    Impl(const Config& cfg, std::span<const BoundWeight> weights, DeviceContext& device,
         cudaStream_t s, std::uint32_t max_context)
        : config(cfg), bound(weights), stream(s), dev(device.device()),
          view(device.execution_view()),
          hidden(int(cfg.hidden_size)), q_heads(int(cfg.num_attention_heads)),
          head_dim(int(cfg.head_dim)), kv_heads(int(cfg.num_key_value_heads)),
          rel(int(cfg.attention.relative_taps)), window(int(cfg.sliding_window)),
          vocab(int(cfg.vocab_size)), experts(int(cfg.moe.num_experts)),
          top_k(int(cfg.moe.num_experts_per_tok)), width(int(cfg.moe.intermediate_size)),
          eps(cfg.rms_norm_eps), softcap(cfg.final_logit_softcapping),
          scale(1.0f / std::sqrt(float(cfg.head_dim))), max_ctx(max_context),
          arena_backing(dev, 32u << 20),
          arena(MemorySpan{arena_backing.data(), arena_backing.size(), dev}) {
        kvd = kv_heads * head_dim;

        const auto vlayers = visit_layers(cfg);
        const std::size_t n0 = cfg.num_hidden_layers, nc = cfg.loop.end - cfg.loop.start;
        visits.resize(vlayers.size());
        for (std::size_t v = 0; v < vlayers.size(); ++v) {
            VPlan& vp = visits[v];
            vp.layer  = vlayers[v];
            vp.pass   = v < n0 ? 0 : (v - n0 < nc ? 1 : 2);
            const bool looped = cfg.loop.loops(vp.layer);
            vp.save_prelude = looped && vp.layer == cfg.loop.start && vp.pass == 0;
            vp.restart      = looped && vp.layer == cfg.loop.start && vp.pass > 0;
            vp.control      = looped && vp.pass > 0;
            vp.start        = vp.layer % cfg.attn_res_block_size == 0;
        }

        // Persistent scratch.
        states      = DeviceBuffer(dev, std::size_t(kern::kMaxStates) * hidden * 4);
        messages    = DeviceBuffer(dev, std::size_t(4) * hidden * 4);
        raw_attn    = DeviceBuffer(dev, std::size_t(3) * hidden * 4);
        raw_ffn     = DeviceBuffer(dev, std::size_t(hidden) * 4);
        in_f32      = DeviceBuffer(dev, std::size_t(3) * hidden * 4);
        ffn_f32     = DeviceBuffer(dev, std::size_t(hidden) * 4);
        in_bf16     = DeviceBuffer(dev, std::size_t(3) * hidden * 2);
        ffn_bf16    = DeviceBuffer(dev, std::size_t(hidden) * 2);
        stage       = DeviceBuffer(dev, 64);
        logits_f32  = DeviceBuffer(dev, std::size_t(vocab) * 4);
        logits_bf16 = DeviceBuffer(dev, std::size_t(vocab) * 2);

        const int qrows = 2 * q_heads * head_dim + q_heads * rel;
        qgr        = DeviceBuffer(dev, std::size_t(qrows) * 4);
        kv_rows    = DeviceBuffer(dev, std::size_t(2) * kvd * 4);
        taps_rows  = DeviceBuffer(dev, std::size_t(2) * kv_heads * kTaps * 4);
        q_flat     = DeviceBuffer(dev, std::size_t(q_heads) * head_dim * 4);
        r_flat     = DeviceBuffer(dev, std::size_t(q_heads) * rel * 4);
        sinks      = DeviceBuffer(dev, std::size_t(q_heads) * 4 + 256);
        attn_bf16  = DeviceBuffer(dev, std::size_t(q_heads) * head_dim * 2);
        ctrl_state = DeviceBuffer(dev, std::size_t(3) * cfg.loop.controller_rank * 4 + 256);
        ctrl_fg    = DeviceBuffer(dev, std::size_t(6) * hidden * 4);
        ctrl_concat= DeviceBuffer(dev, std::size_t(3) * 2 * hidden * 4);
        gu_bf16    = DeviceBuffer(dev, std::size_t(top_k) * 2 * width * 2 + 256);
        gu_f32     = DeviceBuffer(dev, std::size_t(top_k) * 2 * width * 4);
        mid_bf16   = DeviceBuffer(dev, std::size_t(top_k) * width * 2 + 256);
        down_bf16  = DeviceBuffer(dev, std::size_t(top_k) * hidden * 2 + 256);
        down_f32   = DeviceBuffer(dev, std::size_t(top_k) * hidden * 4);
        route_scratch = DeviceBuffer(dev, (std::size_t(top_k) * 3 + experts) * 4 + 256);
        route_ptr_slot = DeviceBuffer(dev, std::size_t(top_k) * 9 * 8);
        // Garbage in any buffer a kernel may read (partial fills, slot rings)
        // leaks nondeterministically into logits; zero the persistent scratch.
        for (DeviceBuffer* b : {&states, &messages, &raw_attn, &raw_ffn, &in_f32,
                                &ffn_f32, &in_bf16, &ffn_bf16, &stage, &logits_f32,
                                &logits_bf16, &qgr, &kv_rows, &taps_rows, &q_flat,
                                &r_flat, &sinks, &attn_bf16, &ctrl_state, &ctrl_fg,
                                &ctrl_concat, &gu_bf16, &gu_f32, &mid_bf16,
                                &down_bf16, &down_f32, &route_scratch, &route_ptr_slot})
            cudaMemset(b->data(), 0, b->size());

        std::vector<std::int32_t> id(8);
        for (int i = 0; i < 8; ++i) { id[i] = i; }
        identity = DeviceBuffer(dev, id.size() * 4);
        cudaMemcpy(identity.data(), id.data(), id.size() * 4, cudaMemcpyHostToDevice);

        graphs_enabled = std::getenv("INK2_NO_GRAPHS") == nullptr &&
                         std::getenv("INK2_EXEC_DEBUG") == nullptr;
        if (s != view.stream()) { graphs_enabled = false; }  // capture requires one stream
        expert_ptr_tables.reserve(config.num_hidden_layers);

        build_plans();
        embed_table  = native_weight(find("text/token_embedding").view);
        head_mat     = Mat{winput(find("text/output_head"))};
        f_embed_norm = f32("text/embedding_norm");
        f_final_query = f32("text/final_residual_query");
        f_final_norm  = f32("text/final_norm");
        st.cur = 0;  // the embedding lands in buffer 0 (reference embed)
    }

    [[nodiscard]] const BoundWeight& find(std::string_view name) const {
        for (const auto& bw : bound) { if (bw.name == name) { return bw; } }
        throw std::invalid_argument(std::string("inkling2: missing weight ") + std::string(name));
    }

    // The binder records one mathematical use per binding; adopt its policy verbatim.
    [[nodiscard]] static ops::WeightInput winput(const BoundWeight& bw) {
        const auto& use = bw.uses.at(0);
        return {bw.view, use.policy, use.activation_input_divisor};
    }

    // Optional direct FP32 tensor: per-stream QKV input norms exist only on full-attention
    // layers (measured: artifact ships them for layers 5/11/17/23 only); sliding layers
    // fall back to input_norm at use sites.
    [[nodiscard]] const float* f32_opt(std::string_view name) const {
        for (const auto& bw : bound) { if (bw.name == name) { return f32(name); } }
        return nullptr;
    }

    // Direct FP32 tensor payload (norms, biases, small f32 tensors).
    [[nodiscard]] const float* f32(std::string_view name) const {
        const Tensor t = parameter_tensor(find(name));
        if (t.dtype != DType::FP32) {
            throw std::invalid_argument(std::string("inkling2: not fp32: ") + std::string(name));
        }
        return static_cast<const float*>(t.data);
    }

    Mat mat(std::string_view name) const { return Mat{winput(find(name))}; }

    void build_plans() {
        layers.resize(config.num_hidden_layers);
        for (std::uint32_t l = 0; l < config.num_hidden_layers; ++l) {
            LPlan& p = layers[l];
            p.full   = config.is_full_attention(l);
            p.loops  = config.loop.loops(l);
            const std::string lp = "text/layers/" + std::to_string(l);
            const std::string ap = lp + "/attention/";
            const std::string cp = lp + "/controller/";

            p.query    = mat(ap + "query");
            p.gate     = mat(ap + "gate");
            p.relative = mat(ap + "relative_projection");
            p.key      = mat(ap + "key");
            p.value    = mat(ap + "value");
            p.output   = mat(ap + "output");
            p.query_rows  = 2 * q_heads * head_dim + q_heads * rel;
            p.tap_columns = kv_heads * kTaps;

            p.query_norm = f32_opt(ap + "query_input_norm");
            p.key_norm   = f32_opt(ap + "key_input_norm");
            p.value_norm = f32_opt(ap + "value_input_norm");
            // input_norm is absent on the 4 full layers (artifact: 20 layers); full
            // layers norm via the q/k/v *_input_norm rows, so null is never read there.
            p.input_norm = f32_opt(lp + "/input_norm");
            p.query_clip    = f32(ap + "query_clip");
            p.relative_clip = f32(ap + "relative_clip");
            p.distances     = f32(ap + "relative_bias");
            p.key_conv    = f32(ap + "key_convolution");
            p.value_conv  = f32(ap + "value_convolution");
            p.output_conv = f32(ap + "post_convolution");
            p.ffn_conv    = f32(lp + "/ffn_post_convolution");
            p.ffn_norm    = f32(lp + "/post_attention_norm");
            p.key_taps    = f32_opt(ap + "key_convolution_projection");
            p.value_taps  = f32_opt(ap + "value_convolution_projection");
            // Layer 0 has no residual_query (artifact: layers 1..23 only); its
            // attention input is the normed embedding, so the pointer stays null.
            p.residual_query     = f32_opt(lp + "/residual_query");
            p.ffn_residual_query = f32(lp + "/ffn_residual_query");
            p.router      = f32(lp + "/moe/router");
            p.router_bias = f32(lp + "/moe/router_bias");
            p.sink        = f32(ap + "sink_projection");

            if (p.loops) {
                p.loop_queries = f32(cp + "message_query");
                static constexpr const char* kQKV[3] = {"q", "k", "v"};
                for (int i = 0; i < 3; ++i) {
                    const std::string s = kQKV[i];
                    p.ctrl_down[i]   = f32(cp + s + "_down");
                    p.ctrl_gain[i]   = f32(cp + s + "_gain");
                    p.ctrl_forget[i] = f32(cp + s + "_forget");
                }
                p.ffn_down   = f32(cp + "ffn_down");
                p.ffn_gain   = f32(cp + "ffn_gain");
                p.ffn_forget = f32(cp + "ffn_forget");
            }
            p.expert_gate.resize(config.moe.num_experts);
            p.expert_up.resize(config.moe.num_experts);
            p.expert_down.resize(config.moe.num_experts);
            const std::string ep = lp + "/moe/experts/";
            for (std::uint32_t e = 0; e < config.moe.num_experts; ++e) {
                const std::string base = ep + std::to_string(e);
                p.expert_gate[e] = mat(base + "/gate");
                p.expert_up[e]   = mat(base + "/up");
                p.expert_down[e] = mat(base + "/down");
            }
            // Device pointer table [expert][kind][plane] for graph-mode routing (route_ptrs).
            const std::size_t ptrs = std::size_t(config.moe.num_experts) * 9;
            std::vector<const void*> tab(ptrs);
            for (std::uint32_t e = 0; e < config.moe.num_experts; ++e) {
                const Mat* m[3] = { &p.expert_gate[e], &p.expert_up[e], &p.expert_down[e] };
                for (int kind = 0; kind < 3; ++kind) {
                    const Weight& w = m[kind]->w.weight;
                    tab[std::size_t(e) * 9 + std::size_t(kind) * 3 + 0] = w.qdata;
                    tab[std::size_t(e) * 9 + std::size_t(kind) * 3 + 1] = w.qhigh;
                    tab[std::size_t(e) * 9 + std::size_t(kind) * 3 + 2] = w.scales;
                }
            }
            expert_ptr_tables.push_back(DeviceBuffer(dev, ptrs * sizeof(void*)));
            cudaMemcpy(expert_ptr_tables.back().data(), tab.data(), ptrs * sizeof(void*),
                       cudaMemcpyHostToDevice);
            p.expert_ptrs = static_cast<const void**>(expert_ptr_tables.back().data());

            // The artifact ships residual_query pre-tiled to [3][hidden] for full
            // layers with three DISTINCT rows (measured 2026-11-25, weight/000170);
            // no engine-side replication.
            p.query_mix = p.residual_query;
        }
    }

    [[nodiscard]] Tensor vbf16(void* p, std::initializer_list<std::int32_t> s) const {
        return Tensor(p, dev, DType::BF16, s);
    }

    void linear(const Mat& m, const Tensor& x, Tensor& y, MemoryArena& a) {
        ops::linear(x, m.w.weight, y, m.w.policy, a, view);
    }

    // Convert one bf16 row (k wide) into f32 at out (row distance out_stride).
    void row_to_f32(const void* bf, int k, float* out, std::int64_t out_stride) {
        kern::bf16_rows_f32(static_cast<const __nv_bfloat16*>(bf), k,
                            static_cast<const std::int32_t*>(identity.data()), 1,
                            out, out_stride, stream);
    }

    // History site for visit v: o = 0 K, 1 V, 2 attention output, 3 MoE output.
    [[nodiscard]] kern::History site(Session& s, std::size_t v, int o) {
        kern::History h;
        float* base = s.store->site[v];
        const std::size_t off =
            o < 2 ? std::size_t(o) * kHistory * kvd
                  : std::size_t(2) * kHistory * kvd + std::size_t(o - 2) * kHistory * hidden;
        h.ring        = base + off;
        h.seq_stride  = 0;
        h.slot_stride = o < 2 ? kvd : hidden;
        h.slots       = kHistory;
        return h;
    }

    // MixArgs over the live residual states (cur last), as the reference mix_args.
    kern::MixArgs mix_args(const float* queries, int query_count) {
        kern::MixArgs m;
        float* S = static_cast<float*>(states.data());
        int n = 0;
        for (int b : st.blocks) { m.states[n++] = S + std::size_t(b) * hidden; }
        m.states[n++] = S + std::size_t(st.cur) * hidden;
        m.count        = n;
        m.queries      = queries;
        m.query_count  = query_count;
        m.tokens       = 1;
        m.hidden       = hidden;
        m.eps          = eps;
        m.query_stride = hidden;
        return m;
    }

    // The next attention input mix (raw into the controller plane for controlled visits;
    // normed otherwise). Outputs are left unset for the fused (residual) form; callers set them.
    kern::MixArgs attention_mix(const LPlan& np, bool ctl) {
        kern::MixArgs m = mix_args(np.query_mix, np.full ? 3 : 1);
        if (ctl) {
            m.raw = static_cast<float*>(raw_attn.data());
        } else if (np.full) {
            m.norm[0] = np.query_norm; m.norm[1] = np.key_norm; m.norm[2] = np.value_norm;
        } else {
            m.norm[0] = m.norm[1] = m.norm[2] = np.input_norm;
        }
        return m;
    }

    kern::MixArgs ffn_mix(const LPlan& np, bool ctl) {
        kern::MixArgs m = mix_args(np.ffn_residual_query, 1);
        if (ctl) { m.raw = static_cast<float*>(raw_ffn.data()); return m; }
        m.norm[0] = np.ffn_norm;
        return m;
    }

    kern::MixArgs output_mix() {
        kern::MixArgs m = mix_args(f_final_query, 1);
        m.norm[0] = f_final_norm;
        return m;
    }

    // The controllers of `count` inputs, as the reference control(): concat the rms-normed
    // x/message, run the low-rank state, apply forget/gain, norm into the outputs.
    void control(std::int32_t count, const float* const x[3], const float* const message[3],
                 const float* const norm[3], const float* const down[3], const float* const gain[3],
                 const float* const forget[3], __nv_bfloat16* out_bf16, float* out_f32) {
        const int rank = int(config.loop.controller_rank);
        kern::ControlInputs ci;
        for (int i = 0; i < count; ++i) { ci.x[i] = x[i]; ci.message[i] = message[i]; }
        ci.count = count; ci.tokens = 1; ci.hidden = hidden;
        ci.concat = static_cast<float*>(ctrl_concat.data());
        kern::control_inputs(ci, stream);

        float* state = static_cast<float*>(ctrl_state.data());
        kern::GemvBatch b1;
        for (int i = 0; i < count; ++i) {
            b1.w[i]   = down[i];
            b1.x[i]   = ci.concat + std::size_t(i) * 2 * hidden;
            b1.out[i] = state + std::size_t(i) * rank;
        }
        b1.count = count;
        kern::gemm_f32_batch(rank, 1, 2 * hidden, b1, stream);

        float* fg = static_cast<float*>(ctrl_fg.data());
        kern::GemvBatch b2;
        int j = 0;
        for (int i = 0; i < count; ++i) {
            b2.w[j] = forget[i]; b2.x[j] = state + std::size_t(i) * rank;
            b2.out[j] = fg + std::size_t(j) * hidden; ++j;
            b2.w[j] = gain[i];   b2.x[j] = state + std::size_t(i) * rank;
            b2.out[j] = fg + std::size_t(j) * hidden; ++j;
        }
        b2.count = j;
        kern::gemm_f32_batch(hidden, 1, rank, b2, stream);

        kern::ControlApply ca;
        for (int i = 0; i < count; ++i) {
            ca.x[i] = x[i]; ca.message[i] = message[i];
            ca.forget[i] = fg + std::size_t(2 * i) * hidden;
            ca.gain[i]   = fg + std::size_t(2 * i + 1) * hidden;
            ca.norm[i]   = norm[i];
        }
        ca.count = count; ca.tokens = 1; ca.hidden = hidden; ca.eps = eps;
        ca.query_stride = hidden;
        ca.out = out_bf16; ca.f32 = out_f32;
        kern::control_apply(ca, stream);
    }

    void visit(Session& session, std::size_t v, const kern::Rows& rows);
};

// ---- visit: the 18 steps, in the reference order ----------------------------------------------

void Executor::Impl::visit(Session& session, std::size_t v, const kern::Rows& rows) {
    const VPlan& vp = visits[v];
    const LPlan& p  = layers[vp.layer];
    const bool ctl  = vp.control;
    float* S   = static_cast<float*>(states.data());
    float* MSG = static_cast<float*>(messages.data());

    // 1. State machine around the loop entry.
    if (vp.save_prelude) {
        st.prelude_blocks = st.blocks;
        st.prelude_cur    = st.cur;
    }
    if (vp.restart) {
        st.bank.assign(st.blocks.begin() + std::ptrdiff_t(st.prelude_blocks.size() + 1),
                       st.blocks.end());
        st.bank_last = st.cur;
        st.blocks    = st.prelude_blocks;
        st.cur       = st.prelude_cur;
        inputs_ready = false;
    }

    // 2. The controller mixes stored states into the visit's messages (loop layers only).
    if (ctl) {
        kern::MixArgs m;
        int n = 0;
        for (int b : st.bank) { m.states[n++] = S + std::size_t(b) * hidden; }
        m.states[n++] = S + std::size_t(st.bank_last) * hidden;
        m.count        = n;
        m.queries      = p.loop_queries;      // [4][hidden]
        m.query_count  = 4;
        m.tokens       = 1;
        m.hidden       = hidden;
        m.eps          = eps;
        m.query_stride = hidden;
        m.raw          = MSG;                 // messages in, refreshed messages out
        kern::mix(m, stream);
    }

    // 3. Attention input: layer 0 uses the (already normed) embedding; else the depth mix.
    const bool layer0 = vp.layer == 0;
    if (layer0) {
        kern::rms_norm(S + std::size_t(st.cur) * hidden, p.input_norm, hidden, 1, eps,
                       static_cast<__nv_bfloat16*>(in_bf16.data()), stream, {},
                       static_cast<float*>(in_f32.data()));
    } else if (!inputs_ready) {
        kern::MixArgs m = attention_mix(p, ctl);
        if (!ctl) {
            m.out = static_cast<__nv_bfloat16*>(in_bf16.data());
            m.f32 = static_cast<float*>(in_f32.data());
        }
        kern::mix(m, stream);
    }
    inputs_ready = false;

    // 4. The controller on the attention input (3 inputs), overwriting the input planes.
    int distinct = p.full ? 3 : 1;
    if (ctl) {
        const float* xf[3]; const float* msg[3]; const float* nm[3];
        const float* qkv_norm[3] = { p.query_norm, p.key_norm, p.value_norm };
        const float* rawp = static_cast<float*>(raw_attn.data());
        for (int i = 0; i < 3; ++i) {
            xf[i]  = p.full ? rawp + std::size_t(i) * hidden : rawp;
            msg[i] = MSG + std::size_t(i) * hidden;
            nm[i]  = p.full ? qkv_norm[i] : p.input_norm;
        }
        control(3, xf, msg, nm, p.ctrl_down.data(), p.ctrl_gain.data(), p.ctrl_forget.data(),
                static_cast<__nv_bfloat16*>(in_bf16.data()), static_cast<float*>(in_f32.data()));
        distinct = 3;
    }
    const int ki = distinct == 3 ? 1 : 0;
    const int vi = distinct == 3 ? 2 : 0;

    auto scope = arena.scope();
    float* qgr  = static_cast<float*>(this->qgr.data());
    float* kvr  = static_cast<float*>(kv_rows.data());
    float* taps = static_cast<float*>(taps_rows.data());

    // 5. Projections into the packed row buffers (engine linear replaces the GGUF product()).
    {
        const Tensor xq = vbf16(in_bf16.data(), {hidden, 1});
        const Tensor xk = vbf16(static_cast<__nv_bfloat16*>(in_bf16.data()) +
                                std::size_t(ki) * hidden, {hidden, 1});
        const Tensor xv = vbf16(static_cast<__nv_bfloat16*>(in_bf16.data()) +
                                std::size_t(vi) * hidden, {hidden, 1});
        Tensor y = arena.alloc(DType::BF16, {q_heads * head_dim, 1});
        linear(p.query, xq, y, arena);
        row_to_f32(y.data, q_heads * head_dim, qgr, p.query_rows);
        y = arena.alloc(DType::BF16, {q_heads * head_dim, 1});
        linear(p.gate, xq, y, arena);
        row_to_f32(y.data, q_heads * head_dim, qgr + q_heads * head_dim, p.query_rows);
        y = arena.alloc(DType::BF16, {q_heads * rel, 1});
        linear(p.relative, xq, y, arena);
        row_to_f32(y.data, q_heads * rel, qgr + 2 * q_heads * head_dim, p.query_rows);
        y = arena.alloc(DType::BF16, {kvd, 1});
        linear(p.key, xk, y, arena);
        row_to_f32(y.data, kvd, kvr, 2 * kvd);
        y = arena.alloc(DType::BF16, {kvd, 1});
        linear(p.value, xv, y, arena);
        row_to_f32(y.data, kvd, kvr + kvd, 2 * kvd);
        // Sinks are a direct FP32 tensor: keep the donor's FP32 path (gemm_f32), not bf16 linear.
        kern::gemm_f32(q_heads, 1, hidden, p.sink,
                       static_cast<float*>(in_f32.data()), static_cast<float*>(sinks.data()),
                       stream, 0, 0);
    }
    if (p.key_taps) {
        kern::gemm_f32(kv_heads * kTaps, 1, hidden, p.key_taps,
                       static_cast<float*>(in_f32.data()) + std::size_t(ki) * hidden, taps,
                       stream, 0, 0);
        kern::gemm_f32(kv_heads * kTaps, 1, hidden, p.value_taps,
                       static_cast<float*>(in_f32.data()) + std::size_t(vi) * hidden,
                       taps + kv_heads * kTaps, stream, 0, 0);
    }

    // 6. Q and the relative feature.
    kern::QueryArgs qa;
    qa.qgr = qgr; qa.stride = p.query_rows; qa.relative_column = 2 * q_heads * head_dim;
    qa.tokens = 1; qa.heads = q_heads;
    qa.query_clip = p.query_clip; qa.relative_clip = p.relative_clip;
    qa.qk_clip = config.attention.qk_clip; qa.rel_clip = config.attention.relative_clip;
    qa.log_scale = p.full;
    qa.alpha = config.attention.log_scale_alpha; qa.floor = config.attention.log_scale_floor;
    qa.positions = rows.positions;
    qa.q_out = static_cast<float*>(q_flat.data());
    qa.r_out = static_cast<float*>(r_flat.data());
    kern::query_post(qa, stream);

    // 7. K/V: convolutions (+ dynamic taps), clip/norm, into the cache slot and history.
    const std::size_t pi = v;
    kern::KvArgs ka;
    ka.rows  = rows;
    ka.kv    = kvr; ka.kv_stride = 2 * kvd;
    ka.k_taps = p.key_taps; ka.v_taps = p.value_taps;
    ka.taps_stride = p.tap_columns;
    ka.k_conv = p.key_conv; ka.v_conv = p.value_conv;
    ka.k_history = site(session, pi, 0);
    ka.v_history = site(session, pi, 1);
    ka.keep = true;
    ka.kv_heads = kv_heads;
    ka.qk_clip = config.attention.qk_clip; ka.eps = eps;
    ka.k_cache = session.store->kp[pi];
    ka.v_cache = session.store->vp[pi];
    ka.head_stride = session.store->head_stride[pi];
    ka.ring = session.store->ring[pi];
    kern::kv_post(ka, stream);

    // 8. Attention.
    kern::AttentionArgs aa;
    aa.rows = rows;
    aa.q = static_cast<float*>(q_flat.data());
    aa.r = static_cast<float*>(r_flat.data());
    aa.sinks = static_cast<float*>(sinks.data());
    aa.table = p.distances;
    aa.extent = p.full ? int(config.attention.relative_extent) : window;
    aa.k_cache = session.store->kp[pi];
    aa.v_cache = session.store->vp[pi];
    aa.head_stride = session.store->head_stride[pi];
    aa.k_plane = f16_plane(aa.k_cache, aa.head_stride);
    aa.v_plane = f16_plane(aa.v_cache, aa.head_stride);
    aa.ring   = session.store->ring[pi];
    aa.window = p.full ? 0 : window;
    aa.query_heads = q_heads; aa.kv_heads = kv_heads;
    aa.scale  = scale;
    aa.gate   = qgr + q_heads * head_dim; aa.gate_stride = p.query_rows;
    aa.out    = static_cast<__nv_bfloat16*>(attn_bf16.data());
    kern::attention(aa, stream);

    // 9. Output projection; the residual update with the fused FFN mix.
    {
        const Tensor xa = vbf16(attn_bf16.data(), {q_heads * head_dim, 1});
        Tensor y = arena.alloc(DType::BF16, {hidden, 1});
        linear(p.output, xa, y, arena);
        // (the product y lands in ffn-site order after conv; residual reads f32 rows)
        float* yf = static_cast<float*>(down_f32.data());  // scratch: [top_k][hidden], row 0
        row_to_f32(y.data, hidden, yf, hidden);
        if (vp.start) {
            st.blocks.push_back(st.cur);
            st.cur = st.acquire();
        }
        kern::ResidualArgs ra;
        ra.rows    = rows;
        ra.y       = yf;
        ra.conv    = p.output_conv;
        ra.history = site(session, pi, 2);
        ra.keep    = true;
        ra.start   = vp.start;  // a block start: cur takes y as a fresh row, not an addend
        ra.cur     = S + std::size_t(st.cur) * hidden;
        ra.hidden  = hidden;
        ra.mix     = ffn_mix(p, ctl);
        if (!ctl) {
            ra.mix.out = static_cast<__nv_bfloat16*>(ffn_bf16.data());
            ra.mix.f32 = static_cast<float*>(ffn_f32.data());
        }
        kern::residual(ra, stream);
    }

    // 10. FFN-side controller (loop layers), overwriting the FFN planes.
    if (ctl) {
        const float* xf[1]  = { static_cast<float*>(raw_ffn.data()) };
        const float* msg[1] = { MSG + std::size_t(3) * hidden };
        const float* nm[1]  = { p.ffn_norm };
        const float* dn[1]  = { p.ffn_down };
        const float* gn[1]  = { p.ffn_gain };
        const float* fk[1]  = { p.ffn_forget };
        control(1, xf, msg, nm, dn, gn, fk, static_cast<__nv_bfloat16*>(ffn_bf16.data()),
                static_cast<float*>(ffn_f32.data()));
    }

    // 11. Router, route, experts.
    float* ff  = static_cast<float*>(ffn_f32.data());
    auto* rid  = static_cast<std::int32_t*>(route_scratch.data());
    float* rw  = reinterpret_cast<float*>(rid + top_k);
    float* rl  = rw + top_k;   // [experts] logits
    float* ps  = rl + experts; // [top_k] gate_up pair scale
    {  // router gemm feeds route(), which only reads the logits
        kern::gemm_f32(experts, 1, hidden, p.router, ff, rl, stream);
        kern::RouteArgs rt;
        rt.x = ff; rt.hidden = hidden; rt.tokens = 1;
        rt.router = p.router; rt.bias = p.router_bias;
        rt.experts = experts; rt.top_k = top_k; rt.scale = config.moe.router_scale;
        rt.logits = rl; rt.ids = rid; rt.weights = rw; rt.pair_scale = ps;
        kern::route(rt, stream);
    }
    // Routing stays device-side (graph mode): expert weight planes are fetched through the
    // per-visit pointer slots. No DtoH, no stream sync inside the step.
    void** pslot = static_cast<void**>(route_ptr_slot.data());
    kern::route_ptrs(rid, static_cast<const void* const*>(p.expert_ptrs), pslot, top_k, stream);
    {
        const Tensor xin = vbf16(ffn_bf16.data(), {hidden, 1});
        float* gu = static_cast<float*>(gu_f32.data());
        for (int t = 0; t < top_k; ++t) {
            Tensor yg = arena.alloc(DType::BF16, {width, 1});
            linear(with_slot(p.expert_gate[0], pslot + std::size_t(t) * 9), xin, yg, arena);
            row_to_f32(yg.data, width, gu + std::size_t(t) * 2 * width, 2 * width);
            Tensor yu = arena.alloc(DType::BF16, {width, 1});
            linear(with_slot(p.expert_up[0], pslot + std::size_t(t) * 9 + 3), xin, yu, arena);
            row_to_f32(yu.data, width, gu + std::size_t(t) * 2 * width + width, 2 * width);
        }
        kern::situ_rows(gu, 2 * width, top_k, width, ps, config.moe.situ_beta,
                        config.moe.situ_linear_beta,
                        static_cast<__nv_bfloat16*>(mid_bf16.data()), stream);
        for (int t = 0; t < top_k; ++t) {
            const Tensor xm = vbf16(static_cast<__nv_bfloat16*>(mid_bf16.data()) +
                                    std::size_t(t) * width, {width, 1});
            Tensor yd = arena.alloc(DType::BF16, {hidden, 1});
            linear(with_slot(p.expert_down[0], pslot + std::size_t(t) * 9 + 6), xm, yd, arena);
            row_to_f32(yd.data, hidden,
                       static_cast<float*>(down_f32.data()) + std::size_t(t) * hidden, hidden);
        }
    }

    // 12. Residual with the MoE output and the next visit's mix (or the output mix).
    {
        kern::ResidualArgs ra;
        ra.rows        = rows;
        ra.moe_rows    = static_cast<float*>(down_f32.data());
        ra.moe_weights = rw;
        ra.top_k       = top_k;
        ra.conv        = p.ffn_conv;
        ra.history     = site(session, pi, 3);
        ra.keep        = true;
        ra.cur         = S + std::size_t(st.cur) * hidden;
        ra.hidden      = hidden;
        const bool last = v + 1 == visits.size();
        if (last) {
            ra.mix = output_mix();
            ra.mix.out = static_cast<__nv_bfloat16*>(ffn_bf16.data());
        } else if (v + 1 < visits.size() && !visits[v + 1].restart && visits[v + 1].layer > 0) {
            const bool next_ctl = visits[v + 1].control;
            ra.mix = attention_mix(layers[visits[v + 1].layer], next_ctl);
            if (!next_ctl) {
                ra.mix.out = static_cast<__nv_bfloat16*>(in_bf16.data());
                ra.mix.f32 = static_cast<float*>(in_f32.data());
            }
            inputs_ready = true;
        }
        kern::residual(ra, stream);
    }
}

// ---- public API ------------------------------------------------------------------------------

Executor::Executor(const Config& cfg, std::span<const BoundWeight> weights,
                   DeviceContext& device, cudaStream_t stream, std::uint32_t max_context)
    : impl(std::make_unique<Impl>(cfg, weights, device, stream, max_context)) {}

Executor::~Executor() = default;

std::unique_ptr<Session> Executor::make_session() const {
    return std::make_unique<Session>(impl->config, impl->dev, impl->visits.size(), impl->max_ctx);
}

void Executor::decode_step(Session& session, std::uint32_t token, std::int64_t position) {
    Impl& im = *impl;
    auto scope = im.arena.scope();

    // A fresh call: the reference builds new Call states per forward. The embedding lands in
    // buffer 0; residual bookkeeping and the fused-input flag are per-call.
    im.st = Impl::StateBook{};
    im.st.cur = 0;
    im.inputs_ready = false;

    // Embedding row into state 0, then the embedding norm (reference embed()).
    std::int32_t pos32 = std::int32_t(position);
    std::int32_t tok32 = std::int32_t(token);
    cudaMemcpyAsync(im.stage.data(), &tok32, 4, cudaMemcpyHostToDevice, im.stream);
    cudaMemcpyAsync(static_cast<char*>(im.stage.data()) + 4, &pos32, 4,
                    cudaMemcpyHostToDevice, im.stream);

    // Token and position landed in the fixed stage buffer; the step reads them device-side, so
    // one captured graph serves every step (values are never baked into kernel arguments).
    ++im.step_calls;
    if (im.graph_ready) {
        im.graph_exec.launch(im.stream);
        session.advance(1);
        return;
    }

    auto body = [&] {
    const Tensor ids = Tensor(im.stage.data(), im.dev, DType::I32, {1});
    Tensor emb = im.arena.alloc(DType::BF16, {im.hidden, 1});  // ops::embedding wants [d,T]
    ops::embedding(ids, im.embed_table, emb, im.view);
    im.row_to_f32(emb.data, im.hidden, static_cast<float*>(im.states.data()), im.hidden);
    kern::rms_norm(static_cast<float*>(im.states.data()), im.f_embed_norm,
                   im.hidden, 1, im.eps, nullptr, im.stream, {},
                   static_cast<float*>(im.states.data()));

    kern::Rows rows;
    rows.tokens   = 1;
    rows.chunk    = false;
    rows.sequence = 0;
    rows.positions = reinterpret_cast<const std::int32_t*>(static_cast<char*>(im.stage.data()) + 4);

    static const bool exec_debug = std::getenv("INK2_EXEC_DEBUG") != nullptr;
    for (std::size_t v = 0; v < im.visits.size(); ++v) {
        if (exec_debug) {
            // Mirror of reference INK2_DEBUG: rms of current state row at visit start.
            std::vector<float> h(im.hidden);
            cudaMemcpy(h.data(), static_cast<float*>(im.states.data()) +
                                    std::size_t(im.st.cur) * im.hidden,
                       im.hidden * sizeof(float), cudaMemcpyDeviceToHost);
            double s = 0.0;
            for (float f : h) s += double(f) * double(f);
            printf("v%03zu L%02up%u rms=%.6f blocks=%zu\n", v,
                   im.visits[v].layer, im.visits[v].pass,
                   std::sqrt(s / double(im.hidden)), im.st.blocks.size());
            fflush(stdout);
        }
        im.visit(session, v, rows);
    }

    // Head: the final mix landed in ffn_bf16 (step 12 of the last visit).
    {
        const Tensor x = im.vbf16(im.ffn_bf16.data(), {im.hidden, 1});
        Tensor y = im.arena.alloc(DType::BF16, {im.vocab, 1});
        im.linear(im.head_mat, x, y, im.arena);
        im.row_to_f32(y.data, im.vocab, static_cast<float*>(im.logits_f32.data()), im.vocab);
        kern::softcap_bf16(static_cast<const float*>(im.logits_f32.data()),
                           static_cast<__nv_bfloat16*>(im.logits_bf16.data()), im.vocab,
                           im.softcap, im.stream);
    }
    };

    // Capture on the second call (the first warms allocator, routes, and scratch); capture does
    // not execute, so this step still runs through body() directly.
    if (im.graphs_enabled && im.step_calls == 2) {
        {
            DecodeGraphCapture capture(im.stream);
            body();
            im.graph_def = capture.finish();
        }
        im.graph_exec.instantiate(im.graph_def);
        im.graph_ready = true;
        // Capture only records; the replay executes this step, so body() must not run again
        // (a second pass would re-consume the per-step residual state pool).
        im.graph_exec.launch(im.stream);
        session.advance(1);
        return;
    }
    body();
    session.advance(1);
}

const void* Executor::logits() const noexcept { return impl->logits_bf16.data(); }

} // namespace ninfer::models::inkling2::execution
