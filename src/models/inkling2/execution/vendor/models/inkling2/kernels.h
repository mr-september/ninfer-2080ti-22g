#pragma once

// Inkling 2's own kernels: the depth mixes and their norms, the loop controllers, the attention's
// clips, short convolutions and relative-bias attention with per-query sinks, routing, SiTU and
// the residual updates. Activations are token-major FP32 rows h[t][i]; projection inputs are BF16
// rows x[t][k] (a [K, T] Tensor) or, for at most eight rows, the GGUF vector kernels' q8_1 form.
//
// A call's rows (struct Rows) belong to one sequence or, with `sequences`, each to its own, at
// logical positions `positions`. A prompt chunk's rows lie at the physical cache positions
// first + t; with `starts`, its rows from `shared` on are branches, each continuing the sequence
// at first + shared (logical positions restart there), row t's branch beginning at row starts[t].
// A decode or verification call's rows (chunk false) are physical and logical alike, a
// sequence's rows consecutive and in position order.
//
// The short convolutions keep, per visit and sequence, a ring of the last `slots` raw inputs of
// each of their four sites (K, V, the attention's output and the MoE's), position p in slot
// p % slots, FP32. A convolution reads the inputs of the three positions before a row from the
// call's own rows where they lie in the call, else from the ring (zeros before position 0).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "models/inkling2/kv_format.h"

namespace ninfer::models::inkling2::kernels {

// Rows in the GGUF vector kernels' q8_1 activation form (ggml_bridge.h's vector activation):
// int8 values [T][k] and (d, sum) half2 pairs [T][k / 32].
struct Q8Rows {
    std::int8_t* qs = nullptr;
    half2* ds       = nullptr;
};

// Where a call's rows sit (see the file comment).
struct Rows {
    const std::int32_t* positions = nullptr; // [T], logical
    const std::int32_t* sequences = nullptr; // [T]; null: every row is `sequence`'s
    int sequence = 0;
    int tokens   = 0;
    bool chunk   = false; // a prompt chunk (physical positions first + t)
    int first    = 0;
    int shared   = 0;
    const std::int32_t* starts = nullptr; // [T], branches; null: none
};

// One convolution site's history ring: channel 0 of slot 0 of sequence 0, `seq_stride` floats
// between sequences and `slot_stride` between slots.
struct History {
    float* ring             = nullptr;
    std::int64_t seq_stride = 0;
    std::int64_t slot_stride = 0;
    int slots               = 0;
};

// y = x * rsqrt(mean x^2 + eps) * w (w null: no weight) for rows of `hidden` (x rows `hidden`
// apart), written as BF16 rows (out), q8_1 rows (q) and FP32 rows (f32, which may be x itself),
// each when given.
void rms_norm(const float* x, const float* w, int hidden, int tokens, float eps,
              __nv_bfloat16* out, cudaStream_t stream, Q8Rows q = {}, float* f32 = nullptr);

// The depth attention of each row over `count` states (the residual blocks so far, the current
// state last): s_j = rms(state_j[t]) without a weight (kMixEps), weights softmax_j(query . s_j /
// sqrt(hidden)) for each query, mixed[q][t] = sum_j weight_j s_j. Per query, when given: the
// mixed row in FP32 (raw), and the mixed row RMS-normed with norm[q] (eps) as BF16 rows, q8_1
// rows and FP32 rows. Outputs of query q lie q * query_stride elements past their base, its q8_1
// rows (a whole vector activation, values then scales) q * q8_stride bytes past qs and ds.
inline constexpr int kMaxStates  = 8;
inline constexpr int kMaxQueries = 4;
struct MixArgs {
    const float* states[kMaxStates] = {};
    int count = 0;
    const float* queries = nullptr; // [query_count][hidden]
    int query_count      = 0;
    int tokens = 0, hidden = 0;
    float eps  = 1e-6f;
    std::int64_t query_stride = 0; // elements between queries' outputs ([T][hidden] each)
    std::int64_t q8_stride    = 0; // bytes between queries' q8_1 activations
    float* raw = nullptr;
    const float* norm[kMaxQueries] = {};
    __nv_bfloat16* out = nullptr;
    Q8Rows out_q;
    float* f32 = nullptr;
    // Scratch of mix_split_floats(tokens, hidden) for a call of at most kSplitRows rows: each row
    // then spreads over one-warp blocks, which keep their warps' sums there and add them in warp
    // order, so the values equal the one-block row's. Null: one block per row.
    float* split = nullptr;
};
inline constexpr int kSplitRows = 16;
[[nodiscard]] std::size_t mix_split_floats(int tokens, int hidden);
void mix(const MixArgs& args, cudaStream_t stream);

// out[t][m] = sum_k w[m][k] x[t][k] in FP32 for w row-major [m][k] and x [n][k] (rows `x_stride`
// apart, k when 0), out rows `out_stride` apart (m when 0): up to eight rows by the GEMV kernel (a
// fixed grid, so CUDA graphs replay it), more by cuBLAS SGEMM.
void gemm_f32(int m, int n, int k, const float* w, const float* x, float* out, cudaStream_t stream,
              std::int64_t x_stride = 0, std::int64_t out_stride = 0);

// Up to kMaxGemvBatch products of one shape, out[b] = w[b] x[b] as gemm_f32 computes each of them
// (the same values), in one launch for up to eight rows: k <= 128 a warp per output row.
inline constexpr int kMaxGemvBatch = 8;
struct GemvBatch {
    const float* w[kMaxGemvBatch] = {};
    const float* x[kMaxGemvBatch] = {};
    float* out[kMaxGemvBatch]     = {};
    int count = 0;
};
void gemm_f32_batch(int m, int n, int k, const GemvBatch& batch, cudaStream_t stream,
                    std::int64_t x_stride = 0, std::int64_t out_stride = 0);

// The loop controllers of up to four inputs: concat[i][t] = [rms(x_i[t]); rms(message_i[t])]
// (kMixEps, no weight), [count][T][2 hidden] FP32.
struct ControlInputs {
    const float* x[kMaxQueries]       = {}; // [T][hidden] each
    const float* message[kMaxQueries] = {};
    int count = 0, tokens = 0, hidden = 0;
    float* concat = nullptr;
};
void control_inputs(const ControlInputs& args, cudaStream_t stream);

// y = x - tanh(forget) x + tanh(gain) message for each of `count` inputs (forget and gain FP32
// [T][hidden] products of the controller's hidden state), then RMS-normed with norm[i] (eps) into
// BF16 rows, q8_1 rows and FP32 rows as MixArgs writes them (query_stride and q8_stride apart).
struct ControlApply {
    const float* x[kMaxQueries]       = {};
    const float* message[kMaxQueries] = {};
    const float* forget[kMaxQueries]  = {};
    const float* gain[kMaxQueries]    = {};
    const float* norm[kMaxQueries]    = {};
    int count = 0, tokens = 0, hidden = 0;
    float eps                 = 1e-6f;
    std::int64_t query_stride = 0;
    std::int64_t q8_stride    = 0;
    __nv_bfloat16* out = nullptr;
    Q8Rows out_q;
    float* f32 = nullptr;
};
void control_apply(const ControlApply& args, cudaStream_t stream);

// The call's rows: which sequence a row belongs to, and its physical position (a prompt chunk's
// rows lie first + t, a decode or verification call's at their positions).
__device__ __forceinline__ int row_sequence(const Rows& r, int t) {
    return r.sequences != nullptr ? r.sequences[t] : r.sequence;
}

__device__ __forceinline__ int physical(const Rows& r, int t) {
    return r.chunk ? r.first + t : r.positions[t];
}

// Q and the relative feature of each row from the [Q; gate; relative] product (FP32 rows
// `stride` apart, Q at column 0, the relative feature at `relative_column`): per head
//   q = clip(q * query_clip[h], qk_clip) * tau,   r = clip(r * relative_clip[h], relative_clip) * tau
// where clip(x, c) = x * min(1, c / (|x| + 1e-6)) and tau = 1 + alpha log(max(1, (p + 1) / floor))
// at the row's logical position p (1 without `log_scale`). q_out FP32 [T][heads][128], r_out
// FP32 [T][heads][16].
struct QueryArgs {
    const float* qgr = nullptr;
    std::int64_t stride = 0, relative_column = 0;
    int tokens = 0, heads = 0;
    const float* query_clip    = nullptr; // [heads][128]
    const float* relative_clip = nullptr; // [heads][16]
    float qk_clip = 0, rel_clip = 0;
    bool log_scale = false;
    float alpha = 0, floor = 1;
    const std::int32_t* positions = nullptr;
    float* q_out = nullptr;
    float* r_out = nullptr;
};
void query_post(const QueryArgs& args, cudaStream_t stream);

// K and V of each row: their short convolutions over the raw projections (FP32 rows `kv_stride`
// apart, K at column 0, V at kv_heads * 128), with the dynamic per-head taps of a full-attention
// layer (FP32 [T][kv_heads * 4] each, `taps_stride` apart; null: none); then K clipped to an L2
// norm of qk_clip and V RMS-normed without a weight (eps), as F16 into cache slot `slot % ring`
// (ring 0: slot) of the row's sequence, slot its physical position. Cache element
// (d, slot, head, sequence) at base + sequence * seq_stride + head * head_stride + slot * 128 + d.
// With `keep` the rows' raw K and V also go into their history slots (a decode or verification
// call; a prompt chunk keeps them afterwards with keep_history).
struct KvArgs {
    Rows rows;
    const float* kv = nullptr;
    std::int64_t kv_stride = 0;
    const float* k_taps = nullptr;
    const float* v_taps = nullptr;
    std::int64_t taps_stride = 0;
    const float* k_conv = nullptr; // [kv_heads * 128][4]
    const float* v_conv = nullptr;
    History k_history, v_history;
    bool keep = false;
    int kv_heads = 0;
    float qk_clip = 0, eps = 1e-6f;
    __half* k_cache = nullptr;
    __half* v_cache = nullptr;
    std::int64_t seq_stride = 0, head_stride = 0;
    int ring = 0;
    // Row t's values land at slot t (head_stride t wide): a quantized visit's staging scratch,
    // which kv_store then reads; without it, slot `position % ring` (ring 0: position).
    bool row_slots = false;
    // MXFP8 planes (kv_codec.cuh; a full visit of an mxfp8 run): the rows are encoded into them
    // at slot `position`, and k_cache / v_cache are unused.
    KvPlaneView k_plane, v_plane;
};
void kv_post(const KvArgs& args, cudaStream_t stream);

// The attention of each row's heads (128 values, `query_heads / kv_heads` query heads per key and
// value head) over the keys it sees: its sequence's earlier positions (and, in a prompt chunk, the
// chunk's rows its own sees: earlier rows, of a branch row only the shared ones and its branch's),
// within `window` positions of it (0: every one). Key slot: position % ring (ring 0: position).
//   s = scale q . k + (d < extent ? r . table[d] : 0),  d = the row's logical position less the key's
//   out = sum e^(s - m) v / (sum e^(s - m) + e^(sink - m))
// with the row's sink logit of the head (null: none, e^(sink - m) = 0). The output, times sigmoid
// of the gate (FP32 rows `gate_stride` apart, [heads][128]; null: no gate), goes out as BF16 rows
// [T][heads * 128] or their q8_1 form (r may be null when extent is 0).
// A decode or verification call (rows.chunk false) splits each row's keys over `splits` ranges
// merged after; a prompt chunk runs its rows eight to a block, every key in one pass.
// `uniform` (a DSpark block, rows.chunk false, at most eight rows of one sequence whose keys are
// already in the cache): every row sees the keys the last row sees, its position and the `window`
// before it (0: every earlier one), the block's own later rows included; eight rows to a block,
// split over `splits` ranges.
inline constexpr int kMaxRowGroup = 4;
struct AttentionArgs {
    Rows rows;
    const float* q = nullptr; // [T][query_heads][128]
    const float* r = nullptr; // [T][query_heads][16]
    const float* sinks = nullptr;
    std::int64_t sink_stride = 0; // row stride of the sinks ([query_heads] per row)
    const float* table = nullptr; // [extent][16]
    int extent = 0;
    const __half* k_cache = nullptr;
    const __half* v_cache = nullptr;
    // Quantized planes take the caches' place (a full-attention visit of a --kv-dtype rk8v4 /
    // rk4v4 run, whose F16 caches the executor does not allocate): a plane whose coding is not
    // F16 has its codes and scales read instead of the halves.
    KvPlaneView k_plane, v_plane;
    std::int64_t seq_stride = 0, head_stride = 0;
    int ring = 0, window = 0;
    int query_heads = 0, kv_heads = 0;
    float scale = 1.0f;
    int splits  = 1;
    bool uniform = false;
    float* partial = nullptr; // decode: attention_workspace_floats
    // A full visit's decode or verification call (window 0, no ring): up to `row_group` (1 to
    // kMaxRowGroup) consecutive rows of one sequence share a block, which reads each key tile once
    // for them, in `row_groups` blocks (0: one a row; at least the call's groups). Each row splits
    // and reads its keys as a block of its own row does, so its output is the same.
    int row_group = 1, row_groups = 0;
    // A prompt chunk of a full visit (window 0, no ring): attention_workspace_floats(T,
    // query_heads, 1) for the far field's softmax states, computed on tensor cores first: the keys
    // before the chunk at least `extent` before every row of a group of 32 rows. Null: none.
    float* far = nullptr;
    const float* gate = nullptr;
    std::int64_t gate_stride = 0;
    __nv_bfloat16* out = nullptr;
    Q8Rows out_q;
};
[[nodiscard]] std::size_t attention_workspace_floats(int rows, int query_heads, int splits);
void attention(const AttentionArgs& args, cudaStream_t stream);

// The residual update of a product y (FP32 rows [T][hidden]; or a MoE's output, the 2^-32
// fixed-point sums `fixed` [T][hidden] or its pairs' FP32 rows [T * top_k][hidden] weighted by
// `weights`): c = (y + sum_i w[c][i] y(p - 3 + i)) / sqrt(taps + 1) through the site's history,
// then cur = c (`start`: a residual block begins, cur a fresh row) or cur += c. With `keep` the
// rows' y go into their history slots. Then, when mix.query_count > 0, the next depth mix of each
// row as mix() computes it, mix.states naming cur last (`cur` itself). With mix.split (a call of
// at most kSplitRows rows, the mix's scratch with or without a mix) the rows spread over one-warp
// blocks as the split mix's do.
struct ResidualArgs {
    Rows rows;
    const float* y = nullptr;
    const unsigned long long* fixed = nullptr;
    const float* moe_rows = nullptr;
    const float* moe_weights = nullptr;
    int top_k = 0;
    const float* conv = nullptr; // [hidden][4]
    History history;
    bool keep  = false;
    bool start = false;
    float* cur = nullptr;
    int hidden = 0;
    MixArgs mix;
};
void residual(const ResidualArgs& args, cudaStream_t stream);

// A prompt chunk's history of a product site: the products of rows [begin, end) as residual()
// reads them, into their slots (position: the row's logical one). Runs after the chunk's
// residual(), whose reads of the ring it would otherwise overwrite.
void keep_products(const ResidualArgs& args, int begin, int end, cudaStream_t stream);

// A prompt chunk's history: the raw inputs of its rows [begin, end) (FP32 rows `stride` apart,
// `channels` wide) into their slots of one sequence's ring (position: the row's logical one).
void keep_history(const float* x, std::int64_t stride, int channels, const std::int32_t* positions,
                  int begin, int end, const History& history, int sequence, cudaStream_t stream);

// A call's pairs grouped by expert (gguf::MoeRouting's fields, writable): `sorted` the pairs of
// expert e at [bounds[e], bounds[e + 1]), `active` the experts with any, ascending, and for each
// column tile width 16 << w the tiles holding a pair (expert | tile << 16) at tiles + w *
// tile_capacity.
struct RouteSort {
    std::int32_t* bounds       = nullptr; // [experts + 1]
    std::int32_t* sorted       = nullptr; // [pairs]
    std::int32_t* active       = nullptr; // [experts]
    std::int32_t* active_count = nullptr; // [1]
    std::int32_t* tile_count   = nullptr; // [4]
    std::int32_t* tiles        = nullptr; // [4][tile_capacity]
    int tile_capacity          = 0;
};

// The router of each row x (FP32 [T][hidden]): p = sigmoid(router . x), the top_k experts by
// p + bias (lower id first on equal values), weighted p / (sum of the chosen p + 1e-20) * scale *
// down_scale[e] (when given); pair_scale[t * top_k + k] = gate_up_scale[e] (1 without). ids and
// weights [T][top_k]; `logits` the router's FP32 products [T][experts] (gemm_f32).
struct RouteArgs {
    const float* x = nullptr;
    int hidden = 0, tokens = 0;
    const float* router = nullptr;
    const float* bias   = nullptr;
    int experts = 0, top_k = 0;
    float scale = 1.0f;
    const float* gate_up_scale = nullptr;
    const float* down_scale    = nullptr;
    float* logits = nullptr;
    std::int32_t* ids = nullptr;
    float* weights    = nullptr;
    float* pair_scale = nullptr;
    // At most kSortedRouteRows rows of at most kSortedRouteExperts experts, `sorted` set: the
    // call's pairs also grouped by expert there as gguf::moe_sort_routes writes them (its fields of
    // gguf::moe_sorted_routes's view of the same workspace), in the same launch.
    RouteSort sorted;
};
inline constexpr int kSortedRouteRows    = 8;
inline constexpr int kSortedRouteExperts = 128;
void route(const RouteArgs& args, cudaStream_t stream);

// The SiTU middles of `pairs` FP32 [gate; up] rows (2 width each, `stride` apart): out[p][r] =
// bf16(situ(s g, s u)) with s = pair_scale[p] (models/inkling2/situ.h), width % 32 == 0: the
// values the CPU's expert products compute (cpu_experts.h).
void situ_rows(const float* gu, std::int64_t stride, int pairs, int width, const float* pair_scale,
               float gate_cap, float up_cap, __nv_bfloat16* out, cudaStream_t stream);

// BF16 expert products: out[p][r] = sum_j W_e[r][j] x[p / x_div][j] in FP32 for every pair p of
// an active expert e (the routing's grouping: pairs sorted[bounds[e], bounds[e + 1]), experts
// active[0, *active_count), at most max_active of them), W_e the `rows` x `k` BF16 rows at
// table[e] (16-byte aligned), x BF16 rows of k values (a token's for x_div top_k, a pair's own
// for 1), k % 8 == 0. Other pairs' rows of out are left as they are. Each sum adds a lane's
// eight-column runs in column order, then the warp's lanes: the same bits on every call.
void bf16_expert_rows(const void* const* table, int rows, int k, const std::int32_t* bounds,
                      const std::int32_t* sorted, const std::int32_t* active,
                      const std::int32_t* active_count, int max_active, const __nv_bfloat16* x,
                      int x_div, float* out, cudaStream_t stream);

// Rows ids[0, rows) of a BF16 table [*][k] as FP32 rows `out_stride` apart, exactly.
void bf16_rows_f32(const __nv_bfloat16* table, int k, const std::int32_t* ids, int rows,
                   float* out, std::int64_t out_stride, cudaStream_t stream);

// out = bf16(cap tanh(in / cap)) for `n` logits.
void softcap_bf16(const float* in, __nv_bfloat16* out, std::int64_t n, float cap,
                  cudaStream_t stream);
// The same over `rows` rows of `width` logits, rows `in_stride` and `out_stride` values apart.
void softcap_bf16(const float* in, std::int64_t in_stride, __nv_bfloat16* out,
                  std::int64_t out_stride, int width, int rows, float cap, cudaStream_t stream);

void f32_to_bf16(const float* in, __nv_bfloat16* out, std::int64_t n, cudaStream_t stream);

// h[(row + i) * hidden + c] = rows[i][c] for `count` BF16 rows: an image's span rows in place of
// their tokens' embeddings.
void place_rows(const __nv_bfloat16* rows, int count, int hidden, float* h, cudaStream_t stream);

// Copies the K and V rows of positions [begin, end) of one sequence from one ring to another:
// position p lives in slot p % ring of each ([kv heads][slots][128] halves, heads head_stride
// elements apart).
struct RingCopyArgs {
    const __half* src_k = nullptr;
    const __half* src_v = nullptr;
    __half* dst_k       = nullptr;
    __half* dst_v       = nullptr;
    std::int64_t src_head_stride = 0, dst_head_stride = 0;
    int src_ring = 0, dst_ring = 0, kv_heads = 0, begin = 0, end = 0;
};
void ring_copy(const RingCopyArgs& args, cudaStream_t stream);

// ---- The DSpark drafter (executor.h) -------------------------------------------------------------

// The drafter's feature mixes of rows [begin, end) of a call: mix m of row t is
//   x = sum_j softmax_j(fuse[m]) taps[j][t]   (taps FP32 [T][hidden] each, j < tap_count)
// RMS-normed with norm[m] (eps). Mixes below `context_count` go out as BF16 rows (out, mix m at
// m * T * hidden) or q8_1 rows (out_q, mix m q8_stride bytes on); mix `context_count` (the
// recurrent head's context) goes out FP32 into the row's slot (its position % slots) of its
// sequence's ring (ring + sequence * seq_stride + slot * hidden), for rows from `ring_begin` on.
inline constexpr int kMaxSparkTaps  = 8;
inline constexpr int kMaxSparkMixes = 8;
struct SparkMixArgs {
    Rows rows;
    const float* taps[kMaxSparkTaps] = {};
    int tap_count = 0;
    const float* fuse = nullptr; // [context_count + 1][tap_count]
    const float* norm[kMaxSparkMixes] = {};
    int context_count = 0, hidden = 0, begin = 0, end = 0, ring_begin = 0;
    float eps = 1e-6f;
    __nv_bfloat16* out = nullptr;
    Q8Rows out_q;
    std::int64_t q8_stride = 0;
    float* ring = nullptr;
    std::int64_t seq_stride = 0;
    int slots = 0;
};
void spark_mix(const SparkMixArgs& args, cudaStream_t stream);

// Per head of 128 values, rows [begin, end): x RMS-normed with norm (eps), then rotated at the row's
// logical position (NeoX halves, theta^(-2i/128) for pair (i, i + 64)). Queries (`heads` of them,
// FP32 rows `stride` apart from `column`) are rotated in place; keys (`heads` KV heads at `column`)
// and the values after them (at column + heads * 128, not normed) go as F16 into cache slot
// `position % ring` (ring 0: position) of the row's sequence, laid out as kv_post's caches.
struct SparkRotateArgs {
    Rows rows;
    float* x = nullptr;
    std::int64_t stride = 0, column = 0;
    int heads = 0, begin = 0, end = 0;
    const float* norm = nullptr;
    float eps = 1e-6f, theta = 10000.0f;
    __half* k_cache = nullptr; // null: queries
    __half* v_cache = nullptr;
    std::int64_t seq_stride = 0, head_stride = 0;
    int ring = 0;
};
void spark_rotate(const SparkRotateArgs& args, cudaStream_t stream);

// h[i] += y[i] (FP32 rows `y_stride` apart, `hidden` wide), `rows` rows.
void add_rows(float* h, const float* y, std::int64_t y_stride, int hidden, int rows,
              cudaStream_t stream);

// out[t][c] = bf16(silu(gu[t][c]) * gu[t][width + c]) for FP32 [gate; up] rows (2 width each).
void silu_mul(const float* gu, int width, int rows, __nv_bfloat16* out, cudaStream_t stream);

// The recurrent head's head products (prepared once): W [rows][k] FP32 rows into M BF16 [rows][256]
// and c FP32 [rows]: m[v][r] = bf16(p[v][r]) and c[v] = p[v][256] for products p FP32 [n][257].
void spark_table(const float* p, int n, __nv_bfloat16* m, float* c, cudaStream_t stream);
// out[r][k] = hidden_out[k][r] for r < rank, out[rank][k] = bias[k]: the head's inputs.
void spark_head_inputs(const float* hidden_out, const float* bias, int hidden, int rank, float* out,
                       cudaStream_t stream);

// One step of the recurrent head (rank 256) over draft row `row` of a block:
//   joint  = W [state; embed[prev]; hidden[row]; context] + bias     (W FP32 [768][512 + 2 hidden])
//   logits = cap tanh((base[row] + c + M tanh(joint[512:768])) / cap)    over `vocab`
//   prev   = the first maximum; confidence[row - 1] = its probability; drafts[row - 1] = prev
//   state  = tanh(joint[256:512]) + sigmoid(joint[0:256]) (state - tanh(joint[256:512]))
// The context is the ring row of the position before the block's first (rows.positions[0] - 1) of
// rows.sequences[0]. `joint`, `logits` and `partials` are scratch (spark_step_floats).
struct SparkStepArgs {
    const float* weight = nullptr;
    const float* bias   = nullptr;
    const __nv_bfloat16* embed = nullptr; // [vocab][256]
    const float* hidden = nullptr;        // [block][hidden], final-normed
    const float* ring   = nullptr;        // the contexts, as SparkMixArgs writes them
    std::int64_t seq_stride = 0;
    int slots = 0;
    const std::int32_t* positions = nullptr;
    const std::int32_t* sequences = nullptr;
    const float* base = nullptr; // [block][vocab_stride] FP32 head products of `hidden`
    std::int64_t vocab_stride = 0;
    const __nv_bfloat16* m = nullptr;
    const float* c = nullptr;
    int vocab = 0, hidden_size = 0, row = 0;
    float cap = 30.0f;
    float* state = nullptr;          // [256]
    std::int32_t* prev = nullptr;    // [1]
    std::int32_t* drafts = nullptr;  // [block - 1]
    float* confidence = nullptr;     // [block - 1]
    float* joint = nullptr;
    float* logits = nullptr;
    float* partials = nullptr;
};
[[nodiscard]] std::size_t spark_step_floats(int vocab);
void spark_step(const SparkStepArgs& args, cudaStream_t stream);

// sums[i] += products[i] for i < n when *flag is nonzero: the 2^-32 fixed-point sums of the
// routed experts the CPU computed for a call (ExpertDelivery), added as the expert kernels add
// theirs, modulo 2^64.
void add_fixed(unsigned long long* sums, const void* products, const std::int32_t* flag,
               std::int64_t n, cudaStream_t stream);

} // namespace ninfer::models::inkling2::kernels
