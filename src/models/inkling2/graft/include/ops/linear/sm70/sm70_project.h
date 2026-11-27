#pragma once

// sm_70 wide projections of any dense weight format (sm70_dequantize.h) through the FP16 cuBLAS
// route of sm70_gemm.h, with the epilogues the fused Linear Ops need: a parent's row ranges stored
// into or added onto BF16 planes, and SwiGLU over a [gate; up] pair. Callers gate on
// sm70::takes(tokens) && sm70::dequantizable(weight).

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear_policy.h"
#include "ops/linear/sm70/sm70_gemm.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops::detail::sm70 {

// Parent rows [row_begin, row_begin + rows) land at destination rows from row_offset of a BF16
// plane with `leading` rows per column: stored, or added onto (the residual epilogue). With `fp32`
// set they land unrounded in that FP32 plane instead (a tensor-parallel rank's partial sum).
struct Destination {
    __nv_bfloat16* data     = nullptr;
    std::int32_t leading    = 0;
    std::int32_t row_offset = 0;
    std::int32_t row_begin  = 0;
    std::int32_t rows       = 0;
    bool accumulate         = false;
    float* fp32             = nullptr;
};

inline constexpr int kMaxDestinations = 4;

// The columns a wide call converts and multiplies at once: every prompt chunk in one pass, a
// Vision item's patches in several, which bounds its FP16 activations and FP32 product plane.
inline constexpr std::int32_t kWideColumns = 8192;

// The activations of one call in FP16 (or, for the small-T route's direct widths, as the caller's
// BF16 plane) and the slab scratch every parent of up to `max_rows` rows shares. A wide call of
// more than kWideColumns columns keeps its BF16 plane and converts each pass's columns into `pass`.
struct Activation {
    const __half* data        = nullptr;
    const __nv_bfloat16* bf16 = nullptr;
    __half* pass              = nullptr;
    // The small-T route's flags of its FP16 plane: nonzero for each 64-value chunk of a column
    // holding a value beyond FP16 ([T][k / 64]; sm70_small.h).
    const std::uint8_t* beyond_chunks = nullptr;
    std::int32_t k            = 0;
    std::int32_t tokens       = 0;
    std::int32_t max_rows     = 0;
    void* scratch             = nullptr;
    // The integer route's quantized form (sm70_integer.h), when the call takes it.
    const std::int8_t* codes      = nullptr;
    const float* code_scales      = nullptr;
    const std::int32_t* code_sums = nullptr;
};

// Workspace of prepare() for parents of at most `max_rows` rows.
[[nodiscard]] std::size_t workspace_bytes(std::int32_t max_rows, std::int32_t k,
                                          std::int32_t tokens);

[[nodiscard]] Activation prepare(const Tensor& x, std::int32_t max_rows, WorkspaceArena& workspace,
                                 cudaStream_t stream);

// Every row of `w` into the destination that takes it (at most kMaxDestinations).
void project(const Activation& x, const Weight& w, std::span<const Destination> destinations,
             cudaStream_t stream);

// out[i, t] = silu(gate_i . x_t) * (up_i . x_t), with gate rows [0, rows) and up rows [rows,
// 2 rows) of one parent (`up` null) or rows of two weights. `out` is BF16 [rows, T].
void swiglu(const Activation& x, const Weight& gate, const Weight* up, Tensor& out,
            cudaStream_t stream);

// --- Both sm_70 routes ------------------------------------------------------------------------
// The wide route above and the small-T one (sm70_small.h) together take every width of a dense
// format on sm_70; the *_for functions pick the one a call's width takes.

// Whether a dense weight of this format and input width has both routes (whole chunks of the
// small-T route: 128 values for T2, 64 for the other formats).
[[nodiscard]] bool covered_format(QType format, std::int32_t k);
[[nodiscard]] bool covers(const Weight& w, std::int32_t tokens);

// Workspace over [min_tokens, max_tokens] for parents of up to `max_rows` rows when the format is
// covered, the integer route's (sm70_integer.h) included where `policy` admits it; otherwise
// routes(min_tokens, max_tokens), the Op's own routes.
template <class Routes>
[[nodiscard]] std::size_t capacity(QType format, std::int32_t max_rows, std::int32_t k,
                                   std::int32_t min_tokens, std::int32_t max_tokens,
                                   LinearPolicy policy, Routes&& routes);

// The activations of a call; `integer` (sm70::integer_takes for the call's weights) adds the
// integer route's quantized form.
[[nodiscard]] Activation prepare_for(const Tensor& x, std::int32_t max_rows,
                                     WorkspaceArena& workspace, cudaStream_t stream,
                                     bool integer = false);
// The activations of a call no wider than the small-T route's direct widths, which need no
// workspace (callers without an arena check direct_tokens first).
[[nodiscard]] bool direct_tokens(std::int32_t tokens);
[[nodiscard]] Activation prepare_direct(const Tensor& x);
void project_for(const Activation& x, const Weight& w, std::span<const Destination> destinations,
                 cudaStream_t stream);
void swiglu_for(const Activation& x, const Weight& gate, const Weight* up, Tensor& out,
                cudaStream_t stream);

[[nodiscard]] std::size_t covered_bytes(QType format, std::int32_t max_rows, std::int32_t k,
                                        std::int32_t min_tokens, std::int32_t max_tokens,
                                        LinearPolicy policy);

template <class Routes>
std::size_t capacity(QType format, std::int32_t max_rows, std::int32_t k, std::int32_t min_tokens,
                     std::int32_t max_tokens, LinearPolicy policy, Routes&& routes) {
    if (!enabled() || !covered_format(format, k)) { return routes(min_tokens, max_tokens); }
    return covered_bytes(format, max_rows, k, min_tokens, max_tokens, policy);
}

} // namespace ninfer::ops::detail::sm70
