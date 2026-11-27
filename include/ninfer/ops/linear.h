#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"

#include "core/device.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * @brief Permitted private activation-compute profiles for a linear projection.
 *
 * The policy constrains private route selection; it does not select a kernel or prescribe a
 * particular MMA instruction. The public activation and output tensors remain BF16 for every
 * policy.
 */
enum class LinearPolicy : std::uint8_t {
    A16Only, ///< Admit only A16 compute profiles.
    AllowA8, ///< Admit either A16 or A8 compute profiles.
    AllowA4, ///< Admit A16, A8 or A4 compute profiles.
    /// Admit either A16 or integer-A8 compute profiles. Distinct from AllowA8, which selects the
    /// FP8 activation path against FP8 weights; this one quantises activations to s8 with one
    /// scale per weight group and feeds groupwise-int weights to the integer tensor cores. Held
    /// to the same A8 activation allowance.
    AllowA8Int,
    /// AllowA8Int plus the integer small-T route for the 27B gate_up at decode and verify widths
    /// (ops/linear/q4/q4_small_t_mma_i8.cuh). Separate because that route is a quality trade the
    /// prefill one is not asked to carry: it applies s8 activation quantisation to every decode
    /// step rather than to full prefill tiles only, so it is opt-in per engine.
    AllowA8IntDecode,
    /// AllowA8Int plus the cuBLAS prefill route at wide token counts: the weight is materialised as
    /// int8 with one scale per row and the GEMM handed to cuBLAS, which runs it about twice as fast
    /// as this fork's own integer mainloop can. It is a further quality trade -- per-row weight
    /// scales and per-token activation scales, where the integer route carries per-group ones -- so
    /// it is opt-in per engine, and it falls back to AllowA8Int below the width where it pays.
    AllowPrefillCublas,
    /// AllowA8Int's counterpart for block-scaled NVFP4 weights: the stored E2M1 codes widen
    /// losslessly to s8 and feed the integer tensor cores with s8-quantised activations, one FP32
    /// scale per (token, 32 columns) and the per-16 E4M3 weight scales rescaling in FP32. It is the
    /// same A8-activation quality trade the Q4/Q5 integer routes carry, opt-in per engine with
    /// --nvfp4-w4a8-int.
    AllowNvfp4Int8,
};

[[nodiscard]] constexpr bool valid_linear_policy(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8 ||
           policy == LinearPolicy::AllowA4 || policy == LinearPolicy::AllowA8Int ||
           policy == LinearPolicy::AllowA8IntDecode || policy == LinearPolicy::AllowPrefillCublas ||
           policy == LinearPolicy::AllowNvfp4Int8;
}

[[nodiscard]] constexpr bool allows_a8(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowA8 || policy == LinearPolicy::AllowA4;
}

/**
 * Returns the caller-owned transient capacity required by Linear for every T in the inclusive
 * `[min_tokens,max_tokens]` interval. Invalid registered profiles, policies, or intervals throw;
 * a legal route that requires no transient storage returns zero.
 */
[[nodiscard]] std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                          std::int32_t input_rows,
                                                          LinearPolicy policy,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens);

/**
 * Y[n,t] = sum_k decode(W)[n,k] * X[k,t].
 *
 * X is BF16 [K,T], W represents [N,K], and Y is BF16 [N,T]. Activations and output are
 * contiguous, non-null and 16-byte aligned; T is positive. Output overlaps neither X nor
 * any weight plane. Inputs are preserved. There is no bias, activation or transpose.
 *
 * Supports registered weight geometries and column domains for contiguous BF16, RowSplit
 * Q4/Q5/Q6-G64 and Q8-G32 with FP16 scales and row-scaled FP8.
 * Unsupported format/geometry/layout combinations throw.
 *
 * policy permits private A16/A8 computation; input and output remain BF16. Permission
 * does not require activation quantization. Workspace is caller-owned transient storage,
 * sized by linear_workspace_capacity_bytes(), disjoint from all operands. No hidden allocation.
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            MemoryArena& workspace, CudaExecutionView execution);

/** A16-only projection with the same operand requirements and no transient workspace. */
void linear(const Tensor& x, const Weight& w, Tensor& out, CudaExecutionView execution);

} // namespace ninfer::ops
