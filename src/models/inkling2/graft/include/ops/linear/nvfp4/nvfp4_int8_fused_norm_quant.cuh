#pragma once

// The unified chain's fused [RMSNorm -> int8 activation quantise] edge kernel.
//
// Why it exists: every int8 projection route quantises its activations in its own pass, and the
// normed inputs (the attention and GDN input projections, and every SwiGLU gate_up) pay a second
// kernel for the RMSNorm that produced the plane's source. The two passes touch the same
// hidden-sized row, so they merge into one launch that never writes the BF16 intermediate to
// global memory. Only the unified chain calls this: the own chain's files are frozen and keep the
// standalone pair.
//
// Bit-for-bit contract. The norm half reproduces the launcher the model
// layer calls today -- `launcher/rmsnorm.cu` sends an aligned D=5120 row with the unit-offset
// epilogue to `rmsnorm_cta_bf16x2_kernel<Offset, 256, 10, true, 5120>` -- so this kernel uses the
// same 256-thread block per row, the same pair ownership `pair = threadIdx.x + k * 256` over ten
// trips, the same `sum += x.x * x.x + x.y * x.y` accumulation order, the same `block_reduce_sum`,
// the same `rsqrtf(sum / 5120 + eps)`, the same `rmsnorm_epilogue<Offset>` expression, and the same
// single `__floats2bfloat162_rn` rounding. The rounded image lands in shared memory instead of in
// the global `h` buffer, and the quantise half then calls the very group function both chains
// already share (`nvfp4_int8_quantize_group`: per 32-column group, four uint4 in the fixed word
// order, integer abs-max, `scale = amax / 127`, `inv = 127 / amax`, round-to-nearest clamp to
// +-127). Groups are independent of each other and of the block shape, so sweeping the 160 groups
// of K=5120 with 256 threads yields the codes and scales the standalone 128-thread quantiser would
// have written from the same BF16 row.
//
// The plan is the same construction the W4A4 fused route documents at
// `attn_input_proj/nvfp4/nvfp4_attn_input_fused_rmsnorm_w4a4.cu`, applied to the int8 planes.
// Nothing here introduces a quantiser, a group width, or a layout of its own: the two-plane form
// (codes[token * k + g * 32], scales[token * (k / 32) + g]) is the one the grouped and narrow
// mainloops already read, and it is never handed to the own chain's MMQ route.

#include "core/device_launch.h"
#include "core/tensor.h"
#include "ops/common/warp.cuh"
#include "ops/kernel/rmsnorm.cuh"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail::unified {

namespace fused_detail {

constexpr int kBlock = 256;  // the rmsnorm launcher's block for D=5120, not a free choice

// The hidden of the 27B's normed inputs: attention/GDN input projections and both SwiGLU gate_ups.
constexpr int kHidden = 5120;

// The GDN output projection's source width: 48 value heads of 128 for this model. The gated edge is
// instantiated only for this (K, D) pair, and its eligibility predicate refuses anything else, so a
// model with a different value width simply keeps the standalone pair rather than routing through a
// kernel that was never sized for it.
constexpr int kGdnValueWidth = 6144;
constexpr int kGdnHeadDim    = 128;
constexpr int kGdnBlock      = 512;

} // namespace fused_detail

// One block per addressed token row. Rows at or past `tokens` are the padded tail the wide
// mainloops address: their planes are zeroed exactly as the standalone quantiser zeroes them,
// and x is not touched (its rows stop at the caller's real token count).
template <int K = fused_detail::kHidden>
__launch_bounds__(fused_detail::kBlock) __global__
    void rmsnorm_int8_quantize_kernel(const __nv_bfloat162* __restrict__ residual,
                                      const __nv_bfloat162* __restrict__ norm_weight,
                                      std::int8_t* __restrict__ codes, float* __restrict__ scales,
                                      std::int32_t tokens, float eps) {
    constexpr int kPairsPerRow    = K / 2;
    constexpr int kPairsPerThread = kPairsPerRow / fused_detail::kBlock;
    constexpr int kGroups         = K / 32;
    static_assert(kPairsPerRow % fused_detail::kBlock == 0, "the row must split into whole trips");

    const std::int32_t token = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t tid   = static_cast<std::int32_t>(threadIdx.x);
    if (token >= tokens) {
        zero_nvfp4_int8_activation_row(codes, scales, token, K, kGroups);
        return;
    }

    // ---- norm half: the D=5120 unit-offset launcher's ownership, order, reduction and rounding ----
    const std::int64_t row_base = static_cast<std::int64_t>(token) * kPairsPerRow;
    __nv_bfloat162 values[kPairsPerThread];
    __nv_bfloat162 weights[kPairsPerThread];
    float sum = 0.0F;
#pragma unroll
    for (int k = 0; k < kPairsPerThread; ++k) {
        const int pair = tid + k * fused_detail::kBlock;
        values[k]      = residual[row_base + pair];
        weights[k]     = norm_weight[pair];
        const float2 xf = __bfloat1622float2(values[k]);
        sum += xf.x * xf.x + xf.y * xf.y;
    }

    __shared__ float warp_sums[fused_detail::kBlock / kWarpSize];
    __shared__ float inv_shared;
    const float block_sum = block_reduce_sum<fused_detail::kBlock>(sum, warp_sums);
    if (tid == 0) {
        inv_shared = rsqrtf(block_sum / static_cast<float>(K) + eps);
    }
    __syncthreads();
    const float inv = inv_shared;

    // The exact BF16 image the standalone quantiser would have read back from the global `h`.
    __shared__ alignas(32) __nv_bfloat16 staged[K];
#pragma unroll
    for (int k = 0; k < kPairsPerThread; ++k) {
        const int pair    = tid + k * fused_detail::kBlock;
        const float2 xf   = __bfloat1622float2(values[k]);
        const float2 wf   = __bfloat1622float2(weights[k]);
        reinterpret_cast<__nv_bfloat162*>(staged)[pair] = __floats2bfloat162_rn(
            rmsnorm_epilogue<RmsEpilogue::Offset>(xf.x, inv, wf.x, 0.0F),
            rmsnorm_epilogue<RmsEpilogue::Offset>(xf.y, inv, wf.y, 0.0F));
    }
    __syncthreads();

    // ---- quantise half: the shared per-32 group function, reading the staged image ----
    for (int g = tid; g < kGroups; g += fused_detail::kBlock) {
        float scale = 0.0F;
        uint4 packed[2];
        nvfp4_int8_quantize_group(reinterpret_cast<const uint4*>(staged + g * 32), scale, packed);
        scales[static_cast<std::size_t>(token) * kGroups + g] = scale;
        std::int8_t* dst = codes + static_cast<std::size_t>(token) * K + g * 32;
        reinterpret_cast<uint4*>(dst)[0] = packed[0];
        reinterpret_cast<uint4*>(dst)[1] = packed[1];
    }
}

// The GDN output projection's input is not a plain RMSNorm row: it is a per-head norm with a SiLU
// gate, `rmsnorm_warp_bf16x2_kernel<Gated, Block, Prefetch>`, where one warp owns one 128-wide head
// row and reduces with warp shuffles. The block size and the prefetch flag are scheduling choices
// (the kernel's own comment says so), so this pass is free to pick its own; what it may not change
// is the per-row ownership, the k-ascending `sum += xf.x * xf.x + xf.y * xf.y` order inside the
// four-trip window, `rsqrtf(sum / 128 + eps)`, and the single `__floats2bfloat162_rn` rounding of
// `rmsnorm_epilogue<Gated>`. Those are reproduced here verbatim, and the resulting BF16 image lands
// in shared memory instead of in the global `on` buffer, from which the same shared per-32 group
// function writes the two planes.
//
// One block per addressed token: a warp takes the head rows whose index is congruent to it, so the
// token's whole K-wide row is staged before any group is quantised. Group ownership is free (groups
// are independent of each other and of the reduction), which is what lets one block shape serve both
// halves.
template <int K, int D, int Block>
__launch_bounds__(Block) __global__
    void gated_rmsnorm_int8_quantize_kernel(const __nv_bfloat162* __restrict__ source,
                                            const __nv_bfloat162* __restrict__ norm_weight,
                                            const __nv_bfloat162* __restrict__ gate,
                                            std::int8_t* __restrict__ codes, float* __restrict__ scales,
                                            std::int32_t tokens, float eps) {
    constexpr int kHeads            = K / D;
    constexpr int kWarps            = Block / kWarpSize;
    constexpr int kHeadsPerWarp     = kHeads / kWarps;
    constexpr int kPairs            = D / 2;
    constexpr int kMaxPairsPerLane  = 4;   // the standalone kernel's window
    constexpr int kGroups           = K / 32;
    static_assert(kHeads % kWarps == 0, "head rows must divide across the block's warps");
    static_assert((D % 64) == 0, "the four-trip window covers D up to 256");

    const std::int32_t token = static_cast<std::int32_t>(blockIdx.x);
    if (token >= tokens) {
        // Block-cooperative, like the standalone quantiser's own tail path.
        zero_nvfp4_int8_activation_row(codes, scales, token, K, kGroups);
        return;
    }
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;

    __shared__ alignas(32) __nv_bfloat16 staged[K];
    // A token's row is the whole K-wide span (kPairs is the *head's* pair count), so the token
    // stride is K/2 pairs -- the first token cannot tell the two apart, which is why a wrong stride
    // here only shows up from T=2 on.
    const std::int64_t row0 =
        static_cast<std::int64_t>(token) * (K / 2);
#pragma unroll
    for (int h = 0; h < kHeadsPerWarp; ++h) {
        const int  head  = warp + h * kWarps;
        const auto base  = row0 + static_cast<std::int64_t>(head) * kPairs;
        __nv_bfloat162 values[kMaxPairsPerLane];
        __nv_bfloat162 weights[kMaxPairsPerLane];
        __nv_bfloat162 gates[kMaxPairsPerLane];
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < kMaxPairsPerLane; ++k) {
            const int pair = lane + k * kWarpSize;
            if (pair < kPairs) {
                values[k] = source[base + pair];
                weights[k] = norm_weight[pair];
                gates[k]   = gate[base + pair];
                const float2 xf = __bfloat1622float2(values[k]);
                sum += xf.x * xf.x + xf.y * xf.y;
            }
        }
        sum       = warp_reduce_sum(sum);
        float inv = lane == 0 ? rsqrtf(sum / static_cast<float>(D) + eps) : 0.0f;
        inv       = __shfl_sync(kFullWarpMask, inv, 0);
#pragma unroll
        for (int k = 0; k < kMaxPairsPerLane; ++k) {
            const int pair = lane + k * kWarpSize;
            if (pair < kPairs) {
                const float2 xf = __bfloat1622float2(values[k]);
                const float2 wf = __bfloat1622float2(weights[k]);
                const float2 zf = __bfloat1622float2(gates[k]);
                reinterpret_cast<__nv_bfloat162*>(staged)[
                    static_cast<std::size_t>(head) * kPairs + pair] = __floats2bfloat162_rn(
                    rmsnorm_epilogue<RmsEpilogue::Gated>(xf.x, inv, wf.x, zf.x),
                    rmsnorm_epilogue<RmsEpilogue::Gated>(xf.y, inv, wf.y, zf.y));
            }
        }
    }
    __syncthreads();

    const std::int32_t tid = static_cast<std::int32_t>(threadIdx.x);
    for (int g = tid; g < kGroups; g += Block) {
        float scale = 0.0F;
        uint4 packed[2];
        nvfp4_int8_quantize_group(reinterpret_cast<const uint4*>(staged + g * 32), scale, packed);
        scales[static_cast<std::size_t>(token) * kGroups + g] = scale;
        std::int8_t* dst = codes + static_cast<std::size_t>(token) * K + g * 32;
        reinterpret_cast<uint4*>(dst)[0] = packed[0];
        reinterpret_cast<uint4*>(dst)[1] = packed[1];
    }
}

// The host-callable launchers live in nvfp4_int8_fused_norm_quant.cu / .h, so that a route or a
// test can reach the pass without pulling warp intrinsics into a C++ compile.

} // namespace ninfer::ops::detail::unified
