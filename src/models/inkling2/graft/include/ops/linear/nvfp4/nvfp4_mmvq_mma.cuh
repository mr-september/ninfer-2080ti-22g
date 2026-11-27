#pragma once

// The MMVQ-shaped NVFP4 mainloop (see nvfp4_mmvq_host.h for what is ported, from where, and which
// three things had to change for ninfer's planes and epilogue contract).
//
// Arithmetic destination is deliberately the one the other two int8 routes already hit -- E2M1 codes
// widened to the exact integer grid s8 can hold (v = 2*e2m1), one NVFP4 G16 group rescaling the
// partial sum in FP32 through the E4M3 LUT, one per-(token, 32 columns) FP32 scale per group, FP32
// accumulation, a single BF16 rounding at the store -- so a value produced here and one produced by
// nvfp4_int8_grouped_kernel agree bit for bit, not merely to within a tolerance. Three things make
// that true rather than merely plausible:
//
//   * The 16-column group is the same unit both accumulation paths sum over. One `mma.m16n8k16.s8`
//     covers sixteen k at a time, which is exactly one NVFP4 group, and so does one trip here --
//     four `dp4a` folding sixteen products into one int32. The integers therefore add in the same
//     width, over the same terms, in the same order.
//   * The rescale is one product per (group, column): the weight scale is E4M3 already halved by the
//     LUT (nvfp4_e4m3_scale.h), the activation scale is the single 32-block this group sits inside,
//     and they are applied with one fmaf against the int32 sum -- the same expression the grouped
//     mainloop's `w0 * xa0` feeds to its own fmaf.
//   * Nothing between here and the store rounds: the accumulator is FP32, the cross-warp reduction
//     is FP32, and the adapter owns the one conversion.

#include "ops/linear/common/output.cuh"
#include "ops/linear/nvfp4/nvfp4_codec_unified.cuh"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_mma.cuh" // e2m1_word, the E4M3 LUT
#include "ops/linear/nvfp4/nvfp4_mmvq_host.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail::mmvq {

namespace g = ninfer::ops::detail::unified;

// One CUDA block owns R output rows outright. `NC` columns are retired together, so a row's weight
// bytes are fetched once and reused across every column -- the whole reason this shape beats a token
// tile at the widths that reach it, where a tile would pay for columns the caller never asked for.
//
// `tokens` is the caller's real width and the only columns stored; NC >= tokens pads the instruction
// stream, not any plane, so the store is the only thing that needs a guard. A width above kMaxTokens
// never gets here (see nvfp4_mmvq_launch.cuh), which is llama.cpp's crossover rather than a
// limitation of this kernel.
//
// R is a *load-sharing* factor, not a tile: the block's activation reads are the expensive stream
// (NC * k bytes per row against k/2 bytes of weight), so the column loop sits outside the row loop
// and one fetched activation word feeds every row the block owns. Each row still walks the same
// groups in the same order, so its accumulator sees the same terms in the same sequence as R = 1 and
// the result is bit-identical, not merely close.
template <int NC, int R, class Output>
__global__ __launch_bounds__(kThreads) void nvfp4_mmvq_kernel(
    const std::uint8_t* __restrict__ weight_codes, const std::uint8_t* __restrict__ weight_scales,
    const std::int8_t* __restrict__ x_codes, const float* __restrict__ x_scales, float alpha_half,
    Output output, std::int32_t k, std::int32_t tokens) {
    static_assert(NC >= 1 && NC <= kMaxTokens, "NC is one of the instantiated width stages");
    static_assert(R >= 1 && R <= kMaxRowsPerBlock, "R is one of the instantiated row stages");

    const std::int32_t row_begin = static_cast<std::int32_t>(blockIdx.x) * R;
    const std::int32_t tid  = static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t lane = tid & (kWarpSize - 1);
    const std::int32_t warp = tid / kWarpSize;

    const std::int32_t groups   = k / kGroupCodes; // one E4M3 scale each
    const std::int32_t x_groups = k / 32;          // activation scale stride
    const std::size_t row_bytes = static_cast<std::size_t>(k / 2);

    // The scale byte address is the only per-trip address in this kernel that is not affine in g,
    // because nvfp4_scale_byte_offset interleaves shifts and masks. It does split additively into a
    // part that is fixed for the whole block and a part that depends only on the group index, so the
    // row terms are computed once here instead of once per trip. Nothing else changes: the sum below
    // plus the group term is the same integer the shared helper returns for every (row, group, k).
    std::int32_t scale_row[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const std::int32_t row = row_begin + r;
        scale_row[r] =
            (row >> 7) * ((k >> 6) << 9) + (row & 31) * 16 + ((row & 127) >> 5) * 4;
    }
    // Trip n visits g = tid + n * kThreads. kThreads is a multiple of 4, so the group's position
    // inside its four-byte cluster is the thread's own (tid & 3) on every trip, and only the cluster
    // index advances.
    const std::int32_t scale_group_lo = tid & 3;

    float acc[R][NC];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < NC; ++j) { acc[r][j] = 0.0F; }
    }

    // Deliberately not unrolled: unrolling two trips does pair the weight loads, but it costs enough
    // registers to drop block residency, and every width in this band then loses back more than the
    // pairing saves. Decode calls live in this band.
    for (std::int32_t g = tid; g < groups; g += kThreads) {
        unsigned q[R][4];
        float w_scale[R];
        const std::int32_t scale_group = ((g >> 2) << 9) + scale_group_lo;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            // An eight-byte read is exactly what a sixteen-column NVFP4 group is. K % 64 == 0 is
            // already this route's alignment gate, so every group of every row starts on an
            // eight-byte boundary and the device-side weight planes are read for no other reason.
            const uint2 packed = *reinterpret_cast<const uint2*>(
                weight_codes + static_cast<std::size_t>(row_begin + r) * row_bytes +
                static_cast<std::size_t>(g) * kGroupBytes);
            // e2m1_word emits consecutive codes, so these four words are the group's sixteen
            // operands in k order: q[c] holds k = g*16 + 4c .. 4c+3.
            g::e2m1_word(packed.x, q[r][0], q[r][1]);
            g::e2m1_word(packed.y, q[r][2], q[r][3]);

            // One E4M3 byte per group through the shared LUT -- the trailing /2 and NaN -> 0 are
            // llama's ue4m3 semantics and live in the table, exactly as they do for other routes.
            const unsigned raw = weight_scales[scale_row[r] + scale_group];
            w_scale[r] = g::kNvfp4E4M3ScaleLut[raw];
        }

        const std::int32_t kbase = g * kGroupCodes;
        // Which 32-column activation scale this group sits in: a group never straddles two, because
        // 32 is a whole number of 16s.
        const int x_block = kbase >> 5;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const uint4 xw = *reinterpret_cast<const uint4*>(
                x_codes + static_cast<std::size_t>(j) * k + kbase);
            const float x_scale = x_scales[static_cast<std::size_t>(j) * x_groups + x_block];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int sum = 0;
                sum = __dp4a(static_cast<int>(q[r][0]), static_cast<int>(xw.x), sum);
                sum = __dp4a(static_cast<int>(q[r][1]), static_cast<int>(xw.y), sum);
                sum = __dp4a(static_cast<int>(q[r][2]), static_cast<int>(xw.z), sum);
                sum = __dp4a(static_cast<int>(q[r][3]), static_cast<int>(xw.w), sum);
                // One product per (group, column): the int32 sum is exact, both scales are exact, and
                // this fmaf is the same operation the grouped mainloop spends on the same sixteen
                // terms.
                acc[r][j] = fmaf(w_scale[r] * x_scale, static_cast<float>(sum), acc[r][j]);
            }
        }
    }

    // llama's reduction, minus what it needed and we do not: every warp reduces by shuffle and parks
    // its column partials in shared memory for warp 0 to absorb. The rows this block owns share the
    // shuffle but not a reduction axis -- each row's columns are independent sums.
    __shared__ float reduced[kWarps][R][NC > 1 ? NC : 1];
#pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int j = 0; j < NC; ++j) {
                acc[r][j] += __shfl_xor_sync(0xFFFFFFFFu, acc[r][j], off);
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int j = 0; j < NC; ++j) { reduced[warp][r][j] = acc[r][j]; }
        }
    }
    __syncthreads();

    if (tid >= NC) { return; }
    const auto epilogue = linear_output_tile<R>(output, row_begin);
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float total = reduced[0][r][tid];
#pragma unroll
        for (int w = 1; w < kWarps; ++w) { total += reduced[w][r][tid]; }
        if (tid < tokens) { epilogue.store(row_begin + r, tid, total * alpha_half); }
    }
}

} // namespace ninfer::ops::detail::mmvq
