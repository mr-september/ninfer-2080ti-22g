#pragma once

// The unified chain's narrow-band mainloop: one warp group per row set, sixteen-column weight groups,
// `dp4a` accumulation, one E4M3 scale per group and one FP32 activation scale per (token, 32 columns)
// applied as a single rescale product per (group, column). The shape is the MMVQ-shaped kernel's (see
// nvfp4_mmvq_mma.cuh for why each of those choices is the one the grouped mainloop also makes); this
// file exists so the unified route reaches that shape through its own kernel, with a column stage per
// width and activation planes sized to the call.
//
// Bit-for-bit identity with that kernel is the acceptance criterion, and it is a construction: for any
// output element (row, column) the FP32 addition sequence here is term-for-term the sequence it
// runs -- same ascending walk of the sixteen-column groups, same `fmaf(w_scale * x_scale, sum, acc)`
// expression against the same LUT byte and the same `(column, kbase >> 5)` activation scale, same five
// butterfly rounds over the same thirty-two lanes, same `reduced[0] -> [1] -> [2] -> [3]` absorption,
// same single BF16 rounding in the same adapter. NC being exact removes columns MMVQ computed and
// then discarded at the store; R changes which block owns a row, not what a row accumulates.
//
// That construction is also why this band uses 32-column activation groups: both chains quantise
// once per call into the same two planes, and the wide grouped mainloop folds per 32-column group
// like the own chain does, so the two chains leave the same bytes above the crossover.

#include "ops/linear/common/output.cuh"
#include "ops/linear/nvfp4/nvfp4_codec_unified.cuh"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_mma.cuh" // e2m1_word, the E4M3 LUT
#include "ops/linear/nvfp4/nvfp4_int8_narrow_host.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail::narrow {

namespace g = ninfer::ops::detail::unified;

template <int NC, int R, class Output>
__global__ __launch_bounds__(kThreads) void nvfp4_int8_narrow_kernel(
    const std::uint8_t* __restrict__ weight_codes, const std::uint8_t* __restrict__ weight_scales,
    const std::int8_t* __restrict__ x_codes, const float* __restrict__ x_scales, float alpha_half,
    Output output, std::int32_t k, std::int32_t tokens) {
    static_assert(NC >= 1 && NC <= kMaxTokens, "NC is one of the instantiated column stages");
    static_assert(R >= 1 && R <= kMaxRowsPerBlock, "R is one of the instantiated row stages");

    const std::int32_t row_begin = static_cast<std::int32_t>(blockIdx.x) * R;
    const std::int32_t tid  = static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t lane = tid & (kWarpSize - 1);
    const std::int32_t warp = tid / kWarpSize;

    const std::int32_t groups   = k / kGroupCodes; // one E4M3 scale each
    const std::int32_t x_groups = k / 32;          // activation scale stride
    const std::size_t row_bytes = static_cast<std::size_t>(k / 2);

    // Same additive split of the scale byte address as the MMVQ kernel, for the same reason: the row
    // terms are fixed for the whole block and a trip advances g by a multiple of four, so the group's
    // slot inside its four-byte cluster is the thread's own tid & 3 on every trip.
    std::int32_t scale_row[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const std::int32_t row = row_begin + r;
        scale_row[r] =
            (row >> 7) * ((k >> 6) << 9) + (row & 31) * 16 + ((row & 127) >> 5) * 4;
    }
    const std::int32_t scale_group_lo = tid & 3;

    float acc[R][NC];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < NC; ++j) { acc[r][j] = 0.0F; }
    }

    // Not unrolled: pairing trips costs registers, and registers are what this block's residency is
    // made of -- the same budget the column stage is sized against.
    for (std::int32_t g_index = tid; g_index < groups; g_index += kThreads) {
        unsigned q[R][4];
        float w_scale[R];
        const std::int32_t scale_group = ((g_index >> 2) << 9) + scale_group_lo;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint2 packed = *reinterpret_cast<const uint2*>(
                weight_codes + static_cast<std::size_t>(row_begin + r) * row_bytes +
                static_cast<std::size_t>(g_index) * kGroupBytes);
            g::e2m1_word(packed.x, q[r][0], q[r][1]);
            g::e2m1_word(packed.y, q[r][2], q[r][3]);

            const unsigned raw = weight_scales[scale_row[r] + scale_group];
            w_scale[r] = g::kNvfp4E4M3ScaleLut[raw];
        }

        const std::int32_t kbase = g_index * kGroupCodes;
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
                acc[r][j] = fmaf(w_scale[r] * x_scale, static_cast<float>(sum), acc[r][j]);
            }
        }
    }

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

} // namespace ninfer::ops::detail::narrow
