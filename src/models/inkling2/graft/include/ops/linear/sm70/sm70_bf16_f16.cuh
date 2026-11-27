#pragma once

// BF16 to FP16 on sm_70, whose tensor cores multiply FP16 operands only: the saturating conversion
// and what saturation takes from a value beyond FP16 (sm70_gemm.h describes how the wide products
// add it back; the Volta attention kernels do the same per tile).

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail::sm70 {

// Two BF16 values in one word (lower k in the low half) as two FP16 values in one word, each
// saturated to +-65504.
__device__ __forceinline__ unsigned bf16x2_to_f16x2(unsigned word) {
    const float lo  = fminf(fmaxf(__uint_as_float(word << 16), -65504.0f), 65504.0f);
    const float hi  = fminf(fmaxf(__uint_as_float(word & 0xffff0000u), -65504.0f), 65504.0f);
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}

// Whether either BF16 half of `word` has a magnitude of 2^16 or more (infinities and NaNs too),
// which bf16x2_to_f16x2 saturates. Below 2^16 a BF16 value is at most 65280. With the signs and
// mantissas masked off, adding 113 to an exponent field of 143 or more carries into its sign bit.
__device__ __forceinline__ bool bf16x2_beyond_f16(unsigned word) {
    return (((word & 0x7f807f80u) + 0x38803880u) & 0x80008000u) != 0u;
}

// What bf16x2_to_f16x2's saturation takes from one BF16 value: zero unless it is beyond FP16.
__device__ __forceinline__ float saturated_excess(float x) {
    return x - fminf(fmaxf(x, -65504.0f), 65504.0f);
}

// sum_e w[e] * saturated_excess(x[e]) over a row's FP16 weights and a column's BF16 values.
__device__ __forceinline__ float saturated_part(const __half* w, const __nv_bfloat16* x,
                                                std::int32_t k) {
    float sum = 0.0f;
    for (std::int32_t e = 0; e < k; e += 2) {
        const unsigned word = *reinterpret_cast<const unsigned*>(x + e);
        if (!bf16x2_beyond_f16(word)) { continue; }
        const float2 weights = __half22float2(*reinterpret_cast<const __half2*>(w + e));
        sum = fmaf(weights.x, saturated_excess(__uint_as_float(word << 16)), sum);
        sum = fmaf(weights.y, saturated_excess(__uint_as_float(word & 0xffff0000u)), sum);
    }
    return sum;
}

} // namespace ninfer::ops::detail::sm70
