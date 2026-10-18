#pragma once

// Weight-operand helpers shared by the grouped-integer MMA kernels (Q4/Q5/Q6/Q8).

#include "ops/common/cuda/mma.cuh"
#include "ops/common/numeric/convert.cuh"

namespace ninfer::ops::common {

// --- Dequantized-integer weight operands -----------------------------------------------------
// The grouped-integer formats feed the tensor cores exact small integers (|q| <= 128) and apply
// the group scale in FP32 afterwards. Native-BF16 targets keep them as BF16 pairs. Turing has no
// BF16 tensor core and no cheap BF16 pack, so it builds exact FP16 pairs from the magic-number
// identity half(0x6400 | m) == 1024 + m and runs the FP16 tensor core; only the BF16 activation
// fragments are converted per MMA.
#if defined(NINFER_SM75)
__device__ __forceinline__ unsigned integer_pair_operand(int q0, int q1) {
    const unsigned biased = (static_cast<unsigned>(q0 + 128) & 0xffu) |
                            ((static_cast<unsigned>(q1 + 128) & 0xffu) << 16) | 0x64006400u;
    const unsigned bias = 0x64806480u; // half2(1152)
    const __half2 result = __hsub2(*reinterpret_cast<const __half2*>(&biased),
                                   *reinterpret_cast<const __half2*>(&bias));
    return *reinterpret_cast<const unsigned*>(&result);
}

// Two signed bytes (low 16 bits of `values`) to an exact FP16 pair.
__device__ __forceinline__ unsigned s8x2_pair_operand(unsigned values) {
    const unsigned biased = (__byte_perm(values, 0x64646464u, 0x5140) ^ 0x00800080u);
    const unsigned bias   = 0x64806480u; // half2(1152)
    const __half2 result  = __hsub2(*reinterpret_cast<const __half2*>(&biased),
                                    *reinterpret_cast<const __half2*>(&bias));
    return *reinterpret_cast<const unsigned*>(&result);
}

__device__ __forceinline__ void mma_integer_weights(float& c0, float& c1, float& c2, float& c3,
                                                    unsigned a0, unsigned a1, unsigned a2,
                                                    unsigned a3, unsigned b0, unsigned b1) {
    mma_f16(c0, c1, c2, c3, a0, a1, a2, a3, bf16x2_to_f16x2_operand(b0),
            bf16x2_to_f16x2_operand(b1));
}
#else
__device__ __forceinline__ unsigned integer_pair_operand(int q0, int q1) {
    union {
        __nv_bfloat162 value;
        unsigned bits;
    } pair;
    pair.value = __floats2bfloat162_rn(static_cast<float>(q0), static_cast<float>(q1));
    return pair.bits;
}

__device__ __forceinline__ void mma_integer_weights(float& c0, float& c1, float& c2, float& c3,
                                                    unsigned a0, unsigned a1, unsigned a2,
                                                    unsigned a3, unsigned b0, unsigned b1) {
    mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
}
#endif

// Two signed bytes (low 16 bits of `values`) as the exact MMA weight-operand pair.
__device__ __forceinline__ unsigned int8x2_to_mma_pair(unsigned values) {
#if defined(NINFER_SM75)
    return s8x2_pair_operand(values);
#else
    return int8x2_to_bf16x2_bits(values);
#endif
}

} // namespace ninfer::ops::common
