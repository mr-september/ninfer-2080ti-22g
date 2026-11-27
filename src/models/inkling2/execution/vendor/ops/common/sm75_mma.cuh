#pragma once

// Turing (sm_75) tensor cores have the FP16 m16n8k8 and INT8 m8n8k16 shapes only: no BF16 or TF32
// operands and no m16n8k16/m16n8k32. A 75 build's mma.cuh helpers take the forms below, which keep
// every caller's fragment contract.
//
// - FP16 m16n8k16 is two m16n8k8 over k 0..7 ({a0,a1}, b0) and k 8..15 ({a2,a3}, b1): the k8
//   fragments are exactly the halves of the k16 ones.
// - INT8 m16n8k32 is four m8n8k16: m8n8k16 holds one row g with k = 4*tig..4*tig+3 for A and the
//   same k of column g for B, which is a0 (row g), a1 (row g+8), a2/a3 (k + 16) and b0/b1 (k + 16)
//   of m16n8k32. Integer accumulation is exact, so the result is identical.
// - BF16 and TF32 are not narrowed. A BF16 value (8 significant bits) is exactly an FP16 value when
//   it is zero or its exponent is in [-17, 15]; a TF32 value (11 significant bits) when it is zero
//   or its exponent is in [-14, 15]. The tile runs on FP16 m16n8k8 with FP32 accumulation, which
//   forms the same exact products the native instruction does for every output whose row of A and
//   column of B qualify. When some do not, the warp also computes the tile in FP32 from the same
//   operands (pre_sm80.cuh, the products a 70 build forms), and exactly the outputs with a
//   non-qualifying row or column take that result. Correctness therefore never depends on the range
//   of the data, only the speed does, and an output's bits depend on its own operands alone,
//   whatever else shares its tile.

#include "ops/common/pre_sm80.cuh"

#include <cuda_fp16.h>

#if defined(NINFER_SM75_FORMS)
namespace ninfer::ops::sm75 {

__device__ __forceinline__ void mma_f16_k8(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                           unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ void mma_f16_k8_f16acc(unsigned& d0, unsigned& d1, unsigned a0,
                                                  unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
                 : "+r"(d0), "+r"(d1)
                 : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ void mma_s8_k16(int& c0, int& c1, unsigned a, unsigned b) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a), "r"(b));
}

// Both BF16 halves are zero or have a biased exponent in [110, 142].
__device__ __forceinline__ bool bf16x2_exact_in_fp16(unsigned x) {
    const unsigned lo = x & 0xffffu, hi = x >> 16;
    const bool lo_ok = (lo & 0x7fffu) == 0u || ((lo >> 7) & 0xffu) - 110u <= 32u;
    const bool hi_ok = (hi & 0x7fffu) == 0u || ((hi >> 7) & 0xffu) - 110u <= 32u;
    return lo_ok && hi_ok;
}

__device__ __forceinline__ unsigned half2_bits(float lo, float hi) {
    const __half2 h = __halves2half2(__float2half_rn(lo), __float2half_rn(hi));
    return *reinterpret_cast<const unsigned*>(&h);
}

// Exact for operands that passed bf16x2_exact_in_fp16.
__device__ __forceinline__ unsigned bf16x2_to_half2(unsigned x) {
    const float2 value = pre_sm80::bf16x2_bits_to_float2(x);
    return half2_bits(value.x, value.y);
}

__device__ __forceinline__ bool tf32_exact_in_fp16(unsigned x) {
    const unsigned v = x & 0x7fffe000u;
    return v == 0u || ((v >> 23) & 0xffu) - 113u <= 29u;
}

// The FP32 tile products of the rare route, out of line: inlining them at every MMA site multiplies
// the size of the kernels (and their ptxas time) several times over.
template <class AOperands, class BOperands>
static __device__ __noinline__ float4 mma_k16_fp32(float4 acc, unsigned a0, unsigned a1,
                                                   unsigned a2, unsigned a3, unsigned b0,
                                                   unsigned b1) {
    pre_sm80::mma_m16n8k16<AOperands, BOperands>(acc.x, acc.y, acc.z, acc.w, a0, a1, a2, a3, b0,
                                                 b1);
    return acc;
}

static __device__ __noinline__ float4 mma_tf32_fp32(float4 acc, unsigned a0, unsigned a1,
                                                    unsigned a2, unsigned a3, unsigned b0,
                                                    unsigned b1) {
    pre_sm80::mma_m16n8k8_tf32(acc.x, acc.y, acc.z, acc.w, a0, a1, a2, a3, b0, b1);
    return acc;
}

// The lane's outputs (rows g and g + 8, columns 2t and 2t + 1) whose own row of A or column of B
// holds an operand FP16 cannot represent take the FP32 tile product. The ballots say which pieces
// qualify: row g in quad g of `rows_lo`, row g + 8 in quad g of `rows_hi`, column n in quad n of
// `cols`. An output's route therefore depends on its own row and column alone, never on the other
// rows and columns that share its tile, so every tiling of a product gives the same bits.
__device__ __forceinline__ void take_inexact_outputs(float& c0, float& c1, float& c2, float& c3,
                                                     float4 exact, unsigned rows_lo,
                                                     unsigned rows_hi, unsigned cols) {
    const unsigned lane = pre_sm80::lane_id();
    const unsigned g = lane >> 2, t = lane & 3u;
    const auto quad = [](unsigned mask, unsigned q) { return ((mask >> (4u * q)) & 0xfu) == 0xfu; };
    const bool row_lo = quad(rows_lo, g), row_hi = quad(rows_hi, g);
    const bool col_0 = quad(cols, 2u * t), col_1 = quad(cols, 2u * t + 1u);
    if (!(row_lo && col_0)) { c0 = exact.x; }
    if (!(row_lo && col_1)) { c1 = exact.y; }
    if (!(row_hi && col_0)) { c2 = exact.z; }
    if (!(row_hi && col_1)) { c3 = exact.w; }
}

// A warp's B fragments of two k16 steps (four BF16 pairs a lane, as a groupwise small-T kernel
// loads them) as FP16 pairs in `half`, when every lane's values are FP16 values; false otherwise.
// With A operands that are FP16 values too (decoded integer codes), mma_f16 on these forms the
// bits mma_bf16 would, at a fraction of its cost.
__device__ __forceinline__ bool bf16_fragments_as_f16(uint4 bits, uint4& half) {
    if (!__all_sync(0xffffffffu, bf16x2_exact_in_fp16(bits.x) && bf16x2_exact_in_fp16(bits.y) &&
                                     bf16x2_exact_in_fp16(bits.z) &&
                                     bf16x2_exact_in_fp16(bits.w))) {
        return false;
    }
    half = make_uint4(bf16x2_to_half2(bits.x), bf16x2_to_half2(bits.y), bf16x2_to_half2(bits.z),
                      bf16x2_to_half2(bits.w));
    return true;
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
    mma_f16_k8(c0, c1, c2, c3, a0, a1, b0);
    mma_f16_k8(c0, c1, c2, c3, a2, a3, b1);
}

// FP16 accumulation: two k8 steps round the partial sum to FP16 once more than k16.
__device__ __forceinline__ void mma_f16_f16acc(unsigned& d0, unsigned& d1, unsigned a0, unsigned a1,
                                               unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    mma_f16_k8_f16acc(d0, d1, a0, a1, b0);
    mma_f16_k8_f16acc(d0, d1, a2, a3, b1);
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    mma_s8_k16(c0, c1, a0, b0);
    mma_s8_k16(c0, c1, a2, b1);
    mma_s8_k16(c2, c3, a1, b0);
    mma_s8_k16(c2, c3, a3, b1);
}

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    constexpr unsigned kAll = 0xffffffffu;
    const unsigned rows_lo =
        __ballot_sync(kAll, bf16x2_exact_in_fp16(a0) && bf16x2_exact_in_fp16(a2));
    const unsigned rows_hi =
        __ballot_sync(kAll, bf16x2_exact_in_fp16(a1) && bf16x2_exact_in_fp16(a3));
    const unsigned cols = __ballot_sync(kAll, bf16x2_exact_in_fp16(b0) && bf16x2_exact_in_fp16(b1));
    const float4 acc    = make_float4(c0, c1, c2, c3);
    mma_f16_k8(c0, c1, c2, c3, bf16x2_to_half2(a0), bf16x2_to_half2(a1), bf16x2_to_half2(b0));
    mma_f16_k8(c0, c1, c2, c3, bf16x2_to_half2(a2), bf16x2_to_half2(a3), bf16x2_to_half2(b1));
    if ((rows_lo & rows_hi & cols) != kAll) {
        take_inexact_outputs(c0, c1, c2, c3,
                             mma_k16_fp32<pre_sm80::Bf16Operands, pre_sm80::Bf16Operands>(
                                 acc, a0, a1, a2, a3, b0, b1),
                             rows_lo, rows_hi, cols);
    }
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
    constexpr unsigned kAll = 0xffffffffu;
    const unsigned rows_lo  = __ballot_sync(kAll, tf32_exact_in_fp16(a0) && tf32_exact_in_fp16(a2));
    const unsigned rows_hi  = __ballot_sync(kAll, tf32_exact_in_fp16(a1) && tf32_exact_in_fp16(a3));
    const unsigned cols     = __ballot_sync(kAll, tf32_exact_in_fp16(b0) && tf32_exact_in_fp16(b1));
    const float4 acc        = make_float4(c0, c1, c2, c3);
    // FP16 m16n8k8 holds k = 2*tig, 2*tig+1 where TF32 holds k = tig, tig+4. Pairing them the same
    // way in A and B permutes k alike on both sides, which leaves every dot product unchanged.
    using pre_sm80::tf32_operand;
    mma_f16_k8(c0, c1, c2, c3, half2_bits(tf32_operand(a0), tf32_operand(a2)),
               half2_bits(tf32_operand(a1), tf32_operand(a3)),
               half2_bits(tf32_operand(b0), tf32_operand(b1)));
    if ((rows_lo & rows_hi & cols) != kAll) {
        take_inexact_outputs(c0, c1, c2, c3, mma_tf32_fp32(acc, a0, a1, a2, a3, b0, b1), rows_lo,
                             rows_hi, cols);
    }
}

} // namespace ninfer::ops::sm75
#endif
