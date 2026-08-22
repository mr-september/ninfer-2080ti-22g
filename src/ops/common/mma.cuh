#pragma once

#include "core/arch_traits.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops {

__device__ __forceinline__ void ldmatrix_x1(unsigned& r0, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                 : "=r"(r0)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x1_t(unsigned& r0, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                 : "=r"(r0)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

// Convert a 32-bit register holding 2x BF16 values to 2x FP16 values.
__device__ __forceinline__ unsigned bf162_to_f162(unsigned b) {
    const auto* bf_ptr = reinterpret_cast<const __nv_bfloat162*>(&b);
    float2 f = __bfloat1622float2(*bf_ptr);
    half2 h  = __float22half2_rn(f);
    return *reinterpret_cast<const unsigned*>(&h);
}

// Turing-native FP16 MMA: 16x8x8
__device__ __forceinline__ void mma_f16_m16n8k8(float& c0, float& c1, float& c2, float& c3,
                                                unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
}

// Turing-native INT8 MMA: 8x8x16
__device__ __forceinline__ void mma_s8_m8n8k16(int& c0, int& c1, unsigned a0, unsigned b0) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(b0));
}

__device__ __forceinline__ float bf16_low_to_float(unsigned packed) {
    return __uint_as_float(packed << 16);
}
__device__ __forceinline__ float bf16_high_to_float(unsigned packed) {
    return __uint_as_float(packed & 0xffff0000U);
}

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
#if defined(NINFER_SM75)
    const int lane    = threadIdx.x & 31;
    const int r       = lane >> 2;
    const int t       = lane & 3;
    const int base_a  = 4 * r;
    const int base_b0 = 8 * t;
    const int base_b1 = 8 * t + 4;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

#pragma unroll
    for (int k_idx = 0; k_idx < 4; ++k_idx) {
        const unsigned reg_a0 = __shfl_sync(0xffffffff, a0, base_a + k_idx);
        const unsigned reg_a1 = __shfl_sync(0xffffffff, a1, base_a + k_idx);
        const unsigned reg_a2 = __shfl_sync(0xffffffff, a2, base_a + k_idx);
        const unsigned reg_a3 = __shfl_sync(0xffffffff, a3, base_a + k_idx);

        const unsigned reg_b0_n0 = __shfl_sync(0xffffffff, b0, base_b0 + k_idx);
        const unsigned reg_b1_n0 = __shfl_sync(0xffffffff, b1, base_b0 + k_idx);
        const unsigned reg_b0_n1 = __shfl_sync(0xffffffff, b0, base_b1 + k_idx);
        const unsigned reg_b1_n1 = __shfl_sync(0xffffffff, b1, base_b1 + k_idx);

        const float a_r_k0  = bf16_low_to_float(reg_a0);
        const float a_r_k1  = bf16_high_to_float(reg_a0);
        const float a_r8_k0 = bf16_low_to_float(reg_a1);
        const float a_r8_k1 = bf16_high_to_float(reg_a1);

        const float a_r_k8  = bf16_low_to_float(reg_a2);
        const float a_r_k9  = bf16_high_to_float(reg_a2);
        const float a_r8_k8 = bf16_low_to_float(reg_a3);
        const float a_r8_k9 = bf16_high_to_float(reg_a3);

        const float b_n0_k0 = bf16_low_to_float(reg_b0_n0);
        const float b_n0_k1 = bf16_high_to_float(reg_b0_n0);
        const float b_n0_k8 = bf16_low_to_float(reg_b1_n0);
        const float b_n0_k9 = bf16_high_to_float(reg_b1_n0);

        const float b_n1_k0 = bf16_low_to_float(reg_b0_n1);
        const float b_n1_k1 = bf16_high_to_float(reg_b0_n1);
        const float b_n1_k8 = bf16_low_to_float(reg_b1_n1);
        const float b_n1_k9 = bf16_high_to_float(reg_b1_n1);

        sum0 += a_r_k0  * b_n0_k0 + a_r_k1  * b_n0_k1 + a_r_k8  * b_n0_k8 + a_r_k9  * b_n0_k9;
        sum1 += a_r_k0  * b_n1_k0 + a_r_k1  * b_n1_k1 + a_r_k8  * b_n1_k8 + a_r_k9  * b_n1_k9;
        sum2 += a_r8_k0 * b_n0_k0 + a_r8_k1 * b_n0_k1 + a_r8_k8 * b_n0_k8 + a_r8_k9 * b_n0_k9;
        sum3 += a_r8_k0 * b_n1_k0 + a_r8_k1 * b_n1_k1 + a_r8_k8 * b_n1_k8 + a_r8_k9 * b_n1_k9;
    }

    c0 += sum0;
    c1 += sum1;
    c2 += sum2;
    c3 += sum3;
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
#if defined(NINFER_SM75)
    // Decompose m16n8k16 into two Turing m16n8k8 operations:
    // (A_k0: a0 [rows 0..7], a2 [rows 8..15]) x B_k0 [b0]
    // (A_k1: a1 [rows 0..7], a3 [rows 8..15]) x B_k1 [b1]
    mma_f16_m16n8k8(c0, c1, c2, c3, a0, a2, b0);
    mma_f16_m16n8k8(c0, c1, c2, c3, a1, a3, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if defined(NINFER_SM75)
    // Turing decomposition of m16n8k32 into 4x m8n8k16 operations:
    // (M0, K0), (M0, K1), (M1, K0), (M1, K1)
    mma_s8_m8n8k16(c0, c1, a0, b0);
    mma_s8_m8n8k16(c0, c1, a1, b1);
    mma_s8_m8n8k16(c2, c3, a2, b0);
    mma_s8_m8n8k16(c2, c3, a3, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
#if !defined(NINFER_SM75) && !defined(NINFER_SM86)
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    // Fallback if ever called on pre-sm_90
    (void)c0; (void)c1; (void)c2; (void)c3;
    (void)a0; (void)a1; (void)a2; (void)a3;
    (void)b0; (void)b1;
#endif
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
#if defined(NINFER_SM75)
    // On SM75 (Turing), TF32 Tensor Cores are unavailable.
    // We compute the exact m16n8k8 row.col FP32 tile via warp shuffles + SIMT FMAs.
    // Thread T (r = T / 4 in [0..7], t = T % 4 in [0..3]):
    // Target C registers: c0 = C(r, 2t), c1 = C(r, 2t+1), c2 = C(r+8, 2t), c3 = C(r+8, 2t+1)
    const int lane    = threadIdx.x & 31;
    const int r       = lane >> 2;
    const int t       = lane & 3;
    const int base_a  = 4 * r;
    const int base_b0 = 8 * t;
    const int base_b1 = 8 * t + 4;

    const float fa0 = __uint_as_float(a0);
    const float fa1 = __uint_as_float(a1);
    const float fa2 = __uint_as_float(a2);
    const float fa3 = __uint_as_float(a3);
    const float fb0 = __uint_as_float(b0);
    const float fb1 = __uint_as_float(b1);

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

#pragma unroll
    for (int k_idx = 0; k_idx < 4; ++k_idx) {
        const float a_r_k   = __shfl_sync(0xffffffff, fa0, base_a + k_idx);
        const float a_r8_k  = __shfl_sync(0xffffffff, fa1, base_a + k_idx);
        const float a_r_k4  = __shfl_sync(0xffffffff, fa2, base_a + k_idx);
        const float a_r8_k4 = __shfl_sync(0xffffffff, fa3, base_a + k_idx);

        const float b_n0_k  = __shfl_sync(0xffffffff, fb0, base_b0 + k_idx);
        const float b_n0_k4 = __shfl_sync(0xffffffff, fb1, base_b0 + k_idx);
        const float b_n1_k  = __shfl_sync(0xffffffff, fb0, base_b1 + k_idx);
        const float b_n1_k4 = __shfl_sync(0xffffffff, fb1, base_b1 + k_idx);

        sum0 += a_r_k  * b_n0_k  + a_r_k4  * b_n0_k4;
        sum1 += a_r_k  * b_n1_k  + a_r_k4  * b_n1_k4;
        sum2 += a_r8_k * b_n0_k  + a_r8_k4 * b_n0_k4;
        sum3 += a_r8_k * b_n1_k  + a_r8_k4 * b_n1_k4;
    }

    c0 += sum0;
    c1 += sum1;
    c2 += sum2;
    c3 += sum3;
#else
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}

__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
#if !defined(NINFER_SM75) && !defined(NINFER_SM86)
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
#else
    (void)c0; (void)c1; (void)c2; (void)c3;
    (void)a0; (void)a1; (void)a2; (void)a3;
    (void)b0; (void)b1; (void)sfa; (void)sfb;
#endif
}

} // namespace ninfer::ops
