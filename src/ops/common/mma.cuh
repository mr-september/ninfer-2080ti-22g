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

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
#if defined(NINFER_SM75)
    // Emulate BF16 m16n8k16 via two Turing FP16 m16n8k8 operations:
    // (A_k0: a0 [rows 0..7], a2 [rows 8..15]) x B_k0 [b0]
    // (A_k1: a1 [rows 0..7], a3 [rows 8..15]) x B_k1 [b1]
    unsigned fa0 = bf162_to_f162(a0);
    unsigned fa2 = bf162_to_f162(a2);
    unsigned fb0 = bf162_to_f162(b0);
    mma_f16_m16n8k8(c0, c1, c2, c3, fa0, fa2, fb0);

    unsigned fa1 = bf162_to_f162(a1);
    unsigned fa3 = bf162_to_f162(a3);
    unsigned fb1 = bf162_to_f162(b1);
    mma_f16_m16n8k8(c0, c1, c2, c3, fa1, fa3, fb1);
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
    // Convert float bits to half2 and compute via m16n8k8 FP16 TC.
    // fa0: row 0 (k=t),   fa2: row 0 (k=t+4)
    // fa1: row 1 (k=t),   fa3: row 1 (k=t+4)
    // fb0: col (k=t),     fb1: col (k=t+4)
    float fa0 = __uint_as_float(a0);
    float fa1 = __uint_as_float(a1);
    float fa2 = __uint_as_float(a2);
    float fa3 = __uint_as_float(a3);
    float fb0 = __uint_as_float(b0);
    float fb1 = __uint_as_float(b1);

    half2 ha0 = __floats2half2_rn(fa0, fa2);
    half2 ha1 = __floats2half2_rn(fa1, fa3);
    half2 hb0 = __floats2half2_rn(fb0, fb1);

    unsigned ua0 = *reinterpret_cast<const unsigned*>(&ha0);
    unsigned ua1 = *reinterpret_cast<const unsigned*>(&ha1);
    unsigned ub0 = *reinterpret_cast<const unsigned*>(&hb0);

    mma_f16_m16n8k8(c0, c1, c2, c3, ua0, ua1, ub0);
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
