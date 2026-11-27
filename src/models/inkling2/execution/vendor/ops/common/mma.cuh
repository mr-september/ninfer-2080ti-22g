#pragma once

#include "ops/common/memory.cuh"
#include "ops/common/sm75_mma.cuh"

namespace ninfer::ops {

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
#if defined(NINFER_SM70_DEVICE)
    unsigned r[2];
    sm70::ldmatrix<2>(r, addr);
    r0 = r[0];
    r1 = r[1];
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
#endif
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
#if defined(NINFER_SM70_DEVICE)
    unsigned r[4];
    sm70::ldmatrix<4>(r, addr);
    r0 = r[0];
    r1 = r[1];
    r2 = r[2];
    r3 = r[3];
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
#endif
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
#if defined(NINFER_SM70_DEVICE)
    unsigned r[2];
    sm70::ldmatrix_trans<2>(r, addr);
    r0 = r[0];
    r1 = r[1];
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
#endif
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned addr) {
#if defined(NINFER_SM70_DEVICE)
    unsigned r[4];
    sm70::ldmatrix_trans<4>(r, addr);
    r0 = r[0];
    r1 = r[1];
    r2 = r[2];
    r3 = r[3];
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
#endif
}

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
#if defined(NINFER_SM70_DEVICE)
    pre_sm80::mma_m16n8k16<pre_sm80::Bf16Operands>(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#elif defined(NINFER_SM75_DEVICE)
    sm75::mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
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
#if defined(NINFER_SM70_DEVICE)
    pre_sm80::mma_m16n8k16<pre_sm80::F16Operands>(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#elif defined(NINFER_SM75_DEVICE)
    sm75::mma_f16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// FP16 accumulation: twice the FP32-accumulator rate on GeForce parts. d0 holds the row-gid pair
// and d1 the row-gid+8 pair of the fragment, in the same order as mma_f16's c0..c3. Turing sums the
// fragment as two k8 steps, which round the partial sum to FP16 once more.
__device__ __forceinline__ void mma_f16_f16acc(unsigned& d0, unsigned& d1, unsigned a0,
                                               unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                               unsigned b1) {
#if defined(NINFER_SM70_DEVICE)
    sm70::mma_m16n8k16_f16acc(d0, d1, a0, a1, a2, a3, b0, b1);
#elif defined(NINFER_SM75_DEVICE)
    sm75::mma_f16_f16acc(d0, d1, a0, a1, a2, a3, b0, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(d0), "+r"(d1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if defined(NINFER_SM70_DEVICE)
    sm70::mma_m16n8k32_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#elif defined(NINFER_SM75_DEVICE)
    sm75::mma_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// On RTX 5090, block-scaled FP8 avoids the plain form's half-rate FP32 accumulation.
// UE8M0 0x7f is exactly 1: preserve the raw E4M3 dot product and keep the callers'
// existing row scales in FP32 after the MMA. No operand repacking is needed.
__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
#if defined(NINFER_PRE_BLACKWELL_DEVICE)
    __trap();
#else
    constexpr unsigned kUnitScale8          = 0x7Fu;
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
                 "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                 "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(kUnitScale8),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(kUnitScale8), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
#endif
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
#if defined(NINFER_SM70_DEVICE)
    pre_sm80::mma_m16n8k8_tf32(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#elif defined(NINFER_SM75_DEVICE)
    sm75::mma_tf32_bits(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
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
#if defined(NINFER_PRE_BLACKWELL_DEVICE)
    __trap();
#else
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
#endif
}

} // namespace ninfer::ops
