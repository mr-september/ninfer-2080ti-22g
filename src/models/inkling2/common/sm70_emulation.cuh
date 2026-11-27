#pragma once

// Volta (sm_70) has none of the Ampere warp primitives the kernels are written against: no
// cp.async, no ldmatrix, no m16n8 mma.sync shapes and no integer tensor cores. A 70 build keeps
// every kernel and replaces those primitives with the exact equivalents below and in pre_sm80.cuh,
// built from warp shuffles, shared-memory loads, FP32 FMA and dp4a. Each one reproduces the PTX
// fragment layout of the instruction it stands for, so the callers' register bookkeeping is
// unchanged. The hot prefill and attention paths of a 70 build take their own routes; these keep
// every other kernel correct. Turing (sm_75) has ldmatrix and the FP16/INT8 m16n8k8 and m8n8k16
// shapes, and a 75 build takes its own forms (sm75_mma.cuh).

#include "ops/common/pre_sm80.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750
#    define NINFER_SM70_DEVICE 1
#endif

namespace ninfer::ops::sm70 {

__device__ __forceinline__ unsigned lds32(unsigned addr) {
    unsigned value;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(value) : "r"(addr));
    return value;
}

__device__ __forceinline__ unsigned lds16(unsigned addr) {
    unsigned short value;
    asm volatile("ld.shared.b16 %0, [%1];" : "=h"(value) : "r"(addr));
    return value;
}

// ldmatrix.m8n8.b16: thread 8i+r names row r of matrix i; every thread receives, from matrix i,
// the 32-bit word at row lane/4, columns 2(lane%4) and 2(lane%4)+1.
template <int Matrices>
__device__ __forceinline__ void ldmatrix(unsigned (&r)[Matrices], unsigned addr) {
    const unsigned lane = pre_sm80::lane_id();
#pragma unroll
    for (int i = 0; i < Matrices; ++i) {
        const unsigned row = __shfl_sync(0xffffffffu, addr, (i << 3) | (lane >> 2));
        r[i]               = lds32(row + ((lane & 3u) << 2));
    }
}

// ldmatrix.m8n8.trans.b16: every thread receives, from matrix i, the elements at rows 2(lane%4)
// and 2(lane%4)+1 of column lane/4, low half first.
template <int Matrices>
__device__ __forceinline__ void ldmatrix_trans(unsigned (&r)[Matrices], unsigned addr) {
    const unsigned lane   = pre_sm80::lane_id();
    const unsigned column = (lane >> 2) << 1;
#pragma unroll
    for (int i = 0; i < Matrices; ++i) {
        const unsigned row0 = __shfl_sync(0xffffffffu, addr, (i << 3) | ((lane & 3u) << 1));
        const unsigned row1 = __shfl_sync(0xffffffffu, addr, (i << 3) | ((lane & 3u) << 1) | 1u);
        r[i]                = lds16(row0 + column) | (lds16(row1 + column) << 16);
    }
}

// mma.m16n8k16.row.col.f16.f16.f16.f16: the same product with FP16 accumulators, d0 holding the
// row-g pair and d1 the row-(g+8) pair. The emulation accumulates in FP32 and rounds once.
__device__ __forceinline__ void mma_m16n8k16_f16acc(unsigned& d0, unsigned& d1, unsigned a0,
                                                    unsigned a1, unsigned a2, unsigned a3,
                                                    unsigned b0, unsigned b1) {
    const float2 lo = pre_sm80::f16x2_bits_to_float2(d0);
    const float2 hi = pre_sm80::f16x2_bits_to_float2(d1);
    float c0 = lo.x, c1 = lo.y, c2 = hi.x, c3 = hi.y;
    pre_sm80::mma_m16n8k16<pre_sm80::F16Operands>(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    const __half2 out_lo = __floats2half2_rn(c0, c1);
    const __half2 out_hi = __floats2half2_rn(c2, c3);
    d0                   = *reinterpret_cast<const unsigned*>(&out_lo);
    d1                   = *reinterpret_cast<const unsigned*>(&out_hi);
}

// mma.m16n8k8.row.col.f32.f16.f16.f32: thread (g, t) holds A[g][2t..], A[g+8][2t..] and
// B[2t..][g]; the accumulators match the k16 shape.
__device__ __forceinline__ void mma_m16n8k8_f16(float& c0, float& c1, float& c2, float& c3,
                                                unsigned a0, unsigned a1, unsigned b0) {
    const unsigned lane = pre_sm80::lane_id();
    const unsigned quad = lane & ~3u;
    const unsigned col0 = (lane & 3u) << 3;
#pragma unroll 1
    for (unsigned j = 0; j < 4; ++j) {
        const float2 a_g  = pre_sm80::f16x2_bits_to_float2(__shfl_sync(0xffffffffu, a0, quad | j));
        const float2 a_g8 = pre_sm80::f16x2_bits_to_float2(__shfl_sync(0xffffffffu, a1, quad | j));
        const float2 b_n0 = pre_sm80::f16x2_bits_to_float2(__shfl_sync(0xffffffffu, b0, col0 | j));
        const float2 b_n1 =
            pre_sm80::f16x2_bits_to_float2(__shfl_sync(0xffffffffu, b0, (col0 + 4u) | j));
        c0                = fmaf(a_g.y, b_n0.y, fmaf(a_g.x, b_n0.x, c0));
        c1                = fmaf(a_g.y, b_n1.y, fmaf(a_g.x, b_n1.x, c1));
        c2                = fmaf(a_g8.y, b_n0.y, fmaf(a_g8.x, b_n0.x, c2));
        c3                = fmaf(a_g8.y, b_n1.y, fmaf(a_g8.x, b_n1.x, c3));
    }
}

// mma.m16n8k32.row.col.s32.s8.s8.s32: the k16 layout with four int8 codes per register, so every
// pass covers k = 4j..4j+3 and 4j+16..4j+19 through dp4a. Integer accumulation is exact.
__device__ __forceinline__ void mma_m16n8k32_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0,
                                                unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                                unsigned b1) {
    const unsigned lane = pre_sm80::lane_id();
    const unsigned quad = lane & ~3u;
    const unsigned col0 = (lane & 3u) << 3;
#pragma unroll 1
    for (unsigned j = 0; j < 4; ++j) {
        const int a_g_lo  = static_cast<int>(__shfl_sync(0xffffffffu, a0, quad | j));
        const int a_g8_lo = static_cast<int>(__shfl_sync(0xffffffffu, a1, quad | j));
        const int a_g_hi  = static_cast<int>(__shfl_sync(0xffffffffu, a2, quad | j));
        const int a_g8_hi = static_cast<int>(__shfl_sync(0xffffffffu, a3, quad | j));
        const int b_n0_lo = static_cast<int>(__shfl_sync(0xffffffffu, b0, col0 | j));
        const int b_n0_hi = static_cast<int>(__shfl_sync(0xffffffffu, b1, col0 | j));
        const int b_n1_lo = static_cast<int>(__shfl_sync(0xffffffffu, b0, (col0 + 4u) | j));
        const int b_n1_hi = static_cast<int>(__shfl_sync(0xffffffffu, b1, (col0 + 4u) | j));
        c0                = __dp4a(a_g_hi, b_n0_hi, __dp4a(a_g_lo, b_n0_lo, c0));
        c1                = __dp4a(a_g_hi, b_n1_hi, __dp4a(a_g_lo, b_n1_lo, c1));
        c2                = __dp4a(a_g8_hi, b_n0_hi, __dp4a(a_g8_lo, b_n0_lo, c2));
        c3                = __dp4a(a_g8_hi, b_n1_hi, __dp4a(a_g8_lo, b_n1_lo, c3));
    }
}

// mma.m16n8k16.row.col.s32.s8.s8.s32: thread (g, t) holds A[g][4t..4t+3], A[g+8][4t..] and
// B[4t..4t+3][g].
__device__ __forceinline__ void mma_m16n8k16_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0,
                                                unsigned a1, unsigned b0) {
    const unsigned lane = pre_sm80::lane_id();
    const unsigned quad = lane & ~3u;
    const unsigned col0 = (lane & 3u) << 3;
#pragma unroll 1
    for (unsigned j = 0; j < 4; ++j) {
        const int a_g  = static_cast<int>(__shfl_sync(0xffffffffu, a0, quad | j));
        const int a_g8 = static_cast<int>(__shfl_sync(0xffffffffu, a1, quad | j));
        const int b_n0 = static_cast<int>(__shfl_sync(0xffffffffu, b0, col0 | j));
        const int b_n1 = static_cast<int>(__shfl_sync(0xffffffffu, b0, (col0 + 4u) | j));
        c0             = __dp4a(a_g, b_n0, c0);
        c1             = __dp4a(a_g, b_n1, c1);
        c2             = __dp4a(a_g8, b_n0, c2);
        c3             = __dp4a(a_g8, b_n1, c3);
    }
}

} // namespace ninfer::ops::sm70
