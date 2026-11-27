#pragma once

// Exact warp-level forms of sm_80 instructions that neither Volta (sm_70) nor Turing (sm_75) has:
// cp.async as a plain copy, redux.sync as a shuffle butterfly, and FP32 products of BF16, FP16 and
// TF32 m16n8 fragments formed from shuffles and FMA in the PTX fragment layouts. A 70 build runs
// its tensor-core shapes on them (sm70_emulation.cuh); a 75 build keeps its native FP16 and INT8
// shapes and takes these where an operand is not an FP16 value (sm75_mma.cuh). Each product's
// summation order is fixed, so both builds form the same bits from the same operands.

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
// Device code without cp.async, redux.sync, BF16/TF32 MMA and the packed conversions.
#    define NINFER_PRE_SM80_DEVICE 1
#endif

// Turing device code: the Turing forms of the sm_80 instructions (sm75_mma.cuh) replace them. A
// build of several architectures selects them in its Turing image alone, so its other images keep
// their own instructions (ops/common/device_tier.h).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 800
#    define NINFER_SM75_DEVICE 1
#endif

// Device code without Blackwell's block-scaled FP8 and FP4 tensor-core instructions, TMA and the
// mbarrier transaction forms. A build of several architectures compiles a kernel built on them into
// every image; the host launches it only on a card that runs the 120a code
// (ops/common/device_tier.h), and the other images hold a trap in its place.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
#    define NINFER_PRE_BLACKWELL_DEVICE 1
#endif

// Device code that has Turing's FP16 m16n8k8 and INT8 m8n8k16 MMA (every image but sm_70's), and
// the host pass. The Turing forms (sm75_mma.cuh) are defined there, so a Turing kernel variant
// compiles in every image that may hold it; a Volta image holds none.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
#    define NINFER_SM75_FORMS 1
#endif

namespace ninfer::ops::pre_sm80 {

__device__ __forceinline__ unsigned lane_id() {
    unsigned lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}

__device__ __forceinline__ float2 bf16x2_bits_to_float2(unsigned bits) {
    return make_float2(__uint_as_float(bits << 16), __uint_as_float(bits & 0xffff0000u));
}

__device__ __forceinline__ float2 f16x2_bits_to_float2(unsigned bits) {
    __half2 value;
    *reinterpret_cast<unsigned*>(&value) = bits;
    return __half22float2(value);
}

struct Bf16Operands {
    __device__ __forceinline__ static float2 widen(unsigned bits) {
        return bf16x2_bits_to_float2(bits);
    }
};

struct F16Operands {
    __device__ __forceinline__ static float2 widen(unsigned bits) {
        return f16x2_bits_to_float2(bits);
    }
};

// mma.m16n8k16.row.col with FP32 accumulators, A and B widened by their own operand types. Thread
// (g = lane/4, t = lane%4) holds A[g][2t..], A[g+8][2t..], A[g][2t+8..], A[g+8][2t+8..],
// B[2t..][g], B[2t+8..][g] and the accumulators C[g][2t..2t+1], C[g+8][2t..2t+1]. Row g of A lives
// in quad g; column n of B in quad n. Each pass j gathers lane j of the thread's own quad (its two
// A rows) and lane j of quads 2t and 2t+1 (its two B columns), which together cover k = 2j, 2j+1,
// 2j+8 and 2j+9.
template <class AOperands, class BOperands = AOperands>
__device__ __forceinline__ void mma_m16n8k16(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    const unsigned lane = lane_id();
    const unsigned quad = lane & ~3u;
    const unsigned col0 = (lane & 3u) << 3;
#pragma unroll 1
    for (unsigned j = 0; j < 4; ++j) {
        const float2 a_g_lo  = AOperands::widen(__shfl_sync(0xffffffffu, a0, quad | j));
        const float2 a_g8_lo = AOperands::widen(__shfl_sync(0xffffffffu, a1, quad | j));
        const float2 a_g_hi  = AOperands::widen(__shfl_sync(0xffffffffu, a2, quad | j));
        const float2 a_g8_hi = AOperands::widen(__shfl_sync(0xffffffffu, a3, quad | j));
        const float2 b_n0_lo = BOperands::widen(__shfl_sync(0xffffffffu, b0, col0 | j));
        const float2 b_n0_hi = BOperands::widen(__shfl_sync(0xffffffffu, b1, col0 | j));
        const float2 b_n1_lo = BOperands::widen(__shfl_sync(0xffffffffu, b0, (col0 + 4u) | j));
        const float2 b_n1_hi = BOperands::widen(__shfl_sync(0xffffffffu, b1, (col0 + 4u) | j));
        c0                   = fmaf(a_g_lo.x, b_n0_lo.x, c0);
        c0                   = fmaf(a_g_lo.y, b_n0_lo.y, c0);
        c0                   = fmaf(a_g_hi.x, b_n0_hi.x, c0);
        c0                   = fmaf(a_g_hi.y, b_n0_hi.y, c0);
        c1                   = fmaf(a_g_lo.x, b_n1_lo.x, c1);
        c1                   = fmaf(a_g_lo.y, b_n1_lo.y, c1);
        c1                   = fmaf(a_g_hi.x, b_n1_hi.x, c1);
        c1                   = fmaf(a_g_hi.y, b_n1_hi.y, c1);
        c2                   = fmaf(a_g8_lo.x, b_n0_lo.x, c2);
        c2                   = fmaf(a_g8_lo.y, b_n0_lo.y, c2);
        c2                   = fmaf(a_g8_hi.x, b_n0_hi.x, c2);
        c2                   = fmaf(a_g8_hi.y, b_n0_hi.y, c2);
        c3                   = fmaf(a_g8_lo.x, b_n1_lo.x, c3);
        c3                   = fmaf(a_g8_lo.y, b_n1_lo.y, c3);
        c3                   = fmaf(a_g8_hi.x, b_n1_hi.x, c3);
        c3                   = fmaf(a_g8_hi.y, b_n1_hi.y, c3);
    }
}

// mma.m16n8k8.row.col.f32.tf32.tf32.f32: thread (g, t) holds A[g][t], A[g+8][t], A[g][t+4],
// A[g+8][t+4], B[t][g] and B[t+4][g]. The tensor core ignores the low 13 mantissa bits of each
// operand, which these forms reproduce by masking them.
__device__ __forceinline__ float tf32_operand(unsigned bits) {
    return __uint_as_float(bits & 0xffffe000u);
}

__device__ __forceinline__ void mma_m16n8k8_tf32(float& c0, float& c1, float& c2, float& c3,
                                                 unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                                 unsigned b0, unsigned b1) {
    const unsigned lane = lane_id();
    const unsigned quad = lane & ~3u;
    const unsigned col0 = (lane & 3u) << 3;
#pragma unroll 1
    for (unsigned j = 0; j < 4; ++j) {
        const float a_g_lo  = tf32_operand(__shfl_sync(0xffffffffu, a0, quad | j));
        const float a_g8_lo = tf32_operand(__shfl_sync(0xffffffffu, a1, quad | j));
        const float a_g_hi  = tf32_operand(__shfl_sync(0xffffffffu, a2, quad | j));
        const float a_g8_hi = tf32_operand(__shfl_sync(0xffffffffu, a3, quad | j));
        const float b_n0_lo = tf32_operand(__shfl_sync(0xffffffffu, b0, col0 | j));
        const float b_n0_hi = tf32_operand(__shfl_sync(0xffffffffu, b1, col0 | j));
        const float b_n1_lo = tf32_operand(__shfl_sync(0xffffffffu, b0, (col0 + 4u) | j));
        const float b_n1_hi = tf32_operand(__shfl_sync(0xffffffffu, b1, (col0 + 4u) | j));
        c0                  = fmaf(a_g_hi, b_n0_hi, fmaf(a_g_lo, b_n0_lo, c0));
        c1                  = fmaf(a_g_hi, b_n1_hi, fmaf(a_g_lo, b_n1_lo, c1));
        c2                  = fmaf(a_g8_hi, b_n0_hi, fmaf(a_g8_lo, b_n0_lo, c2));
        c3                  = fmaf(a_g8_hi, b_n1_hi, fmaf(a_g8_lo, b_n1_lo, c3));
    }
}

// __reduce_max_sync (redux.sync, sm_80) over the lanes 0..Width-1 named by mask, as a butterfly.
template <int Width>
__device__ __forceinline__ unsigned reduce_max(unsigned mask, unsigned value) {
    static_assert(Width > 0 && Width <= 32 && (Width & (Width - 1)) == 0);
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        const unsigned other = __shfl_xor_sync(mask, value, offset, Width);
        value                = other > value ? other : value;
    }
    return value;
}

// cp.async with its zero fill, as a plain copy: the data is in shared memory when the call
// returns, which satisfies every later wait_group, and a buffer is never written while a reader
// still needs it (an asynchronous copy could land at any time after it was issued), so the callers'
// barriers order these stores the same way.
template <int Bytes, bool BypassL1>
__device__ __forceinline__ void copy_to_shared(void* smem_dst, const void* gmem_src) {
    if constexpr (Bytes == 16) {
        *static_cast<int4*>(smem_dst) = BypassL1 ? __ldcg(static_cast<const int4*>(gmem_src))
                                                 : *static_cast<const int4*>(gmem_src);
    } else if constexpr (Bytes == 8) {
        *static_cast<int2*>(smem_dst) = *static_cast<const int2*>(gmem_src);
    } else {
        static_assert(Bytes == 4);
        *static_cast<int*>(smem_dst) = *static_cast<const int*>(gmem_src);
    }
}

// Reads only src_bytes and zero-fills the rest; with none it reads nothing, which callers rely on
// to pass an address past the end of their tensor.
template <int Bytes, bool BypassL1>
__device__ __forceinline__ void copy_to_shared_zfill(void* smem_dst, const void* gmem_src,
                                                     int src_bytes) {
    if (src_bytes >= Bytes) {
        copy_to_shared<Bytes, BypassL1>(smem_dst, gmem_src);
        return;
    }
    unsigned char* dst       = static_cast<unsigned char*>(smem_dst);
    const unsigned char* src = static_cast<const unsigned char*>(gmem_src);
#pragma unroll
    for (int i = 0; i < Bytes; ++i) { dst[i] = i < src_bytes ? src[i] : 0; }
}

} // namespace ninfer::ops::pre_sm80
