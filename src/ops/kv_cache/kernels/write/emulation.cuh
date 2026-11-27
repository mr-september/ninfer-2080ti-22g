#pragma once

// EXPERIMENT ONLY (.agent-tmp/kv-ngram-eval, not for commit). Emulates ninfer-all's rk8v4 and
// rk4v4 KV codings inside this tree's INT8 G64 cache so their quality can be measured with the
// existing INT8 attention kernels. Compiled only when NINFER_KV_EMULATION is defined and > 0.
//   1 = rk8v4: keys as INT8 (unchanged), values signed int4 G32 (FP16 absmax/7).
//   2 = rk4v4: keys Lloyd-Max 4-bit G64 (exact: its codes are INT8 values under an FP16 scale),
//      values as in rk8v4.
// Values: the int4-decoded value q*s is written on an INT8 G64 grid with scale S = s_max/16, which
// is exact for the half-group holding the larger scale and adds a second rounding to the other
// half, so the emulated value error is never smaller than the real coding's.
// Lloyd-Max encoder ported from ninfer-all src/ops/kv_cache/int8_g64_codec.cuh at b12331ca
// (Apache-2.0), with the same explicit rounding intrinsics.

#if defined(NINFER_KV_EMULATION) && NINFER_KV_EMULATION > 0

#include "ops/common/cuda/reduce.cuh"

#include <cuda_fp16.h>
#include <cstdint>

namespace ninfer::ops::detail::kv::emulation {

inline constexpr int kMode = NINFER_KV_EMULATION;
static_assert(kMode == 1 || kMode == 2, "NINFER_KV_EMULATION must be 0, 1 or 2");

__device__ __forceinline__ int lloyd4_magnitude(float z) {
    const float a = fabsf(z);
    int m         = 0;
    m += a >= 0.25825f;
    m += a >= 0.52245f;
    m += a >= 0.79960f;
    m += a >= 1.09930f;
    m += a >= 1.43715f;
    m += a >= 1.84355f;
    m += a >= 2.40080f;
    return m;
}

__device__ __forceinline__ int lloyd4_code(int index) {
    constexpr int codes[8] = {6, 18, 31, 44, 58, 75, 96, 127};
    return (index & 8) != 0 ? -codes[index & 7] : codes[index & 7];
}

struct Codes {
    std::int8_t c0;
    std::int8_t c1;
    __half scale;
};

// One rotated G64 key group per warp, lane owns dimensions 64g+lane and 64g+lane+32.
__device__ __forceinline__ Codes lloyd4_key_group(float x0, float x1) {
    constexpr unsigned FullMask = 0xffffffffu;
    float sum_sq                = __fadd_rn(__fmul_rn(x0, x0), __fmul_rn(x1, x1));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq = __fadd_rn(sum_sq, __shfl_xor_sync(FullMask, sum_sq, offset));
    }
    const float rms     = __fsqrt_rn(__fmul_rn(sum_sq, 1.0f / 64.0f));
    const float inv_rms = rms > 0.0f ? __fdiv_rn(1.0f, rms) : 0.0f;
    const int index0    = (x0 < 0.0f ? 8 : 0) | lloyd4_magnitude(__fmul_rn(x0, inv_rms));
    const int index1    = (x1 < 0.0f ? 8 : 0) | lloyd4_magnitude(__fmul_rn(x1, inv_rms));
    const float c0      = static_cast<float>(lloyd4_code(index0));
    const float c1      = static_cast<float>(lloyd4_code(index1));
    float xc            = __fadd_rn(__fmul_rn(x0, c0), __fmul_rn(x1, c1));
    float cc            = __fadd_rn(__fmul_rn(c0, c0), __fmul_rn(c1, c1));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        xc = __fadd_rn(xc, __shfl_xor_sync(FullMask, xc, offset));
        cc = __fadd_rn(cc, __shfl_xor_sync(FullMask, cc, offset));
    }
    return {static_cast<std::int8_t>(c0), static_cast<std::int8_t>(c1),
            __float2half_rn(rms > 0.0f ? __fdiv_rn(xc, cc) : 0.0f)};
}

__device__ __forceinline__ float int4_decode(float x, float& scale_out) {
    constexpr unsigned FullMask = 0xffffffffu;
    const float absmax          = common::warp_max(fabsf(x), FullMask);
    const __half scale          = __float2half_rn(absmax > 0.0f ? absmax / 7.0f : 0.0f);
    const float s               = __half2float(scale);
    scale_out                   = s;
    if (s <= 0.0f) { return 0.0f; }
    int q = __float2int_rn(x * (1.0f / s));
    q     = max(-7, min(7, q));
    return static_cast<float>(q) * s;
}

// One G64 value group per warp: v0 lanes are the int4 coding's first G32 group, v1 the second.
__device__ __forceinline__ Codes int4_value_group(float v0, float v1) {
    float s_lo = 0.0f;
    float s_hi = 0.0f;
    const float d0   = int4_decode(v0, s_lo);
    const float d1   = int4_decode(v1, s_hi);
    const __half S   = __float2half_rn(fmaxf(s_lo, s_hi) * 0.0625f);
    const float sf   = __half2float(S);
    const float inv  = sf > 0.0f ? 1.0f / sf : 0.0f;
    const int q0     = inv == 0.0f ? 0 : max(-127, min(127, __float2int_rn(d0 * inv)));
    const int q1     = inv == 0.0f ? 0 : max(-127, min(127, __float2int_rn(d1 * inv)));
    return {static_cast<std::int8_t>(q0), static_cast<std::int8_t>(q1), S};
}

} // namespace ninfer::ops::detail::kv::emulation

#endif
