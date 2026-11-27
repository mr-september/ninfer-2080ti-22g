#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/fp8/fp8_a16_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_e4m3_scale.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::detail::unified {

__device__ __forceinline__ float decode_nvfp4_e4m3(std::uint8_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = static_cast<std::uint16_t>(storage) | (static_cast<std::uint16_t>(storage) << 8);
    return static_cast<float2>(value).x;
}

// The ue4m3 scale table for the A16 decode path, expanded from the same list the int8 routes use
// (nvfp4_e4m3_scale.h). Over the reachable domain 0x00..0x7E every entry is exactly half the vendor
// __nv_fp8x2_e4m3 value; both users below compensate with a power-of-two fold (2^15), so every
// product stays bit-identical.
__device__ constexpr float kNvfp4E4M3ScaleA16[256] = {
    NINFER_E4M3_SCALE_LIST(NINFER_E4M3_SCALE_VALUE)};

// The A16 decode reads an E2M1 pair and its group's E4M3 scale in units whose product the kernel
// rescales by kNvfp4A16CoefficientFold, a power of two, so every product is the same FP32 value.
// Blackwell converts both in one instruction each and keeps the vendor forms (fold 1). On older
// targets ptxas outlines both conversions into calls with BSSY/BSYNC bookkeeping, so they read the
// integer expansion below (e2m1 * 2^-14) and the ue4m3 table (e4m3 / 2), folded by 2^15.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
inline constexpr float kNvfp4A16CoefficientFold = 1.0F;

__device__ __forceinline__ float2 decode_nvfp4_e2m1x2_raw(std::uint8_t storage) {
    __nv_fp4x2_e2m1 value;
    value.__x = storage;
    return static_cast<float2>(value);
}

__device__ __forceinline__ float nvfp4_e4m3_scale_a16(std::uint8_t storage) {
    return decode_nvfp4_e4m3(storage);
}

__device__ __forceinline__ float2 decode_nvfp4_e2m1x2(std::uint8_t storage) {
    return decode_nvfp4_e2m1x2_raw(storage);
}
#else
inline constexpr float kNvfp4A16CoefficientFold = 32768.0F;

// The FP16 pattern carries the e2m1 significand in [11:9] and the sign in [15], and FP16
// subnormals line the exp==0 codes up on a common 2^-14 factor, so the conversion yields
// e2m1 * 2^-14 exactly.
__device__ __forceinline__ float2 decode_nvfp4_e2m1x2_raw(std::uint8_t storage) {
    const unsigned v = static_cast<unsigned>(storage) | (static_cast<unsigned>(storage) << 12);
    const unsigned r = ((v << 9) & 0x0E000E00u) | ((v & 0x00080008u) << 12);
    __half2 packed;
    packed.x = __ushort_as_half(static_cast<unsigned short>(r & 0xFFFFu));
    packed.y = __ushort_as_half(static_cast<unsigned short>(r >> 16));
    return __half22float2(packed);
}

__device__ __forceinline__ float nvfp4_e4m3_scale_a16(std::uint8_t storage) {
    return kNvfp4E4M3ScaleA16[storage];
}

// The expansion rescaled by 2^14, a power of two: the E2M1 values themselves.
__device__ __forceinline__ float2 decode_nvfp4_e2m1x2(std::uint8_t storage) {
    const float2 raw = decode_nvfp4_e2m1x2_raw(storage);
    float2 result;
    result.x = raw.x * 16384.0F;
    result.y = raw.y * 16384.0F;
    return result;
}
#endif

__device__ __forceinline__ unsigned nvfp4_scaled_pair_bf16(std::uint8_t code, std::uint8_t scale) {
#if NINFER_NATIVE_FP8_FP4_PAIR_CVT
    // E2M1 times a finite E4M3 scale needs at most six significand bits, so the BF16 product is
    // exact and equals the FP32 expansion below.
    unsigned values, multiplier, result;
    asm("{\n"
        ".reg .b8 packed;\n"
        "mov.b32 {packed, _, _, _}, %1;\n"
        "cvt.rn.bf16x2.e2m1x2 %0, packed;\n"
        "}\n"
        : "=r"(values)
        : "r"(static_cast<unsigned>(code)));
    const auto scale_pair = static_cast<std::uint16_t>(scale | (static_cast<unsigned>(scale) << 8));
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(multiplier) : "h"(scale_pair));
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(result) : "r"(values), "r"(multiplier));
    return result;
#else
    // E2M1 * E4M3 scale to a BF16 pair without either vendor conversion: the code side is the
    // raw bit-trick expansion (e2m1 * 2^-14), the scale side is the ue4m3 table (exactly
    // e4m3fn / 2 over the reachable domain) raised back by 2^15.  Both factors are power-of-two
    // rescalings of exact values, so the product rounds once to the same bits the vendor pair
    // produced (lut_bitcheck.cu is bit-exact over the full domain).
    const float2 values    = decode_nvfp4_e2m1x2_raw(code);
    const float multiplier = nvfp4_e4m3_scale_a16(scale) * kNvfp4A16CoefficientFold;

    union {
        __nv_bfloat162 pair;
        unsigned bits;
    } result;

    result.pair = __floats2bfloat162_rn(values.x * multiplier, values.y * multiplier);
    return result.bits;
#endif
}

// Eight elements at once: four code bytes under the one scale byte they share. The widening pass
// reads a 32-bit code word and one scale byte per sixteen columns anyway, so the scale expansion
// and its table lookup are hoisted out of four pairs here. Each pair still does exactly the
// arithmetic nvfp4_scaled_pair_bf16 does on the same inputs -- same inputs, same operations, same
// rounding -- so the output is bit-identical; the saving is the redundancy, not a change of math.
__device__ __forceinline__ void nvfp4_scaled_quad_bf16(unsigned codes, std::uint8_t scale,
                                                       unsigned out[4]) {
#if NINFER_NATIVE_FP8_FP4_PAIR_CVT
    const auto scale_pair = static_cast<std::uint16_t>(scale | (static_cast<unsigned>(scale) << 8));
    unsigned multiplier;
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(multiplier) : "h"(scale_pair));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned values, result;
        const unsigned code = (codes >> (8 * i)) & 0xFFu;
        asm("{\n"
            ".reg .b8 packed;\n"
            "mov.b32 {packed, _, _, _}, %1;\n"
            "cvt.rn.bf16x2.e2m1x2 %0, packed;\n"
            "}\n"
            : "=r"(values)
            : "r"(code));
        asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(result) : "r"(values), "r"(multiplier));
        out[i] = result;
    }
#else
    const float multiplier = nvfp4_e4m3_scale_a16(scale) * kNvfp4A16CoefficientFold;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 values =
            decode_nvfp4_e2m1x2_raw(static_cast<std::uint8_t>((codes >> (8 * i)) & 0xFFu));
        union {
            __nv_bfloat162 pair;
            unsigned bits;
        } result;
        result.pair = __floats2bfloat162_rn(values.x * multiplier, values.y * multiplier);
        out[i] = result.bits;
    }
#endif
}

__device__ __forceinline__ int nvfp4_a16_shared_col_64(int row, int col) {
    return col ^ ((row & 7) << 3);
}

__device__ __forceinline__ std::int64_t nvfp4_scale_byte_offset(int row, int group, int k) {
    return static_cast<std::int64_t>((row / 128) * (k / 64) + group / 4) * 512 + (row & 31) * 16 +
           ((row & 127) >> 5) * 4 + (group & 3);
}

struct alignas(8) Nvfp4QuantizedK16 {
    std::uint32_t codes_lo;
    std::uint32_t codes_hi;
    std::uint8_t scale;
};

static_assert(alignof(Nvfp4QuantizedK16) == 8);

__device__ __forceinline__ void
pack_nvfp4_e2m1x16(const float2 (&values)[8], std::uint32_t& codes_lo, std::uint32_t& codes_hi) {
#if defined(NINFER_PRE_BLACKWELL_DEVICE)
    // Blackwell's FP4 conversion; an image of another architecture in a build of several never
    // runs the routes that call it.
    codes_lo = codes_hi = 0;
    __trap();
#else
    asm volatile("{\n"
                 ".reg .b8 b0;\n"
                 ".reg .b8 b1;\n"
                 ".reg .b8 b2;\n"
                 ".reg .b8 b3;\n"
                 ".reg .b8 b4;\n"
                 ".reg .b8 b5;\n"
                 ".reg .b8 b6;\n"
                 ".reg .b8 b7;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16;\n"
                 "mov.b32 %0, {b0,b1,b2,b3};\n"
                 "mov.b32 %1, {b4,b5,b6,b7};\n"
                 "}\n"
                 : "=r"(codes_lo), "=r"(codes_hi)
                 : "f"(values[0].x), "f"(values[0].y), "f"(values[1].x), "f"(values[1].y),
                   "f"(values[2].x), "f"(values[2].y), "f"(values[3].x), "f"(values[3].y),
                   "f"(values[4].x), "f"(values[4].y), "f"(values[5].x), "f"(values[5].y),
                   "f"(values[6].x), "f"(values[6].y), "f"(values[7].x), "f"(values[7].y));
#endif
}

// Reciprocal replaces the sixteen per-value divisions with one reciprocal plus multiplies.
// Measured faster on latency-bound small grids (linear MMA route, T < 1024) and measurably
// slower on the throughput-bound TMA route, so callers select it explicitly; the default keeps
// historical behavior. The two formulations are algebraically identical and differ only in
// floating-point rounding (~1 ulp); the resulting output error stays within the A4 allowance.
template <bool Reciprocal = false>
__device__ __forceinline__ Nvfp4QuantizedK16 quantize_nvfp4_k16(const __nv_bfloat16* source,
                                                                float input_scale_divisor) {
    const uint4 packed0                = load_vec<uint4>(source);
    const uint4 packed1                = load_vec<uint4>(source + 8);
    const std::uint32_t represented[8] = {
        packed0.x, packed0.y, packed0.z, packed0.w, packed1.x, packed1.y, packed1.z, packed1.w,
    };

    float2 values[8];
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = bf16x2_bits_to_float2(represented[pair]);
        max_abs      = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs      = fmaxf(max_abs, fabsf(values[pair].y));
    }

    Nvfp4QuantizedK16 result{};
    const float scale_unencoded = __fdiv_rn(input_scale_divisor * max_abs, 6.0F);
    result.scale                = __nv_cvt_float_to_fp8(scale_unencoded, __NV_SATFINITE, __NV_E4M3);
    if (result.scale == 0) { return result; }

    const float decoded_scale = decode_nvfp4_e4m3(result.scale);
    if constexpr (Reciprocal) {
        const float gain = input_scale_divisor * __fdiv_rn(1.0F, decoded_scale);
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            values[pair].x = values[pair].x * gain;
            values[pair].y = values[pair].y * gain;
        }
    } else {
#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            values[pair].x = __fdiv_rn(values[pair].x * input_scale_divisor, decoded_scale);
            values[pair].y = __fdiv_rn(values[pair].y * input_scale_divisor, decoded_scale);
        }
    }
    pack_nvfp4_e2m1x16(values, result.codes_lo, result.codes_hi);
    return result;
}

} // namespace ninfer::ops::detail::unified
