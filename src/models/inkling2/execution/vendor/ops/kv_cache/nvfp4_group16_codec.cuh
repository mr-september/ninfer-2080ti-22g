#pragma once

// D256 KV-cache NVFP4 codec. Unlike weight artifacts, cache rows have no matrix-level divisor:
// every represented value is exactly E2M1(code) * E4M3(scale) for its contiguous G16 group.

#include "ops/kernel/paged_kv_address.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKVCacheNvfp4HeadDim        = 256;
inline constexpr int kKVCacheNvfp4Group          = 16;
inline constexpr int kKVCacheNvfp4Groups         = 16;
inline constexpr int kKVCacheNvfp4CodeBytes      = 128;
inline constexpr float kKVCacheNvfp4MaxFinite    = 6.0F;
inline constexpr float kKVCacheNvfp4ScaleMinimum = 0x1p-9F;
inline constexpr float kKVCacheNvfp4ScaleMaximum = 448.0F;

__device__ __forceinline__ std::int64_t
kv_cache_nvfp4_code_index(int physical_page, int kv_head, int d, int page_offset, int kv_heads) {
    return paged_kv_element_offset<kKVCacheNvfp4CodeBytes>(physical_page, kv_head, page_offset,
                                                           d >> 1, kv_heads);
}

__device__ __forceinline__ std::int64_t kv_cache_nvfp4_scale_index(int physical_page, int kv_head,
                                                                   int group, int page_offset,
                                                                   int kv_heads) {
    return paged_kv_element_offset<kKVCacheNvfp4Groups>(physical_page, kv_head, page_offset, group,
                                                        kv_heads);
}

__device__ __forceinline__ std::int64_t kv_cache_nvfp4_src_index(int kv_head, int d, int token,
                                                                 int kv_heads) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kKVCacheNvfp4HeadDim) *
               (static_cast<std::int64_t>(kv_head) + static_cast<std::int64_t>(kv_heads) * token);
}

struct KVCacheNvfp4QuantizedGroup16 {
    std::uint32_t codes_lo = 0;
    std::uint32_t codes_hi = 0;
    std::uint8_t scale     = 0;
};

__device__ __forceinline__ KVCacheNvfp4QuantizedGroup16
kv_cache_nvfp4_quantize_group16(const float* source) {
    float2 values[8];
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = make_float2(source[2 * pair], source[2 * pair + 1]);
        max_abs      = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs      = fmaxf(max_abs, fabsf(values[pair].y));
    }

    KVCacheNvfp4QuantizedGroup16 result{};
    if (max_abs == 0.0F) return result;

    const float raw_scale = __fdiv_rn(max_abs, kKVCacheNvfp4MaxFinite);
    const float bounded =
        fminf(kKVCacheNvfp4ScaleMaximum, fmaxf(kKVCacheNvfp4ScaleMinimum, raw_scale));
    result.scale                  = __nv_cvt_float_to_fp8(bounded, __NV_SATFINITE, __NV_E4M3);
    const float represented_scale = detail::decode_nvfp4_e4m3(result.scale);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair].x = __fdiv_rn(values[pair].x, represented_scale);
        values[pair].y = __fdiv_rn(values[pair].y, represented_scale);
    }
    detail::pack_nvfp4_e2m1x16(values, result.codes_lo, result.codes_hi);
    return result;
}

// Two E2M1 codes (the low byte of `codes`, low nibble first) as FP16 by integer operations: every
// E2M1 magnitude is an FP16 with a zero low byte, so a byte permute looks the high bytes up and
// the sign bits are moved in. This is for targets without a hardware E2M1 conversion.
__device__ __forceinline__ __half2 kv_cache_e2m1x2_to_half2(unsigned codes) {
    const unsigned select = ((codes & 0x7u) << 4) | ((codes & 0x70u) << 8);
    const unsigned bits   = __byte_perm(0x3E3C3800u, 0x46444240u, select) | ((codes & 0x8u) << 12) |
                            ((codes & 0x80u) << 24);
    return *reinterpret_cast<const __half2*>(&bits);
}

// The E2M1 pair and E4M3 scale conversions of the dequantizers below. Blackwell converts both in
// one instruction each and keeps the vendor forms; older targets take the integer expansions, which
// return the same values (see kv_cache_nvfp4_dequant_f16x8).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
__device__ __forceinline__ __half2 kv_cache_nvfp4_pair_to_half2(std::uint8_t byte) {
    __nv_fp4x2_e2m1 encoded;
    encoded.__x = byte;
    return static_cast<__half2>(encoded);
}

__device__ __forceinline__ float kv_cache_nvfp4_scale_to_float(std::uint8_t scale_code) {
    __nv_fp8_e4m3 encoded;
    encoded.__x = scale_code;
    return static_cast<float>(encoded);
}
#else
__device__ __forceinline__ __half2 kv_cache_nvfp4_pair_to_half2(std::uint8_t byte) {
    return kv_cache_e2m1x2_to_half2(byte);
}

__device__ __forceinline__ float kv_cache_nvfp4_scale_to_float(std::uint8_t scale_code) {
    return detail::kNvfp4E4M3ScaleA16[scale_code] * 2.0F;
}
#endif

__device__ __forceinline__ int4 kv_cache_nvfp4_dequant_f16x8(const std::uint8_t* codes,
                                                             std::uint8_t scale_code) {
    // Every finite E2M1 value times a legal nonnegative E4M3 cache scale is exactly representable
    // in FP16 (at most four product fraction bits and magnitude <= 2688). Half2 multiplication is
    // therefore the exact FP16 expansion boundary, not an additional approximation.
    //
    // Before Blackwell neither operand goes through the vendor conversions. On sm_86/sm_89 ptxas
    // outlines __nv_fp4x2_e2m1 and __nv_fp8_e4m3 into calls carrying BSSY/BSYNC and a divide
    // fixup, and this path is issue-bound on them: the nvfp4 prompt-attention kernel held 88
    // CALL.REL where its bf16 counterpart holds 3. kv_cache_e2m1x2_to_half2 above is the expansion
    // the sm70 path already runs; the scale comes from the ue4m3 table the weight codec reads,
    // whose entries are exactly half the vendor value, so doubling it is exact.
    //
    // Bit-exactness: an E2M1 magnitude is an FP16 with a zero low byte, so the permuted pattern is
    // the same value the vendor conversion returns, and both scale routes land on the same
    // representable half. The one input the two disagree on is a 0x7F scale byte, which the
    // quantizer cannot emit (it converts a value clamped into [2^-9, 448] with __NV_SATFINITE).
    const float scale_f  = kv_cache_nvfp4_scale_to_float(scale_code);
    const __half2 scale2 = __float2half2_rn(scale_f);
    const std::uint32_t packed = load_vec<std::uint32_t>(codes);
    const std::uint8_t* bytes  = reinterpret_cast<const std::uint8_t*>(&packed);
    unsigned half_bits[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const __half2 value = __hmul2(kv_cache_nvfp4_pair_to_half2(bytes[pair]), scale2);
        half_bits[pair]     = *reinterpret_cast<const unsigned*>(&value);
    }
    return make_int4(static_cast<int>(half_bits[0]), static_cast<int>(half_bits[1]),
                     static_cast<int>(half_bits[2]), static_cast<int>(half_bits[3]));
}

// BF16 counterpart of kv_cache_nvfp4_dequant_f16x8, for QK MMA operands on hardware with no FP4
// tensor-core path (sm_86/sm_89): the nvfp4-format prompt kernel dequantizes K to BF16 (matching
// Q's storage) instead of FP16.
__device__ __forceinline__ int4 kv_cache_nvfp4_dequant_bf16x8(const std::uint8_t* codes,
                                                              std::uint8_t scale_code) {
    // Same two substitutions as the FP16 form above, and the same reasoning: the E2M1 half2 the
    // sm70 path builds carries the exact value, so widening it to FP32 and multiplying by the
    // table-derived scale reproduces the vendor expression bit for bit.
    const float scale_f        = kv_cache_nvfp4_scale_to_float(scale_code);
    const std::uint32_t packed = load_vec<std::uint32_t>(codes);
    const std::uint8_t* bytes  = reinterpret_cast<const std::uint8_t*>(&packed);
    unsigned bf16_bits[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const float2 decoded = __half22float2(kv_cache_nvfp4_pair_to_half2(bytes[pair]));
        const __nv_bfloat162 value =
            __floats2bfloat162_rn(decoded.x * scale_f, decoded.y * scale_f);
        bf16_bits[pair] = *reinterpret_cast<const unsigned*>(&value);
    }
    return make_int4(static_cast<int>(bf16_bits[0]), static_cast<int>(bf16_bits[1]),
                     static_cast<int>(bf16_bits[2]), static_cast<int>(bf16_bits[3]));
}

struct KVCacheNvfp4DequantizedF16x16 {
    int4 lo;
    int4 hi;
};

__device__ __forceinline__ KVCacheNvfp4DequantizedF16x16
kv_cache_nvfp4_dequant_f16x16(const std::uint8_t* codes, std::uint8_t scale_code) {
    const float scale_f  = kv_cache_nvfp4_scale_to_float(scale_code);
    const __half2 scale2 = __float2half2_rn(scale_f);
    const int2 packed         = load_vec<int2>(codes);
    const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(&packed);
    unsigned half_bits[8];
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        const __half2 value = __hmul2(kv_cache_nvfp4_pair_to_half2(bytes[pair]), scale2);
        half_bits[pair]     = *reinterpret_cast<const unsigned*>(&value);
    }
    return {
        make_int4(static_cast<int>(half_bits[0]), static_cast<int>(half_bits[1]),
                  static_cast<int>(half_bits[2]), static_cast<int>(half_bits[3])),
        make_int4(static_cast<int>(half_bits[4]), static_cast<int>(half_bits[5]),
                  static_cast<int>(half_bits[6]), static_cast<int>(half_bits[7])),
    };
}

} // namespace ninfer::ops
