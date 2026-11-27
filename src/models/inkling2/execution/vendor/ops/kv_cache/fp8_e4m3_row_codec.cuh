#pragma once

// E4M3FN row-scaled D256 KV-cache codec shared by standalone append and causal Attention.
// Persistent scales cross one FP16 represented-value boundary before codes are formed.

#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKVCacheFp8HeadDim        = 256;
inline constexpr int kKVCacheFp8Group          = 256;
inline constexpr int kKVCacheFp8Groups         = 1;
inline constexpr float kKVCacheFp8MaxFinite    = 448.0F;
inline constexpr float kKVCacheFp8ScaleMinimum = 0x1p-24F;
inline constexpr float kKVCacheFp8ScaleMaximum = 65504.0F;

__device__ __forceinline__ std::int64_t
kv_cache_fp8_code_index(int physical_page, int kv_head, int d, int page_offset, int kv_heads) {
    return paged_kv_element_offset<kKVCacheFp8HeadDim>(physical_page, kv_head, page_offset, d,
                                                       kv_heads);
}

__device__ __forceinline__ std::int64_t kv_cache_fp8_scale_index(int physical_page, int kv_head,
                                                                 int page_offset, int kv_heads) {
    return paged_kv_element_offset<kKVCacheFp8Groups>(physical_page, kv_head, page_offset, 0,
                                                      kv_heads);
}

__device__ __forceinline__ std::int64_t kv_cache_fp8_src_index(int kv_head, int d, int token,
                                                               int kv_heads) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kKVCacheFp8HeadDim) *
               (static_cast<std::int64_t>(kv_head) + static_cast<std::int64_t>(kv_heads) * token);
}

struct KVCacheFp8QuantParams {
    __half scale;
    float inverse_scale;
};

__device__ __forceinline__ KVCacheFp8QuantParams kv_cache_fp8_quant_params(float absmax) {
    if (absmax == 0.0F) { return {.scale = __float2half_rn(0.0F), .inverse_scale = 0.0F}; }
    const float raw_scale = absmax / kKVCacheFp8MaxFinite;
    const float bounded = fminf(kKVCacheFp8ScaleMaximum, fmaxf(kKVCacheFp8ScaleMinimum, raw_scale));
    const __half scale  = __float2half_rn(bounded);
    const float represented_scale = __half2float(scale);
    return {.scale = scale, .inverse_scale = 1.0F / represented_scale};
}

__device__ __forceinline__ std::uint8_t kv_cache_fp8_quant_code(float x, float inverse_scale) {
    if (inverse_scale == 0.0F) { return 0; }
    return __nv_cvt_float_to_fp8(x * inverse_scale, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ std::uint16_t kv_cache_fp8_quant_code2(float x0, float x1,
                                                                  float inverse_scale) {
    if (inverse_scale == 0.0F) { return 0; }
    return __nv_cvt_float2_to_fp8x2(make_float2(x0 * inverse_scale, x1 * inverse_scale),
                                    __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ __half2 kv_cache_fp8_code2_to_half2(std::uint16_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = storage;
    return static_cast<__half2>(value);
}

// Two E4M3 codes (the low 16 bits of `codes`) as FP16 values times 2^-8, by integer operations:
// moved into an FP16's exponent and mantissa fields, a code's bits are its value times 2^-8,
// subnormals included (E4M3 NaN, which the cache never stores, reads as 480). Multiplying by a
// scale times 2^8 then rounds once, exactly as the converted code times the scale does; this is
// for targets without a hardware E4M3 conversion.
__device__ __forceinline__ __half2 kv_cache_fp8_code2_to_half2_x2m8(unsigned codes) {
    const unsigned pair = __byte_perm(codes, 0u, 0x4140);
    const unsigned bits = ((pair & 0x007f007fu) << 7) | ((pair & 0x00800080u) << 8);
    return *reinterpret_cast<const __half2*>(&bits);
}

__device__ __forceinline__ __half2 kv_cache_fp8_dequant_code2_to_half2(std::uint16_t storage,
                                                                       __half scale) {
    // The integer expansion above carries the code times 2^-8 exactly, so the 2^8 that the vendor
    // conversion applies rides in a second half2 operand instead: scaling by a power of two is
    // exact, and the final multiply rounds once, exactly where the converted code times the scale
    // rounds. This is the sm70 path's form, applied where the vendor conversion is not free either:
    // sm_89 and newer convert E4M3 pairs in one instruction and keep it.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    return __hmul2(kv_cache_fp8_code2_to_half2(storage), __halves2half2(scale, scale));
#else
    const __half2 bias = __float2half2_rn(256.0F);
    return __hmul2(__hmul2(kv_cache_fp8_code2_to_half2_x2m8(storage), bias),
                   __halves2half2(scale, scale));
#endif
}

// Dequantizes 8 contiguous E4M3 codes (16 bytes) to FP16, for PV MMA operands on hardware with
// no FP8 tensor-core path (sm_86/sm_89): the row's scale is a single value shared by every code.
__device__ __forceinline__ int4 kv_cache_fp8_dequant_f16x8(const std::uint8_t* codes,
                                                           __half scale) {
    const int2 raw         = load_vec<int2>(codes);
    const std::uint16_t* c = reinterpret_cast<const std::uint16_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 value2 = kv_cache_fp8_dequant_code2_to_half2(c[i], scale);
        packed[i]            = *reinterpret_cast<const unsigned*>(&value2);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// Dequantizes 8 contiguous E4M3 codes (16 bytes) to BF16, for QK MMA operands on hardware with
// no FP8 tensor-core path: sm_86/sm_89 must run QK on ordinary BF16 Tensor Cores instead of the
// Blackwell-only mma.sync...kind::f8f6f4, so K needs a real (not just reinterpreted) wide copy.
__device__ __forceinline__ int4 kv_cache_fp8_dequant_bf16x8(const std::uint8_t* codes,
                                                             __half scale) {
    // The integer expansion carries the code times 2^-8 exactly, so the scale absorbs the 2^8
    // (exact, a power of two) and the product is the same FP32 value the vendor conversion feeds
    // the bf16 rounding. The K plane of k8v4 reads through here, and that path is issue-bound on
    // the outlined E4M3 conversion the same way the nvfp4 codec was. sm_89 and newer keep the
    // one-instruction vendor conversion.
    const int2 raw         = load_vec<int2>(codes);
    const std::uint16_t* c = reinterpret_cast<const std::uint16_t*>(&raw);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    const float scale_f = __half2float(scale);
#else
    const float scale_f = __half2float(scale) * 256.0F;
#endif
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
        __nv_fp8x2_e4m3 value;
        value.__x            = c[i];
        const float2 decoded = static_cast<float2>(value);
#else
        const float2 decoded = __half22float2(kv_cache_fp8_code2_to_half2_x2m8(c[i]));
#endif
        const __nv_bfloat162 value2 =
            __floats2bfloat162_rn(decoded.x * scale_f, decoded.y * scale_f);
        packed[i] = *reinterpret_cast<const unsigned*>(&value2);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

} // namespace ninfer::ops
