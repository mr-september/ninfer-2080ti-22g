#pragma once

// Device side of the KV row codings (kv_row.h): a warp encoding or decoding one D-wide block of a
// row (one KV head of one position), and the eight-value chunk reads an attention's tile loader
// makes. Chunk c (dimensions [8c, 8c + 8)) of a row has its codes at byte c * bytes-per-chunk and
// its scale at index c / chunks-per-group of the row's scales, whatever block it is in; where the
// codes and scales lie is the family's addressing.

#include "ops/common/warp.cuh"
#include "ops/kv_cache/e8_root_codec.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/kv_cache/kv_row.h"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::kv_row {

// Eight F16 values as one 16-byte word.
__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
    uint4 w;
    __half2* h = reinterpret_cast<__half2*>(&w);
#pragma unroll
    for (int j = 0; j < 4; ++j) { h[j] = __floats2half2_rn(f[2 * j], f[2 * j + 1]); }
    return w;
}

__device__ __forceinline__ void int8x8_to_float(uint2 raw, float s, float (&f)[8]) {
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) { f[j] = float(c[j]) * s; }
}

// Signed nibbles, dimension order (low nibble first).
__device__ __forceinline__ void int4x8_to_float(unsigned raw, float s, float (&f)[8]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int nibble = int((raw >> (4 * j)) & 0xfu);
        f[j]             = float((nibble ^ 8) - 8) * s;
    }
}

// MXFP8, the release's rule (vLLM's mxfp8_quantize for Inkling 2): a group of 32 values has the
// scale 2^(e - 127), e the exponent of amax / 448 rounded up to a power of two (0 when amax / 448
// is at most 2^-127, at most 254, 255 for NaN); a value's code is x * 2^(127 - e) clamped to
// +-448 and rounded to E4M3, nearest even. Inkling 2's keys are clipped to qk_clip and its values
// RMS-normed, so their exponents stay far below 246, the largest the decode below represents.
__device__ __forceinline__ unsigned mxfp8_exponent(float amax) {
    const float raw = amax * (1.0f / 448.0f);
    if (raw != raw) { return 255u; }
    if (raw <= 0x1p-127f) { return 0u; }
    const unsigned e = ((__float_as_uint(raw) + 0x007FFFFFu) & 0x7F800000u) >> 23;
    return min(e, 254u);
}

// The code of x in a group of exponent e.
__device__ __forceinline__ std::uint8_t mxfp8_code(float x, unsigned e) {
    // 2^(127 - e): e 0 gives 2^127, as the rule's inverse does; NaN groups encode zeros times NaN.
    const float inverse = e == 255u ? 0.0f : __uint_as_float((254u - e) << 23);
    const float v       = fminf(fmaxf(x * inverse, -448.0f), 448.0f);
    return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

// The group's scale times 2^8, for codes read as FP16 times 2^-8 (ops' x2m8 expansion): exact
// for e up to 246; 255 is NaN.
__device__ __forceinline__ float mxfp8_scale_x256(unsigned e) {
    return e == 255u ? __uint_as_float(0x7FC00000u) : __uint_as_float((e + 8u) << 23);
}

// Chunks of eight values. Per coding: Raw, the bytes of one chunk; Scale, the chunk's group scale;
// load(), the chunk's bytes from its row; scale(), the chunk's scale; decode(), the eight values.
template <KvCoding C>
struct Chunk;

template <>
struct Chunk<KvCoding::Int8G64> {
    using Raw = uint2;
    using Scale = __half;
    static constexpr int kBytes = 8, kPerGroup = 8; // 64-dimension groups
    __device__ static Raw load(const std::uint8_t* codes, int c) {
        return *reinterpret_cast<const uint2*>(codes + c * kBytes);
    }
    __device__ static Scale scale(const std::uint8_t* scales, int c) {
        return reinterpret_cast<const __half*>(scales)[c / kPerGroup];
    }
    __device__ static void decode(Raw r, Scale s, float (&f)[8]) { int8x8_to_float(r, __half2float(s), f); }
};

template <>
struct Chunk<KvCoding::Int4G32> {
    using Raw = unsigned;
    using Scale = __half;
    static constexpr int kBytes = 4, kPerGroup = 4; // 32-dimension groups
    __device__ static Raw load(const std::uint8_t* codes, int c) {
        return *reinterpret_cast<const unsigned*>(codes + c * kBytes);
    }
    __device__ static Scale scale(const std::uint8_t* scales, int c) {
        return reinterpret_cast<const __half*>(scales)[c / kPerGroup];
    }
    __device__ static void decode(Raw r, Scale s, float (&f)[8]) { int4x8_to_float(r, __half2float(s), f); }
};

template <>
struct Chunk<KvCoding::Int4E8> : Chunk<KvCoding::Int4G32> {
    static constexpr int kPerGroup = 8; // 64-dimension groups
    __device__ static Scale scale(const std::uint8_t* scales, int c) {
        return reinterpret_cast<const __half*>(scales)[c / kPerGroup];
    }
};

template <>
struct Chunk<KvCoding::Lloyd4> {
    using Raw = unsigned;
    using Scale = __half;
    static constexpr int kBytes = 4, kPerGroup = 8; // 64-dimension groups
    __device__ static Raw load(const std::uint8_t* codes, int c) {
        return *reinterpret_cast<const unsigned*>(codes + c * kBytes);
    }
    __device__ static Scale scale(const std::uint8_t* scales, int c) {
        return reinterpret_cast<const __half*>(scales)[c / kPerGroup];
    }
    __device__ static void decode(Raw r, Scale s, float (&f)[8]) {
        const uint2 codes =
            make_uint2(kv_cache_lloyd4_expand4(r & 0xffffu), kv_cache_lloyd4_expand4(r >> 16));
        int8x8_to_float(codes, __half2float(s), f);
    }
};

template <>
struct Chunk<KvCoding::RootE8> {
    using Raw = unsigned short;
    using Scale = __half;
    static constexpr int kBytes = 2, kPerGroup = 8; // 64-dimension groups
    __device__ static Raw load(const std::uint8_t* codes, int c) {
        return *reinterpret_cast<const unsigned short*>(codes + c * kBytes);
    }
    __device__ static Scale scale(const std::uint8_t* scales, int c) {
        return reinterpret_cast<const __half*>(scales)[c / kPerGroup];
    }
    __device__ static void decode(Raw r, Scale s, float (&f)[8]) {
        int8x8_to_float(kv_cache_e8_root_decode_block(r & 0xffu, r >> 8), __half2float(s), f);
    }
};

template <>
struct Chunk<KvCoding::Fp8Row> {
    using Raw = uint2;
    using Scale = __half;
    static constexpr int kBytes = 8, kPerGroup = 32; // 256-dimension groups
    __device__ static Raw load(const std::uint8_t* codes, int c) {
        return *reinterpret_cast<const uint2*>(codes + c * kBytes);
    }
    __device__ static Scale scale(const std::uint8_t* scales, int c) {
        return reinterpret_cast<const __half*>(scales)[c / kPerGroup];
    }
    __device__ static void decode(Raw r, Scale s, float (&f)[8]) {
        const std::uint16_t* pairs = reinterpret_cast<const std::uint16_t*>(&r);
        const float sf             = __half2float(s);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 v = __half22float2(kv_cache_fp8_code2_to_half2(pairs[j]));
            f[2 * j]       = v.x * sf;
            f[2 * j + 1]   = v.y * sf;
        }
    }
};

template <>
struct Chunk<KvCoding::Nvfp4G16> {
    using Raw = unsigned;
    using Scale = std::uint8_t;
    static constexpr int kBytes = 4, kPerGroup = 2; // 16-dimension groups
    __device__ static Raw load(const std::uint8_t* codes, int c) {
        return *reinterpret_cast<const unsigned*>(codes + c * kBytes);
    }
    __device__ static Scale scale(const std::uint8_t* scales, int c) { return scales[c / kPerGroup]; }
    __device__ static void decode(Raw r, Scale s, float (&f)[8]) {
        // E2M1 times an E4M3 scale is exact in FP16 (nvfp4_group16_codec.cuh) and in FP32.
        __nv_fp8_e4m3 scale;
        scale.__x      = s;
        const float sf = static_cast<float>(scale);
        const std::uint8_t* bytes = reinterpret_cast<const std::uint8_t*>(&r);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            __nv_fp4x2_e2m1 pair;
            pair.__x       = bytes[j];
            const float2 v = static_cast<float2>(pair);
            f[2 * j]       = v.x * sf;
            f[2 * j + 1]   = v.y * sf;
        }
    }
};

template <>
struct Chunk<KvCoding::Mxfp8> {
    using Raw = uint2;
    using Scale = std::uint8_t;
    static constexpr int kBytes = 8, kPerGroup = 4; // 32-dimension groups
    __device__ static Raw load(const std::uint8_t* codes, int c) {
        return *reinterpret_cast<const uint2*>(codes + c * kBytes);
    }
    __device__ static Scale scale(const std::uint8_t* scales, int c) { return scales[c / kPerGroup]; }
    __device__ static void decode(Raw r, Scale s, float (&f)[8]) {
        // Codes times 2^-8 as FP16 pairs, then times the group's scale times 2^8: exact in FP32.
        const float sf = mxfp8_scale_x256(s);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned word = j < 2 ? r.x : r.y;
            const float2 pair =
                __half22float2(kv_cache_fp8_code2_to_half2_x2m8(word >> (16 * (j & 1))));
            f[2 * j]     = pair.x * sf;
            f[2 * j + 1] = pair.y * sf;
        }
    }
};

// The unnormalized 128-point Sylvester-Hadamard transform of a warp's 128-wide row (lane l holds
// dimensions l + 32 r in x[r]): the lane bits by shuffles, then the two register bits.
__device__ __forceinline__ void hadamard128(float (&x)[4], int lane) {
    hadamard_d32_columns_inplace<4>(x, lane);
#pragma unroll
    for (int span = 1; span < 4; span <<= 1) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            if ((r & span) != 0) { continue; }
            const float a = x[r], b = x[r + span];
            x[r]          = __fadd_rn(a, b);
            x[r + span]   = __fsub_rn(a, b);
        }
    }
}

// ---- One block per warp -------------------------------------------------------------------------
// A block of D = 32 R dimensions: lane l holds dimensions l + 32 r in x[r] (the layout of the
// engine's append kernels and of normalized_hadamard_d256_inplace). `scratch` is the warp's D floats
// of shared memory; only Nvfp4G16's encoder and decode_block use it.

// Encodes x (already rotated for a rotated plane) into the block's codes and scales.
template <KvCoding C, int R>
__device__ __forceinline__ void encode_block(const float (&x)[R], std::uint8_t* codes,
                                             std::uint8_t* scales, float* scratch, int lane) {
    static_assert(R == 4 || R == 8, "blocks of 128 or 256 dimensions");
    constexpr unsigned kAll = 0xffffffffu;
    if constexpr (C == KvCoding::Int8G64) {
#pragma unroll
        for (int g = 0; g < R / 2; ++g) {
            const float a = warp_max(fmaxf(fabsf(x[2 * g]), fabsf(x[2 * g + 1])), kAll);
            const auto q  = kv_cache_int8_quant_params(a);
            auto* c       = reinterpret_cast<std::int8_t*>(codes);
            c[64 * g + lane]      = kv_cache_int8_quant_code(x[2 * g], q.inverse_scale);
            c[64 * g + 32 + lane] = kv_cache_int8_quant_code(x[2 * g + 1], q.inverse_scale);
            if (lane == 0) { reinterpret_cast<__half*>(scales)[g] = q.scale; }
        }
    } else if constexpr (C == KvCoding::Int4G32) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const auto q        = kv_cache_int4_quant_params(warp_max(fabsf(x[r]), kAll));
            const std::int8_t c = kv_cache_int4_quant_code(x[r], q.inverse_scale);
            const int partner   = __shfl_xor_sync(kAll, int(c), 1);
            if ((lane & 1) == 0) {
                codes[16 * r + (lane >> 1)] = kv_cache_int4_pack(c, std::int8_t(partner));
            }
            if (lane == 0) { reinterpret_cast<__half*>(scales)[r] = q.scale; }
        }
    } else if constexpr (C == KvCoding::Lloyd4) {
#pragma unroll
        for (int g = 0; g < R / 2; ++g) {
            const auto e       = kv_cache_lloyd4_encode_group(x[2 * g], x[2 * g + 1]);
            const int partner0 = __shfl_xor_sync(kAll, e.index0, 1);
            const int partner1 = __shfl_xor_sync(kAll, e.index1, 1);
            if ((lane & 1) == 0) {
                codes[32 * g + (lane >> 1)]      = std::uint8_t(e.index0 | (partner0 << 4));
                codes[32 * g + 16 + (lane >> 1)] = std::uint8_t(e.index1 | (partner1 << 4));
            }
            if (lane == 0) { reinterpret_cast<__half*>(scales)[g] = e.scale; }
        }
    } else if constexpr (C == KvCoding::Mxfp8) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const unsigned e     = mxfp8_exponent(warp_max(fabsf(x[r]), kAll));
            codes[32 * r + lane] = mxfp8_code(x[r], e);
            if (lane == 0) { scales[r] = std::uint8_t(e); }
        }
    } else if constexpr (C == KvCoding::Int4E8) {
#pragma unroll
        for (int g = 0; g < R / 2; ++g) {
            const float a        = warp_max(fmaxf(fabsf(x[2 * g]), fabsf(x[2 * g + 1])), kAll);
            const auto q         = kv_cache_int4_quant_params(a);
            const std::int8_t c0 = kv_cache_int4_e8_key_code(x[2 * g], q.inverse_scale, lane);
            const std::int8_t c1 = kv_cache_int4_e8_key_code(x[2 * g + 1], q.inverse_scale, lane);
            const int partner0   = __shfl_xor_sync(kAll, int(c0), 1);
            const int partner1   = __shfl_xor_sync(kAll, int(c1), 1);
            if ((lane & 1) == 0) {
                codes[32 * g + (lane >> 1)]      = kv_cache_int4_pack(c0, std::int8_t(partner0));
                codes[32 * g + 16 + (lane >> 1)] = kv_cache_int4_pack(c1, std::int8_t(partner1));
            }
            if (lane == 0) { reinterpret_cast<__half*>(scales)[g] = q.scale; }
        }
    } else if constexpr (C == KvCoding::RootE8) {
#pragma unroll
        for (int g = 0; g < R / 2; ++g) {
            const float a = warp_max(fmaxf(fabsf(x[2 * g]), fabsf(x[2 * g + 1])), kAll);
            const auto q  = kv_cache_int4_quant_params(a);
            kv_cache_e8_root_store_key_group(codes, 0, g, x[2 * g], x[2 * g + 1],
                                             __half2float(q.scale), lane);
            if (lane == 0) { reinterpret_cast<__half*>(scales)[g] = q.scale; }
        }
    } else if constexpr (C == KvCoding::Fp8Row) {
        static_assert(R == 8, "Fp8Row keeps one scale per 256 dimensions");
        float a = 0.0f;
#pragma unroll
        for (int r = 0; r < R; ++r) { a = fmaxf(a, fabsf(x[r])); }
        const auto q = kv_cache_fp8_quant_params(warp_max(a, kAll));
#pragma unroll
        for (int r = 0; r < R; ++r) { codes[32 * r + lane] = kv_cache_fp8_quant_code(x[r], q.inverse_scale); }
        if (lane == 0) { reinterpret_cast<__half*>(scales)[0] = q.scale; }
    } else if constexpr (C == KvCoding::Nvfp4G16) {
#pragma unroll
        for (int r = 0; r < R; ++r) { scratch[32 * r + lane] = x[r]; }
        __syncwarp();
        if (lane < 2 * R) { // one lane per 16 dimensions
            const auto q = kv_cache_nvfp4_quantize_group16(scratch + 16 * lane);
            *reinterpret_cast<uint2*>(codes + 8 * lane) = make_uint2(q.codes_lo, q.codes_hi);
            scales[lane] = q.scale;
        }
        __syncwarp();
    }
}

// The block's represented values, in the encoder's layout (rotated still, for a rotated plane).
template <KvCoding C, int R>
__device__ __forceinline__ void decode_block(const std::uint8_t* codes, const std::uint8_t* scales,
                                             float (&x)[R], float* scratch, int lane) {
    static_assert(R == 4 || R == 8, "blocks of 128 or 256 dimensions");
    using K = Chunk<C>;
    __syncwarp();
    if (lane < 4 * R) { // chunk `lane`: dimensions [8 lane, 8 lane + 8)
        float f[8];
        K::decode(K::load(codes, lane), K::scale(scales, lane), f);
#pragma unroll
        for (int j = 0; j < 8; ++j) { scratch[8 * lane + j] = f[j]; }
    }
    __syncwarp();
#pragma unroll
    for (int r = 0; r < R; ++r) { x[r] = scratch[32 * r + lane]; }
    __syncwarp();
}

// Chunk `c` of a row of coding C, whose codes and scales begin at `codes` / `scales`, as one
// 16-byte word of halves: a tile loader that knows the coding at compile time.
template <KvCoding C>
__device__ __forceinline__ uint4 chunk_word(const std::uint8_t* codes, const std::uint8_t* scales,
                                            int c) {
    float f[8];
    Chunk<C>::decode(Chunk<C>::load(codes, c), Chunk<C>::scale(scales, c), f);
    return pack8(f);
}

} // namespace ninfer::ops::kv_row
