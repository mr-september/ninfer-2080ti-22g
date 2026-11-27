#pragma once

// The vector kernel: products of a GGUF matrix with one to eight q8 activation columns. A warp owns
// two rows; lane l takes the 32-value slices l, l + 32, ... of each. Every slice of a row is decoded
// once into eight int8x4 words in value order (word i holds values 4i..4i+3) plus the factors of its
// type, then dotted with that slice of every column. The activation is planar: int8 values
// [columns][K] and one (d, sum of x) half2 per 32 values [columns][K / 32], ggml's q8_1 numbers.
//
// The decoders follow ggml-quants.c's dequantize_row_* exactly, with float slice factors where
// llama.cpp's vector kernels round integer scales (the 2- and 3-bit i-quants).

#include "ggml_bridge_vec_launch.cuh"
#include "../../common/fixed_point.cuh"
#include "../../common/device_route_lookup.h"

#include <algorithm>
#include <type_traits>

namespace ninfer::ops::gguf::detail {

// Types whose vector kernels are also built for the static K of the 27B text matrices.
template <ggml_type type>
inline constexpr bool kVecStaticK =
    type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ4_XS ||
    type == GGML_TYPE_Q4_K || type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ2_XS ||
    type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_Q2_K;

inline constexpr int kVecWarps = 4;

namespace vec {

enum class VectorFamily { Dense, Moe, MoeUp };

template <ggml_type type, VectorFamily Family = VectorFamily::Dense>
bool cooperative_loads(int width) {
    if (width < 1 || width > 8) { return false; }
    if constexpr (Family != VectorFamily::Dense) {
        if (width != 1 && width != 2 && width != 4 && width != 8) { return false; }
        if constexpr (Family == VectorFamily::MoeUp) {
            if (width > 4) { return false; }
        }
    }
    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K) {
        constexpr bool q4 = type == GGML_TYPE_Q4_K;
        constexpr std::string_view key = Family == VectorFamily::Dense
            ? (q4 ? "gguf_vector/q4_k" : "gguf_vector/q6_k")
            : Family == VectorFamily::Moe
                ? (q4 ? "gguf_moe_vector/q4_k" : "gguf_moe_vector/q6_k")
                : (q4 ? "gguf_moe_up/q4_k" : "gguf_moe_up/q6_k");
        return device_route_schedule(key, width) == "cooperative";
    }
    return false;
}

struct Slice {
    int w[8];
    float f[6];
    int h[8]; // a second word set (Q3_K's high bits)
};

__device__ __forceinline__ std::uint32_t u32_a2(const std::uint8_t* p) {
    const auto* h = reinterpret_cast<const std::uint16_t*>(p);
    return std::uint32_t(h[0]) | (std::uint32_t(h[1]) << 16);
}

__device__ __forceinline__ std::uint32_t u32_a4(const std::uint8_t* p) {
    return *reinterpret_cast<const std::uint32_t*>(p);
}

__device__ __forceinline__ float half_at(const std::uint8_t* p) {
    return __half2float(*reinterpret_cast<const half*>(p));
}

// Negates the bytes of `g` whose bit is set in the low nibble of `bits`. Every grid this serves has
// no zero byte, so the per-byte two's-complement increment cannot carry.
__device__ __forceinline__ int negate_bytes(std::uint32_t g, std::uint32_t bits) {
    const std::uint32_t ones = ((bits & 0xF) * 0x00204081u) & 0x01010101u;
    return int((g ^ (ones * 0xFFu)) + ones);
}

// ksigns_iq2xs: seven stored sign bits and an eighth that makes the parity even.
__device__ __forceinline__ std::uint32_t ksigns(std::uint32_t v) {
    return v ^ ((__popc(v) & 1) << 7);
}

__device__ __forceinline__ int dot4(const int* w, const int* a, int acc) {
#pragma unroll
    for (int i = 0; i < 4; ++i) { acc = ggml_cuda_dp4a(w[i], a[i], acc); }
    return acc;
}

__device__ __forceinline__ int sum4(const int* a) {
    int acc = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) { acc = ggml_cuda_dp4a(a[i], 0x01010101, acc); }
    return acc;
}

__device__ __forceinline__ int sum2(const int* a) {
    return ggml_cuda_dp4a(a[1], 0x01010101, ggml_cuda_dp4a(a[0], 0x01010101, 0));
}

// The 6-bit scale and min of sub-block j of a Q4_K / Q5_K block (ggml's get_scale_min_k4).
__device__ __forceinline__ void scale_min_k4(const std::uint8_t* s, int j, int& sc, int& m) {
    if (j < 4) {
        sc = s[j] & 63;
        m  = s[j + 4] & 63;
    } else {
        sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
        m  = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

// The product of one slice of a type the CPU also computes (Q2_0, Q4_0, Q5_0, Q8_0, IQ4_NL,
// IQ4_XS, IQ3_XXS, IQ3_S, IQ2_XXS; IQ2_S, IQ1_S, IQ1_M, the K-quants with a minimum and Q6_K spell
// out their forms): the slice factor times the activation scale, rounded, times the integer dot,
// rounded, never fused with the sum it joins, so that gguf_expert_cpu reproduces every bit.
__device__ __forceinline__ float cpu_product(float factor, float d, int dot) {
    return __fmul_rn(__fmul_rn(factor, d), float(dot));
}

// Q4_K and Q5_K: the scaled dot less the minimum times the activation's sum, each step rounded.
__device__ __forceinline__ float cpu_product_min(float factor, float d, int dot, float minimum,
                                                 float sum) {
    return __fsub_rn(cpu_product(factor, d, dot), __fmul_rn(minimum, sum));
}

// dot() is the contribution of one slice to one column: d and sum are the column's slice scale and
// the sum of its unquantized values.
template <ggml_type type>
struct Decoder;

template <>
struct Decoder<GGML_TYPE_Q8_0> {
    static constexpr int kBlockElems = 32, kBlockBytes = 34, kTableWords = 0;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int, const std::uint32_t*,
                                                  Slice& s) {
        s.f[0] = half_at(b);
#pragma unroll
        for (int i = 0; i < 8; ++i) { s.w[i] = int(u32_a2(b + 2 + 4 * i)); }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return cpu_product(s.f[0], d, dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

// Q4_0: codes c in 0..15, value d * (c - 8); value j < 16 is the low nibble of byte j, value
// j + 16 its high nibble.
template <>
struct Decoder<GGML_TYPE_Q4_0> {
    static constexpr int kBlockElems = 32, kBlockBytes = 18, kTableWords = 0;

    __device__ static const std::uint32_t* table() { return nullptr; }

    // The 18-byte block starts 2-byte aligned: five aligned words cover it (the last one may reach
    // two bytes past it, never past a 4-byte boundary of the allocation), against nine 2-byte loads.
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int, const std::uint32_t*,
                                                  Slice& s) {
        const auto at       = reinterpret_cast<std::uintptr_t>(b);
        const auto* w       = reinterpret_cast<const std::uint32_t*>(at & ~std::uintptr_t(3));
        const unsigned skew = unsigned(at & 2) * 8; // 0 or 16
        std::uint32_t word[5];
#pragma unroll
        for (int i = 0; i < 5; ++i) { word[i] = w[i]; }
        const std::uint32_t d = __funnelshift_rc(word[0], word[1], skew) & 0xFFFFu;
        s.f[0]                = __half2float(__ushort_as_half(static_cast<unsigned short>(d)));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t q = __funnelshift_rc(word[j], word[j + 1], 16 + skew);
            s.w[j]                = int(__vsubss4(q & 0x0F0F0F0Fu, 0x08080808u));
            s.w[j + 4]            = int(__vsubss4((q >> 4) & 0x0F0F0F0Fu, 0x08080808u));
        }
    }

    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return cpu_product(s.f[0], d, dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

// Q5_0: Q4_0's nibbles with a fifth bit per value in qh (bit j for value j), value d * (c - 16).
template <>
struct Decoder<GGML_TYPE_Q5_0> {
    static constexpr int kBlockElems = 32, kBlockBytes = 22, kTableWords = 0;

    __device__ static const std::uint32_t* table() { return nullptr; }

    // The four bits of `h` as bit 4 of four bytes.
    __device__ __forceinline__ static std::uint32_t high_bits(std::uint32_t h) {
        return ((h << 4) & 0x00000010u) | ((h << 11) & 0x00001000u) | ((h << 18) & 0x00100000u) |
               ((h << 25) & 0x10000000u);
    }

    __device__ __forceinline__ static void decode(const std::uint8_t* b, int, const std::uint32_t*,
                                                  Slice& s) {
        s.f[0]                 = half_at(b);
        const std::uint32_t qh = u32_a2(b + 2);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t q  = u32_a2(b + 6 + 4 * j);
            const std::uint32_t lo = (q & 0x0F0F0F0Fu) | high_bits(qh >> (4 * j));
            const std::uint32_t hi = ((q >> 4) & 0x0F0F0F0Fu) | high_bits(qh >> (16 + 4 * j));
            s.w[j]                 = int(__vsubss4(lo, 0x10101010u));
            s.w[j + 4]             = int(__vsubss4(hi, 0x10101010u));
        }
    }

    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return cpu_product(s.f[0], d, dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

// Q2_0: 64 two-bit codes c, value d * (c - 1); value j is bits 2(j % 4) of byte j / 4. A block is
// two 32-value slices.
template <>
struct Decoder<GGML_TYPE_Q2_0> {
    static constexpr int kBlockElems = 64, kBlockBytes = 18, kTableWords = 0;

    __device__ static const std::uint32_t* table() { return nullptr; }

    // The four codes of byte `x` as four bytes, minus one each.
    __device__ __forceinline__ static int spread(std::uint32_t x) {
        x &= 0xFFu;
        return int(__vsubss4((x | (x << 6) | (x << 12) | (x << 18)) & 0x03030303u, 0x01010101u));
    }

    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t*, Slice& s) {
        s.f[0] = half_at(b);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const std::uint32_t q = u32_a2(b + 2 + 8 * u + 4 * h);
#pragma unroll
            for (int byte = 0; byte < 4; ++byte) { s.w[4 * h + byte] = spread(q >> (8 * byte)); }
        }
    }

    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return cpu_product(s.f[0], d, dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }

    // The transposed form for activations stored by decode_quantize<true>: word 4h + k holds the
    // raw codes of values 16h + k, 16h + 4 + k, 16h + 8 + k and 16h + 12 + k, two operations per
    // word instead of a spread per byte, and the codes' offset of one comes off as the slice's sum
    // of quantized activations. The integer dot is the same, so the product is dot()'s bit for bit.
    static constexpr bool kTransposed = true;

    // `b` is global memory (device or mapped host): read-only loads.
    __device__ __forceinline__ static void decode_t(const std::uint8_t* b, int u, Slice& s) {
        const auto* h16 = reinterpret_cast<const unsigned short*>(b);
        s.f[0]          = __half2float(__ushort_as_half(__ldg(h16)));
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const std::uint32_t q = std::uint32_t(__ldg(h16 + 1 + 4 * u + 2 * h)) |
                                    (std::uint32_t(__ldg(h16 + 2 + 4 * u + 2 * h)) << 16);
#pragma unroll
            for (int k = 0; k < 4; ++k) { s.w[4 * h + k] = int((q >> (2 * k)) & 0x03030303u); }
        }
    }

    __device__ __forceinline__ static float dot_t(const Slice& s, const int* a, float d, int qsum) {
        return cpu_product(s.f[0], d, dot4(s.w + 4, a + 4, dot4(s.w, a, 0)) - qsum);
    }
};

// Decoders with a transposed form (decode_t / dot_t over decode_quantize<true> activations).
template <class D, class = void>
struct transposed : std::false_type {};
template <class D>
struct transposed<D, std::void_t<decltype(D::kTransposed)>> : std::bool_constant<D::kTransposed> {};

// A slice and its product in a decoder's transposed form when it has one; `sum` serves the plain
// form and `qsum` the transposed one.
template <class D, bool Cooperative = false>
__device__ __forceinline__ void decode_words(const std::uint8_t* b, int sub,
                                             const std::uint32_t* table, Slice& s) {
    if constexpr (Cooperative && (std::is_same_v<D, Decoder<GGML_TYPE_Q4_K>> ||
                                  std::is_same_v<D, Decoder<GGML_TYPE_Q6_K>>)) {
        D::template decode<true>(b, sub, table, s);
    } else {
        D::decode(b, sub, table, s);
    }
}

template <class D, bool Cooperative = false>
__device__ __forceinline__ void decode_slice(const std::uint8_t* b, int sub,
                                             const std::uint32_t* table, Slice& s) {
    if constexpr (transposed<D>::value) {
        D::decode_t(b, sub, s);
    } else {
        decode_words<D, Cooperative>(b, sub, table, s);
    }
}
template <class D>
__device__ __forceinline__ float dot_slice(const Slice& s, const int* a, float d, float sum,
                                           int qsum) {
    if constexpr (transposed<D>::value) {
        return D::dot_t(s, a, d, qsum);
    } else {
        return D::dot(s, a, d, sum);
    }
}

template <>
struct Decoder<GGML_TYPE_IQ4_NL> {
    static constexpr int kBlockElems = 32, kBlockBytes = 18, kTableWords = 0;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int, const std::uint32_t*,
                                                  Slice& s) {
        s.f[0] = half_at(b);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int2 v = get_int_from_table_16(int(u32_a2(b + 2 + 4 * j)), kvalues_iq4nl);
            s.w[j]       = v.x;
            s.w[j + 4]   = v.y;
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return cpu_product(s.f[0], d, dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

// MXFP4: an E8M0 exponent byte e, then value j < 16 the low nibble of byte j + 1 and value j + 16
// its high nibble, each an E2M1 code's value (kvalues_mxfp4 holds them doubled) times 2^(e - 127).
// The 17-byte block starts at any byte: five aligned words cover its codes, the last reaching at
// most three bytes past the block (into the next block or the bound device tail).
template <>
struct Decoder<GGML_TYPE_MXFP4> {
    static constexpr int kBlockElems = 32, kBlockBytes = 17, kTableWords = 0;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int, const std::uint32_t*,
                                                  Slice& s) {
        s.f[0]              = ggml_cuda_e8m0_to_fp32(b[0]) * 0.5f;
        const auto at       = reinterpret_cast<std::uintptr_t>(b + 1);
        const auto* w       = reinterpret_cast<const std::uint32_t*>(at & ~std::uintptr_t(3));
        const unsigned skew = unsigned(at & 3) * 8;
        std::uint32_t word[5];
#pragma unroll
        for (int i = 0; i < 5; ++i) { word[i] = w[i]; }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int2 v = get_int_from_table_16(int(__funnelshift_r(word[j], word[j + 1], skew)),
                                                 kvalues_mxfp4);
            s.w[j]       = v.x;
            s.w[j + 4]   = v.y;
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return cpu_product(s.f[0], d, dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

// NVFP4: four UE4M3 scale bytes (one per 16 values) and 32 bytes of E2M1 codes; value j of
// sub-block s is the low nibble of code byte 8 s + j for j < 8 and the high nibble of code byte
// 8 s + j - 8 otherwise. A 64-value block is two slices of two sub-blocks each. Blocks of 36 bytes
// in rows of whole blocks start 4-byte aligned. The product is Q6_K's form over the two halves.
template <>
struct Decoder<GGML_TYPE_NVFP4> {
    static constexpr int kBlockElems = 64, kBlockBytes = 36, kTableWords = 0;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u, const std::uint32_t*,
                                                  Slice& s) {
        const std::uint32_t d = u32_a4(b);
        s.f[0]                = ue4m3_half(d >> (16 * u));
        s.f[1]                = ue4m3_half(d >> (16 * u + 8));
        // Code word k holds bytes 4k..4k+3 of the slice's 16: low nibbles are values 4k..4k+3 of
        // its sub-block (k < 2) or of the second (k >= 2, values 16 + 4(k - 2) ...), high nibbles
        // the same eight positions later.
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int2 v = get_int_from_table_16(int(u32_a4(b + 4 + 16 * u + 4 * k)), kvalues_mxfp4);
            s.w[4 * (k / 2) + k % 2]     = v.x;
            s.w[4 * (k / 2) + 2 + k % 2] = v.y;
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return __fmul_rn(d, __fadd_rn(__fmul_rn(s.f[0], float(dot4(s.w, a, 0))),
                                      __fmul_rn(s.f[1], float(dot4(s.w + 4, a + 4, 0)))));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ4_XS> {
    static constexpr int kBlockElems = 256, kBlockBytes = 136, kTableWords = 0;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t*, Slice& s) {
        const std::uint32_t sh = *reinterpret_cast<const std::uint16_t*>(b + 2);
        const int ls           = ((b[4 + u / 2] >> (4 * (u & 1))) & 0xF) | (((sh >> (2 * u)) & 3) << 4);
        s.f[0]                 = half_at(b) * float(ls - 32);
        const uint2 lo         = *reinterpret_cast<const uint2*>(b + 8 + 16 * u);
        const uint2 hi         = *reinterpret_cast<const uint2*>(b + 16 + 16 * u);
        const std::uint32_t q[4] = {lo.x, lo.y, hi.x, hi.y};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int2 v = get_int_from_table_16(int(q[j]), kvalues_iq4nl);
            s.w[j]       = v.x;
            s.w[j + 4]   = v.y;
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return cpu_product(s.f[0], d, dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_Q4_K> {
    static constexpr int kBlockElems = 256, kBlockBytes = 144, kTableWords = 0;
    __device__ static const std::uint32_t* table() { return nullptr; }
    template <bool Cooperative = false>
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t*, Slice& s) {
        const float2 dm = __half22float2(*reinterpret_cast<const half2*>(b));
        int sc, m;
        scale_min_k4(b + 4, u, sc, m);
        s.f[0]              = dm.x * float(sc);
        s.f[1]              = dm.y * float(m);
        const uint4* qs     = reinterpret_cast<const uint4*>(b + 16 + 32 * (u / 2));
        uint4 a0, a1;
        if constexpr (Cooperative) {
            const unsigned mask = 0xFFu << (threadIdx.x & 24);
            const uint4 own = qs[u & 1];
            const uint4 peer = make_uint4(__shfl_xor_sync(mask, own.x, 1, 8),
                                          __shfl_xor_sync(mask, own.y, 1, 8),
                                          __shfl_xor_sync(mask, own.z, 1, 8),
                                          __shfl_xor_sync(mask, own.w, 1, 8));
            a0 = (u & 1) ? peer : own;
            a1 = (u & 1) ? own : peer;
        } else {
            a0 = qs[0];
            a1 = qs[1];
        }
        const int shift     = 4 * (u & 1);
        const std::uint32_t v[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) { s.w[i] = int((v[i] >> shift) & 0x0F0F0F0Fu); }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float sum) {
        return cpu_product_min(s.f[0], d, dot4(s.w + 4, a + 4, dot4(s.w, a, 0)), s.f[1], sum);
    }
};

template <>
struct Decoder<GGML_TYPE_Q5_K> {
    static constexpr int kBlockElems = 256, kBlockBytes = 176, kTableWords = 0;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t*, Slice& s) {
        const float2 dm = __half22float2(*reinterpret_cast<const half2*>(b));
        int sc, m;
        scale_min_k4(b + 4, u, sc, m);
        s.f[0]          = dm.x * float(sc);
        s.f[1]          = dm.y * float(m);
        const uint4* qh = reinterpret_cast<const uint4*>(b + 16);
        const uint4* ql = reinterpret_cast<const uint4*>(b + 48 + 32 * (u / 2));
        const uint4 h0 = qh[0], h1 = qh[1], l0 = ql[0], l1 = ql[1];
        const std::uint32_t hv[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
        const std::uint32_t lv[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
        const int shift = 4 * (u & 1);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s.w[i] = int(((lv[i] >> shift) & 0x0F0F0F0Fu) | (((hv[i] >> u) << 4) & 0x10101010u));
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float sum) {
        return cpu_product_min(s.f[0], d, dot4(s.w + 4, a + 4, dot4(s.w, a, 0)), s.f[1], sum);
    }
};

template <>
struct Decoder<GGML_TYPE_Q6_K> {
    static constexpr int kBlockElems = 256, kBlockBytes = 210, kTableWords = 0;
    __device__ static const std::uint32_t* table() { return nullptr; }
    template <bool Cooperative = false>
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t*, Slice& s) {
        const int h = u / 4, q = u % 4;
        const float d        = half_at(b + 208);
        const auto* sc       = reinterpret_cast<const std::int8_t*>(b + 192 + 8 * h);
        s.f[0]               = d * float(sc[2 * q]);
        s.f[1]               = d * float(sc[2 * q + 1]);
        const std::uint8_t* ql = b + 64 * h + 32 * (q & 1);
        const std::uint8_t* qh = b + 128 + 32 * h;
        if constexpr (Cooperative) {
            const unsigned mask = 0xFu << (threadIdx.x & 28);
#pragma unroll
            for (int base = 0; base < 8; base += 4) {
                const std::uint32_t hi = u32_a2(qh + 4 * (base + q));
                const std::uint32_t lo0 = u32_a2(ql + 4 * (base + (q >> 1)));
                const std::uint32_t lo1 = u32_a2(ql + 4 * (base + 2 + (q >> 1)));
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const std::uint32_t high = __shfl_sync(mask, hi, j, 4);
                    const std::uint32_t low = __shfl_sync(mask, j < 2 ? lo0 : lo1,
                                                         (q & 1) + 2 * (j & 1), 4);
                    const std::uint32_t v = ((low >> (4 * (q >> 1))) & 0x0F0F0F0Fu) |
                                            (((high >> (2 * q)) << 4) & 0x30303030u);
                    s.w[base + j] = int(v ^ 0xE0E0E0E0u ^ ((v & 0x20202020u) * 6u));
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const std::uint32_t v = ((u32_a2(ql + 4 * i) >> (4 * (q >> 1))) & 0x0F0F0F0Fu) |
                                        (((u32_a2(qh + 4 * i) >> (2 * q)) << 4) & 0x30303030u);
                // v - 32 per byte, v in [0, 64): flip the high bits by whether v is below 32.
                s.w[i] = int(v ^ 0xE0E0E0E0u ^ ((v & 0x20202020u) * 6u));
            }
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return __fmul_rn(d, __fadd_rn(__fmul_rn(s.f[0], float(dot4(s.w, a, 0))),
                                      __fmul_rn(s.f[1], float(dot4(s.w + 4, a + 4, 0)))));
    }
};

template <>
struct Decoder<GGML_TYPE_Q2_K> {
    static constexpr int kBlockElems = 256, kBlockBytes = 84, kTableWords = 0;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t*, Slice& s) {
        const int h = u / 4, j = u % 4;
        const float2 dm      = __half22float2(*reinterpret_cast<const half2*>(b + 80));
        const std::uint32_t lo = b[8 * h + 2 * j], hi = b[8 * h + 2 * j + 1];
        s.f[0]               = dm.x * float(lo & 0xF);
        s.f[1]               = dm.x * float(hi & 0xF);
        s.f[2]               = dm.y * float(lo >> 4);
        s.f[3]               = dm.y * float(hi >> 4);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            s.w[i] = int((u32_a4(b + 16 + 32 * h + 4 * i) >> (2 * j)) & 0x03030303u);
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return d * (s.f[0] * float(dot4(s.w, a, 0)) + s.f[1] * float(dot4(s.w + 4, a + 4, 0)) -
                    s.f[2] * float(sum4(a)) - s.f[3] * float(sum4(a + 4)));
    }
};

template <>
struct Decoder<GGML_TYPE_Q3_K> {
    static constexpr int kBlockElems = 256, kBlockBytes = 110, kTableWords = 0;
    __device__ static const std::uint32_t* table() { return nullptr; }
    __device__ __forceinline__ static int scale(const std::uint8_t* b, int k) {
        return (((b[96 + k % 8] >> (4 * (k / 8))) & 0xF) | (((b[104 + k % 4] >> (2 * (k / 4))) & 3) << 4)) - 32;
    }
    // A value is (low two bits) + 4 (high bit) - 4. The slice keeps the low bits (w) and four
    // times the high bit (h) apart, two operations a word each, and the product subtracts four
    // times the activations' sum: the same integers as the value bytes'. A block is only 2-byte
    // aligned, so its words come from aligned loads shifted into place.
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t*, Slice& s) {
        const int h = u / 4, j = u % 4;
        const float d = half_at(b + 108);
        s.f[0]        = d * float(scale(b, 8 * h + 2 * j));
        s.f[1]        = d * float(scale(b, 8 * h + 2 * j + 1));
        const auto at        = reinterpret_cast<std::uintptr_t>(b);
        const auto* words    = reinterpret_cast<const std::uint32_t*>(at & ~std::uintptr_t(3));
        const unsigned shift = unsigned(at & 3) * 8;
        std::uint32_t hm[9], qs[9];
#pragma unroll
        for (int i = 0; i < 9; ++i) {
            hm[i] = words[i];
            qs[i] = words[8 + 8 * h + i];
        }
        // The high bit of value 4i + k sits at bit 4h + j of byte k of hmask word i; rotated to
        // bit 2 of its byte.
        const unsigned hshift = unsigned(4 * h + j + 30) & 31;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const std::uint32_t q  = __funnelshift_r(qs[i], qs[i + 1], shift);
            const std::uint32_t hb = __funnelshift_r(hm[i], hm[i + 1], shift);
            s.w[i]                 = int((q >> (2 * j)) & 0x03030303u);
            s.h[i]                 = int(__funnelshift_r(hb, hb, hshift) & 0x04040404u);
        }
    }
    // Q6_K's form, each step rounded, as gguf_expert_cpu computes it.
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        const int lo = dot4(s.h, a, dot4(s.w, a, 0)) - 4 * sum4(a);
        const int hi = dot4(s.h + 4, a + 4, dot4(s.w + 4, a + 4, 0)) - 4 * sum4(a + 4);
        return __fmul_rn(d, __fadd_rn(__fmul_rn(s.f[0], float(lo)), __fmul_rn(s.f[1], float(hi))));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ2_XXS> {
    static constexpr int kBlockElems = 256, kBlockBytes = 66, kTableWords = 512;
    __device__ static const std::uint32_t* table() {
        return reinterpret_cast<const std::uint32_t*>(iq2xxs_grid);
    }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t* tab, Slice& s) {
        const std::uint32_t idx = u32_a2(b + 2 + 8 * u);
        const std::uint32_t aux = u32_a2(b + 6 + 8 * u);
        s.f[0]                  = half_at(b) * (0.5f + float(aux >> 28)) * 0.25f;
        const auto* grid        = reinterpret_cast<const uint2*>(tab);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g          = grid[(idx >> (8 * l)) & 0xFF];
            const std::uint32_t sg = ksigns((aux >> (7 * l)) & 0x7F);
            s.w[2 * l]             = negate_bytes(g.x, sg);
            s.w[2 * l + 1]         = negate_bytes(g.y, sg >> 4);
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return cpu_product(s.f[0], d, dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ2_XS> {
    static constexpr int kBlockElems = 256, kBlockBytes = 74, kTableWords = 1024;
    __device__ static const std::uint32_t* table() {
        return reinterpret_cast<const std::uint32_t*>(iq2xs_grid);
    }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t* tab, Slice& s) {
        const float d          = half_at(b);
        const std::uint32_t sc = b[66 + u];
        s.f[0]                 = d * (0.5f + float(sc & 0xF)) * 0.25f;
        s.f[1]                 = d * (0.5f + float(sc >> 4)) * 0.25f;
        const std::uint32_t q01 = u32_a2(b + 2 + 8 * u), q23 = u32_a2(b + 6 + 8 * u);
        const auto* grid        = reinterpret_cast<const uint2*>(tab);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t q  = ((l < 2 ? q01 : q23) >> (16 * (l & 1))) & 0xFFFF;
            const uint2 g          = grid[q & 511];
            const std::uint32_t sg = ksigns(q >> 9);
            s.w[2 * l]             = negate_bytes(g.x, sg);
            s.w[2 * l + 1]         = negate_bytes(g.y, sg >> 4);
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return d * (s.f[0] * float(dot4(s.w, a, 0)) + s.f[1] * float(dot4(s.w + 4, a + 4, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ2_S> {
    static constexpr int kBlockElems = 256, kBlockBytes = 82, kTableWords = 2048;
    __device__ static const std::uint32_t* table() {
        return reinterpret_cast<const std::uint32_t*>(iq2s_grid);
    }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t* tab, Slice& s) {
        const float d          = half_at(b);
        const std::uint32_t sc = b[74 + u];
        s.f[0]                 = d * (0.5f + float(sc & 0xF)) * 0.25f;
        s.f[1]                 = d * (0.5f + float(sc >> 4)) * 0.25f;
        const std::uint32_t qs = u32_a2(b + 2 + 4 * u);
        const std::uint32_t sg = u32_a2(b + 34 + 4 * u);
        const std::uint32_t qh = b[66 + u];
        const auto* grid       = reinterpret_cast<const uint2*>(tab);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 g = grid[((qs >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300)];
            s.w[2 * l]     = negate_bytes(g.x, sg >> (8 * l));
            s.w[2 * l + 1] = negate_bytes(g.y, sg >> (8 * l + 4));
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        // Spelled out as the CPU's product computes it (gguf_expert_cpu).
        return __fmul_rn(d, __fmaf_rn(s.f[0], float(dot4(s.w, a, 0)),
                                      __fmul_rn(s.f[1], float(dot4(s.w + 4, a + 4, 0)))));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ3_XXS> {
    static constexpr int kBlockElems = 256, kBlockBytes = 98, kTableWords = 256;
    __device__ static const std::uint32_t* table() { return iq3xxs_grid; }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t* grid, Slice& s) {
        const std::uint32_t q0  = u32_a2(b + 2 + 8 * u);
        const std::uint32_t q1  = u32_a2(b + 6 + 8 * u);
        const std::uint32_t aux = u32_a2(b + 66 + 4 * u);
        s.f[0]                  = half_at(b) * (0.5f + float(aux >> 28)) * 0.5f;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const std::uint32_t q  = p < 2 ? q0 : q1;
            const std::uint32_t sg = ksigns((aux >> (7 * p)) & 0x7F);
            s.w[2 * p]     = negate_bytes(grid[(q >> (16 * (p & 1))) & 0xFF], sg);
            s.w[2 * p + 1] = negate_bytes(grid[(q >> (16 * (p & 1) + 8)) & 0xFF], sg >> 4);
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return cpu_product(s.f[0], d, dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ3_S> {
    static constexpr int kBlockElems = 256, kBlockBytes = 110, kTableWords = 512;
    __device__ static const std::uint32_t* table() { return iq3s_grid; }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t* grid, Slice& s) {
        const std::uint32_t q0 = u32_a2(b + 2 + 8 * u);
        const std::uint32_t q1 = u32_a2(b + 6 + 8 * u);
        const std::uint32_t qh = b[66 + u];
        const std::uint32_t sg = u32_a2(b + 74 + 4 * u);
        const int sc           = (b[106 + u / 2] >> (4 * (u & 1))) & 0xF;
        s.f[0]                 = half_at(b) * float(1 + 2 * sc);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const std::uint32_t idx = (((e < 4 ? q0 : q1) >> (8 * (e & 3))) & 0xFF) | ((qh << (8 - e)) & 0x100);
            s.w[e]                  = negate_bytes(grid[idx], sg >> (4 * e));
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        return cpu_product(s.f[0], d, dot4(s.w + 4, a + 4, dot4(s.w, a, 0)));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ1_S> {
    static constexpr int kBlockElems = 256, kBlockBytes = 50, kTableWords = 2048;
    __device__ static const std::uint32_t* table() { return iq1s_grid_gpu; }
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t* grid, Slice& s) {
        const std::uint32_t qs = u32_a2(b + 2 + 4 * u);
        const std::uint32_t qh = *reinterpret_cast<const std::uint16_t*>(b + 34 + 2 * u);
        s.f[0] = half_at(b) * float(((qh >> 11) & 0x0E) + 1);
        s.f[1] = -1.0f + IQ1S_DELTA - float(qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t g = grid[((qs >> (8 * l)) & 0xFF) | (((qh >> (3 * l)) & 7) << 8)];
            s.w[2 * l]            = int(g & 0x0F0F0F0Fu);
            s.w[2 * l + 1]        = int((g >> 4) & 0x0F0F0F0Fu);
        }
    }
    // Spelled out as the CPU's product computes it (gguf_expert_cpu).
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float sum) {
        return __fmul_rn(s.f[0], __fmaf_rn(d, float(dot4(s.w + 4, a + 4, dot4(s.w, a, 0))),
                                           __fmul_rn(s.f[1], sum)));
    }
};

template <>
struct Decoder<GGML_TYPE_IQ1_M> {
    static constexpr int kBlockElems = 256, kBlockBytes = 56, kTableWords = 2048;
    __device__ static const std::uint32_t* table() { return iq1s_grid_gpu; }
    // A value is d (2 s + 1) (g + delta) with g in -1..1 and delta +-1/8 per eight values. Eight
    // times g + delta is the integer 8 c - 7 (delta 1/8) or 8 c - 9 (delta -1/8) for the grid's
    // code c = g + 1, so the words hold those integers and the factors d (2 s + 1) / 8, both exact:
    // each half's dot is ggml's exactly, and the product is IQ2_S's.
    __device__ __forceinline__ static void decode(const std::uint8_t* b, int u,
                                                  const std::uint32_t* grid, Slice& s) {
        const auto* sc = reinterpret_cast<const std::uint16_t*>(b + 48);
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        const float d          = __half2float(scale.f16);
        const int tmp          = sc[u / 2] >> (6 * (u % 2));
        s.f[0]                 = d * float(2 * (tmp & 7) + 1) * 0.125f;
        s.f[1]                 = d * float(2 * ((tmp >> 3) & 7) + 1) * 0.125f;
        const std::uint32_t qs = u32_a4(b + 4 * u);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t qhl    = b[32 + 2 * u + l / 2] >> (4 * (l % 2));
            const std::uint32_t g      = grid[((qs >> (8 * l)) & 0xFF) | ((qhl & 7) << 8)];
            const std::uint32_t offset = (qhl & 0x08) ? 0x09090909u : 0x07070707u;
            s.w[2 * l]                 = int(__vsub4((g & 0x0F0F0F0Fu) << 3, offset));
            s.w[2 * l + 1]             = int(__vsub4(((g >> 4) & 0x0F0F0F0Fu) << 3, offset));
        }
    }
    __device__ __forceinline__ static float dot(const Slice& s, const int* a, float d, float) {
        // Spelled out as the CPU's product computes it (gguf_expert_cpu).
        return __fmul_rn(d, __fmaf_rn(s.f[0], float(dot4(s.w, a, 0)),
                                      __fmul_rn(s.f[1], float(dot4(s.w + 4, a + 4, 0)))));
    }
};

// KS: the static K the slice loop unrolls over, or 0 for the runtime K. Fused: the weight is one
// [gate; up] parent and each output row is silu(gate row) * up row.
template <ggml_type type, int T, bool Fused, int KS, bool Cooperative = false>
__launch_bounds__(kVecWarps * 32) __global__ void kernel(VecArgs p) {
    using D                 = Decoder<type>;
    constexpr int kTable    = D::kTableWords > 0 ? D::kTableWords : 4;
    constexpr int kPerBlock = D::kBlockElems / 32;
    constexpr int kRows     = 2;
    __shared__ __align__(16) std::uint32_t table[kTable];
    if constexpr (D::kTableWords > 0) {
        const std::uint32_t* src = D::table();
        for (int i = threadIdx.x; i < D::kTableWords; i += kVecWarps * 32) { table[i] = src[i]; }
        __syncthreads();
    }
    const int warp   = threadIdx.x >> 5;
    const int lane   = threadIdx.x & 31;
    const int k      = KS != 0 ? KS : p.k;
    const int slices = k / 32;
    const int step   = Fused ? 1 : kRows;
    const std::int64_t lane_offset = std::int64_t(lane / kPerBlock) * D::kBlockBytes;
    const int sub                  = lane % kPerBlock;

    for (int row0 = (blockIdx.x * kVecWarps + warp) * step; row0 < p.rows;
         row0 += gridDim.x * kVecWarps * step) {
        const std::uint8_t* rowp[kRows];
        if constexpr (Fused) {
            rowp[0] = p.weight + std::int64_t(row0) * p.row_bytes + lane_offset;
            rowp[1] = p.weight + std::int64_t(p.rows + row0) * p.row_bytes + lane_offset;
        } else {
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                rowp[r] = p.weight + std::int64_t(min(row0 + r, p.rows - 1)) * p.row_bytes + lane_offset;
            }
        }
        float acc[kRows][T];
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) { acc[r][j] = 0.0f; }
        }
        const auto slice = [&](int it) {
            const int s = lane + 32 * it;
            int a[T][8];
            float d[T], sum[T];
#pragma unroll
            for (int j = 0; j < T; ++j) {
                const float2 ds = __half22float2(p.ds[std::int64_t(j) * slices + s]);
                d[j]            = ds.x;
                sum[j]          = ds.y;
                const auto* q   = reinterpret_cast<const int4*>(p.qs + std::int64_t(j) * k + 32 * s);
                const int4 v0 = q[0], v1 = q[1];
                a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                Slice sl;
                const auto* block = rowp[r] + std::int64_t(32 / kPerBlock) * it * D::kBlockBytes;
                decode_words<D, Cooperative>(block, sub, table, sl);
#pragma unroll
                for (int j = 0; j < T; ++j) { acc[r][j] += D::dot(sl, a[j], d[j], sum[j]); }
            }
        };
        if constexpr (KS != 0) {
            static_assert(KS % 1024 == 0);
#pragma unroll
            for (int it = 0; it < KS / 1024; ++it) { slice(it); }
        } else {
            for (int it = 0; lane + 32 * it < slices; ++it) { slice(it); }
        }
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) {
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) {
                    acc[r][j] += __shfl_xor_sync(0xFFFFFFFFu, acc[r][j], o);
                }
            }
        }
        if (lane == 0) {
            if constexpr (Fused) {
#pragma unroll
                for (int j = 0; j < T; ++j) { vec_store(p.out, row0, j, vec_gate(acc[0][j], p.out.activation) * acc[1][j]); }
            } else {
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    if (row0 + r < p.rows) {
#pragma unroll
                        for (int j = 0; j < T; ++j) { vec_store(p.out, row0 + r, j, acc[r][j]); }
                    }
                }
            }
        }
    }
}

template <ggml_type type, int T, bool Fused, int KS, bool Cooperative = false>
void launch(const VecArgs& args, cudaStream_t stream) {
    static const int resident = [] {
        int blocks = 0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel<type, T, Fused, KS, Cooperative>,
                                                            kVecWarps * 32, 0),
              "vector occupancy");
        return std::max(blocks, 1);
    }();
    const int per_block = kVecWarps * (Fused ? 1 : 2);
    const int groups    = (args.rows + per_block - 1) / per_block;
    const int blocks    = std::min(groups, resident * device_facts().sm_count);
    kernel<type, T, Fused, KS, Cooperative><<<blocks, kVecWarps * 32, 0, stream>>>(args);
    check(cudaGetLastError(), "vector product launch");
}

template <ggml_type type, int T, bool Fused, bool Cooperative = false>
void launch_k(const VecArgs& args, cudaStream_t stream) {
    if constexpr (kVecStaticK<type>) {
        switch (args.k) {
        case 5120: return launch<type, T, Fused, 5120, Cooperative>(args, stream);
        case 6144: return launch<type, T, Fused, 6144, Cooperative>(args, stream);
        case 10240: return launch<type, T, Fused, 10240, Cooperative>(args, stream);
        case 17408: return launch<type, T, Fused, 17408, Cooperative>(args, stream);
        default: break;
        }
    }
    launch<type, T, Fused, 0, Cooperative>(args, stream);
}

template <ggml_type type, bool Fused, bool Cooperative = false>
void launch_columns(const VecArgs& args, int columns, cudaStream_t stream) {
    switch (columns) {
    case 1: return launch_k<type, 1, Fused, Cooperative>(args, stream);
    case 2: return launch_k<type, 2, Fused, Cooperative>(args, stream);
    case 3: return launch_k<type, 3, Fused, Cooperative>(args, stream);
    case 4: return launch_k<type, 4, Fused, Cooperative>(args, stream);
    case 5: return launch_k<type, 5, Fused, Cooperative>(args, stream);
    case 6: return launch_k<type, 6, Fused, Cooperative>(args, stream);
    case 7: return launch_k<type, 7, Fused, Cooperative>(args, stream);
    case 8: return launch_k<type, 8, Fused, Cooperative>(args, stream);
    default: break;
    }
    throw std::invalid_argument("gguf vector product: columns must be 1..8");
}

// The MoE form: block (x, y) takes rows of the y-th active expert for every pair routed to it, up
// to T of them per pass over the rows; a pass over fewer skips the missing columns, and every
// column's arithmetic is the same at any T. A pair's activation column is pair / column_group.
// Fused: the first table holds the gate rows and the second the up rows, and each output row is
// silu(gate) * up.
template <ggml_type type, int T, bool Fused, bool Cooperative = false>
__launch_bounds__(kVecWarps * 32) __global__ void moe_kernel(MoeVecArgs p) {
    using D                 = Decoder<type>;
    constexpr int kTable    = D::kTableWords > 0 ? D::kTableWords : 4;
    constexpr int kPerBlock = D::kBlockElems / 32;
    constexpr int kRows     = 2;
    // Uniform per block, so it may leave before the barrier below.
    if (int(blockIdx.y) >= *p.active_count) { return; }
    __shared__ __align__(16) std::uint32_t table[kTable];
    if constexpr (D::kTableWords > 0) {
        const std::uint32_t* src = D::table();
        for (int i = threadIdx.x; i < D::kTableWords; i += kVecWarps * 32) { table[i] = src[i]; }
        __syncthreads();
    }
    const int expert               = p.active[blockIdx.y];
    const int begin                = p.bounds[expert];
    const int end                  = p.bounds[expert + 1];
    const std::uint8_t* first      = p.first[expert];
    const std::uint8_t* second     = Fused ? p.second[expert] : nullptr;
    const int warp                 = threadIdx.x >> 5;
    const int lane                 = threadIdx.x & 31;
    const int k                    = p.k;
    const int slices               = k / 32;
    const int step                 = Fused ? 1 : kRows;
    const std::int64_t lane_offset = std::int64_t(lane / kPerBlock) * D::kBlockBytes;
    const int sub                  = lane % kPerBlock;

    for (int row0 = (blockIdx.x * kVecWarps + warp) * step; row0 < p.rows;
         row0 += gridDim.x * kVecWarps * step) {
        const std::uint8_t* rowp[kRows];
        if constexpr (Fused) {
            rowp[0] = first + std::int64_t(row0) * p.row_bytes + lane_offset;
            rowp[1] = second + std::int64_t(row0) * p.row_bytes + lane_offset;
        } else {
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                rowp[r] =
                    first + std::int64_t(min(row0 + r, p.rows - 1)) * p.row_bytes + lane_offset;
            }
        }
        for (int c0 = begin; c0 < end; c0 += T) {
            const int n = min(T, end - c0);
            int pair[T], column[T];
#pragma unroll
            for (int j = 0; j < T; ++j) {
                pair[j]   = p.sorted[c0 + min(j, n - 1)];
                column[j] = pair[j] / p.column_group;
            }
            float acc[kRows][T];
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
#pragma unroll
                for (int j = 0; j < T; ++j) { acc[r][j] = 0.0f; }
            }
            for (int it = 0; lane + 32 * it < slices; ++it) {
                const int s = lane + 32 * it;
                int a[T][8];
                float d[T], sum[T];
#pragma unroll
                for (int j = 0; j < T; ++j) {
                    if (j >= n) { break; }
                    const float2 ds = __half22float2(p.ds[std::int64_t(column[j]) * slices + s]);
                    d[j]            = ds.x;
                    sum[j]          = ds.y;
                    const auto* q =
                        reinterpret_cast<const int4*>(p.qs + std::int64_t(column[j]) * k + 32 * s);
                    const int4 v0 = q[0], v1 = q[1];
                    a[j][0] = v0.x;
                    a[j][1] = v0.y;
                    a[j][2] = v0.z;
                    a[j][3] = v0.w;
                    a[j][4] = v1.x;
                    a[j][5] = v1.y;
                    a[j][6] = v1.z;
                    a[j][7] = v1.w;
                }
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    Slice sl;
                    decode_words<D, Cooperative>(
                        rowp[r] + std::int64_t(32 / kPerBlock) * it * D::kBlockBytes, sub, table, sl);
#pragma unroll
                    for (int j = 0; j < T; ++j) {
                        if (j >= n) { break; }
                        acc[r][j] += D::dot(sl, a[j], d[j], sum[j]);
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
#pragma unroll
                for (int j = 0; j < T; ++j) {
                    if (j >= n) { break; }
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) {
                        acc[r][j] += __shfl_xor_sync(0xFFFFFFFFu, acc[r][j], o);
                    }
                }
            }
            if (lane != 0) { continue; }
#pragma unroll
            for (int j = 0; j < T; ++j) {
                if (j >= n) { break; }
                if constexpr (Fused) {
                    p.out_bf16[std::int64_t(pair[j]) * p.rows + row0] =
                        __float2bfloat16(vec_gate(acc[0][j], p.activation) * acc[1][j]);
                } else {
#pragma unroll
                    for (int r = 0; r < kRows; ++r) {
                        const int row = row0 + r;
                        if (row >= p.rows) { break; }
                        const std::int64_t at = std::int64_t(pair[j]) * p.rows + row;
                        float value           = acc[r][j];
                        if (p.gate != nullptr) { value *= vec_gate(p.gate[at], p.activation); }
                        if (p.weighted != nullptr) {
                            // 2^-32 fixed point, saturated far beyond any real activation.
                            atomicAdd(p.weighted + std::int64_t(pair[j] / p.per_token) * p.rows +
                                          row,
                                      static_cast<unsigned long long>(
                                          fixed_product(value, p.weights[pair[j]])));
                        } else if (p.out_f32 != nullptr) {
                            p.out_f32[at] = value;
                        } else {
                            p.out_bf16[at] = __float2bfloat16(value);
                        }
                    }
                }
            }
        }
    }
}

template <ggml_type type, int T, bool Fused, bool Cooperative = false>
void moe_launch_chunk(const MoeVecArgs& args, int max_active, cudaStream_t stream) {
    const int per_block = kVecWarps * (Fused ? 1 : 2);
    const int groups    = (args.rows + per_block - 1) / per_block;
    moe_kernel<type, T, Fused, Cooperative><<<dim3(groups, max_active, 1), kVecWarps * 32, 0, stream>>>(args);
    check(cudaGetLastError(), "moe product launch");
}

template <ggml_type type, bool Fused, bool Cooperative = false>
void moe_launch_fused(const MoeVecArgs& args, int max_active, int chunk, cudaStream_t stream) {
    switch (chunk) {
    case 1:
        return moe_launch_chunk<type, 1, Fused, Cooperative>(args, max_active, stream);
    case 2:
        return moe_launch_chunk<type, 2, Fused, Cooperative>(args, max_active, stream);
    case 4:
        return moe_launch_chunk<type, 4, Fused, Cooperative>(args, max_active, stream);
    case 8:
        return moe_launch_chunk<type, 8, Fused, Cooperative>(args, max_active, stream);
    default:
        break;
    }
    throw std::invalid_argument("gguf moe product: chunk must be 1, 2, 4 or 8");
}

// --- The MoE of one decode or verification call in few launches --------------------------------
// moe_decode_up computes silu(gate . x) * (up . x) of every pair of an expert in one block row;
// moe_decode_down runs the down rows, adds each weighted pair into the 2^-32 fixed-point sum and,
// when asked, in its last block adds external products and stores y. The pairs of an expert are
// found by scanning the call's ids, or listed once for the call by moe_decode_prep, which also
// quantizes the tokens' inputs once (else every block quantizes its own); prepared up blocks
// quantize their middles for the down kernel. No kernel needs a sort or a memset. 2-bit decoding is
// integer-throughput bound, so a decoder with a transposed form (Q2_0) takes its activations
// transposed. Every row's arithmetic is moe_kernel's, so the sums are bit for bit those of the
// separate kernels.

inline constexpr int kDecodeTokens = 8;
inline constexpr int kDecodeK      = 2560;

// q8_1 of n columns of k BF16 values (16-byte aligned) into shared planes: quantize_vector_kernel's
// numbers, bit for bit. A thread takes eight consecutive values, four threads a 32-value group, and
// up to four rounds of loads are in flight at once. The group sum is the warp butterfly's (offsets
// 16, 8, 4, 2, 1 over the group's 32 values): its two cross-thread levels pair thread q with q ^ 2
// and q ^ 1 value by value, the other three are within the thread.
template <bool Transposed = false>
__device__ __forceinline__ void decode_quantize(const __nv_bfloat16* x, std::int64_t stride,
                                                const int* columns, int n, int k, std::int8_t* qs,
                                                half2* ds, int* qsum = nullptr) {
    constexpr int kRounds = 4;
    const int chunks      = k / 8;   // per column
    const int total       = n * chunks;
    const int groups      = k / QK8_1;
    for (int base = 0; base < total; base += kRounds * int(blockDim.x)) {
        uint4 raw[kRounds];
#pragma unroll
        for (int r = 0; r < kRounds; ++r) {
            const int c = base + r * int(blockDim.x) + int(threadIdx.x);
            raw[r]      = make_uint4(0, 0, 0, 0);
            if (c < total) {
                const int j = c / chunks, i = c % chunks;
                // Through L2: a middle was written by other blocks of this launch.
                raw[r] = __ldcg(reinterpret_cast<const uint4*>(x + std::int64_t(columns[j]) * stride + 8 * i));
            }
        }
#pragma unroll
        for (int r = 0; r < kRounds; ++r) {
            // Whole quads are valid or not (k is a multiple of 32), and every lane shuffles.
            const int c = base + r * int(blockDim.x) + int(threadIdx.x);
            float v[8];
            const auto* pairs2 = reinterpret_cast<const __nv_bfloat162*>(&raw[r]);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 f = __bfloat1622float2(pairs2[e]);
                v[2 * e]       = f.x;
                v[2 * e + 1]   = f.y;
            }
            float amax = 0.0f;
#pragma unroll
            for (int e = 0; e < 8; ++e) { amax = fmaxf(amax, fabsf(v[e])); }
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 2));
            float t[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) { t[e] = v[e] + __shfl_xor_sync(0xFFFFFFFFu, v[e], 2); }
#pragma unroll
            for (int e = 0; e < 8; ++e) { t[e] += __shfl_xor_sync(0xFFFFFFFFu, t[e], 1); }
            const float s4[4] = {t[0] + t[4], t[1] + t[5], t[2] + t[6], t[3] + t[7]};
            const float s2[2] = {s4[0] + s4[2], s4[1] + s4[3]};
            const float sum   = s2[0] + s2[1];
            const float d = amax / 127.0f;
            std::uint32_t lo = 0, hi = 0;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const auto q = std::uint32_t(std::uint8_t(
                    amax == 0.0f ? std::int8_t(0) : static_cast<std::int8_t>(roundf(v[e] / d))));
                if (e < 4) {
                    lo |= q << (8 * e);
                } else {
                    hi |= q << (8 * (e - 4));
                }
            }
            int words[2] = {int(lo), int(hi)};
            int total_q  = 0;
            if constexpr (Transposed) {
                // The 16 values of a half slice are this thread's two words and its partner's
                // (lane ^ 1): A0..A3 in value order. Word k of the half becomes byte k of each.
                const std::uint32_t plo = __shfl_xor_sync(0xFFFFFFFFu, lo, 1);
                const std::uint32_t phi = __shfl_xor_sync(0xFFFFFFFFu, hi, 1);
                const bool odd          = (threadIdx.x & 1) != 0;
                const std::uint32_t a0 = odd ? plo : lo, a1 = odd ? phi : hi;
                const std::uint32_t a2 = odd ? lo : plo, a3 = odd ? hi : phi;
                const std::uint32_t t01 = __byte_perm(a0, a1, odd ? 0x7362 : 0x5140);
                const std::uint32_t t23 = __byte_perm(a2, a3, odd ? 0x7362 : 0x5140);
                words[0] = int(__byte_perm(t01, t23, 0x5410));
                words[1] = int(__byte_perm(t01, t23, 0x7632));
                total_q  = __dp4a(int(lo), 0x01010101, __dp4a(int(hi), 0x01010101, 0));
                total_q += __shfl_xor_sync(0xFFFFFFFFu, total_q, 1);
                total_q += __shfl_xor_sync(0xFFFFFFFFu, total_q, 2);
            }
            if (c >= total) { continue; }
            const int j = c / chunks, i = c % chunks;
            *reinterpret_cast<uint2*>(qs + j * k + 8 * i) =
                make_uint2(std::uint32_t(words[0]), std::uint32_t(words[1]));
            if (i % 4 == 0) {
                ds[j * groups + i / 4] = make_half2(d, sum);
                if constexpr (Transposed) { qsum[j * groups + i / 4] = total_q; }
            }
        }
    }
}

// The pairs block row `slot` serves, found by one whole warp (lane `lane`): none when its id is
// negative or an earlier slot of the same source holds the same expert, else the slots holding it
// in ascending order. ids == nullptr: every slot is expert 0 (the shared expert). A single thread
// scanning made a verification's blocks quadratic in its slots.
__device__ __forceinline__ int decode_pairs(const std::int32_t* ids, int slots, int slot,
                                            int per_token, int* pairs, int* columns, int& expert,
                                            int lane) {
    const int e = ids != nullptr ? ids[slot] : 0;
    expert      = e;
    if (e < 0) { return 0; }
    for (int base = 0; base < slot; base += 32) {
        const int q = base + lane;
        if (__any_sync(0xFFFFFFFFu, q < slot && (ids != nullptr ? ids[q] : 0) == e)) { return 0; }
    }
    int n = 0;
    for (int base = slot; base < slots; base += 32) {
        const int q          = base + lane;
        const bool mine      = q < slots && (ids != nullptr ? ids[q] : 0) == e;
        const unsigned found = __ballot_sync(0xFFFFFFFFu, mine);
        const int at         = n + __popc(found & ((1u << lane) - 1u));
        if (mine && at < kDecodeTokens) {
            pairs[at]   = q;
            columns[at] = q / per_token;
        }
        n += __popc(found);
    }
    return min(n, kDecodeTokens);
}

// Each expert-major down block takes four passes of row pairs, so its inputs are quantized once per
// 32 down rows rather than per pass.
inline constexpr int kDecodeRowPasses = 4;

// The up kernel's blocks: sixteen warps over 32 consecutive rows (a quantization group of the
// middle), two rows each, so a lane's unrolled loads fit its registers at full occupancy.
inline constexpr int kUpWarps  = 16;
inline constexpr int kUpPasses = 2;
inline constexpr int kUpRows   = kUpWarps * kUpPasses;

// Block (x, y) takes rows [32 x, 32 x + 32) of slot y's gate and up, warp w rows 16 pass + w;
// every pass's rows and every slice are unrolled with unconditional loads, so a lane's weight
// loads are all issued before its products. Lane l accumulates slices l, l + 32, ... in that
// order, so every row's sums are moe_kernel's. With prepared middles the block then quantizes
// its 32 middle values of every pair itself (one warp per pair, quantize_vector_kernel's
// arithmetic), else it stores the middles.
template <ggml_type type, int T, bool Cooperative = false>
__launch_bounds__(kUpWarps * 32, 1) __global__ void moe_decode_up_kernel(MoeDecodeUpArgs p) {
    using D                   = Decoder<type>;
    constexpr int kTable      = D::kTableWords > 0 ? D::kTableWords : 4;
    constexpr int kPerBlock   = D::kBlockElems / 32;
    constexpr int k           = kDecodeK; // the launch checks p.k
    constexpr int slices      = k / 32;
    constexpr int kSliceIters = (slices + 31) / 32;
    constexpr bool kT         = transposed<D>::value;
    // Unprepared: the call's columns (slots / per_token), quantized, and their slice sums, sized
    // at launch.
    extern __shared__ __align__(16) std::int8_t decode_shared[];
    const int all_columns = p.slots / p.per_token;
    std::int8_t* qs       = decode_shared;
    auto* ds              = reinterpret_cast<half2*>(decode_shared + all_columns * k);
    auto* qsum            = reinterpret_cast<int*>(ds + all_columns * slices);
    __shared__ __align__(16) std::uint32_t table[kTable];
    __shared__ int pairs[kDecodeTokens], columns[kDecodeTokens], count, expert_shared;
    __shared__ float group[kDecodeTokens][kUpRows];
    if (p.zero != nullptr && blockIdx.y == 0) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < p.zero_count;
             i += gridDim.x * blockDim.x) {
            p.zero[i] = 0;
        }
    }
    if (threadIdx.x < 32) {
        const int lane = int(threadIdx.x);
        int expert = -1, c = 0;
        if (p.records != nullptr) {
            const MoeDecodeRecord& record = p.records[blockIdx.y];
            c      = record.n;
            expert = record.expert;
            if (lane < c) {
                pairs[lane]   = record.pairs[lane];
                columns[lane] = record.columns[lane];
            }
        } else {
            c = decode_pairs(p.ids, p.slots, blockIdx.y, p.per_token, pairs, columns, expert, lane);
        }
        if (lane == 0) {
            count         = c;
            expert_shared = expert;
        }
    }
    __syncthreads();
    const int n      = count;
    const int expert = expert_shared;
    // Scanning ids, block row y lists slot y for the down kernel: every slot in the first stage,
    // the slots it serves in the second (whose others keep the first stage's records).
    if (p.write_records != nullptr && blockIdx.x == 0 && threadIdx.x < 32 &&
        (n > 0 || p.record_source == 0)) {
        MoeDecodeRecord& record = p.write_records[blockIdx.y];
        if (int(threadIdx.x) < n) {
            record.pairs[threadIdx.x]   = pairs[threadIdx.x];
            record.columns[threadIdx.x] = columns[threadIdx.x];
        }
        if (threadIdx.x == 0) {
            record.expert = expert;
            record.n      = n;
            record.source = p.record_source;
        }
    }
    if (n == 0) { return; }
    if constexpr (D::kTableWords > 0) {
        const std::uint32_t* src = D::table();
        for (int i = threadIdx.x; i < D::kTableWords; i += blockDim.x) { table[i] = src[i]; }
    }
    // Prepared inputs are read where moe_decode_prep left them, by token; else this block
    // quantizes its pairs' tokens into shared memory, by pair.
    const bool prepared = p.qs != nullptr;
    if (!prepared) { decode_quantize<kT>(p.m, p.k, columns, n, k, qs, ds, qsum); }
    __syncthreads();
    const std::int8_t* q_base  = prepared ? p.qs : qs;
    const half2* d_base        = prepared ? p.ds : ds;
    const int* s_base          = prepared ? p.qsum : qsum;
    const std::uint8_t* first  = p.gate[expert];
    const std::uint8_t* second = p.up[expert];
    const int warp             = threadIdx.x >> 5;
    const int lane             = threadIdx.x & 31;
    const int row_begin        = blockIdx.x * kUpRows;
    for (int c0 = 0; c0 < n; c0 += T) {
        const int m_cols = min(T, n - c0);
        float acc[kUpPasses][2][T];
#pragma unroll
        for (int pass = 0; pass < kUpPasses; ++pass) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
#pragma unroll
                for (int j = 0; j < T; ++j) { acc[pass][r][j] = 0.0f; }
            }
        }
        // Every load is unconditional (rows and slices clamped into range) and only the sums
        // are predicated, so nothing orders one pass's loads after another's products.
#pragma unroll
        for (int it = 0; it < kSliceIters; ++it) {
            const bool live = lane + 32 * it < slices;
            const int s     = min(lane + 32 * it, slices - 1);
            int a[T][8], qs_sum[T];
            float d[T], sum[T];
#pragma unroll
            for (int j = 0; j < T; ++j) {
                const int c     = c0 + min(j, m_cols - 1);
                const int col   = prepared ? columns[c] : c;
                const float2 dv = __half22float2(d_base[col * slices + s]);
                d[j]            = dv.x;
                sum[j]          = dv.y;
                qs_sum[j]       = kT ? s_base[col * slices + s] : 0;
                const auto* q   = reinterpret_cast<const int4*>(q_base + col * k + 32 * s);
                const int4 v0 = q[0], v1 = q[1];
                a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
            }
#pragma unroll
            for (int pass = 0; pass < kUpPasses; ++pass) {
                const int row = min(row_begin + kUpWarps * pass + warp, p.rows - 1);
                // Slice s of the row (a dead lane reads the last one).
                const std::int64_t at = std::int64_t(row) * p.row_bytes +
                                        std::int64_t(s / kPerBlock) * D::kBlockBytes;
                Slice gate_slice, up_slice;
                decode_slice<D, Cooperative>(first + at, s % kPerBlock, table, gate_slice);
                decode_slice<D, Cooperative>(second + at, s % kPerBlock, table, up_slice);
#pragma unroll
                for (int j = 0; j < T; ++j) {
                    const float g = dot_slice<D>(gate_slice, a[j], d[j], sum[j], qs_sum[j]);
                    const float u = dot_slice<D>(up_slice, a[j], d[j], sum[j], qs_sum[j]);
                    if (live) {
                        acc[pass][0][j] += g;
                        acc[pass][1][j] += u;
                    }
                }
            }
        }
#pragma unroll
        for (int pass = 0; pass < kUpPasses; ++pass) {
            const int local = kUpWarps * pass + warp;
            const int row   = row_begin + local;
#pragma unroll
            for (int r = 0; r < 2; ++r) {
#pragma unroll
                for (int j = 0; j < T; ++j) {
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) {
                        acc[pass][r][j] += __shfl_xor_sync(0xFFFFFFFFu, acc[pass][r][j], o);
                    }
                }
            }
            if (lane != 0 || row >= p.rows) { continue; }
#pragma unroll
            for (int j = 0; j < T; ++j) {
                if (j >= m_cols) { break; }
                const __nv_bfloat16 v = __float2bfloat16(vec_silu(acc[pass][0][j]) * acc[pass][1][j]);
                if (p.middle_qs != nullptr) {
                    group[c0 + j][local] = __bfloat162float(v);
                } else {
                    p.middle[std::int64_t(pairs[c0 + j]) * p.rows + row] = v;
                }
            }
        }
    }
    if (p.middle_qs == nullptr) { return; }
    __syncthreads();
    // Warp w quantizes pair w's 32 values, lane l value l: the warp butterfly of
    // quantize_vector_kernel (offsets 16 .. 1) for the maximum and the sum.
    for (int j = warp; j < n; j += kUpWarps) {
        const float v = group[j][lane];
        float amax    = fabsf(v);
        float total   = v;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
            total += __shfl_xor_sync(0xFFFFFFFFu, total, o);
        }
        const float d = amax / 127.0f;
        const int q   = amax == 0.0f ? 0 : int(static_cast<std::int8_t>(roundf(v / d)));
        int qtotal    = q;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { qtotal += __shfl_xor_sync(0xFFFFFFFFu, qtotal, o); }
        const int pair  = pairs[j];
        const int g     = blockIdx.x; // the group: 32 rows
        constexpr int kMiddle = 640, groups = kMiddle / QK8_1;
        // Transposed: value v of a 16-value half h sits at word 4h + v % 4, byte (v % 16) / 4.
        const int at = p.middle_transposed
                           ? (lane >> 4) * 16 + (lane & 3) * 4 + ((lane >> 2) & 3)
                           : lane;
        p.middle_qs[std::int64_t(pair) * kMiddle + 32 * g + at] = static_cast<std::int8_t>(q);
        if (lane == 0) {
            p.middle_ds[pair * groups + g]   = make_half2(d, total);
            p.middle_qsum[pair * groups + g] = qtotal;
        }
    }
}

// Block (x, y) takes kDecodeRowPasses passes of kVecWarps row pairs of slot y's down rows; every
// pass and slice is unrolled with unconditional loads as in moe_decode_up, and a warp's two rows
// share each input load.
template <ggml_type type, int T>
__launch_bounds__(kVecWarps * 32) __global__ void moe_decode_down_kernel(MoeDecodeDownArgs p) {
    using D                 = Decoder<type>;
    constexpr int kTable    = D::kTableWords > 0 ? D::kTableWords : 4;
    constexpr int kPerBlock = D::kBlockElems / 32;
    constexpr int kRows     = 2;
    constexpr int kWidth    = 640;
    constexpr int slices    = kWidth / 32;
    constexpr bool kT       = transposed<D>::value;
    __shared__ __align__(16) std::int8_t qs[kDecodeTokens * kWidth];
    __shared__ half2 ds[kDecodeTokens * (kWidth / QK8_1)];
    __shared__ int qsum[kDecodeTokens * (kWidth / QK8_1)];
    __shared__ __align__(16) std::uint32_t table[kTable];
    __shared__ int pairs[kDecodeTokens], slots_of[kDecodeTokens], count, expert_shared, source;
    __shared__ bool last;
    if (threadIdx.x < 32) {
        const int lane = int(threadIdx.x);
        int expert = -1, n = 0, from = 0;
        if (p.records != nullptr) {
            const MoeDecodeRecord& record = p.records[blockIdx.y];
            n      = record.n;
            expert = record.expert;
            from   = record.source;
            if (lane < n) { pairs[lane] = record.pairs[lane]; }
        } else {
            n = decode_pairs(p.ids_a, p.slots, blockIdx.y, p.per_token, pairs, slots_of, expert,
                             lane);
            if (n == 0 && p.ids_b != nullptr && p.ids_a[blockIdx.y] < 0) {
                n    = decode_pairs(p.ids_b, p.slots, blockIdx.y, p.per_token, pairs, slots_of,
                                    expert, lane);
                from = 1;
            }
        }
        __syncwarp();
        // decode_pairs filled columns with the token; the middle is indexed by pair.
        if (lane < n) { slots_of[lane] = pairs[lane]; }
        if (lane == 0) {
            count         = n;
            expert_shared = expert;
            source        = from;
        }
    }
    __syncthreads();
    const int n = count;
    if (n > 0) {
        if constexpr (D::kTableWords > 0) {
            const std::uint32_t* src = D::table();
            for (int i = threadIdx.x; i < D::kTableWords; i += kVecWarps * 32) { table[i] = src[i]; }
        }
        // Prepared middles are read by pair where the up launches left them; else
        // this block quantizes its pairs' middles into shared memory.
        const bool prepared = p.qs != nullptr;
        if (!prepared) { decode_quantize<kT>(p.middle, kWidth, slots_of, n, kWidth, qs, ds, qsum); }
        __syncthreads();
        const std::int8_t* q_base = prepared ? p.qs : qs;
        const half2* d_base       = prepared ? p.ds : ds;
        const int* s_base         = prepared ? p.qsum : qsum;
        const std::uint8_t* first =
            (source == 0 ? p.table_a : p.table_b)[expert_shared];
        const int warp                 = threadIdx.x >> 5;
        const int lane                 = threadIdx.x & 31;
        const int stride               = gridDim.x * kVecWarps * kRows;
        const int row_base             = (blockIdx.x * kVecWarps + warp) * kRows;
        // slices < 32: one pass over them. As in moe_decode_up, every load is unconditional (a
        // lane past the last slice reads the last one) and only the sums are predicated.
        const bool live                = lane < slices;
        const int s                    = min(lane, slices - 1);
        for (int c0 = 0; c0 < n; c0 += T) {
            const int m_cols = min(T, n - c0);
            float acc[kDecodeRowPasses][kRows][T];
#pragma unroll
            for (int pass = 0; pass < kDecodeRowPasses; ++pass) {
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
#pragma unroll
                    for (int j = 0; j < T; ++j) { acc[pass][r][j] = 0.0f; }
                }
            }
            {
                int a[T][8], qs_sum[T];
                float d[T], sum[T];
#pragma unroll
                for (int j = 0; j < T; ++j) {
                    const int c     = c0 + min(j, m_cols - 1);
                    const int col   = prepared ? pairs[c] : c;
                    const float2 dv = __half22float2(d_base[col * slices + s]);
                    d[j]            = dv.x;
                    sum[j]          = dv.y;
                    qs_sum[j]       = kT ? s_base[col * slices + s] : 0;
                    const auto* q   = reinterpret_cast<const int4*>(q_base + col * kWidth + 32 * s);
                    const int4 v0 = q[0], v1 = q[1];
                    a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                    a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
                }
#pragma unroll
                for (int pass = 0; pass < kDecodeRowPasses; ++pass) {
                    const int row0 = row_base + pass * stride;
#pragma unroll
                    for (int r = 0; r < kRows; ++r) {
                        Slice sl;
                        decode_slice<D>(first + std::int64_t(min(row0 + r, p.rows - 1)) * p.row_bytes +
                                            std::int64_t(s / kPerBlock) * D::kBlockBytes,
                                        s % kPerBlock, table, sl);
#pragma unroll
                        for (int j = 0; j < T; ++j) {
                            const float v = dot_slice<D>(sl, a[j], d[j], sum[j], qs_sum[j]);
                            if (live) { acc[pass][r][j] += v; }
                        }
                    }
                }
            }
#pragma unroll
            for (int pass = 0; pass < kDecodeRowPasses; ++pass) {
                const int row0 = row_base + pass * stride;
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
#pragma unroll
                    for (int j = 0; j < T; ++j) {
#pragma unroll
                        for (int o = 16; o > 0; o >>= 1) {
                            acc[pass][r][j] += __shfl_xor_sync(0xFFFFFFFFu, acc[pass][r][j], o);
                        }
                    }
                }
                if (lane != 0 || row0 >= p.rows) { continue; }
#pragma unroll
                for (int j = 0; j < T; ++j) {
                    if (j >= m_cols) { break; }
                    const int pair = pairs[c0 + j];
#pragma unroll
                    for (int r = 0; r < kRows; ++r) {
                        const int row = row0 + r;
                        if (row >= p.rows) { break; }
                        atomicAdd(p.fixed + std::int64_t(pair / p.per_token) * p.rows + row,
                                  static_cast<unsigned long long>(
                                      fixed_product(acc[pass][r][j], p.weights[pair])));
                    }
                }
            }
        }
    }
    if (p.y == nullptr) { return; }
    // The last block to finish adds the external products and stores y. Every thread's sums are
    // made visible before its block counts itself done.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) { last = atomicAdd(p.counter, 1u) == gridDim.x * gridDim.y - 1; }
    __syncthreads();
    if (!last) { return; }
    __threadfence();
    const int products = p.product_count != nullptr ? *p.product_count : 0;
    for (int i = 0; i < products; ++i) {
        const int pair = p.product_pairs[i];
        for (int r = threadIdx.x; r < p.rows; r += blockDim.x) {
            atomicAdd(p.fixed + std::int64_t(pair / p.per_token) * p.rows + r,
                      static_cast<unsigned long long>(fixed_product(
                          p.products[std::int64_t(i) * p.rows + r], p.weights[pair])));
        }
    }
    __threadfence();
    __syncthreads();
    for (int i = threadIdx.x; i < p.tokens * p.rows; i += blockDim.x) {
        // Through L2: other blocks' sums never passed this block's L1.
        p.y[i] = static_cast<float>(static_cast<double>(static_cast<long long>(__ldcg(p.fixed + i))) *
                                    (1.0 / 4294967296.0));
    }
    if (threadIdx.x == 0) { *p.counter = 0; }
}

// The prepared down product of one token, row-major: block x takes rows [16 x, 16 x + 16) of
// every slot's expert, warp w rows 16 x + 2 w and the next. A row's weighted products of all the
// call's pairs are summed in fixed point in registers and added to the sum once, and kDownGroup
// slots' loads are in flight together. (A verification's experts serve several pairs each; the
// expert-major kernel reads their rows once for all of them.)
inline constexpr int kDownWarps = 8;
inline constexpr int kDownGroup = 4;
inline constexpr int kDownSlots = kDecodeTokens * 10;

template <ggml_type type, int T>
__launch_bounds__(kDownWarps * 32) __global__ void moe_decode_down_rows_kernel(MoeDecodeDownArgs p) {
    using D                 = Decoder<type>;
    constexpr int kTable    = D::kTableWords > 0 ? D::kTableWords : 4;
    constexpr int kPerBlock = D::kBlockElems / 32;
    constexpr int kRows     = 2;
    constexpr int kWidth    = 640;
    constexpr int slices    = kWidth / 32;
    __shared__ MoeDecodeRecord records[kDownSlots];
    __shared__ const std::uint8_t* bases[kDownSlots];
    __shared__ __align__(16) std::uint32_t table[kTable];
    for (int i = threadIdx.x; i < p.slots * int(sizeof(MoeDecodeRecord) / 4); i += blockDim.x) {
        reinterpret_cast<int*>(records)[i] = reinterpret_cast<const int*>(p.records)[i];
    }
    if constexpr (D::kTableWords > 0) {
        const std::uint32_t* src = D::table();
        for (int i = threadIdx.x; i < D::kTableWords; i += blockDim.x) { table[i] = src[i]; }
    }
    __syncthreads();
    // Every slot's rows, a slot without pairs pointing at a served one (its loads are then
    // harmless and its sums predicated off).
    __shared__ int any;
    if (threadIdx.x == 0) {
        any = -1;
        for (int y = 0; y < p.slots && any < 0; ++y) { any = records[y].n > 0 ? y : -1; }
    }
    __syncthreads();
    if (any < 0) { return; }
    for (int y = threadIdx.x; y < p.slots; y += blockDim.x) {
        const MoeDecodeRecord& record = records[records[y].n > 0 ? y : any];
        bases[y] = (record.source == 0 ? p.table_a : p.table_b)[record.expert];
    }
    __syncthreads();
    const int warp     = threadIdx.x >> 5;
    const int lane     = threadIdx.x & 31;
    const bool live    = lane < slices;
    const int s        = min(lane, slices - 1);
    const int row0     = (blockIdx.x * kDownWarps + warp) * kRows;
    const std::int64_t at[kRows] = {
        std::int64_t(min(row0, p.rows - 1)) * p.row_bytes + std::int64_t(s / kPerBlock) * D::kBlockBytes,
        std::int64_t(min(row0 + 1, p.rows - 1)) * p.row_bytes +
            std::int64_t(s / kPerBlock) * D::kBlockBytes};
    unsigned long long total[kRows][T];
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
#pragma unroll
        for (int t = 0; t < T; ++t) { total[r][t] = 0; }
    }
    // One pair per slot at a time (the c-th of its expert's); a slot with more pairs (verification)
    // takes further rounds.
    int rounds = 1;
    for (int y = 0; y < p.slots; ++y) { rounds = max(rounds, records[y].n); }
    for (int c = 0; c < rounds; ++c) {
        for (int y0 = 0; y0 < p.slots; y0 += kDownGroup) {
            float acc[kDownGroup][kRows];
            int pair[kDownGroup];
            bool valid[kDownGroup];
#pragma unroll
            for (int g = 0; g < kDownGroup; ++g) {
                const int y   = min(y0 + g, p.slots - 1);
                valid[g]      = y0 + g < p.slots && c < records[y].n;
                pair[g]       = valid[g] ? records[y].pairs[c] : 0;
                const float2 dv = __half22float2(p.ds[pair[g] * slices + s]);
                const int qs_sum = transposed<D>::value ? p.qsum[pair[g] * slices + s] : 0;
                const auto* q    = reinterpret_cast<const int4*>(p.qs + pair[g] * kWidth + 32 * s);
                const int4 v0 = q[0], v1 = q[1];
                const int a[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
                const std::uint8_t* base = bases[y];
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    Slice sl;
                    decode_slice<D>(base + at[r], s % kPerBlock, table, sl);
                    const float v = dot_slice<D>(sl, a, dv.x, dv.y, qs_sum);
                    acc[g][r]     = live ? v : 0.0f;
                }
            }
#pragma unroll
            for (int g = 0; g < kDownGroup; ++g) {
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    // The same butterfly as moe_kernel's: only live lanes contributed.
                    float v = acc[g][r];
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xFFFFFFFFu, v, o); }
                    acc[g][r] = v;
                }
                if (!valid[g]) { continue; }
                const int token = pair[g] / p.per_token;
                const float w   = p.weights[pair[g]];
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    const auto f = static_cast<unsigned long long>(fixed_product(acc[g][r], w));
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        if (t == token) { total[r][t] += f; }
                    }
                }
            }
        }
    }
    if (lane != 0) { return; }
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        if (row0 + r >= p.rows) { break; }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            if (t < p.tokens) { atomicAdd(p.fixed + std::int64_t(t) * p.rows + row0 + r, total[r][t]); }
        }
    }
}

template <ggml_type type, int T, bool Cooperative = false>
void moe_decode_up_chunk(const MoeDecodeUpArgs& args, cudaStream_t stream) {
    if (args.middle_qs != nullptr && args.rows % kUpRows != 0) {
        throw std::invalid_argument("gguf moe decode: quantized middles need whole groups");
    }
    const int groups    = (args.rows + kUpRows - 1) / kUpRows;
    const int columns   = args.slots / args.per_token;
    const std::size_t shared_bytes =
        args.qs != nullptr ? 0
                           : std::size_t(columns) * kDecodeK +
                                 std::size_t(columns) * (kDecodeK / QK8_1) * (sizeof(half2) + sizeof(int));
    moe_decode_up_kernel<type, T, Cooperative>
        <<<dim3(groups, args.slots, 1), kUpWarps * 32, shared_bytes, stream>>>(args);
    check(cudaGetLastError(), "moe decode up launch");
}

template <ggml_type type, int T>
void moe_decode_down_chunk(const MoeDecodeDownArgs& args, cudaStream_t stream) {
    if (T == 1 && args.records != nullptr && args.y == nullptr) {
        if (args.slots > kDownSlots) { throw std::invalid_argument("gguf moe decode: too many slots"); }
        const int per_block = kDownWarps * 2;
        moe_decode_down_rows_kernel<type, T>
            <<<(args.rows + per_block - 1) / per_block, kDownWarps * 32, 0, stream>>>(args);
        check(cudaGetLastError(), "moe decode down launch");
        return;
    }
    const int per_block = 2 * kVecWarps * kDecodeRowPasses;
    const int groups    = (args.rows + per_block - 1) / per_block;
    moe_decode_down_kernel<type, T><<<dim3(groups, args.slots, 1), kVecWarps * 32, 0, stream>>>(args);
    check(cudaGetLastError(), "moe decode down launch");
}

// moe_decode_prep: blocks 0 .. tokens - 1 quantize one token's input each, the last block lists
// every slot of the first stage's set and zeroes the sum.
template <bool Transposed>
__launch_bounds__(256) __global__ void moe_decode_prep_kernel(MoeDecodePrepArgs p) {
    __shared__ int column;
    if (int(blockIdx.x) < p.tokens) {
        if (threadIdx.x == 0) { column = blockIdx.x; }
        __syncthreads();
        constexpr int groups = kDecodeK / QK8_1;
        decode_quantize<Transposed>(p.m, kDecodeK, &column, 1, kDecodeK,
                                    p.qs + std::int64_t(blockIdx.x) * kDecodeK,
                                    p.ds + blockIdx.x * groups, p.qsum + blockIdx.x * groups);
        return;
    }
    if (p.zero != nullptr) {
        for (int i = threadIdx.x; i < p.zero_count; i += blockDim.x) { p.zero[i] = 0; }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int slot = warp; slot < p.slots; slot += blockDim.x >> 5) {
        MoeDecodeRecord& record = p.records[slot];
        int expert              = -1;
        const int n = decode_pairs(p.ids, p.slots, slot, p.per_token, record.pairs, record.columns,
                                   expert, lane);
        if (lane == 0) {
            record.expert = expert;
            record.n      = n;
            record.source = 0;
        }
    }
}

} // namespace vec

template <ggml_type type>
void moe_decode_up_launch(const MoeDecodeUpArgs& args, int chunk, cudaStream_t stream) {
    using D = vec::Decoder<type>;
    if (args.k != vec::kDecodeK || args.k % D::kBlockElems != 0 || args.slots <= 0 ||
        args.slots > 65535) {
        throw std::invalid_argument("gguf moe decode: unsupported geometry");
    }
    // Eight columns at once spill the unrolled loads out of registers: wider calls take four at
    // a time.
    const auto launch = [&]<bool Cooperative>() {
        switch (chunk) {
        case 1: return vec::moe_decode_up_chunk<type, 1, Cooperative>(args, stream);
        case 2: return vec::moe_decode_up_chunk<type, 2, Cooperative>(args, stream);
        default: return vec::moe_decode_up_chunk<type, 4, Cooperative>(args, stream);
        }
    };
    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K) {
        if (vec::cooperative_loads<type, vec::VectorFamily::MoeUp>(chunk == 1 || chunk == 2 ? chunk : 4)) {
            return launch.template operator()<true>();
        }
    }
    launch.template operator()<false>();
}

template <ggml_type type>
void moe_decode_down_launch(const MoeDecodeDownArgs& args, int chunk, cudaStream_t stream) {
    using D = vec::Decoder<type>;
    if (640 % D::kBlockElems != 0 || args.slots <= 0 || args.slots > 65535) {
        throw std::invalid_argument("gguf moe decode: unsupported geometry");
    }
    switch (chunk) {
    case 1: return vec::moe_decode_down_chunk<type, 1>(args, stream);
    case 2: return vec::moe_decode_down_chunk<type, 2>(args, stream);
    case 4: return vec::moe_decode_down_chunk<type, 4>(args, stream);
    default: return vec::moe_decode_down_chunk<type, 8>(args, stream);
    }
}

template <ggml_type type>
void moe_decode_prep_launch(const MoeDecodePrepArgs& args, cudaStream_t stream) {
    if (args.tokens <= 0 || args.tokens > vec::kDecodeTokens || args.slots <= 0 ||
        args.records == nullptr || args.qs == nullptr) {
        throw std::invalid_argument("gguf moe decode: invalid preparation");
    }
    vec::moe_decode_prep_kernel<vec::transposed<vec::Decoder<type>>::value>
        <<<args.tokens + 1, 256, 0, stream>>>(args);
    check(cudaGetLastError(), "moe decode prep launch");
}

template <ggml_type type>
void moe_launch(const MoeVecArgs& args, int max_active, int chunk, bool fused,
                cudaStream_t stream) {
    using D = vec::Decoder<type>;
    if (args.k % D::kBlockElems != 0 || args.row_bytes % D::kBlockBytes != 0 ||
        args.row_bytes < std::int64_t(args.k / D::kBlockElems) * D::kBlockBytes || args.rows <= 0 ||
        max_active <= 0 || max_active > 65535) {
        throw std::invalid_argument("gguf moe product: unsupported geometry");
    }
    const auto launch = [&]<bool Cooperative>() {
        if (fused) {
            vec::moe_launch_fused<type, true, Cooperative>(args, max_active, chunk, stream);
        } else {
            vec::moe_launch_fused<type, false, Cooperative>(args, max_active, chunk, stream);
        }
    };
    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K) {
        if (vec::cooperative_loads<type, vec::VectorFamily::Moe>(chunk)) { return launch.template operator()<true>(); }
    }
    launch.template operator()<false>();
}

template <ggml_type type>
void vec_launch(const VecArgs& args, int columns, bool fused, cudaStream_t stream) {
    using D = vec::Decoder<type>;
    if (args.k % D::kBlockElems != 0 || args.row_bytes % D::kBlockBytes != 0 ||
        args.row_bytes < std::int64_t(args.k / D::kBlockElems) * D::kBlockBytes || args.rows <= 0) {
        throw std::invalid_argument("gguf vector product: unsupported geometry");
    }
    const auto launch = [&]<bool Cooperative>() {
        if (fused) {
            vec::launch_columns<type, true, Cooperative>(args, columns, stream);
        } else {
            vec::launch_columns<type, false, Cooperative>(args, columns, stream);
        }
    };
    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K) {
        if (vec::cooperative_loads<type>(columns)) { return launch.template operator()<true>(); }
    }
    launch.template operator()<false>();
}

namespace vec {

// Value j of word e (value 4e + j of the slice) as dequantize_row_* gives it, from the decoded
// slice: each type's dot product above, with one activation set to one.
template <ggml_type type>
__device__ __forceinline__ float slice_value(const Slice& s, int e, int j) {
    const float c = float(static_cast<std::int8_t>(std::uint32_t(s.w[e]) >> (8 * j)));
    if constexpr (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        return s.f[0] * c - s.f[1];
    } else if constexpr (type == GGML_TYPE_Q6_K || type == GGML_TYPE_IQ2_XS ||
                         type == GGML_TYPE_IQ2_S || type == GGML_TYPE_NVFP4) {
        return (e < 4 ? s.f[0] : s.f[1]) * c;
    } else if constexpr (type == GGML_TYPE_Q2_K) {
        return e < 4 ? s.f[0] * c - s.f[2] : s.f[1] * c - s.f[3];
    } else if constexpr (type == GGML_TYPE_Q3_K) {
        const float h = float(static_cast<std::int8_t>(std::uint32_t(s.h[e]) >> (8 * j)));
        return (e < 4 ? s.f[0] : s.f[1]) * (c + h - 4.0f);
    } else if constexpr (type == GGML_TYPE_IQ1_S) {
        return s.f[0] * (c + s.f[1]);
    } else {
        return s.f[0] * c;
    }
}

template <typename dst_t>
__device__ __forceinline__ void store8(dst_t* out, const float (&v)[8]) {
    if constexpr (std::is_same_v<dst_t, float>) {
        reinterpret_cast<float4*>(out)[0] = make_float4(v[0], v[1], v[2], v[3]);
        reinterpret_cast<float4*>(out)[1] = make_float4(v[4], v[5], v[6], v[7]);
    } else {
        uint4 packed;
        std::uint32_t* words = reinterpret_cast<std::uint32_t*>(&packed);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if constexpr (std::is_same_v<dst_t, half>) {
                const half2 pair = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
                words[i]         = *reinterpret_cast<const std::uint32_t*>(&pair);
            } else {
                const __nv_bfloat162 pair = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
                words[i]                  = *reinterpret_cast<const std::uint32_t*>(&pair);
            }
        }
        *reinterpret_cast<uint4*>(out) = packed;
    }
}

// One thread a 32-value slice; consecutive threads take consecutive slices of a row, so a warp
// reads consecutive blocks and writes 2 KiB of FP16 in 16-byte stores. The generic kernel
// (ggml_bridge.cu) runs one 32-thread block a super-block with 2-byte stores, about a quarter of
// the write rate on a Tesla V100.
template <ggml_type type, typename dst_t>
__launch_bounds__(256) __global__
    void slice_dequantize_kernel(const std::uint8_t* __restrict__ weight, std::int64_t row_bytes,
                                 const std::int32_t* __restrict__ row_ids, int rows, int k,
                                 dst_t* __restrict__ out, std::int64_t out_row_stride) {
    using D                 = Decoder<type>;
    constexpr int kTable    = D::kTableWords > 0 ? D::kTableWords : 4;
    constexpr int kPerBlock = D::kBlockElems / 32;
    __shared__ __align__(16) std::uint32_t table[kTable];
    if constexpr (D::kTableWords > 0) {
        const std::uint32_t* src = D::table();
        for (int i = threadIdx.x; i < D::kTableWords; i += blockDim.x) { table[i] = src[i]; }
        __syncthreads();
    }
    const int slices          = k / 32;
    const std::int64_t total  = std::int64_t(rows) * slices;
    for (std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += std::int64_t(gridDim.x) * blockDim.x) {
        const int row         = int(i / slices);
        const int slice       = int(i % slices);
        const std::int64_t at = row_ids != nullptr ? row_ids[row] : row;
        Slice s;
        D::decode(weight + at * row_bytes + std::int64_t(slice / kPerBlock) * D::kBlockBytes,
                  slice % kPerBlock, table, s);
        dst_t* y = out + std::int64_t(row) * out_row_stride + 32 * std::int64_t(slice);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            float v[8];
#pragma unroll
            for (int t = 0; t < 8; ++t) { v[t] = slice_value<type>(s, 2 * q + t / 4, t % 4); }
            store8(y + 8 * q, v);
        }
    }
}

} // namespace vec

template <ggml_type type, typename dst_t>
void slice_dequantize(const void* weight, std::int64_t row_bytes, int k, const std::int32_t* row_ids,
                      int rows, dst_t* out, std::int64_t out_row_stride, cudaStream_t stream) {
    using D = vec::Decoder<type>;
    if (!kSliceDequantize<type> || k % D::kBlockElems != 0 || rows <= 0 ||
        out_row_stride % 8 != 0 || reinterpret_cast<std::uintptr_t>(out) % 16 != 0) {
        throw std::invalid_argument("gguf slice dequantize: unsupported type or geometry");
    }
    static const int resident = [] {
        int blocks = 0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &blocks, vec::slice_dequantize_kernel<type, dst_t>, 256, 0),
              "slice dequantize occupancy");
        return std::max(blocks, 1);
    }();
    const std::int64_t threads = std::int64_t(rows) * (k / 32);
    const int blocks = int(std::min<std::int64_t>((threads + 255) / 256,
                                                  std::int64_t(resident) * device_facts().sm_count));
    vec::slice_dequantize_kernel<type, dst_t><<<blocks, 256, 0, stream>>>(
        static_cast<const std::uint8_t*>(weight), row_bytes, row_ids, rows, k, out, out_row_stride);
    check(cudaGetLastError(), "slice dequantize launch");
}

} // namespace ninfer::ops::gguf::detail

#include "ggml_bridge_sm70.cuh"

#define NINFER_GGUF_SLICE_DEQUANTIZE_INSTANCE(TYPE, DST)                                           \
    template void ninfer::ops::gguf::detail::slice_dequantize<TYPE, DST>(                          \
        const void*, std::int64_t, int, const std::int32_t*, int, DST*, std::int64_t, cudaStream_t)

#define NINFER_GGUF_VECTOR_INSTANCE(TYPE)                                                          \
    template void ninfer::ops::gguf::detail::vec_launch<TYPE>(                                     \
        const ninfer::ops::gguf::detail::VecArgs&, int, bool, cudaStream_t);                       \
    template void ninfer::ops::gguf::detail::moe_launch<TYPE>(                                     \
        const ninfer::ops::gguf::detail::MoeVecArgs&, int, int, bool, cudaStream_t);               \
    template void ninfer::ops::gguf::detail::moe_decode_up_launch<TYPE>(                           \
        const ninfer::ops::gguf::MoeDecodeUpArgs&, int, cudaStream_t);                             \
    template void ninfer::ops::gguf::detail::moe_decode_down_launch<TYPE>(                         \
        const ninfer::ops::gguf::MoeDecodeDownArgs&, int, cudaStream_t);                           \
    template void ninfer::ops::gguf::detail::moe_decode_prep_launch<TYPE>(                         \
        const ninfer::ops::gguf::MoeDecodePrepArgs&, cudaStream_t);                                \
    NINFER_GGUF_SLICE_DEQUANTIZE_INSTANCE(TYPE, half);                                             \
    NINFER_GGUF_SLICE_DEQUANTIZE_INSTANCE(TYPE, __nv_bfloat16);                                    \
    NINFER_GGUF_SLICE_DEQUANTIZE_INSTANCE(TYPE, float)
