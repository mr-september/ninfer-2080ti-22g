#pragma once

// The small-T product of an sm_70 build (sm70_small.h) on Volta's tensor cores. A warp owns a
// 32-row tile: mma.m8n8k4 runs four 8x8x4 products at once, one per quad pair, and with every
// quad pair holding the same activation fragment ("mirrored" B) the four together cover 32 weight
// rows times 8 columns per instruction. Each lane decodes the weights of its own row straight into
// its A fragment, so no weight passes through shared memory; the warps of a CTA take K in turns of
// kUnroll consecutive chunks (a lane reads kUnroll records of its row at once) and add their
// partial tiles in shared memory at the end. A row-split format's codes enter the products unscaled
// (small integers, exact in FP16) and each group's FP32 partial products are scaled by the group's
// scale as the group ends.
//
// PTX fragment layouts of mma.m8n8k4.row.col (quad pair q: lanes 4q..4q+3 and 4q+16..4q+19,
// h = lane / 16): A row (lane % 4) + 4h with k 0..3; B column (lane % 4) + 4h with k 0..3; FP32
// accumulator i at row (lane & 1) + (i & 2) + 4h, column (i & 4) + (lane & 2) + (i & 1).
//
// The dot product runs over every eight values of K in the order 0, 4, 1, 5, 2, 6, 3, 7 ("P8"):
// the two products of an eight-value group take k (0, 4, 1, 5) and (2, 6, 3, 7). In that order the
// 16-bit lanes of a packed code word hold codes j and j + 4, which one LOP3 or PRMT moves into an
// FP16 pair, so a decoder spends one or two instructions per pair of weights. The activations are
// stored in the same order: the prepared FP16 plane of wider calls (convert_activation_p8), and the
// shared-memory stage of direct BF16 calls.

#include "ops/linear/sm70/sm70_dequantize.h"
#include "ops/linear/sm70/sm70_bf16_f16.cuh"
#include "ops/linear/sm70/sm70_gemm.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::detail::sm70 {

// Warps per CTA, splitting K: eight, or sixteen where too few row tiles would leave the SMs short
// of warps (kSmallWideWarps, see small_warps in sm70_small.cu).
inline constexpr int kSmallWarps     = 8;
inline constexpr int kSmallWideWarps = 16;
inline constexpr int kSmallTileRows  = 32;
inline constexpr int kSmallChunk     = 64; // the K granularity every format's chunk divides

__device__ __forceinline__ void hmma884(float (&c)[8], unsigned a0, unsigned a1, unsigned b0,
                                        unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]),
                   "+f"(c[6]), "+f"(c[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#else
    (void)c;
    (void)a0;
    (void)a1;
    (void)b0;
    (void)b1;
#endif
}

__device__ __forceinline__ unsigned half2_bits(__half2 value) {
    return *reinterpret_cast<const unsigned*>(&value);
}

__device__ __forceinline__ __half2 bits_half2(unsigned bits) {
    return *reinterpret_cast<const __half2*>(&bits);
}

// (a & b) | c in one instruction.
__device__ __forceinline__ unsigned and_or(unsigned a, unsigned b, unsigned c) {
    unsigned d;
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ unsigned permute(unsigned a, unsigned b, unsigned selector) {
    return __byte_perm(a, b, selector);
}

inline constexpr unsigned kMagic = 0x64006400u; // FP16 1024 in both lanes: 0x6400 | u is 1024 + u

// The FP16 pair (u0 - bias, u1 - bias) of a magic pair (1024 + u0, 1024 + u1), exact.
__device__ __forceinline__ unsigned unbias(unsigned magic, __half2 bias) {
    return half2_bits(__hsub2(bits_half2(magic), bias));
}

// Four FP16 pairs of one P8 group: a[0..1] the first product's A fragment, a[2..3] the second's.
using Fragments = unsigned[4];

// Decoder contract: kChunk values of a row per record chunk (load), kUnroll chunks per lane step,
// decode(raw, g, a) the fragments of P8 group g of a chunk. A format with group scales sets
// kScaleGroups to the P8 groups one scale covers and scale(raw, s) to the lane's scale of the
// chunk's s-th scale group, and decodes unscaled codes; kScaleGroups == 0 decodes final values (up
// to the row's RowFactor).

// Row-split integer formats (row_split_k128_v1, row-major or panel-permuted): a lane's 64-k chunk
// is one G64 group (Q2, Q4, Q5, Q6), two G32 groups (Q8) or half a G128 group (T2).
struct RowSplitView {
    const std::uint8_t* base    = nullptr;
    const std::uint8_t* high    = nullptr;
    const __half* scales        = nullptr;
    std::int32_t groups_per_row = 0;
    std::int32_t panel_shift    = 0;

    __device__ __forceinline__ const std::uint8_t* record(const std::uint8_t* plane, int bytes,
                                                          std::int32_t row, int group) const {
        const std::int64_t mask = (std::int64_t{1} << panel_shift) - 1;
        return plane + ((row & ~mask) * groups_per_row + (row & mask)) * bytes +
               static_cast<std::int64_t>(group) * (bytes << panel_shift);
    }

    __device__ __forceinline__ float scale(std::int32_t row, int group) const {
        return __half2float(scales[static_cast<std::int64_t>(row) * groups_per_row + group]);
    }
};

// Nibble codes j and j + 4 of word w, ORed onto c[j], in the lanes of pair j (j = 0..3).
__device__ __forceinline__ void nibble_pairs(unsigned w, const unsigned (&c)[4], unsigned (&h)[4]) {
    h[0] = and_or(w, 0x000f000fu, c[0]);
    h[1] = and_or(w >> 4, 0x000f000fu, c[1]);
    h[2] = and_or(w >> 8, 0x000f000fu, c[2]);
    h[3] = and_or(w >> 12, 0x000f000fu, c[3]);
}

struct Q4Decoder {
    static constexpr int kChunk       = 64;
    static constexpr int kUnroll      = 2;
    static constexpr int kScaleGroups = 8;
    RowSplitView view;

    struct Raw {
        uint4 codes[2];
        float scale;
    };

    __device__ __forceinline__ float scale(const Raw& raw, int) const { return raw.scale; }

    __device__ __forceinline__ Raw load(std::int32_t row, int chunk) const {
        const uint4* p = reinterpret_cast<const uint4*>(view.record(view.base, 32, row, chunk));
        Raw raw;
        raw.codes[0] = __ldg(p);
        raw.codes[1] = __ldg(p + 1);
        raw.scale    = view.scale(row, chunk);
        return raw;
    }

    // Group g: codes 8g..8g+7 in the nibbles of word g, two's complement (bit 3 flipped to
    // offset 8).
    __device__ __forceinline__ void decode(const Raw& raw, int g, Fragments& a) const {
        const unsigned w        = reinterpret_cast<const unsigned*>(raw.codes)[g] ^ 0x88888888u;
        const unsigned magic[4] = {kMagic, kMagic, kMagic, kMagic};
        unsigned h[4];
        nibble_pairs(w, magic, h);
        const __half2 bias = __float2half2_rn(1032.0f);
#pragma unroll
        for (int j = 0; j < 4; ++j) { a[j] = unbias(h[j], bias); }
    }
};

struct Q5Decoder {
    static constexpr int kChunk       = 64;
    static constexpr int kUnroll      = 2;
    static constexpr int kScaleGroups = 8;
    RowSplitView view;

    struct Raw {
        uint4 codes[2];
        uint2 high;
        float scale;
    };

    __device__ __forceinline__ float scale(const Raw& raw, int) const { return raw.scale; }

    __device__ __forceinline__ Raw load(std::int32_t row, int chunk) const {
        const uint4* p = reinterpret_cast<const uint4*>(view.record(view.base, 32, row, chunk));
        Raw raw;
        raw.codes[0] = __ldg(p);
        raw.codes[1] = __ldg(p + 1);
        raw.high     = __ldg(reinterpret_cast<const uint2*>(view.record(view.high, 8, row, chunk)));
        raw.scale    = view.scale(row, chunk);
        return raw;
    }

    // Five-bit two's complement codes: high byte g holds bit 4 of codes 8g..8g+7, flipped to
    // offset 16. Spread as hb << 4 | hb << 16, bit j lands at bits 4 + j and 16 + j: shifted right
    // by j, bits 4 and 20 are codes j and j + 4.
    __device__ __forceinline__ void decode(const Raw& raw, int g, Fragments& a) const {
        const unsigned w = reinterpret_cast<const unsigned*>(raw.codes)[g];
        const unsigned hb =
            (~reinterpret_cast<const unsigned*>(&raw.high)[g >> 2] >> ((g & 3) * 8)) & 0xffu;
        const unsigned y    = hb * 0x10010u;
        const unsigned c[4] = {and_or(y, 0x00100010u, kMagic), and_or(y >> 1, 0x00100010u, kMagic),
                               and_or(y >> 2, 0x00100010u, kMagic),
                               and_or(y >> 3, 0x00100010u, kMagic)};
        unsigned h[4];
        nibble_pairs(w, c, h);
        const __half2 bias = __float2half2_rn(1040.0f);
#pragma unroll
        for (int j = 0; j < 4; ++j) { a[j] = unbias(h[j], bias); }
    }
};

struct Q6Decoder {
    static constexpr int kChunk       = 64;
    static constexpr int kUnroll      = 2;
    static constexpr int kScaleGroups = 8;
    RowSplitView view;

    struct Raw {
        uint4 codes[2];
        uint4 high;
        float scale;
    };

    __device__ __forceinline__ float scale(const Raw& raw, int) const { return raw.scale; }

    __device__ __forceinline__ Raw load(std::int32_t row, int chunk) const {
        const uint4* p = reinterpret_cast<const uint4*>(view.record(view.base, 32, row, chunk));
        Raw raw;
        raw.codes[0] = __ldg(p);
        raw.codes[1] = __ldg(p + 1);
        raw.high  = __ldg(reinterpret_cast<const uint4*>(view.record(view.high, 16, row, chunk)));
        raw.scale = view.scale(row, chunk);
        return raw;
    }

    // Six-bit two's complement codes: high bytes 2g and 2g + 1 hold bits 4..5 of codes 8g..8g+3
    // and 8g+4..8g+7, two bits each, bit 5 flipped to offset 32. Spread as lo << 4 | hi << 20,
    // field j lands at bits 4 + 2j and 20 + 2j.
    __device__ __forceinline__ void decode(const Raw& raw, int g, Fragments& a) const {
        const unsigned w    = reinterpret_cast<const unsigned*>(raw.codes)[g];
        const unsigned word = reinterpret_cast<const unsigned*>(&raw.high)[g >> 1] ^ 0xaaaaaaaau;
        const unsigned y    = permute(word, 0u, (g & 1) ? 0x4342u : 0x4140u) << 4;
        const unsigned c[4] = {and_or(y, 0x00300030u, kMagic), and_or(y >> 2, 0x00300030u, kMagic),
                               and_or(y >> 4, 0x00300030u, kMagic),
                               and_or(y >> 6, 0x00300030u, kMagic)};
        unsigned h[4];
        nibble_pairs(w, c, h);
        const __half2 bias = __float2half2_rn(1056.0f);
#pragma unroll
        for (int j = 0; j < 4; ++j) { a[j] = unbias(h[j], bias); }
    }
};

// Bytes j and j + 4 of (w0, w1) as 1024 + byte in the lanes of pair j.
__device__ __forceinline__ void byte_pairs(unsigned w0, unsigned w1, unsigned (&h)[4]) {
    constexpr unsigned kBytes = 0x64646464u;
    const unsigned even       = permute(w0, w1, 0x5140u); // bytes 0, 4, 1, 5
    const unsigned odd        = permute(w0, w1, 0x7362u); // bytes 2, 6, 3, 7
    h[0]                      = permute(even, kBytes, 0x4140u);
    h[1]                      = permute(even, kBytes, 0x4342u);
    h[2]                      = permute(odd, kBytes, 0x4140u);
    h[3]                      = permute(odd, kBytes, 0x4342u);
}

struct Q8Decoder {
    static constexpr int kChunk       = 64;
    static constexpr int kUnroll      = 2;
    static constexpr int kScaleGroups = 4;
    RowSplitView view;

    struct Raw {
        uint4 codes[4];
        float scale[2];
    };

    __device__ __forceinline__ float scale(const Raw& raw, int s) const { return raw.scale[s]; }

    __device__ __forceinline__ Raw load(std::int32_t row, int chunk) const {
        Raw raw;
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const uint4* p =
                reinterpret_cast<const uint4*>(view.record(view.base, 32, row, 2 * chunk + g));
            raw.codes[2 * g]     = __ldg(p);
            raw.codes[2 * g + 1] = __ldg(p + 1);
            raw.scale[g]         = view.scale(row, 2 * chunk + g);
        }
        return raw;
    }

    // Signed codes, offset to 128 + q.
    __device__ __forceinline__ void decode(const Raw& raw, int g, Fragments& a) const {
        const unsigned* words = reinterpret_cast<const unsigned*>(raw.codes);
        unsigned h[4];
        byte_pairs(words[2 * g] ^ 0x80808080u, words[2 * g + 1] ^ 0x80808080u, h);
        const __half2 bias = __float2half2_rn(1152.0f);
#pragma unroll
        for (int j = 0; j < 4; ++j) { a[j] = unbias(h[j], bias); }
    }
};

// Q2 (u = q + 1) and T2 (two's complement over {0, 1, 3}, flipped to {2, 3, 1} = value + 2):
// four codes per byte. A T2 chunk is a whole G128 group (32 bytes), a Q2 chunk one G64 group.
template <bool Ternary>
struct Quartet2Decoder {
    static constexpr int kChunk       = Ternary ? 128 : 64;
    static constexpr int kUnroll      = Ternary ? 1 : 2;
    static constexpr int kScaleGroups = kChunk / 8;
    RowSplitView view;

    struct Raw {
        uint4 codes[kChunk / 64];
        float scale;
    };

    __device__ __forceinline__ float scale(const Raw& raw, int) const { return raw.scale; }

    __device__ __forceinline__ Raw load(std::int32_t row, int chunk) const {
        Raw raw;
        const uint4* p =
            reinterpret_cast<const uint4*>(view.record(view.base, kChunk / 4, row, chunk));
#pragma unroll
        for (int i = 0; i < kChunk / 64; ++i) { raw.codes[i] = __ldg(p + i); }
        raw.scale = view.scale(row, chunk);
        return raw;
    }

    // Group g: codes 8g..8g+7 in the sixteen bits from bit 16 (g & 1) of word g / 2. Its two bytes
    // spread to the two lanes; shifted right by 2j, bits 0..1 of each lane are codes j and j + 4.
    __device__ __forceinline__ void decode(const Raw& raw, int g, Fragments& a) const {
        unsigned w = reinterpret_cast<const unsigned*>(raw.codes)[g >> 1];
        if constexpr (Ternary) { w ^= 0xaaaaaaaau; }
        const unsigned y = permute(w, 0u, (g & 1) ? 0x4342u : 0x4140u);
        unsigned h[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) { h[j] = and_or(y >> (2 * j), 0x00030003u, kMagic); }
        const __half2 bias = __float2half2_rn(Ternary ? 1026.0f : 1025.0f);
#pragma unroll
        for (int j = 0; j < 4; ++j) { a[j] = unbias(h[j], bias); }
    }
};

// The FP16 pairs (k0, k1), (k2, k3), (k4, k5), (k6, k7) of a group in P8 order.
__device__ __forceinline__ void p8_from_natural(unsigned f01, unsigned f23, unsigned f45,
                                                unsigned f67, Fragments& a) {
    a[0] = permute(f01, f45, 0x5410u);
    a[1] = permute(f01, f45, 0x7632u);
    a[2] = permute(f23, f67, 0x5410u);
    a[3] = permute(f23, f67, 0x7632u);
}

struct Bf16Decoder {
    static constexpr int kChunk       = 64;
    static constexpr int kUnroll      = 1;
    static constexpr int kScaleGroups = 0;
    const __nv_bfloat16* weights      = nullptr;
    std::int32_t k                    = 0;

    struct Raw {
        uint4 v[8];
    };

    __device__ __forceinline__ Raw load(std::int32_t row, int chunk) const {
        const uint4* p = reinterpret_cast<const uint4*>(
            weights + static_cast<std::int64_t>(row) * k + chunk * kSmallChunk);
        Raw raw;
#pragma unroll
        for (int i = 0; i < 8; ++i) { raw.v[i] = __ldg(p + i); }
        return raw;
    }

    __device__ __forceinline__ void decode(const Raw& raw, int g, Fragments& a) const {
        const uint4 v = raw.v[g];
        p8_from_natural(bf16x2_to_f16x2(v.x), bf16x2_to_f16x2(v.y), bf16x2_to_f16x2(v.z),
                        bf16x2_to_f16x2(v.w), a);
    }
};

// E4M3 to FP16 times 2^-8 in both lanes: the code's seven low bits shifted to FP16's exponent and
// mantissa fields, the sign moved to bit 15 (exact for every finite code, subnormals included). The
// factor 256 joins the row's RowFactor.
__device__ __forceinline__ unsigned e4m3_lanes_to_f16x2_scaled(unsigned lanes) {
    return ((lanes & 0x007f007fu) << 7) | ((lanes & 0x00800080u) << 8);
}

struct Fp8Decoder {
    static constexpr int kChunk        = 64;
    static constexpr int kUnroll       = 2;
    static constexpr int kScaleGroups  = 0;
    static constexpr float kValueScale = 256.0f; // e4m3_lanes_to_f16x2_scaled's 2^-8
    const std::uint8_t* codes          = nullptr;
    std::int32_t k                     = 0;

    struct Raw {
        uint4 v[4];
    };

    __device__ __forceinline__ Raw load(std::int32_t row, int chunk) const {
        const uint4* p = reinterpret_cast<const uint4*>(codes + static_cast<std::int64_t>(row) * k +
                                                        chunk * kSmallChunk);
        Raw raw;
#pragma unroll
        for (int i = 0; i < 4; ++i) { raw.v[i] = __ldg(p + i); }
        return raw;
    }

    __device__ __forceinline__ void decode(const Raw& raw, int g, Fragments& a) const {
        const unsigned* words = reinterpret_cast<const unsigned*>(raw.v);
        const unsigned even   = permute(words[2 * g], words[2 * g + 1], 0x5140u);
        const unsigned odd    = permute(words[2 * g], words[2 * g + 1], 0x7362u);
        a[0]                  = e4m3_lanes_to_f16x2_scaled(permute(even, 0u, 0x4140u));
        a[1]                  = e4m3_lanes_to_f16x2_scaled(permute(even, 0u, 0x4342u));
        a[2]                  = e4m3_lanes_to_f16x2_scaled(permute(odd, 0u, 0x4140u));
        a[3]                  = e4m3_lanes_to_f16x2_scaled(permute(odd, 0u, 0x4342u));
    }
};

struct Nvfp4Decoder {
    static constexpr int kChunk       = 64;
    static constexpr int kUnroll      = 2;
    static constexpr int kScaleGroups = 0;
    const std::uint8_t* codes         = nullptr;
    const std::uint8_t* scales        = nullptr;
    std::int32_t k                    = 0;

    struct Raw {
        uint4 v[2];
        unsigned scale_bytes;
    };

    __device__ __forceinline__ Raw load(std::int32_t row, int chunk) const {
        const uint4* p = reinterpret_cast<const uint4*>(
            codes + static_cast<std::int64_t>(row) * (k / 2) + chunk * 32);
        Raw raw;
        raw.v[0]              = __ldg(p);
        raw.v[1]              = __ldg(p + 1);
        const int inner       = row & 127;
        const std::int64_t at = (static_cast<std::int64_t>(row >> 7) * (k / 64) + chunk) * 512 +
                                (inner & 31) * 16 + (inner >> 5) * 4;
        raw.scale_bytes       = __ldg(reinterpret_cast<const unsigned*>(scales + at));
        return raw;
    }

    // Group g: codes 8g..8g+7 in the nibbles of word g, scaled by the block scale of 16-value
    // block g / 2.
    __device__ __forceinline__ void decode(const Raw& raw, int g, Fragments& a) const {
        const unsigned w = reinterpret_cast<const unsigned*>(raw.v)[g];
        __nv_fp8_e4m3 scale_word;
        scale_word.__x =
            static_cast<__nv_fp8_storage_t>((raw.scale_bytes >> ((g >> 1) * 8)) & 0xffu);
        const float scale = float(scale_word);
        unsigned natural[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            __nv_fp4x2_e2m1 pair;
            pair.__x       = static_cast<__nv_fp4x2_storage_t>((w >> (8 * p)) & 0xffu);
            const float2 f = static_cast<float2>(pair);
            natural[p]     = half2_bits(__floats2half2_rn(f.x * scale, f.y * scale));
        }
        p8_from_natural(natural[0], natural[1], natural[2], natural[3], a);
    }
};

// A warp's activation stage of a direct call: one chunk of eight columns in P8 FP16, each column's
// values followed by 8 of padding (no LDS.128 bank conflicts between columns).
template <int Chunk>
inline constexpr int kStageColumn = Chunk + 8;

// The CTA's shared memory: the warps' activation stages (direct calls), and afterwards their
// partial tiles [32 rows][8 * Tiles columns] FP32, summed over the warps.
template <int Tiles, int Chunk, int Warps>
union SmallShared {
    __half stage[Warps][8 * kStageColumn<Chunk>];
    float partial[Warps][kSmallTileRows][8 * Tiles];
};

// Stages chunk `chunk` (Chunk values) of the direct BF16 columns [col0, col0 + cols) in P8 FP16:
// lane l converts values 8g + p and 8g + p + 4 (g = l / 4 + 8 m, p = l % 4) of every column.
// Returns, in every lane, whether the chunk holds a value beyond FP16.
template <int Chunk>
__device__ __forceinline__ bool stage_direct(const std::uint16_t* __restrict__ x, std::int32_t k,
                                             std::int32_t col0, std::int32_t cols, int chunk,
                                             __half* stage) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int p    = lane & 3;
    bool beyond    = false;
    for (int c = 0; c < cols; ++c) {
        const std::uint16_t* column = x + static_cast<std::int64_t>(col0 + c) * k + chunk * Chunk;
#pragma unroll
        for (int m = 0; m < Chunk / 64; ++m) {
            const int g              = (lane >> 2) + 8 * m;
            const std::uint16_t* src = column + 8 * g + p;
            const unsigned word =
                static_cast<unsigned>(__ldg(src)) | (static_cast<unsigned>(__ldg(src + 4)) << 16);
            beyond |= bf16x2_beyond_f16(word);
            *reinterpret_cast<unsigned*>(stage + c * kStageColumn<Chunk> + 8 * g + 2 * p) =
                bf16x2_to_f16x2(word);
        }
    }
    return __any_sync(0xffffffffu, beyond);
}

// Whether chunk `chunk` of any of the tile's live columns holds a value beyond FP16, from the
// prepared plane's flags (one per 64 values), in every lane.
template <int Chunk, int Tiles>
__device__ __forceinline__ bool flagged_chunk(const std::uint8_t* __restrict__ beyond,
                                              std::int32_t k, std::int32_t col0,
                                              const bool (&live)[Tiles], int column, int chunk) {
    const std::int32_t chunks = k / kSmallChunk;
    bool any                  = false;
#pragma unroll
    for (int j = 0; j < Tiles; ++j) {
        if (!live[j]) { continue; }
        const std::uint8_t* flags =
            beyond + static_cast<std::int64_t>(col0 + 8 * j + column) * chunks + chunk * (Chunk / 64);
#pragma unroll
        for (int s = 0; s < Chunk / 64; ++s) { any |= __ldg(flags + s) != 0; }
    }
    return __any_sync(0xffffffffu, any);
}

// Adds to the tile what saturating the chunk's activations to FP16 took from its products: each lane
// forms its own row times the excess (saturated_excess) of every live column in FP32, scaled by its
// own group scales as the fast path scales its groups, and the sums move by shuffles into the
// accumulator layout (row (lane & 1) + (i & 2) + 4h of the lane's quad pair, column (i & 4) +
// (lane & 2) + (i & 1)). xb is the BF16 plane [T][k] in natural order.
template <class Decoder, int Tiles>
__device__ __forceinline__ void add_saturated(const Decoder& dec,
                                              const typename Decoder::Raw& raw,
                                              const std::uint16_t* __restrict__ xb, std::int32_t k,
                                              std::int32_t col0, std::int32_t cols, int chunk,
                                              float (&acc)[Tiles][8]) {
    constexpr int Chunk  = Decoder::kChunk;
    constexpr int Groups = Chunk / 8;
    constexpr int Span   = Decoder::kScaleGroups != 0 ? Decoder::kScaleGroups : 1;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int source     = (lane & 0x1c) | (lane & 1);
#pragma unroll
    for (int j = 0; j < Tiles; ++j) {
#pragma unroll
        for (int cc = 0; cc < 8; ++cc) {
            const std::int32_t c = 8 * j + cc;
            float sum            = 0.0f;
            if (c < cols) {
                const uint4* column = reinterpret_cast<const uint4*>(
                    xb + static_cast<std::int64_t>(col0 + c) * k + chunk * Chunk);
                float part = 0.0f;
#pragma unroll 1
                for (int g = 0; g < Groups; ++g) {
                    // a[p] holds weights p and p + 4 of the group, v its natural pairs.
                    Fragments a;
                    dec.decode(raw, g, a);
                    const uint4 v           = __ldg(column + g);
                    const unsigned words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                    for (int pr = 0; pr < 4; ++pr) {
                        const float2 w = __half22float2(bits_half2(a[pr]));
                        const unsigned lo_word = words[pr / 2];
                        const unsigned hi_word = words[2 + pr / 2];
                        const float lo = __uint_as_float(pr % 2 == 0 ? lo_word << 16
                                                                     : lo_word & 0xffff0000u);
                        const float hi = __uint_as_float(pr % 2 == 0 ? hi_word << 16
                                                                     : hi_word & 0xffff0000u);
                        part = fmaf(w.x, saturated_excess(lo), part);
                        part = fmaf(w.y, saturated_excess(hi), part);
                    }
                    if (g % Span == Span - 1) {
                        if constexpr (Decoder::kScaleGroups != 0) {
                            sum = fmaf(part, dec.scale(raw, g / Span), sum);
                        } else {
                            sum += part;
                        }
                        part = 0.0f;
                    }
                }
            }
            const float row_a = __shfl_sync(0xffffffffu, sum, source);
            const float row_b = __shfl_sync(0xffffffffu, sum, source | 2);
            if ((lane & 2) == (cc & 2)) {
                acc[j][cc & 5] += row_a;
                acc[j][(cc & 5) | 2] += row_b;
            }
        }
    }
}

// Rows [row0, row0 + 32) of the tile; the lane's row is row0 + 8 q + (lane % 4) + 4 h. Columns
// [col0, col0 + cols) of x: a prepared P8 FP16 plane [T][k] with its chunk flags `beyond`, or
// (Direct, cols <= 8) the caller's BF16 plane, staged per chunk. xb is the BF16 plane (x itself when
// Direct): a chunk with a value beyond FP16 adds what saturation took (add_saturated). `first`
// decodes tile rows below `split`, `second` the rest (the gate and up halves of a SwiGLU tile; one
// decoder when both are the same). The sum over the CTA's warps is left in shared.partial[0].
template <class Decoder, int Tiles, bool Direct, int Warps>
__device__ __forceinline__ void
small_tile(const Decoder& first, const Decoder& second, std::int32_t first_row0,
           std::int32_t second_row0, int split, std::int32_t first_rows, std::int32_t second_rows,
           const std::uint16_t* __restrict__ x, const std::uint16_t* __restrict__ xb,
           const std::uint8_t* __restrict__ beyond, std::int32_t k, std::int32_t col0,
           std::int32_t cols, SmallShared<Tiles, Decoder::kChunk, Warps>& shared) {
    static_assert(!Direct || Tiles == 1);
    constexpr int U      = Decoder::kUnroll;
    constexpr int Chunk  = Decoder::kChunk;
    constexpr int Groups = Chunk / 8;
    constexpr int Scaled = Decoder::kScaleGroups;
    constexpr int Span   = Scaled != 0 ? Scaled : 1; // P8 groups per scale group
    static_assert(Groups % Span == 0);
    const int lane           = static_cast<int>(threadIdx.x) & 31;
    const int warp           = static_cast<int>(threadIdx.x) >> 5;
    const int q              = (lane >> 2) & 3;
    const int h              = lane >> 4;
    const int tile_row       = 8 * q + (lane & 3) + 4 * h;
    const bool upper         = tile_row >= split;
    const Decoder& dec       = upper ? second : first;
    const std::int32_t limit = upper ? second_rows : first_rows;
    std::int32_t row         = upper ? second_row0 + tile_row - split : first_row0 + tile_row;
    row                      = row < limit ? row : limit - 1;
    const int column         = (lane & 3) + 4 * h;
    // The lanes holding the A rows of this lane's accumulator rows (lane & 1) + (i & 2) + 4h.
    const int scale_lane = (lane & ~3) | (lane & 1);
    float acc[Tiles][8];
#pragma unroll
    for (int j = 0; j < Tiles; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { acc[j][i] = 0.0f; }
    }
    bool live[Tiles];
    const std::uint16_t* xcol[Tiles];
#pragma unroll
    for (int j = 0; j < Tiles; ++j) {
        const std::int32_t c = 8 * j + column;
        live[j]              = c < cols;
        xcol[j]              = x + static_cast<std::int64_t>(col0 + (live[j] ? c : 0)) * k;
    }
    const int chunks = k / Chunk;
    __half* stage    = shared.stage[warp];
    // The warps take turns of U consecutive chunks: lane reads of U records of one row at once.
    for (int first_chunk = warp * U; first_chunk < chunks; first_chunk += U * Warps) {
        typename Decoder::Raw raw[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            if (first_chunk + u < chunks) { raw[u] = dec.load(row, first_chunk + u); }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int chunk = first_chunk + u;
            if (chunk >= chunks) { break; }
            bool saturated = false;
            if constexpr (Direct) {
                __syncwarp();
                saturated = stage_direct<Chunk>(x, k, col0, cols, chunk, stage);
                __syncwarp();
            } else {
                saturated = flagged_chunk<Chunk>(beyond, k, col0, live, column, chunk);
            }
            float part[Tiles][8];
#pragma unroll
            for (int g = 0; g < Groups; ++g) {
                if constexpr (Scaled != 0) {
                    if (g % Span == 0) {
#pragma unroll
                        for (int j = 0; j < Tiles; ++j) {
#pragma unroll
                            for (int i = 0; i < 8; ++i) { part[j][i] = 0.0f; }
                        }
                    }
                }
                uint4 b[Tiles];
#pragma unroll
                for (int j = 0; j < Tiles; ++j) {
                    if constexpr (Direct) {
                        b[j] = live[j] ? *reinterpret_cast<const uint4*>(
                                             stage + column * kStageColumn<Chunk> + 8 * g)
                                       : make_uint4(0, 0, 0, 0);
                    } else {
                        b[j] =
                            live[j]
                                ? __ldg(reinterpret_cast<const uint4*>(xcol[j] + chunk * Chunk) + g)
                                : make_uint4(0, 0, 0, 0);
                    }
                }
                Fragments a;
                dec.decode(raw[u], g, a);
#pragma unroll
                for (int j = 0; j < Tiles; ++j) {
                    float (&target)[8] = Scaled != 0 ? part[j] : acc[j];
                    hmma884(target, a[0], a[1], b[j].x, b[j].y);
                    hmma884(target, a[2], a[3], b[j].z, b[j].w);
                }
                if constexpr (Scaled != 0) {
                    if (g % Span == Span - 1) {
                        const float own = dec.scale(raw[u], g / Span);
                        const float s0  = __shfl_sync(0xffffffffu, own, scale_lane);
                        const float s2  = __shfl_sync(0xffffffffu, own, scale_lane | 2);
#pragma unroll
                        for (int j = 0; j < Tiles; ++j) {
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                acc[j][i] = fmaf(part[j][i], (i & 2) ? s2 : s0, acc[j][i]);
                            }
                        }
                    }
                }
            }
            if (saturated) { add_saturated(dec, raw[u], xb, k, col0, cols, chunk, acc); }
        }
    }
    __syncthreads(); // every warp is done with its stage before the partial tiles overwrite them
#pragma unroll
    for (int j = 0; j < Tiles; ++j) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r                = 8 * q + (lane & 1) + (i & 2) + 4 * h;
            const int c                = 8 * j + (i & 4) + (lane & 2) + (i & 1);
            shared.partial[warp][r][c] = acc[j][i];
        }
    }
    __syncthreads();
    if (warp != 0) {
        // Only warp 0 reduces; the others' shares are already in shared memory.
        return;
    }
#pragma unroll
    for (int e = lane; e < kSmallTileRows * 8 * Tiles; e += 32) {
        const int r = e / (8 * Tiles);
        const int c = e % (8 * Tiles);
        float sum   = 0.0f;
#pragma unroll
        for (int w = 1; w < Warps; ++w) { sum += shared.partial[w][r][c]; }
        shared.partial[0][r][c] += sum;
    }
    __syncwarp();
}

} // namespace ninfer::ops::detail::sm70
