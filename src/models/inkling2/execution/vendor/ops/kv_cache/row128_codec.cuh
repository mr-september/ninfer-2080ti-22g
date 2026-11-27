#pragma once

// Device side of the 128-wide KV planes (row128.h) over the engine's row codings
// (kv_row_codec.cuh): a warp encoding one row (one KV head of one position), and the eight-value
// chunk reads an attention's tile loader makes, the coding known at compile time or read from the
// plane.

#include "ops/kv_cache/kv_row_codec.cuh"
#include "ops/kv_cache/row128.h"

#include <cstdint>

namespace ninfer::ops::kv128 {

using kv_row::Chunk;
using kv_row::chunk_word;
using kv_row::hadamard128;
using kv_row::mxfp8_code;
using kv_row::mxfp8_exponent;

// One 128-wide row per warp: lane l holds dimensions l + 32 r of the row in x[r]. Encodes x into
// the row's codes and scales; reading rows back happens in an attention's tile loader, which
// decodes eight-value chunks as it needs them.
template <KvCoding C>
__device__ __forceinline__ void encode_row(const float (&x)[4], std::uint8_t* codes,
                                           std::uint8_t* scales, int lane) {
    static_assert(C == KvCoding::Int8G64 || C == KvCoding::Int4G32 || C == KvCoding::Lloyd4 ||
                      C == KvCoding::Mxfp8,
                  "a 128-wide plane coding");
    kv_row::encode_block<C, 4>(x, codes, scales, nullptr, lane);
}

// Chunk `c` of a row with the plane's coding read at run time. F16 never reaches here.
__device__ __forceinline__ uint4 chunk_word(const KvPlaneView& plane, const std::uint8_t* codes,
                                            const std::uint8_t* scales, int c) {
    switch (plane.coding) {
    case KvCoding::Int8G64: return chunk_word<KvCoding::Int8G64>(codes, scales, c);
    case KvCoding::Int4G32: return chunk_word<KvCoding::Int4G32>(codes, scales, c);
    case KvCoding::Lloyd4: return chunk_word<KvCoding::Lloyd4>(codes, scales, c);
    case KvCoding::Mxfp8: return chunk_word<KvCoding::Mxfp8>(codes, scales, c);
    default: break;
    }
    return make_uint4(0, 0, 0, 0);
}

} // namespace ninfer::ops::kv128
