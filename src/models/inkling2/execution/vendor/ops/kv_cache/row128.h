#pragma once

// Quantized KV planes of 128-wide rows (one KV head of one position), as Inkling 2's full-attention
// visits and Flint's full-attention blocks store them: a row's codes and scales lie in two planes,
// in the engine's row codings (kv_row.h) at D = 128: Int8G64, Int4G32, Lloyd4 (rk4v4 keys) and
// Mxfp8.
//
// A plane stores its rows as they are, or rotated: each row as H x / 8, H the unnormalized
// 128-point Sylvester-Hadamard transform, the attention turning its queries by H / 16 instead (the
// two scales multiply to 1/128, so the scores keep their values). Device side: row128_codec.cuh.

#include "ops/kv_cache/kv_row.h"

#include <cstdint>

namespace ninfer::ops::kv128 {

using kv_row::KvCoding;

// One row: one KV head's 128 values, so 128-wide blocks.
inline constexpr int kKvRowDims = 128;

// Bytes of one row's codes and of its scales.
[[nodiscard]] NINFER_KV_ROW_HOST_DEVICE constexpr int kv_code_bytes(KvCoding c) noexcept {
    return kv_row::code_bytes(c, kKvRowDims);
}

[[nodiscard]] NINFER_KV_ROW_HOST_DEVICE constexpr int kv_scale_bytes(KvCoding c) noexcept {
    return kv_row::scale_bytes(c, kKvRowDims);
}

// A rotated plane stores H x times kRotatedRowScale; its queries are turned by H times
// kRotatedQueryScale (row128_rotate.h).
inline constexpr float kRotatedRowScale = 0.125f, kRotatedQueryScale = 0.0625f;

// A quantized plane: one row's codes at codes + sequence * code_seq + head * code_head +
// slot * code_row, its scales likewise.
struct KvPlaneView {
    std::uint8_t* codes  = nullptr;
    std::uint8_t* scales = nullptr;
    std::int64_t code_row = 0, scale_row = 0, code_head = 0, scale_head = 0, code_seq = 0,
                 scale_seq = 0;
    KvCoding coding = KvCoding::F16;
    bool rotated    = false;
};

} // namespace ninfer::ops::kv128
