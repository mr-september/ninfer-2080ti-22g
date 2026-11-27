#pragma once

// Inkling 2's KV-cache storage (--kv-dtype). A cache row is one KV head of one position, its
// 128 values in one block: codes and scales in two planes of a visit (executor.cpp). The codings
// are the engine's 128-wide row codings (ops/kv_cache/row128.h): Int8G64, Int4G32, Lloyd4 and
// MXFP8.
//
// Only the eight full-attention visits store planes (int8: Int8G64 keys and values; rk8v4:
// Int8G64 keys and Int4G32 values; rk4v4: Lloyd4 keys and Int4G32 values; mxfp8: Mxfp8 keys and
// values); the windowed visits' rings, the convolution histories and the DSpark drafter's caches
// stay F16 whatever the flag says. Rounding the rings' rows to MXFP8 as well raised WikiText's
// mean NLL (+0.029% against +0.024% over 67,977 tokens, +0.091% against +0.015% over 9,718) and
// saves no memory, since a ring's size does not grow with the context. The rows are stored
// unrotated: Inkling's K and V come out of their convolutions clipped and RMS-normed, and rotating
// the keys as Flint's rk4v4 does (row128.h) raised the mean NLL over 67,977 tokens, rk8v4's from
// 1.784735 to 1.784842 and rk4v4's from 1.785611 to 1.785690.

#include "ninfer/kv_cache_storage.h"
#include "ops/kv_cache/row128.h"

#include <cstdint>

namespace ninfer::models::inkling2 {

using ops::kv128::KvCoding;
using ops::kv128::kKvRowDims;
using ops::kv128::kv_code_bytes;
using ops::kv128::kv_scale_bytes;
using ops::kv128::KvPlaneView;

struct KvPlane {
    KvCoding coding = KvCoding::F16;
};

struct KvFormat {
    VendoredKvCacheStorage storage = VendoredKvCacheStorage::BFloat16;
    KvPlane key, value;
    [[nodiscard]] bool quantized() const noexcept {
        return storage != VendoredKvCacheStorage::BFloat16 && storage != VendoredKvCacheStorage::Float16;
    }
    // MXFP8: kv_post encodes the full visits' planes itself; the other quantized formats stage
    // F16 rows for kv_store.
    [[nodiscard]] bool mxfp8() const noexcept { return storage == VendoredKvCacheStorage::Mxfp8; }
    [[nodiscard]] bool staged() const noexcept { return quantized() && !mxfp8(); }
};

[[nodiscard]] KvFormat kv_format(VendoredKvCacheStorage storage);

} // namespace ninfer::models::inkling2
