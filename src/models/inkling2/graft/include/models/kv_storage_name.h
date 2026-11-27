#pragma once

// The --kv-dtype name of a KV-cache storage, for the families' refusals and startup lines.

#include "ninfer/kv_cache_storage.h"

namespace ninfer::models {

[[nodiscard]] constexpr const char* kv_storage_name(KvCacheStorage storage) noexcept {
    switch (storage) {
    case KvCacheStorage::BFloat16: return "bf16";
    case KvCacheStorage::Float16: return "fp16";
    case KvCacheStorage::Int8Group64: return "int8";
    case KvCacheStorage::Fp8E4M3Row256: return "fp8";
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64: return "rk8v4";
    case KvCacheStorage::Nvfp4Group16: return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value: return "k8v4";
    case KvCacheStorage::RotatedLloyd4KeyInt4Value: return "rk4v4";
    case KvCacheStorage::RotatedInt4KeyInt4ValueE8: return "rk4v4-e8";
    case KvCacheStorage::RotatedE8RootKeyInt4Value: return "rk2v4-e8";
    case KvCacheStorage::Bf16Packed: return "bf16p";
    case KvCacheStorage::Mxfp8: return "mxfp8";
    }
    return "unknown";
}

} // namespace ninfer::models
