#pragma once

#include "ninfer/types.h"
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::test {

inline const char* kv_cache_storage_name(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8";
    }
    throw std::invalid_argument("unknown KV storage");
}

inline KvCacheStorage parse_kv_cache_storage(std::string_view name) {
    if (name == "bf16") return KvCacheStorage::BFloat16;
    if (name == "int8") return KvCacheStorage::Int8Group64;
    if (name == "fp8") return KvCacheStorage::Fp8E4M3Row256;
    throw std::invalid_argument("unknown KV dtype: " + std::string(name));
}

} // namespace ninfer::test
