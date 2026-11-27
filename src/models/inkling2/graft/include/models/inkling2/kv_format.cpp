#include "models/inkling2/kv_format.h"

#include "models/kv_storage_name.h"

#include <stdexcept>
#include <string>

namespace ninfer::models::inkling2 {

KvFormat kv_format(KvCacheStorage storage) {
    constexpr KvPlane f16{};
    switch (storage) {
    // The unquantized cache is F16, which --kv-dtype fp16 names and bf16 (the default) selects.
    case KvCacheStorage::BFloat16:
    case KvCacheStorage::Float16: return {storage, f16, f16};
    // The eight full-attention visits only: int8 G64 keys and values.
    case KvCacheStorage::Int8Group64: return {storage, {KvCoding::Int8G64}, {KvCoding::Int8G64}};
    // rk8v4: int8 G64 keys, packed int4 G32 values.
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        return {storage, {KvCoding::Int8G64}, {KvCoding::Int4G32}};
    // rk4v4: 4-bit Lloyd-Max keys over the same value plane.
    case KvCacheStorage::RotatedLloyd4KeyInt4Value:
        return {storage, {KvCoding::Lloyd4}, {KvCoding::Int4G32}};
    // mxfp8: the release's MXFP8 keys and values.
    case KvCacheStorage::Mxfp8: return {storage, {KvCoding::Mxfp8}, {KvCoding::Mxfp8}};
    case KvCacheStorage::Fp8E4M3Row256:
    case KvCacheStorage::Nvfp4Group16:
    case KvCacheStorage::Fp8KeyNvfp4Value:
    case KvCacheStorage::RotatedInt4KeyInt4ValueE8:
    case KvCacheStorage::RotatedE8RootKeyInt4Value:
    case KvCacheStorage::Bf16Packed:
        throw std::invalid_argument(
            "Inkling 2: --kv-dtype " + std::string(kv_storage_name(storage)) +
            " is not built for this family (mxfp8, the default, int8, rk8v4 or rk4v4, or the F16 "
            "cache: bf16 or fp16)");
    }
    throw std::invalid_argument("Inkling 2: unknown KV-cache storage");
}

} // namespace ninfer::models::inkling2
