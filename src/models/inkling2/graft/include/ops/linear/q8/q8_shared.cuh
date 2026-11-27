#pragma once
#include "core/device_launch.h"
#include "ops/common/device_limits.h"

namespace ninfer::ops::detail {
template <int Bytes>
__device__ __forceinline__ unsigned char* q8_shared_storage() {
    static_assert(Bytes <= kMaxSharedBytesPerBlockSm8x);
    if constexpr (Bytes > 48 * 1024) {
        extern __shared__ __align__(16) unsigned char dynamic_shared[];
        return dynamic_shared;
    } else {
        __shared__ __align__(16) unsigned char storage[Bytes];
        return storage;
    }
}

template <int Bytes, auto Kernel>
int q8_prepare_shared() {
    if constexpr (Bytes > 48 * 1024) {
        raise_dynamic_shared_memory(Kernel, Bytes);
        return Bytes;
    } else
        return 0;
}
} // namespace ninfer::ops::detail
