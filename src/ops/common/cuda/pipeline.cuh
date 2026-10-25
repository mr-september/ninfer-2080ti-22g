#pragma once

#include "ops/common/cuda/memory.cuh"

#include <cuda_pipeline.h>

#include <cstdint>

namespace ninfer::ops::common {

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
    // Turing has no cp.async: the copy is a load followed by a store.
    if constexpr (Bytes == 16) {
        *reinterpret_cast<int4*>(smem_dst) = *reinterpret_cast<const int4*>(gmem_src);
    } else if constexpr (Bytes == 8) {
        *reinterpret_cast<int2*>(smem_dst) = *reinterpret_cast<const int2*>(gmem_src);
    } else {
        *reinterpret_cast<int*>(smem_dst) = *reinterpret_cast<const int*>(gmem_src);
    }
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
    if (src_bytes >= Bytes) {
        cp_async<Bytes, Policy>(smem_dst, gmem_src);
    } else if (src_bytes <= 0) {
        if constexpr (Bytes == 16) {
            *reinterpret_cast<int4*>(smem_dst) = make_int4(0, 0, 0, 0);
        } else if constexpr (Bytes == 8) {
            *reinterpret_cast<int2*>(smem_dst) = make_int2(0, 0);
        } else {
            *reinterpret_cast<int*>(smem_dst) = 0;
        }
    } else {
        auto* dst       = reinterpret_cast<std::uint8_t*>(smem_dst);
        const auto* src = reinterpret_cast<const std::uint8_t*>(gmem_src);
        for (int i = 0; i < src_bytes; ++i) { dst[i] = src[i]; }
        for (int i = src_bytes; i < Bytes; ++i) { dst[i] = 0; }
    }
}

__device__ __forceinline__ void cp_commit() {
}

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
}

template <int Bytes>
__device__ __forceinline__ void pipe_copy(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "pipe_copy supports 4, 8, or 16 bytes");
    cp_async<Bytes>(smem_dst, gmem_src);
}

__device__ __forceinline__ void pipe_commit() {
}

template <int Groups>
__device__ __forceinline__ void pipe_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "pipe_wait group count must fit the PTX immediate");
}

} // namespace ninfer::ops::common
