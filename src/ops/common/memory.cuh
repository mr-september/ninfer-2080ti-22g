#pragma once

#include "core/arch_traits.cuh"

#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace ninfer::ops {

enum class Cache { ca, cg };

template <class V, class T>
__device__ __forceinline__ V load_vec(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return *reinterpret_cast<const V*>(ptr);
}

template <class V, class T>
__device__ __forceinline__ V load_ldg(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return __ldg(reinterpret_cast<const V*>(ptr));
}

template <class T, class V>
__device__ __forceinline__ void store_vec(T* ptr, V value) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    *reinterpret_cast<V*>(ptr) = value;
}

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
#if defined(NINFER_SM75)
    if constexpr (Bytes == 16) {
        *reinterpret_cast<int4*>(smem_dst) = *reinterpret_cast<const int4*>(gmem_src);
    } else if constexpr (Bytes == 8) {
        *reinterpret_cast<int2*>(smem_dst) = *reinterpret_cast<const int2*>(gmem_src);
    } else {
        *reinterpret_cast<int*>(smem_dst) = *reinterpret_cast<const int*>(gmem_src);
    }
#else
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes));
    }
#endif
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
#if defined(NINFER_SM75)
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
        auto* dst = reinterpret_cast<std::uint8_t*>(smem_dst);
        const auto* src = reinterpret_cast<const std::uint8_t*>(gmem_src);
        for (int i = 0; i < src_bytes; ++i) {
            dst[i] = src[i];
        }
        for (int i = src_bytes; i < Bytes; ++i) {
            dst[i] = 0;
        }
    }
#else
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes), "r"(src_bytes));
    }
#endif
}

__device__ __forceinline__ void cp_commit() {
#if !defined(NINFER_SM75)
    asm volatile("cp.async.commit_group;\n");
#endif
}

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
#if !defined(NINFER_SM75)
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups));
#endif
}

template <int Bytes>
__device__ __forceinline__ void pipe_copy(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "pipe_copy supports 4, 8, or 16 bytes");
#if defined(NINFER_SM75)
    cp_async<Bytes>(smem_dst, gmem_src);
#else
    __pipeline_memcpy_async(smem_dst, gmem_src, Bytes);
#endif
}

__device__ __forceinline__ void pipe_commit() {
#if !defined(NINFER_SM75)
    __pipeline_commit();
#endif
}

template <int Groups>
__device__ __forceinline__ void pipe_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "pipe_wait group count must fit the PTX immediate");
#if !defined(NINFER_SM75)
    __pipeline_wait_prior(Groups);
#endif
}

} // namespace ninfer::ops
