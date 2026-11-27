#pragma once

#include "ops/common/sm70_emulation.cuh"

#include <cuda_pipeline.h>
#include <cuda_runtime.h>

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

// A global load under the cache policy a cp.async of that policy takes, for code that stages
// through registers where cp.async is absent.
template <class V, Cache Policy>
__device__ __forceinline__ V load_global(const void* gmem_src) {
    if constexpr (Policy == Cache::cg) {
        return __ldcg(static_cast<const V*>(gmem_src));
    } else {
        return __ldca(static_cast<const V*>(gmem_src));
    }
}

// Volta and Turing have no cp.async (sm_80): each copy completes before the call returns
// (pre_sm80.cuh), with the same cache policy and zero fill, and cp_commit/cp_wait are empty.
// Multi-stage rings stay correct there but hide no latency.

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
    static_assert(Policy == Cache::ca || Bytes == 16, "cp.async.cg requires a 16-byte copy");
#if defined(NINFER_PRE_SM80_DEVICE)
    pre_sm80::copy_to_shared<Bytes, Policy == Cache::cg>(smem_dst, gmem_src);
#else
    if constexpr (Policy == Cache::cg) {
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
    static_assert(Policy == Cache::ca || Bytes == 16, "cp.async.cg requires a 16-byte copy");
#if defined(NINFER_PRE_SM80_DEVICE)
    pre_sm80::copy_to_shared_zfill<Bytes, Policy == Cache::cg>(smem_dst, gmem_src, src_bytes);
#else
    if constexpr (Policy == Cache::cg) {
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
#if !defined(NINFER_PRE_SM80_DEVICE)
    asm volatile("cp.async.commit_group;\n");
#endif
}

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
#if !defined(NINFER_PRE_SM80_DEVICE)
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups));
#endif
}

template <int Bytes>
__device__ __forceinline__ void pipe_copy(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "pipe_copy supports 4, 8, or 16 bytes");
    __pipeline_memcpy_async(smem_dst, gmem_src, Bytes);
}

__device__ __forceinline__ void pipe_commit() { __pipeline_commit(); }

template <int Groups>
__device__ __forceinline__ void pipe_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "pipe_wait group count must fit the PTX immediate");
    __pipeline_wait_prior(Groups);
}

} // namespace ninfer::ops
