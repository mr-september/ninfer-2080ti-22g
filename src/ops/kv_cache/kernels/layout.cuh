#pragma once

#include "core/paged_kv_cache.h"

#include <cstdint>

namespace ninfer::ops::detail::kv {

inline constexpr int kPageShift   = 6;
inline constexpr int kPageMask    = kPagedKVPageSize - 1;
inline constexpr int kHeadDim     = 256;
inline constexpr int kInt8Group   = 64;
inline constexpr int kInt8Groups  = kHeadDim / kInt8Group;

static_assert(kPagedKVPageSize == (1 << kPageShift));

template <int Heads>
struct D256Geometry {
    static_assert(Heads == 4);
    static constexpr int KVHeads = Heads;
};

struct DirectMetadata {
    const std::int32_t* table;

    __device__ __forceinline__ int valid_tokens(int width) const { return width; }

    __device__ __forceinline__ const std::int32_t* block_table() const { return table; }
};

template <bool Masked>
struct BatchMetadata {
    static constexpr bool masked = Masked;
    const std::int32_t* tables;
    const std::int32_t* valid_columns;
    const std::int32_t* table_rows;
    std::int32_t table_stride;

    __device__ __forceinline__ int valid_tokens(int width) const {
        if constexpr (Masked) {
            const int valid = valid_columns[0];
            return valid <= 0 ? 0 : (valid < width ? valid : width);
        }
        return width;
    }

    __device__ __forceinline__ const std::int32_t* block_table() const {
        return tables + static_cast<std::int64_t>(table_rows[0]) * table_stride;
    }
};

__device__ __forceinline__ int lookup_page(const std::int32_t* table, int position) {
    return table[position >> kPageShift];
}

// [leading, page offset, head, physical page].
template <int Leading, int Heads>
__device__ __forceinline__ std::int64_t page_major_head_offset(int page, int head) {
    return static_cast<std::int64_t>(Leading) * kPagedKVPageSize *
           (head + static_cast<std::int64_t>(Heads) * page);
}

template <int Leading, int Heads>
__device__ __forceinline__ std::int64_t page_major_offset(int page, int head, int offset,
                                                          int component) {
    return page_major_head_offset<Leading, Heads>(page, head) +
           static_cast<std::int64_t>(Leading) * offset + component;
}

template <int Leading, int Heads>
__device__ __forceinline__ std::int64_t page_major_offset(const std::int32_t* table, int head,
                                                          int position, int component) {
    return page_major_offset<Leading, Heads>(lookup_page(table, position), head,
                                             position & kPageMask, component);
}

// [leading, page offset, physical page, head].
template <int Leading>
__device__ __forceinline__ std::int64_t head_major_offset(int page, int head, int offset,
                                                          int component, int physical_pages) {
    return component + static_cast<std::int64_t>(Leading) *
                           (offset + kPagedKVPageSize *
                                         (page + static_cast<std::int64_t>(physical_pages) * head));
}

// One slot: [feature, ring position, head]. The caller applies the explicit state-slot stride.
template <int Dim, int Capacity>
__device__ __forceinline__ std::int64_t cyclic_offset(int position, int head, int padded_capacity,
                                                      int component = 0) {
    static_assert((Capacity & (Capacity - 1)) == 0);
    return component +
           static_cast<std::int64_t>(Dim) *
               ((position & (Capacity - 1)) + static_cast<std::int64_t>(padded_capacity) * head);
}

template <class Geometry>
__device__ __forceinline__ std::int64_t source_offset(int head, int component, int token) {
    return component + static_cast<std::int64_t>(kHeadDim) *
                           (head + static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

template <int Heads, int CodeExtent, int ScaleExtent, int ValuesPerCode = 1>
struct QuantizedRowLayout {
    __device__ static __forceinline__ std::int64_t code_offset(int page, int head, int component,
                                                               int offset) {
        return page_major_offset<CodeExtent, Heads>(page, head, offset, component / ValuesPerCode);
    }

    __device__ static __forceinline__ std::int64_t scale_offset(int page, int head, int group,
                                                                int offset) {
        return page_major_offset<ScaleExtent, Heads>(page, head, offset, group);
    }

    __device__ static __forceinline__ std::int64_t scale_offset(int page, int head, int offset) {
        static_assert(ScaleExtent == 1);
        return scale_offset(page, head, 0, offset);
    }
};

template <int Heads>
using Int8G64Layout = QuantizedRowLayout<Heads, 256, 4>;

} // namespace ninfer::ops::detail::kv
