#pragma once

#include "core/paged_kv_page.h"

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kPagedKVPageShift = 6;
inline constexpr int kPagedKVPageMask  = kPagedKVPageSize - 1;

static_assert(kPagedKVPageSize == (1 << kPagedKVPageShift));

// The page of a device's planes that holds the rows of `physical_page`, or -1 where another rank's
// planes hold them: a position-split cache gives rank r the pages p with p % stride == r, at
// p / stride (PagedKVLayerView::page_stride).
__device__ __forceinline__ std::int32_t paged_kv_local_page(std::int32_t physical_page,
                                                            std::int32_t stride,
                                                            std::int32_t residue) {
    if (stride == 1) { return physical_page; }
    return physical_page % stride == residue ? physical_page / stride : -1;
}

struct PagedKVDirectMetadata {
    const std::int32_t* table;
    std::int32_t page_stride  = 1;
    std::int32_t page_residue = 0;

    __device__ __forceinline__ std::int32_t valid_tokens(std::int32_t width) const { return width; }

    __device__ __forceinline__ const std::int32_t* block_table() const { return table; }

    __device__ __forceinline__ std::int32_t local_page(std::int32_t physical_page) const {
        return paged_kv_local_page(physical_page, page_stride, page_residue);
    }
};

template <bool Masked>
struct PagedKVBatchMetadata {
    const std::int32_t* tables;
    const std::int32_t* valid_columns;
    const std::int32_t* table_rows;
    std::int32_t table_stride;
    std::int32_t page_stride  = 1;
    std::int32_t page_residue = 0;

    __device__ __forceinline__ std::int32_t valid_tokens(std::int32_t width) const {
        if constexpr (Masked) {
            const std::int32_t valid = valid_columns[0];
            return valid <= 0 ? 0 : (valid < width ? valid : width);
        }
        return width;
    }

    __device__ __forceinline__ const std::int32_t* block_table() const {
        return tables + static_cast<std::int64_t>(table_rows[0]) * table_stride;
    }

    __device__ __forceinline__ std::int32_t local_page(std::int32_t physical_page) const {
        return paged_kv_local_page(physical_page, page_stride, page_residue);
    }
};

__device__ __forceinline__ std::int32_t paged_kv_physical_page(const std::int32_t* block_table,
                                                               std::int32_t position) {
    return block_table[position >> kPagedKVPageShift];
}

template <int LeadingExtent, int HeadExtent>
__device__ __forceinline__ std::int64_t paged_kv_page_head_offset(std::int32_t physical_page,
                                                                  std::int32_t head) {
    return static_cast<std::int64_t>(LeadingExtent) * kPagedKVPageSize *
           (static_cast<std::int64_t>(head) +
            static_cast<std::int64_t>(HeadExtent) * physical_page);
}

template <int LeadingExtent, int HeadExtent>
__device__ __forceinline__ std::int64_t
paged_kv_element_offset(std::int32_t physical_page, std::int32_t head, std::int32_t page_offset,
                        std::int32_t leading) {
    return paged_kv_page_head_offset<LeadingExtent, HeadExtent>(physical_page, head) +
           static_cast<std::int64_t>(LeadingExtent) * page_offset + leading;
}

// The same offset in a plane of `head_extent` heads known only at run time: a tensor-parallel
// rank's run of a layer's KV heads.
template <int LeadingExtent>
__device__ __forceinline__ std::int64_t
paged_kv_element_offset(std::int32_t physical_page, std::int32_t head, std::int32_t page_offset,
                        std::int32_t leading, std::int32_t head_extent) {
    return static_cast<std::int64_t>(LeadingExtent) * kPagedKVPageSize *
               (static_cast<std::int64_t>(head) +
                static_cast<std::int64_t>(head_extent) * physical_page) +
           static_cast<std::int64_t>(LeadingExtent) * page_offset + leading;
}

template <int LeadingExtent, int HeadExtent>
__device__ __forceinline__ std::int64_t
paged_kv_element_offset(const std::int32_t* block_table, std::int32_t head, std::int32_t position,
                        std::int32_t leading) {
    return paged_kv_element_offset<LeadingExtent, HeadExtent>(
        paged_kv_physical_page(block_table, position), head, position & kPagedKVPageMask, leading);
}

} // namespace ninfer::ops
