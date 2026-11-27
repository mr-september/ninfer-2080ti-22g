#pragma once

// The integer tensor-core matrix kernel's launches (ggml_bridge_mmq.cuh, one unit per block type
// in ggml_mmq_*.cu). Included by those units and the dispatch alone.

#include "ggml_bridge_internal.cuh"

namespace ninfer::ops::gguf::detail {

template <ggml_type type>
std::size_t matrix_fixup_bytes_impl(int rows, int columns);

template <ggml_type type>
bool moe_matrix_fits_impl(int rows, int k, bool tail);

template <ggml_type type>
void moe_matrix_product_impl(const void* const* experts, std::int64_t row_bytes, int rows, int k,
                             const MoeRouting& routing, int max_active, int pairs, bool tail,
                             bool aligned, const void* activation, float* out,
                             cudaStream_t stream);

template <ggml_type type, typename Dst = float, bool accumulate = false>
void matrix_product_impl(const void* weight, std::int64_t row_bytes, int rows, int k,
                         const void* activation, int columns, Dst* out,
                         std::int64_t out_column_stride, void* fixup, cudaStream_t stream,
                         bool tail, float* partial = nullptr);

} // namespace ninfer::ops::gguf::detail
