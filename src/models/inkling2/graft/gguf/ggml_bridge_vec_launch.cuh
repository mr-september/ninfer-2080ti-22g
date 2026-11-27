#pragma once

// The vector kernel's launches (ggml_bridge_vec.cuh, one unit per block type in ggml_vec_*.cu),
// their arguments and the device helpers of their stores, and Volta's FP16 matrix products, which
// the same units hold (ggml_bridge_sm70.cuh). Included by those units and the dispatch alone.

#include "ggml_bridge_internal.cuh"

#include "../../common/silu.h"

#include <cstring>

namespace ninfer::ops::gguf::detail {

// An NVFP4 scale byte as ggml-quants.c's ggml_ue4m3_to_fp32 decodes it: half the UE4M3 value (the
// E2M1 table kvalues_mxfp4 holds doubled values), 0 and 0x7F standing for zero. The converters
// refuse scale bytes with bit 7 set, where ggml's CUDA decoder (a signed E4M3 conversion) differs.
__host__ __device__ __forceinline__ float ue4m3_half(std::uint32_t x) {
    x &= 0xFFu;
    if (x == 0 || x == 0x7F) { return 0.0f; }
    const std::uint32_t exponent = (x >> 3) & 0xF, mantissa = x & 7;
    if (exponent == 0) { return float(mantissa) * 0.0009765625f; }
    const std::uint32_t bits = ((exponent + 119u) << 23) | (mantissa << 20);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

// Volta's form of moe_matrix_product (ggml_bridge_sm70.cuh): FP16 tensor-core products over the
// FP16 activation plane of moe_sm70_activation, whose columns are k_padded values apart.
template <ggml_type type>
void moe_matrix_sm70_launch(const void* const* experts, std::int64_t row_bytes, int rows, int k,
                            int k_padded, const MoeRouting& routing, int max_active, int pairs,
                            const void* activation, float* out, cudaStream_t stream);

// Where and how the vector kernel (ggml_bridge_vec.cuh) writes its rows.
struct VecStore {
    __nv_bfloat16* bf16;
    float* f32;
    std::int64_t column_stride;
    Epilogue epilogue;
    const float* gate;
    std::int64_t gate_column_stride;
    int activation; // GateActivation of the gate product and of a fused [gate; up] product
};

struct VecArgs {
    const std::uint8_t* weight;
    std::int64_t row_bytes;
    int rows; // output rows; a fused [gate; up] parent holds 2 * rows
    int k;
    const std::int8_t* qs; // [columns][k]
    const half2* ds;       // [columns][k / 32]
    VecStore out;
};

// The vector kernel's MoE form (ggml_bridge_vec.cuh moe_kernel).
struct MoeVecArgs {
    const std::uint8_t* const* first;  // expert table (the gate table when fused)
    const std::uint8_t* const* second; // fused: the up table
    std::int64_t row_bytes;
    int rows;
    int k;
    const std::int32_t* bounds;
    const std::int32_t* sorted;
    const std::int32_t* active;
    const std::int32_t* active_count;
    int column_group;
    int per_token;
    const std::int8_t* qs; // [columns][k]
    const half2* ds;       // [columns][k / 32]
    __nv_bfloat16* out_bf16;
    float* out_f32;
    unsigned long long* weighted;
    const float* weights;
    const float* gate;
    int activation; // GateActivation of `gate` and of the fused form
};

template <ggml_type type>
void moe_launch(const MoeVecArgs& args, int max_active, int chunk, bool fused, cudaStream_t stream);
template <ggml_type type>
void moe_decode_up_launch(const MoeDecodeUpArgs& args, int chunk, cudaStream_t stream);
template <ggml_type type>
void moe_decode_down_launch(const MoeDecodeDownArgs& args, int chunk, cudaStream_t stream);
template <ggml_type type>
void moe_decode_prep_launch(const MoeDecodePrepArgs& args, cudaStream_t stream);

// Whether slice_dequantize serves the type (every type but IQ1_M).
template <ggml_type type>
inline constexpr bool kSliceDequantize = type != GGML_TYPE_IQ1_M;

// Rows of a GGUF matrix to dst_t through the vector kernels' slice decoders: one thread a 32-value
// slice, written as 16-byte stores (ggml_bridge_vec.cuh).
template <ggml_type type, typename dst_t>
void slice_dequantize(const void* weight, std::int64_t row_bytes, int k, const std::int32_t* row_ids,
                      int rows, dst_t* out, std::int64_t out_row_stride, cudaStream_t stream);

// Byte offset of the scales inside a planar vector activation of `columns` x `k`.
[[nodiscard]] constexpr std::size_t vec_scales_offset(int k, int columns) {
    return (std::size_t(k) * columns + 15) / 16 * 16;
}

template <ggml_type type>
void vec_launch(const VecArgs& args, int columns, bool fused, cudaStream_t stream);

// SiLU and the tanh-approximated GELU from correctly rounded operations (ops/common/silu.h): the
// CPU's GGUF expert products compute the same bits.
__device__ __forceinline__ float vec_silu(float x) { return ops::portable_silu(x); }

// GateActivation: 0 SiLU, 1 tanh-approximated GELU.
__device__ __forceinline__ float vec_gate(float x, int activation) {
    return activation == 0 ? vec_silu(x) : ops::gelu_tanh(x);
}

__device__ __forceinline__ void vec_store(const VecStore& out, int row, int column, float value) {
    const std::int64_t at = column * out.column_stride + row;
    if (out.epilogue == Epilogue::GateProduct) {
        value *= vec_gate(out.gate[column * out.gate_column_stride + row], out.activation);
    }
    if (out.f32 != nullptr) {
        out.f32[at] = out.epilogue == Epilogue::Accumulate ? out.f32[at] + value : value;
        return;
    }
    if (out.epilogue == Epilogue::Accumulate) { value += __bfloat162float(out.bf16[at]); }
    out.bf16[at] = __float2bfloat16(value);
}

} // namespace ninfer::ops::gguf::detail
