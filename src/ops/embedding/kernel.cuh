#pragma once

#include "ops/common/numeric/math.h"
#include "ops/common/quant/integer.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail::embedding {
inline constexpr std::int32_t kEmbedGatherQ6Group = common::Q6RowSplitStorage::kGroupK;
inline constexpr std::int32_t kEmbedGatherQ6NibbleBpr =
    common::Q6RowSplitStorage::kCodeBytesPerGroup;
inline constexpr std::int32_t kEmbedGatherQ6HighBpr = common::Q6RowSplitStorage::kHighBytesPerGroup;
inline constexpr std::int32_t kEmbedGatherQ6GroupsPerBlock = 2;
inline constexpr std::int32_t kEmbedGatherQ8Group          = common::Q8RowSplitStorage::kGroupK;

__global__ void embed_gather_dense_kernel(const std::int32_t* ids, const __nv_bfloat16* table,
                                          __nv_bfloat16* out, std::int32_t d, std::int32_t T) {
    const std::int64_t n      = static_cast<std::int64_t>(d) * T;
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const std::int32_t t = static_cast<std::int32_t>(i / d);
        const std::int32_t k = static_cast<std::int32_t>(i - static_cast<std::int64_t>(t) * d);
        out[i]               = table[static_cast<std::int64_t>(ids[t]) * d + k];
    }
}

__global__ void embed_gather_q6_kernel(const std::int32_t* ids, const std::uint8_t* codes,
                                       const std::uint8_t* high, const std::uint8_t* scales,
                                       __nv_bfloat16* out, std::int32_t d, std::int32_t T,
                                       std::int32_t padded_d) {
    const std::int32_t kg     = padded_d / kEmbedGatherQ6Group;
    const std::int64_t n      = static_cast<std::int64_t>(d) * T;
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;

    for (std::int64_t i = start; i < n; i += stride) {
        const std::int32_t t    = static_cast<std::int32_t>(i / d);
        const std::int32_t k    = static_cast<std::int32_t>(i - static_cast<std::int64_t>(t) * d);
        const std::int32_t row  = ids[t];
        const std::int32_t g    = k / kEmbedGatherQ6Group;
        const std::int32_t lane = k - g * kEmbedGatherQ6Group;
        const std::int64_t group_index = static_cast<std::int64_t>(row) * kg + g;
        const float scale              = common::load_fp16_unaligned(scales + group_index * 2);
        const int code = common::unpack_q6_code(codes + group_index * kEmbedGatherQ6NibbleBpr,
                                                high + group_index * kEmbedGatherQ6HighBpr, lane);
        out[i]         = __float2bfloat16(static_cast<float>(code) * scale);
    }
}

__launch_bounds__(kEmbedGatherQ6Group* kEmbedGatherQ6GroupsPerBlock) __global__
    void embed_gather_q6_grouped_kernel(const std::int32_t* ids, const std::uint8_t* codes,
                                        const std::uint8_t* high, const std::uint8_t* scales,
                                        __nv_bfloat16* out, std::int32_t d, std::int32_t T) {
    const std::int32_t kg           = d / kEmbedGatherQ6Group;
    const std::int32_t group_blocks = common::div_up(kg, kEmbedGatherQ6GroupsPerBlock);
    const std::int32_t t            = static_cast<std::int32_t>(blockIdx.x) / group_blocks;
    const std::int32_t block_group  = static_cast<std::int32_t>(blockIdx.x) - t * group_blocks;
    const std::int32_t group_slot   = threadIdx.x / kEmbedGatherQ6Group;
    const std::int32_t lane         = threadIdx.x - group_slot * kEmbedGatherQ6Group;
    const std::int32_t g            = block_group * kEmbedGatherQ6GroupsPerBlock + group_slot;
    if (g >= kg) { return; }

    const std::int32_t row         = ids[t];
    const std::int64_t group_index = static_cast<std::int64_t>(row) * kg + g;

    int scale_bits = 0;
    if ((lane & 31) == 0) {
        scale_bits = static_cast<int>(scales[group_index * 2]) |
                     (static_cast<int>(scales[group_index * 2 + 1]) << 8);
    }
    scale_bits        = __shfl_sync(0xffffffffu, scale_bits, 0);
    const float scale = __half2float(__ushort_as_half(static_cast<std::uint16_t>(scale_bits)));
    const int code    = common::unpack_q6_code(codes + group_index * kEmbedGatherQ6NibbleBpr,
                                               high + group_index * kEmbedGatherQ6HighBpr, lane);
    const std::int64_t out_idx = static_cast<std::int64_t>(t) * d +
                                 static_cast<std::int64_t>(g) * kEmbedGatherQ6Group + lane;
    out[out_idx] = __float2bfloat16(static_cast<float>(code) * scale);
}

__global__ void embed_gather_q8_kernel(const std::int32_t* ids, const std::uint8_t* codes,
                                       const std::uint8_t* scales, __nv_bfloat16* out,
                                       std::int32_t d, std::int32_t T, std::int32_t padded_d) {
    const std::int32_t kg     = padded_d / kEmbedGatherQ8Group;
    const std::int64_t n      = static_cast<std::int64_t>(d) * T;
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const std::int32_t t    = static_cast<std::int32_t>(i / d);
        const std::int32_t k    = static_cast<std::int32_t>(i - static_cast<std::int64_t>(t) * d);
        const std::int32_t row  = ids[t];
        const std::int32_t g    = k / kEmbedGatherQ8Group;
        const std::int32_t lane = k - g * kEmbedGatherQ8Group;
        const std::int64_t group_index = static_cast<std::int64_t>(row) * kg + g;
        const float scale              = common::load_fp16_unaligned(scales + group_index * 2);
        const auto code = static_cast<std::int8_t>(codes[group_index * kEmbedGatherQ8Group + lane]);
        out[i]          = __float2bfloat16(static_cast<float>(code) * scale);
    }
}

template <int Blocks, int Threads, bool PairStore>
__launch_bounds__(Threads) __global__
    void embed_gather_q8_packed_5120_kernel(const std::int32_t* ids, const std::uint8_t* codes,
                                            const std::uint8_t* scales, __nv_bfloat16* out) {
    constexpr int D = 5120, Values = D / Blocks, Words = Values / 4;
    static_assert(D % Blocks == 0 && Values % 32 == 0);
    const int token      = static_cast<int>(blockIdx.x) / Blocks;
    const int split      = static_cast<int>(blockIdx.x) % Blocks;
    const int row        = ids[token];
    const auto* code_row = codes + static_cast<std::int64_t>(row) * D;
    auto* output         = out + static_cast<std::int64_t>(token) * D;
#pragma unroll
    for (int item = threadIdx.x; item < Words; item += Threads) {
        const int d       = split * Values + item * 4;
        const auto group  = static_cast<std::int64_t>(row) * (D / 32) + d / 32;
        const float scale = common::load_fp16_unaligned(scales + group * 2);
        const auto word   = *reinterpret_cast<const std::uint32_t*>(code_row + d);
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            const auto value =
                common::dequant_q8_pair(static_cast<std::uint16_t>(word >> (pair * 16)), scale);
            if constexpr (PairStore)
                reinterpret_cast<__nv_bfloat162*>(output + d)[pair] = value;
            else {
                output[d + pair * 2]     = value.x;
                output[d + pair * 2 + 1] = value.y;
            }
        }
    }
}

} // namespace ninfer::ops::detail::embedding
