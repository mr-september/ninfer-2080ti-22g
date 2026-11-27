#pragma once

// The slab loop of an sm_70 wide product (sm70_gemm.h) over a caller's dequantizer and epilogue.

#include "core/device_launch.h"
#include "ops/linear/sm70/sm70_bf16_f16.cuh"
#include "ops/linear/sm70/sm70_gemm.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::sm70 {

// The FP32 product of slab row r (FP16 weights [rows][k]) with column t, with what the FP16 plane's
// saturation took from that column added back.
__device__ __forceinline__ float saturated_product(const float* __restrict__ plane,
                                                   const __half* __restrict__ weights,
                                                   const Saturation& saturation, std::int32_t rows,
                                                   std::int32_t r, std::int32_t t) {
    float value = plane[static_cast<std::size_t>(t) * rows + r];
    if (saturation.columns[t] != 0u) {
        value += saturated_part(weights + static_cast<std::size_t>(r) * saturation.k,
                                saturation.x + static_cast<std::size_t>(t) * saturation.k,
                                saturation.k);
    }
    return value;
}

// epilogue(row0 + r, t, product) for every row r of the slab and token t.
template <class Epilogue>
__global__ void epilogue_kernel(const float* __restrict__ plane, const __half* __restrict__ weights,
                                Saturation saturation, std::int32_t rows, std::int32_t tokens,
                                std::int32_t row0, Epilogue epilogue) {
    const std::int32_t r = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= rows) { return; }
    for (std::int32_t t = static_cast<std::int32_t>(blockIdx.y); t < tokens;
         t += static_cast<std::int32_t>(gridDim.y)) {
        epilogue(row0 + r, t, saturated_product(plane, weights, saturation, rows, r, t));
    }
}

// Products of the weight rows [0, rows) with the FP16 activations x16 (`tokens` columns of width
// k), converted from saturation.x with their column flags: dequantize(row0, count, dst) writes rows
// [row0, row0 + count) as FP16 [count][k] into dst, and epilogue(row, token, value) consumes each
// FP32 result. `scratch` holds slab_bytes(rows, k, tokens).
template <class Dequantize, class Epilogue>
void product(const __half* x16, const Saturation& saturation, std::int32_t tokens,
             std::int32_t rows, std::int32_t k, Dequantize&& dequantize, const Epilogue& epilogue,
             void* scratch, cudaStream_t stream) {
    const std::int32_t slab = slab_rows(rows, k);
    auto* weights           = static_cast<__half*>(scratch);
    auto* plane             = reinterpret_cast<float*>(
        static_cast<std::byte*>(scratch) +
        (static_cast<std::size_t>(slab) * static_cast<std::size_t>(k) * sizeof(__half) + 255) /
            256 * 256);
    constexpr int kThreads = 256;
    for (std::int32_t row0 = 0; row0 < rows; row0 += slab) {
        const std::int32_t count = std::min(slab, rows - row0);
        dequantize(row0, count, weights);
        gemm(weights, count, k, x16, tokens, plane, count, stream);
        const dim3 grid(static_cast<unsigned>((count + kThreads - 1) / kThreads),
                        static_cast<unsigned>(std::min<std::int32_t>(tokens, 65535)));
        epilogue_kernel<Epilogue><<<grid, kThreads, 0, stream>>>(plane, weights, saturation, count,
                                                                  tokens, row0, epilogue);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail::sm70
