#pragma once

// General-purpose BF16 x BF16 GEMM fallback for shapes the specialised bf16 linear path does not
// tile (for example a full-precision vocab projection kept as BF16 by a QAT checkpoint), and for
// every FP32 result. Computes out[T, N] = x[T, K] * weight[N, K]^T with an fp32 accumulation and a
// BF16 or FP32 store, matching the contiguous token-major layout of the specialised kernels. Tiling is fixed and the extents are
// runtime, so any (N, K, T) is handled without a per-shape instantiation. Up to 16 tokens over
// 16-byte rows take the GEMV form instead: one warp per output row streams the row once.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

__device__ __forceinline__ void store_result(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
__device__ __forceinline__ void store_result(float* p, float v) { *p = v; }

constexpr std::int32_t kTileN         = 32;
constexpr std::int32_t kTileT         = 32;
constexpr std::int32_t kTileK         = 32;
constexpr std::int32_t kThreads       = 256;
constexpr std::int32_t kThreadCols    = kThreads / 16;      // 16
constexpr std::int32_t kRowsPerThread = kTileN / 16;        // 2
constexpr std::int32_t kColsPerThread = kTileT / kThreadCols;  // 2

template <class Out>
__global__ void bf16_general_gemm_kernel(const __nv_bfloat16* __restrict__ x,
                                         const __nv_bfloat16* __restrict__ weight,
                                         Out* __restrict__ out, std::int32_t N,
                                         std::int32_t K, std::int32_t T) {
    __shared__ __nv_bfloat16 ws[kTileN][kTileK + 1];
    __shared__ __nv_bfloat16 wx[kTileT][kTileK + 1];

    const std::int32_t out_row0 = static_cast<std::int32_t>(blockIdx.x) * kTileN;
    const std::int32_t token0   = static_cast<std::int32_t>(blockIdx.y) * kTileT;
    const std::int32_t tid      = static_cast<std::int32_t>(threadIdx.x);

    const std::int32_t trow = tid / kThreadCols;  // 0..15
    const std::int32_t tcol = tid % kThreadCols;  // 0..15

    float accum[kRowsPerThread][kColsPerThread] = {};

    for (std::int32_t k0 = 0; k0 < K; k0 += kTileK) {
        for (std::int32_t i = tid; i < kTileN * kTileK; i += kThreads) {
            const std::int32_t r  = i / kTileK;
            const std::int32_t c  = i % kTileK;
            const std::int32_t gr = out_row0 + r;
            const std::int32_t gc = k0 + c;
            ws[r][c + 1] = (gr < N && gc < K)
                               ? weight[static_cast<std::int64_t>(gr) * K + gc]
                               : __float2bfloat16_rn(0.0f);
        }
        for (std::int32_t i = tid; i < kTileT * kTileK; i += kThreads) {
            const std::int32_t r  = i / kTileK;
            const std::int32_t c  = i % kTileK;
            const std::int32_t gr = token0 + r;
            const std::int32_t gc = k0 + c;
            wx[r][c + 1] = (gr < T && gc < K)
                               ? x[static_cast<std::int64_t>(gr) * K + gc]
                               : __float2bfloat16_rn(0.0f);
        }
        __syncthreads();

#pragma unroll
        for (std::int32_t rr = 0; rr < kRowsPerThread; ++rr) {
            const std::int32_t row = trow * kRowsPerThread + rr;
#pragma unroll
            for (std::int32_t cc = 0; cc < kColsPerThread; ++cc) {
                const std::int32_t col = tcol * kColsPerThread + cc;
                float sum = accum[rr][cc];
#pragma unroll
                for (std::int32_t kk = 0; kk < kTileK; ++kk) {
                    sum = fmaf(static_cast<float>(wx[col][kk + 1]),
                               static_cast<float>(ws[row][kk + 1]), sum);
                }
                accum[rr][cc] = sum;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (std::int32_t rr = 0; rr < kRowsPerThread; ++rr) {
        const std::int32_t row = out_row0 + trow * kRowsPerThread + rr;
        if (row < N) {
#pragma unroll
            for (std::int32_t cc = 0; cc < kColsPerThread; ++cc) {
                const std::int32_t col = token0 + tcol * kColsPerThread + cc;
                if (col < T) {
                    store_result(out + static_cast<std::int64_t>(col) * N + row, accum[rr][cc]);
                }
            }
        }
    }
}

// One warp per output row: the row in 16-byte vectors (K a multiple of 8), every token's sum in a
// register, FP32 accumulation and a BF16 or FP32 store. A token's sum does not depend on how many
// tokens share the launch, so every width gives each column the bits a one-token call gives it.
// Up to 16 tokens: past eight the SIMT tile took 2.8 ms a round on an RTX 5090 for a DSpine
// drafter's 15 bridge projections ([1024, 5120], [1536, 1024], [5120, 512]) at 16 columns, against
// 0.14 ms for this form at eight.
constexpr std::int32_t kGemvMaxTokens = 16;
constexpr std::int32_t kGemvWarps     = 8;

__device__ __forceinline__ void unpack_bf16x8(const uint4& v, float (&out)[8]) {
    const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(pairs[i]);
        out[2 * i]     = f.x;
        out[2 * i + 1] = f.y;
    }
}

// One warp's output row of the GEMV form: lane `lane` of the warp that owns `row`.
template <int T, class Out>
__device__ __forceinline__ void bf16_general_gemv_row(const __nv_bfloat16* __restrict__ x,
                                                      const __nv_bfloat16* __restrict__ weight,
                                                      Out* __restrict__ out, std::int32_t N,
                                                      std::int32_t K, std::int32_t row,
                                                      std::int32_t lane) {
    const auto* w = reinterpret_cast<const uint4*>(weight + static_cast<std::int64_t>(row) * K);
    const std::int32_t vectors = K / 8;
    float acc[T]               = {};
    for (std::int32_t v = lane; v < vectors; v += 32) {
        float wf[8];
        unpack_bf16x8(__ldg(w + v), wf);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            float xf[8];
            unpack_bf16x8(
                *reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(t) * K + 8 * v), xf);
#pragma unroll
            for (int i = 0; i < 8; ++i) { acc[t] = fmaf(wf[i], xf[i], acc[t]); }
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            store_result(out + static_cast<std::int64_t>(t) * N + row, acc[t]);
        }
    }
}

template <int T, class Out>
__global__ void __launch_bounds__(kGemvWarps * 32)
    bf16_general_gemv_kernel(const __nv_bfloat16* __restrict__ x,
                             const __nv_bfloat16* __restrict__ weight,
                             Out* __restrict__ out, std::int32_t N, std::int32_t K) {
    const std::int32_t row  = static_cast<std::int32_t>(blockIdx.x) * kGemvWarps +
                              static_cast<std::int32_t>(threadIdx.x >> 5);
    const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x & 31);
    if (row >= N) { return; }
    bf16_general_gemv_row<T, Out>(x, weight, out, N, K, row, lane);
}

// Up to kGemvTogether GEMV-form products of one x in one launch: product i takes the blocks from
// first_block[i], kGemvWarps rows each, and every row runs bf16_general_gemv_row as its own launch
// runs it.
constexpr std::int32_t kGemvTogether = 4;

template <class Out>
struct GemvProducts {
    const __nv_bfloat16* weight[kGemvTogether];
    Out* out[kGemvTogether];
    std::int32_t n[kGemvTogether];
    std::int32_t first_block[kGemvTogether];
    std::int32_t count;
};

template <int T, class Out>
__global__ void __launch_bounds__(kGemvWarps * 32)
    bf16_general_gemv_together_kernel(const __nv_bfloat16* __restrict__ x,
                                      const GemvProducts<Out> products, std::int32_t K) {
    const auto block = static_cast<std::int32_t>(blockIdx.x);
    std::int32_t p   = 0;
#pragma unroll
    for (int i = 1; i < kGemvTogether; ++i) {
        if (i < products.count && block >= products.first_block[i]) { p = i; }
    }
    const std::int32_t row = (block - products.first_block[p]) * kGemvWarps +
                             static_cast<std::int32_t>(threadIdx.x >> 5);
    const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x & 31);
    if (row >= products.n[p]) { return; }
    bf16_general_gemv_row<T, Out>(x, products.weight[p], products.out[p], products.n[p], K, row,
                                  lane);
}

} // namespace ninfer::ops::detail
