#pragma once

// The MoE matrix product on Volta (moe_matrix_product where the device runs sm_70 code). Volta has
// no integer tensor cores, so ggml's integer kernels run there on dp4a; this route multiplies FP16
// operands on the FP16 tensor cores instead (mma.m8n8k4 with FP32 accumulators), whose peak is
// twice dp4a's. The activation is an FP16 plane of the pairs' columns in routing order, each column
// scaled by a power of two that keeps its largest magnitude below 2^15, with the inverse factor
// applied to its products (quantize_moe_matrix_activation on Volta): a column keeps BF16's range
// and FP16's precision. The weights are the represented values of the GGUF blocks, decoded with the
// vector kernels' slice decoders and rounded once to FP16, as sm_70's dense GGUF route rounds them.
//
// A CTA owns 128 rows of one expert and one column tile of its pairs (MoeRouting's tile lists of
// 16, 32 or 64 pairs). Each lane decodes the 32-value slices of its own row straight into its A
// fragments, so no weight passes through shared memory; every quad pair reads the same activation
// fragment ("mirrored" B), and one instruction then covers a warp's 32 rows times 8 columns. The
// activation tile is staged in shared memory 128 values of K at a time.
//
// PTX fragment layouts of mma.m8n8k4.row.col (quad pair q: lanes 4q..4q+3 and 4q+16..4q+19,
// h = lane / 16): A row (lane % 4) + 4h with k 0..3; B column (lane % 4) + 4h with k 0..3; FP32
// accumulator i at row (lane & 1) + (i & 2) + 4h, column (i & 4) + (lane & 2) + (i & 1).

#include "ggml_bridge_vec.cuh"

namespace ninfer::ops::gguf::detail {
namespace sm70 {

inline constexpr int kWarps = 4;
inline constexpr int kRows  = 32 * kWarps; // a CTA's rows, 32 per warp
inline constexpr int kStep  = 128;         // activation values of K staged at a time
// Halves per staged column: 272 bytes keep 16-byte stores aligned and put the four columns a
// half-warp reads at one K in distinct banks.
inline constexpr int kStride = kStep + 8;

__device__ __forceinline__ void mma884(float (&c)[8], unsigned a0, unsigned a1, unsigned b0,
                                       unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]),
                   "+f"(c[6]), "+f"(c[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#else
    (void)c;
    (void)a0;
    (void)a1;
    (void)b0;
    (void)b1;
#endif
}

__device__ __forceinline__ unsigned half2_word(float lo, float hi) {
    const __half2 pair = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&pair);
}

// Device code exists only in the sm_70 image; other architectures carry an empty body.
template <ggml_type type, int J>
__launch_bounds__(32 * kWarps) __global__
    void moe_matrix_kernel(const void* const* __restrict__ experts, std::int64_t row_bytes,
                           int rows, int k, int k_padded, const __half* __restrict__ plane,
                           const float* __restrict__ factors, const std::int32_t* __restrict__ sorted,
                           const std::int32_t* __restrict__ bounds,
                           const std::int32_t* __restrict__ tiles,
                           const std::int32_t* __restrict__ tile_count, float* __restrict__ out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    using D                 = vec::Decoder<type>;
    constexpr int kTable    = D::kTableWords > 0 ? D::kTableWords : 4;
    constexpr int kPerBlock = D::kBlockElems / 32;
    constexpr int kSlices   = kStep / 32;
    __shared__ __align__(16) __half stage[J * kStride];
    __shared__ __align__(16) std::uint32_t table[kTable];
    __shared__ std::int32_t pair_of[J];
    __shared__ float factor_of[J];
    if (int(blockIdx.y) >= *tile_count) { return; }
    const int tile    = tiles[blockIdx.y];
    const int expert  = tile & 0xFFFF;
    const int first   = bounds[expert] + (tile >> 16) * J;
    const int columns = min(J, bounds[expert + 1] - first);
    const int tid     = int(threadIdx.x);
    if constexpr (D::kTableWords > 0) {
        const std::uint32_t* src = D::table();
        for (int i = tid; i < D::kTableWords; i += 32 * kWarps) { table[i] = src[i]; }
    }
    for (int c = tid; c < J; c += 32 * kWarps) {
        pair_of[c]   = c < columns ? sorted[first + c] : 0;
        factor_of[c] = c < columns ? factors[first + c] : 0.0f;
    }
    const int lane = tid & 31, h = lane >> 4;
    // The quad pair's first row: rows are whole warps, so a warp is wholly in or out.
    const int base    = int(blockIdx.x) * kRows + (tid >> 5) * 32 + ((lane >> 2) & 3) * 8;
    const bool active = base < rows;
    const std::uint8_t* weight =
        static_cast<const std::uint8_t*>(experts[expert]) +
        std::int64_t(active ? base + (lane & 3) + 4 * h : 0) * row_bytes;
    const int column = (lane & 3) + 4 * h; // the B column of each group of 8
    float acc[J / 8][8];
#pragma unroll
    for (int g = 0; g < J / 8; ++g) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { acc[g][i] = 0.0f; }
    }
    const int slices = k / 32;
    for (int step = 0; step * kStep < k; ++step) {
        __syncthreads();
        for (int i = tid; i < J * (kStep / 8); i += 32 * kWarps) {
            const int c = i / (kStep / 8), q = i % (kStep / 8);
            uint4 v     = make_uint4(0u, 0u, 0u, 0u);
            if (c < columns) {
                v = *reinterpret_cast<const uint4*>(plane + std::int64_t(first + c) * k_padded +
                                                    step * kStep + q * 8);
            }
            *reinterpret_cast<uint4*>(stage + c * kStride + q * 8) = v;
        }
        __syncthreads();
        if (!active) { continue; }
        const int end = min(slices, (step + 1) * kSlices);
        for (int s = step * kSlices; s < end; ++s) {
            vec::Slice d;
            D::decode(weight + std::int64_t(s / kPerBlock) * D::kBlockBytes, s % kPerBlock, table,
                      d);
            unsigned a[8][2];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                a[e][0] = half2_word(vec::slice_value<type>(d, e, 0),
                                     vec::slice_value<type>(d, e, 1));
                a[e][1] = half2_word(vec::slice_value<type>(d, e, 2),
                                     vec::slice_value<type>(d, e, 3));
            }
            const __half* b = stage + column * kStride + (s - step * kSlices) * 32;
#pragma unroll
            for (int g = 0; g < J / 8; ++g) {
                if (g * 8 >= columns) { break; }
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const uint2 v = *reinterpret_cast<const uint2*>(b + g * 8 * kStride + 4 * e);
                    mma884(acc[g], a[e][0], a[e][1], v.x, v.y);
                }
            }
        }
    }
    if (!active) { return; }
#pragma unroll
    for (int g = 0; g < J / 8; ++g) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int c = g * 8 + (i & 4) + (lane & 2) + (i & 1);
            if (c < columns) {
                const int r = base + (lane & 1) + (i & 2) + 4 * h;
                out[std::int64_t(pair_of[c]) * rows + r] = acc[g][i] * factor_of[c];
            }
        }
    }
#else
    (void)experts;
    (void)row_bytes;
    (void)rows;
    (void)k;
    (void)k_padded;
    (void)plane;
    (void)factors;
    (void)sorted;
    (void)bounds;
    (void)tiles;
    (void)tile_count;
    (void)out;
#endif
}

} // namespace sm70

template <ggml_type type>
void moe_matrix_sm70_launch(const void* const* experts, std::int64_t row_bytes, int rows, int k,
                            int k_padded, const MoeRouting& routing, int max_active, int pairs,
                            const void* activation, float* out, cudaStream_t stream) {
    using D = vec::Decoder<type>;
    if (rows <= 0 || rows % 32 != 0 || k % D::kBlockElems != 0 || k_padded % sm70::kStep != 0 ||
        k_padded < k || row_bytes % D::kBlockBytes != 0 ||
        row_bytes < std::int64_t(k / D::kBlockElems) * D::kBlockBytes || max_active <= 0 ||
        pairs <= 0) {
        throw std::invalid_argument("gguf moe matrix product (sm_70): unsupported geometry");
    }
    const auto* plane   = static_cast<const __half*>(activation);
    const auto* factors = reinterpret_cast<const float*>(plane + std::size_t(pairs) * k_padded);
    const auto launch   = [&]<int J>() {
        constexpr int width = J == 16 ? 0 : J == 32 ? 1 : 2;
        static_assert(kMatrixColumnTiles[width] == J, "MoeRouting lists tiles of width 16 << w");
        const int listed = (pairs + J - 1) / J + max_active;
        if (listed > routing.tile_capacity || listed > 65535) {
            throw std::invalid_argument("gguf moe matrix product (sm_70): too few listed tiles");
        }
        const dim3 grid((rows + sm70::kRows - 1) / sm70::kRows, listed, 1);
        sm70::moe_matrix_kernel<type, J><<<grid, 32 * sm70::kWarps, 0, stream>>>(
            experts, row_bytes, rows, k, k_padded, plane, factors, routing.sorted, routing.bounds,
            routing.tiles + std::int64_t(width) * routing.tile_capacity, routing.tile_count + width,
            out);
        check(cudaGetLastError(), "moe matrix product (sm_70) launch");
    };
    // The narrowest tile that holds an active expert's average pairs, as select_moe_launch picks.
    const int average = (pairs + max_active - 1) / max_active;
    if (average <= 16) { return launch.template operator()<16>(); }
    if (average <= 32) { return launch.template operator()<32>(); }
    launch.template operator()<64>();
}

} // namespace ninfer::ops::gguf::detail

#define NINFER_GGUF_SM70_MATRIX_INSTANCE(TYPE)                                                     \
    template void ninfer::ops::gguf::detail::moe_matrix_sm70_launch<TYPE>(                         \
        const void* const*, std::int64_t, int, int, int, const ninfer::ops::gguf::MoeRouting&,     \
        int, int, const void*, float*, cudaStream_t)
