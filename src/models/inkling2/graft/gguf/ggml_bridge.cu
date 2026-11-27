// Entry points into the vendored ggml CUDA kernels and the vector kernel (ggml_bridge_vec.cuh). The
// activation quantizers follow ggml's q8_1, reading BF16 and optionally gathering the input columns.

#include "ggml_bridge.h"
#include "ggml_bridge_mmq_launch.cuh"
#include "ggml_bridge_vec_launch.cuh"

#include "../../common/device_tier.h"
#include "../../common/fixed_point.cuh"

#include "ggml-cuda/dequantize.cuh"

#include <cub/block/block_scan.cuh>

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <mutex>

// ggml's GGML_ABORT/GGML_ASSERT land here; only the vendored headers' own invariants reach it.
extern "C" void ggml_abort(const char* file, int line, const char* fmt, ...) {
    std::fprintf(stderr, "ggml abort at %s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fputc('\n', stderr);
    std::abort();
}

namespace ninfer::ops::gguf {
namespace detail {

[[nodiscard]] inline ggml_type to_ggml(GgmlType type) { return static_cast<ggml_type>(type); }

// The matrix kernel's activation carries one block_q8_1_mmq per 128 values per column; ggml
// over-reads up to one widest column tile past the end, so the plane keeps that much slack.
inline constexpr std::size_t kMatrixActivationSlack = 128 * sizeof(block_q8_1_mmq);
// MoeRouting lists the matrix kernel's work for each of the column tiles (kMatrixColumnTiles).
inline constexpr int kMoeTileWidths = 4;

const DeviceFacts& device_facts() {
    static std::array<DeviceFacts, 16> facts{};
    static std::array<std::once_flag, 16> once;
    int device = 0;
    check(cudaGetDevice(&device), "device");
    if (device < 0 || device >= int(facts.size())) {
        throw std::invalid_argument("gguf: device ordinal out of range");
    }
    std::call_once(once[device], [&] {
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "device properties");
        facts[device].cc                     = 100 * properties.major + 10 * properties.minor;
        facts[device].sm_count               = properties.multiProcessorCount;
        facts[device].shared_per_block_optin = properties.sharedMemPerBlockOptin;
    });
    return facts[device];
}

namespace {

// x BF16 [columns][k] -> the vector kernel's planar q8 activation: ggml's q8_1 numbers, one warp per
// 32 values, element i of column c read from x[c][input_columns[i]] when a gather is given.
constexpr int kQuantizeThreads = 256;

// Value i of column `column` into the planar q8_1 activation; every lane of the warp takes part.
__device__ __forceinline__ void quantize_vector_value(float xi, int i, int column,
                                                      std::int8_t* __restrict__ qs,
                                                      half2* __restrict__ ds, int k) {
    const float amax = warp_reduce_max<QK8_1>(fabsf(xi));
    const float sum  = warp_reduce_sum<QK8_1>(xi);
    const float d    = amax / 127.0f;
    qs[std::int64_t(column) * k + i] = amax == 0.0f ? 0 : static_cast<std::int8_t>(roundf(xi / d));
    if (i % QK8_1 == 0) { ds[std::int64_t(column) * (k / QK8_1) + i / QK8_1] = make_half2(d, sum); }
}

__launch_bounds__(kQuantizeThreads) __global__
    void quantize_vector_kernel(const __nv_bfloat16* __restrict__ x,
                                const std::int32_t* __restrict__ input_columns,
                                std::int8_t* __restrict__ qs, half2* __restrict__ ds, int k) {
    const int i      = blockDim.x * blockIdx.x + threadIdx.x;
    const int column = blockIdx.y;
    if (i >= k) { return; }
    const int source = input_columns != nullptr ? input_columns[i] : i;
    quantize_vector_value(__bfloat162float(x[std::int64_t(column) * k + source]), i, column, qs, ds,
                          k);
}

// FP32 columns: output column c reads x column source[c] (zeros for a negative entry), each value
// multiplied by scale[i] first when given.
__launch_bounds__(kQuantizeThreads) __global__
    void quantize_vector_f32_kernel(const float* __restrict__ x,
                                    const std::int32_t* __restrict__ source,
                                    const float* __restrict__ scale, std::int8_t* __restrict__ qs,
                                    half2* __restrict__ ds, int k) {
    const int i      = blockDim.x * blockIdx.x + threadIdx.x;
    const int column = blockIdx.y;
    if (i >= k) { return; }
    const int from = source != nullptr ? source[column] : column;
    float xi       = 0.0f;
    if (from >= 0) {
        xi = x[std::int64_t(from) * k + i];
        if (scale != nullptr) { xi = __fmul_rn(xi, scale[i]); }
    }
    quantize_vector_value(xi, i, column, qs, ds, k);
}

using VectorLaunch = void (*)(const VecArgs&, int, bool, cudaStream_t);
using MoeLaunch    = void (*)(const MoeVecArgs&, int, int, bool, cudaStream_t);

MoeLaunch moe_launcher(GgmlType type) {
    switch (type) {
    case GgmlType::Q8_0:
        return moe_launch<GGML_TYPE_Q8_0>;
    case GgmlType::Q4_0:
        return moe_launch<GGML_TYPE_Q4_0>;
    case GgmlType::Q5_0:
        return moe_launch<GGML_TYPE_Q5_0>;
    case GgmlType::Q2_0:
        return moe_launch<GGML_TYPE_Q2_0>;
    case GgmlType::Q2_K:
        return moe_launch<GGML_TYPE_Q2_K>;
    case GgmlType::Q3_K:
        return moe_launch<GGML_TYPE_Q3_K>;
    case GgmlType::Q4_K:
        return moe_launch<GGML_TYPE_Q4_K>;
    case GgmlType::Q5_K:
        return moe_launch<GGML_TYPE_Q5_K>;
    case GgmlType::Q6_K:
        return moe_launch<GGML_TYPE_Q6_K>;
    case GgmlType::IQ2_XXS:
        return moe_launch<GGML_TYPE_IQ2_XXS>;
    case GgmlType::IQ2_XS:
        return moe_launch<GGML_TYPE_IQ2_XS>;
    case GgmlType::IQ2_S:
        return moe_launch<GGML_TYPE_IQ2_S>;
    case GgmlType::IQ3_XXS:
        return moe_launch<GGML_TYPE_IQ3_XXS>;
    case GgmlType::IQ3_S:
        return moe_launch<GGML_TYPE_IQ3_S>;
    case GgmlType::IQ1_S:
        return moe_launch<GGML_TYPE_IQ1_S>;
    case GgmlType::IQ1_M:
        return moe_launch<GGML_TYPE_IQ1_M>;
    case GgmlType::IQ4_NL:
        return moe_launch<GGML_TYPE_IQ4_NL>;
    case GgmlType::IQ4_XS:
        return moe_launch<GGML_TYPE_IQ4_XS>;
    case GgmlType::MXFP4:
        return moe_launch<GGML_TYPE_MXFP4>;
    case GgmlType::NVFP4:
        return moe_launch<GGML_TYPE_NVFP4>;
    }
    throw std::invalid_argument("gguf moe product: unsupported block type");
}

template <class Args>
using DecodeLaunch = void (*)(const Args&, int, cudaStream_t);

template <class Fn, template <ggml_type> class Pick>
Fn decode_launcher(GgmlType type) {
    switch (type) {
    case GgmlType::Q8_0: return Pick<GGML_TYPE_Q8_0>::launch;
    case GgmlType::Q4_0: return Pick<GGML_TYPE_Q4_0>::launch;
    case GgmlType::Q5_0: return Pick<GGML_TYPE_Q5_0>::launch;
    case GgmlType::Q2_0: return Pick<GGML_TYPE_Q2_0>::launch;
    case GgmlType::Q2_K: return Pick<GGML_TYPE_Q2_K>::launch;
    case GgmlType::Q3_K: return Pick<GGML_TYPE_Q3_K>::launch;
    case GgmlType::Q4_K: return Pick<GGML_TYPE_Q4_K>::launch;
    case GgmlType::Q5_K: return Pick<GGML_TYPE_Q5_K>::launch;
    case GgmlType::Q6_K: return Pick<GGML_TYPE_Q6_K>::launch;
    case GgmlType::IQ2_XXS: return Pick<GGML_TYPE_IQ2_XXS>::launch;
    case GgmlType::IQ2_XS: return Pick<GGML_TYPE_IQ2_XS>::launch;
    case GgmlType::IQ2_S: return Pick<GGML_TYPE_IQ2_S>::launch;
    case GgmlType::IQ3_XXS: return Pick<GGML_TYPE_IQ3_XXS>::launch;
    case GgmlType::IQ3_S: return Pick<GGML_TYPE_IQ3_S>::launch;
    case GgmlType::IQ1_S: return Pick<GGML_TYPE_IQ1_S>::launch;
    case GgmlType::IQ1_M: return Pick<GGML_TYPE_IQ1_M>::launch;
    case GgmlType::IQ4_NL: return Pick<GGML_TYPE_IQ4_NL>::launch;
    case GgmlType::IQ4_XS: return Pick<GGML_TYPE_IQ4_XS>::launch;
    case GgmlType::MXFP4: return Pick<GGML_TYPE_MXFP4>::launch;
    case GgmlType::NVFP4: return Pick<GGML_TYPE_NVFP4>::launch;
    }
    throw std::invalid_argument("gguf moe decode: unsupported block type");
}

template <ggml_type type>
struct PickUp {
    static constexpr auto launch = &moe_decode_up_launch<type>;
};
template <ggml_type type>
struct PickDown {
    static constexpr auto launch = &moe_decode_down_launch<type>;
};
template <ggml_type type>
struct PickPrep {
    static constexpr auto launch = &moe_decode_prep_launch<type>;
};

// One block: the pairs of each expert counted, scanned into bounds, then placed. The experts with
// pairs are listed in ascending order, and so are the column tiles of each tile width that hold
// any pair (MoeRouting::tiles). With `small_active` (moe_sort_routes_split), the experts with 1 to
// `vector_pairs` pairs are listed there as well, and the column tiles of the others in
// `large_tiles`.
constexpr int kSortThreads = 1024;
static_assert(sizeof(kMatrixColumnTiles) / sizeof(int) == kMoeTileWidths &&
                  kMatrixColumnTiles[0] == 16 && kMatrixColumnTiles[1] == 32 &&
                  kMatrixColumnTiles[2] == 64 && kMatrixColumnTiles[3] == 128,
              "MoeRouting lists tiles of width 16 << w");

int moe_tile_capacity(int experts, int pairs) {
    return (pairs + kMatrixColumnTiles[0] - 1) / kMatrixColumnTiles[0] + std::min(experts, pairs);
}

__launch_bounds__(kSortThreads) __global__
    void moe_sort_kernel(const std::int32_t* __restrict__ ids, int pairs, int experts,
                         std::int32_t* __restrict__ bounds, std::int32_t* __restrict__ sorted,
                         std::int32_t* __restrict__ active,
                         std::int32_t* __restrict__ active_count,
                         std::int32_t* __restrict__ tile_count, std::int32_t* __restrict__ tiles,
                         int tile_capacity, int vector_pairs, std::int32_t* __restrict__ small_active,
                         std::int32_t* __restrict__ small_count,
                         std::int32_t* __restrict__ large_tile_count,
                         std::int32_t* __restrict__ large_tiles) {
    extern __shared__ int shared[];
    const bool split = small_active != nullptr; // uniform over the block
    int* count  = shared;
    int* cursor = shared + experts;
    for (int e = threadIdx.x; e < experts; e += blockDim.x) { count[e] = 0; }
    __syncthreads();
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int e = ids[p];
        if (e >= 0 && e < experts) { atomicAdd(count + e, 1); }
    }
    __syncthreads();
    // Each thread owns a run of consecutive experts; two block scans give every run its first
    // pair and its first active slot.
    using Scan = cub::BlockScan<int, kSortThreads>;
    __shared__ typename Scan::TempStorage scan;
    const int per   = (experts + kSortThreads - 1) / kSortThreads;
    const int first = threadIdx.x * per;
    const auto small = [&](int e) { return count[e] != 0 && count[e] <= vector_pairs; };
    int owned = 0, used = 0, few = 0;
    for (int e = first; e < min(first + per, experts); ++e) {
        owned += count[e];
        used += count[e] != 0;
        few += split && small(e);
    }
    int run = 0, listed = 0, listed_total = 0, small_at = 0, small_total = 0;
    Scan(scan).ExclusiveSum(owned, run);
    __syncthreads();
    Scan(scan).ExclusiveSum(used, listed, listed_total);
    if (split) {
        __syncthreads();
        Scan(scan).ExclusiveSum(few, small_at, small_total);
    }
    for (int e = first; e < min(first + per, experts); ++e) {
        bounds[e] = run;
        cursor[e] = run;
        run += count[e];
        if (count[e] != 0) { active[listed++] = e; }
        if (split && small(e)) { small_active[small_at++] = e; }
    }
    if (threadIdx.x == kSortThreads - 1) {
        bounds[experts] = run;
        *active_count   = listed_total;
        if (split) { *small_count = small_total; }
    }
    // Each width's tiles: a scan over the runs of how many tiles their experts' pairs fill.
#pragma unroll 1
    for (int w = 0; w < kMoeTileWidths; ++w) {
        const int width = 16 << w;
        int filled = 0, large = 0;
        for (int e = first; e < min(first + per, experts); ++e) {
            const int n = (count[e] + width - 1) / width;
            filled += n;
            large += split && count[e] > vector_pairs ? n : 0;
        }
        int at = 0, total = 0;
        __syncthreads();
        Scan(scan).ExclusiveSum(filled, at, total);
        std::int32_t* list = tiles + std::int64_t(w) * tile_capacity;
        for (int e = first; e < min(first + per, experts); ++e) {
            for (int tile = 0; tile * width < count[e]; ++tile) { list[at++] = e | tile << 16; }
        }
        if (threadIdx.x == kSortThreads - 1) { tile_count[w] = total; }
        if (split) {
            int large_at = 0, large_total = 0;
            __syncthreads();
            Scan(scan).ExclusiveSum(large, large_at, large_total);
            std::int32_t* large_list = large_tiles + std::int64_t(w) * tile_capacity;
            for (int e = first; e < min(first + per, experts); ++e) {
                if (count[e] <= vector_pairs) { continue; }
                for (int tile = 0; tile * width < count[e]; ++tile) {
                    large_list[large_at++] = e | tile << 16;
                }
            }
            if (threadIdx.x == kSortThreads - 1) { large_tile_count[w] = large_total; }
        }
    }
    __syncthreads();
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int e = ids[p];
        if (e >= 0 && e < experts) { sorted[atomicAdd(cursor + e, 1)] = p; }
    }
    // Positions past the kept pairs name pair 0, so a consumer that reads every position (the
    // matrix activation's gather) reads a real column.
    for (int p = bounds[experts] + threadIdx.x; p < pairs; p += blockDim.x) { sorted[p] = 0; }
}

VectorLaunch vector_launch(GgmlType type) {
    switch (type) {
    case GgmlType::Q8_0: return vec_launch<GGML_TYPE_Q8_0>;
    case GgmlType::Q4_0:
        return vec_launch<GGML_TYPE_Q4_0>;
    case GgmlType::Q5_0:
        return vec_launch<GGML_TYPE_Q5_0>;
    case GgmlType::Q2_0:
        return vec_launch<GGML_TYPE_Q2_0>;
    case GgmlType::Q2_K: return vec_launch<GGML_TYPE_Q2_K>;
    case GgmlType::Q3_K: return vec_launch<GGML_TYPE_Q3_K>;
    case GgmlType::Q4_K: return vec_launch<GGML_TYPE_Q4_K>;
    case GgmlType::Q5_K: return vec_launch<GGML_TYPE_Q5_K>;
    case GgmlType::Q6_K: return vec_launch<GGML_TYPE_Q6_K>;
    case GgmlType::IQ2_XXS: return vec_launch<GGML_TYPE_IQ2_XXS>;
    case GgmlType::IQ2_XS: return vec_launch<GGML_TYPE_IQ2_XS>;
    case GgmlType::IQ2_S: return vec_launch<GGML_TYPE_IQ2_S>;
    case GgmlType::IQ3_XXS: return vec_launch<GGML_TYPE_IQ3_XXS>;
    case GgmlType::IQ3_S: return vec_launch<GGML_TYPE_IQ3_S>;
    case GgmlType::IQ1_S: return vec_launch<GGML_TYPE_IQ1_S>;
    case GgmlType::IQ1_M: return vec_launch<GGML_TYPE_IQ1_M>;
    case GgmlType::IQ4_NL: return vec_launch<GGML_TYPE_IQ4_NL>;
    case GgmlType::IQ4_XS: return vec_launch<GGML_TYPE_IQ4_XS>;
    case GgmlType::MXFP4: return vec_launch<GGML_TYPE_MXFP4>;
    case GgmlType::NVFP4: return vec_launch<GGML_TYPE_NVFP4>;
    }
    throw std::invalid_argument("gguf vector product: unsupported block type");
}

VecArgs vector_args(const void* weight, std::int64_t row_bytes, int rows, int k,
                    const void* activation, int columns, const VecStore& out) {
    const auto* base = static_cast<const std::uint8_t*>(activation);
    return VecArgs{static_cast<const std::uint8_t*>(weight),
                   row_bytes,
                   rows,
                   k,
                   reinterpret_cast<const std::int8_t*>(base),
                   reinterpret_cast<const half2*>(base + vec_scales_offset(k, columns)),
                   out};
}

constexpr int kMatrixQuantizeThreads = 128;

// Values i0 .. i0 + 3 of column `column` into the matrix kernel's activation; every lane of the
// warp takes part.
template <mmq_q8_1_ds_layout ds_layout>
__device__ __forceinline__ void quantize_matrix_values(float4 xi, std::int64_t i0, int column,
                                                       block_q8_1_mmq* __restrict__ y,
                                                       int columns) {
    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const std::int64_t k_block = i0 / QK8_1_MMQ;
    const std::int64_t iqs     = i0 % QK8_1_MMQ;

    float amax = fabsf(xi.x);
    amax       = fmaxf(amax, fabsf(xi.y));
    amax       = fmaxf(amax, fabsf(xi.z));
    amax       = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }
    float sum = 0.0f;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }
    const float d_inv = 127.0f / amax;
    char4 q;
    q.x           = roundf(xi.x * d_inv);
    q.y           = roundf(xi.y * d_inv);
    q.z           = roundf(xi.z * d_inv);
    q.w           = roundf(xi.w * d_inv);
    const float d = 1.0f / d_inv;

    const std::int64_t ib = k_block * columns + column;
    reinterpret_cast<char4*>(y[ib].qs)[iqs / 4] = q;
    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 == 0 && iqs < 96) {
            y[ib].d2s6[2 + iqs / 16] = sum;
            if (iqs % 64 == 0) { y[ib].d2s6[iqs / 64] = d; }
        }
    } else if (iqs % 32 == 0) {
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            y[ib].ds4[iqs / 32] = make_half2(d, sum);
        } else {
            y[ib].d4[iqs / 32] = d;
        }
    }
}

// Output column c reads x column column_map[c] / column_group (c without a map), its values
// gathered through input_columns when given; values from k up to k_padded are zero.
template <mmq_q8_1_ds_layout ds_layout>
__global__ void quantize_matrix_kernel(const __nv_bfloat16* __restrict__ x,
                                       const std::int32_t* __restrict__ input_columns,
                                       const std::int32_t* __restrict__ column_map,
                                       int column_group, block_q8_1_mmq* __restrict__ y, int k,
                                       int k_padded, int columns) {
    const std::int64_t i0 = (std::int64_t(blockDim.x) * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= k_padded) { return; }
    const int column             = blockIdx.x;
    const int source_column = column_map != nullptr ? column_map[column] / column_group : column;
    const __nv_bfloat16* source = x + std::int64_t(source_column) * k;
    float4 xi;
    if (i0 >= k) {
        xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    } else if (input_columns != nullptr) {
        xi = make_float4(__bfloat162float(source[input_columns[i0 + 0]]),
                         __bfloat162float(source[input_columns[i0 + 1]]),
                         __bfloat162float(source[input_columns[i0 + 2]]),
                         __bfloat162float(source[input_columns[i0 + 3]]));
    } else {
        const uint2 words = *reinterpret_cast<const uint2*>(source + i0);
        const float2 lo   = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words.x));
        const float2 hi   = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words.y));
        xi                = make_float4(lo.x, lo.y, hi.x, hi.y);
    }
    quantize_matrix_values<ds_layout>(xi, i0, column, y, columns);
}

// FP32 columns of a MoE product in routing order: output column c reads x column
// source[column_map[c]] (zeros for a negative entry), each value multiplied by scale[i] first when
// given; values from k up to k_padded are zero. k is a multiple of 4.
template <mmq_q8_1_ds_layout ds_layout>
__global__ void quantize_matrix_f32_kernel(const float* __restrict__ x,
                                           const std::int32_t* __restrict__ source,
                                           const std::int32_t* __restrict__ column_map,
                                           const float* __restrict__ scale,
                                           block_q8_1_mmq* __restrict__ y, int k, int k_padded,
                                           int columns) {
    const std::int64_t i0 = (std::int64_t(blockDim.x) * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= k_padded) { return; }
    const int column = blockIdx.x;
    const int pair   = column_map[column];
    const int from   = source != nullptr ? source[pair] : pair;
    float4 xi        = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (i0 < k && from >= 0) {
        xi = *reinterpret_cast<const float4*>(x + std::int64_t(from) * k + i0);
        if (scale != nullptr) {
            xi = make_float4(__fmul_rn(xi.x, scale[i0 + 0]), __fmul_rn(xi.y, scale[i0 + 1]),
                             __fmul_rn(xi.z, scale[i0 + 2]), __fmul_rn(xi.w, scale[i0 + 3]));
        }
    }
    quantize_matrix_values<ds_layout>(xi, i0, column, y, columns);
}

__global__ void store_plane_kernel(const float* __restrict__ in, std::int64_t in_column_stride,
                                   int rows, VecStore out) {
    const int row    = blockIdx.x * blockDim.x + threadIdx.x;
    const int column = blockIdx.y;
    if (row >= rows) { return; }
    vec_store(out, row, column, in[column * in_column_stride + row]);
}

__global__ void gather_columns_kernel(const __nv_bfloat16* __restrict__ x, int k,
                                      const std::int32_t* __restrict__ input_columns,
                                      __nv_bfloat16* __restrict__ out) {
    const int i      = blockIdx.x * blockDim.x + threadIdx.x;
    const int column = blockIdx.y;
    if (i >= k) { return; }
    out[std::int64_t(column) * k + i] = x[std::int64_t(column) * k + input_columns[i]];
}

cublasHandle_t blas_handle() {
    static std::array<cublasHandle_t, 16> handles{};
    static std::array<std::once_flag, 16> once;
    int device = 0;
    check(cudaGetDevice(&device), "device");
    if (device < 0 || device >= int(handles.size())) {
        throw std::invalid_argument("gguf: device ordinal out of range");
    }
    // The first use can be inside a CUDA graph capture (a verify graph wide enough for the cuBLAS
    // route); relaxed capture mode lets cublasCreate allocate without invalidating it.
    std::call_once(once[device], [&] {
        cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
        check(cudaThreadExchangeStreamCaptureMode(&mode), "capture mode");
        const cublasStatus_t created = cublasCreate(&handles[device]);
        check(cudaThreadExchangeStreamCaptureMode(&mode), "capture mode");
        if (created != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("gguf: cublasCreate failed");
        }
    });
    return handles[device];
}

// Types whose blocks are narrower than the 256-value super-block one kernel block decodes: their
// rows need only be whole blocks, so the last super-block of a row may be partial.
template <ggml_type type>
inline constexpr bool kSmallBlocks =
    type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0 ||
    type == GGML_TYPE_Q2_0 || type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_MXFP4 ||
    type == GGML_TYPE_NVFP4;

template <ggml_type type, int threads, typename dst_t>
__global__ void dequantize_rows_kernel(const char* __restrict__ weight, std::int64_t row_bytes,
                                       const std::int32_t* __restrict__ row_ids,
                                       dst_t* __restrict__ out, std::int64_t out_row_stride,
                                       int k) {
    const int row          = blockIdx.y;
    const std::int64_t src = row_ids != nullptr ? row_ids[row] : row;
    const void* x          = weight + src * row_bytes;
    dst_t* y               = out + std::int64_t(row) * out_row_stride + std::int64_t(blockIdx.x) * QK_K;
    const int count        = min(QK_K, k - int(blockIdx.x) * QK_K);
    if constexpr (type == GGML_TYPE_Q8_0) {
        const block_q8_0* blocks = static_cast<const block_q8_0*>(x) + blockIdx.x * (QK_K / QK8_0);
        for (int i = threadIdx.x; i < count; i += threads) {
            const block_q8_0& b = blocks[i / QK8_0];
            y[i] = ggml_cuda_cast<dst_t>(__half2float(b.d) * b.qs[i % QK8_0]);
        }
    } else if constexpr (type == GGML_TYPE_Q4_0) {
        const block_q4_0* blocks = static_cast<const block_q4_0*>(x) + blockIdx.x * (QK_K / QK4_0);
        for (int i = threadIdx.x; i < count; i += threads) {
            const block_q4_0& b = blocks[i / QK4_0];
            const int j         = i % QK4_0;
            const int code      = j < QK4_0 / 2 ? (b.qs[j] & 0xF) : (b.qs[j - QK4_0 / 2] >> 4);
            y[i]                = ggml_cuda_cast<dst_t>(__half2float(b.d) * float(code - 8));
        }
    } else if constexpr (type == GGML_TYPE_Q5_0) {
        const block_q5_0* blocks = static_cast<const block_q5_0*>(x) + blockIdx.x * (QK_K / QK5_0);
        for (int i = threadIdx.x; i < count; i += threads) {
            const block_q5_0& b = blocks[i / QK5_0];
            const int j         = i % QK5_0;
            std::uint32_t qh;
            memcpy(&qh, b.qh, sizeof(qh));
            const int low  = j < QK5_0 / 2 ? (b.qs[j] & 0xF) : (b.qs[j - QK5_0 / 2] >> 4);
            const int code = low | int(((qh >> j) & 1u) << 4);
            y[i]           = ggml_cuda_cast<dst_t>(__half2float(b.d) * float(code - 16));
        }
    } else if constexpr (type == GGML_TYPE_Q2_0) {
        const block_q2_0* blocks = static_cast<const block_q2_0*>(x) + blockIdx.x * (QK_K / QK2_0);
        for (int i = threadIdx.x; i < count; i += threads) {
            const block_q2_0& b = blocks[i / QK2_0];
            const int j         = i % QK2_0;
            const int code      = (b.qs[j / 4] >> (2 * (j % 4))) & 3;
            y[i]                = ggml_cuda_cast<dst_t>(__half2float(b.d) * float(code - 1));
        }
    } else if constexpr (type == GGML_TYPE_Q2_K) {
        dequantize_q2_K(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_Q3_K) {
        dequantize_q3_K(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_Q4_K) {
        dequantize_q4_K(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_Q5_K) {
        dequantize_q5_K(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_Q6_K) {
        dequantize_q6_K(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
        dequantize_iq2_xxs(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ2_XS) {
        dequantize_iq2_xs(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        dequantize_iq2_s(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        dequantize_iq3_xxs(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        dequantize_iq3_s(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ1_S) {
        dequantize_iq1_s(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ1_M) {
        dequantize_iq1_m(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_IQ4_NL) {
        const block_iq4_nl* blocks =
            static_cast<const block_iq4_nl*>(x) + blockIdx.x * (QK_K / QK4_NL);
        for (int i = threadIdx.x; i < count; i += threads) {
            const block_iq4_nl& b = blocks[i / QK4_NL];
            const int j           = i % QK4_NL;
            const int code        = j < QK4_NL / 2 ? (b.qs[j] & 0xF) : (b.qs[j - QK4_NL / 2] >> 4);
            y[i] = ggml_cuda_cast<dst_t>(__half2float(b.d) * float(kvalues_iq4nl[code]));
        }
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        dequantize_iq4_xs(x, blockIdx.x, y, threadIdx.x);
    } else if constexpr (type == GGML_TYPE_MXFP4) {
        const block_mxfp4* blocks = static_cast<const block_mxfp4*>(x) + blockIdx.x * (QK_K / QK_MXFP4);
        for (int i = threadIdx.x; i < count; i += threads) {
            const block_mxfp4& b = blocks[i / QK_MXFP4];
            const int j          = i % QK_MXFP4;
            const int code       = j < QK_MXFP4 / 2 ? (b.qs[j] & 0xF) : (b.qs[j - QK_MXFP4 / 2] >> 4);
            y[i] = ggml_cuda_cast<dst_t>(ggml_cuda_e8m0_to_fp32(b.e) * 0.5f * float(kvalues_mxfp4[code]));
        }
    } else if constexpr (type == GGML_TYPE_NVFP4) {
        const block_nvfp4* blocks = static_cast<const block_nvfp4*>(x) + blockIdx.x * (QK_K / QK_NVFP4);
        for (int i = threadIdx.x; i < count; i += threads) {
            const block_nvfp4& b = blocks[i / QK_NVFP4];
            const int sub        = (i % QK_NVFP4) / QK_NVFP4_SUB;
            const int j          = i % QK_NVFP4_SUB;
            const int byte       = b.qs[8 * sub + j % 8];
            const int code       = j < 8 ? (byte & 0xF) : (byte >> 4);
            y[i] = ggml_cuda_cast<dst_t>(ue4m3_half(b.d[sub]) * float(kvalues_mxfp4[code]));
        }
    }
}

template <ggml_type type, int threads, typename dst_t>
void launch_dequantize(const void* weight, std::int64_t row_bytes, int k,
                       const std::int32_t* row_ids, int rows, dst_t* out,
                       std::int64_t out_row_stride, cudaStream_t stream) {
    if (k <= 0 || k % (kSmallBlocks<type> ? ggml_cuda_type_traits<type>::qk : QK_K) != 0 ||
        rows <= 0) {
        throw std::invalid_argument("gguf dequantize: K must be whole blocks of the type");
    }
    // The slice decoders write whole 16-byte groups; this kernel stays for the rest.
    if constexpr (kSliceDequantize<type>) {
        if (out_row_stride % 8 == 0 && reinterpret_cast<std::uintptr_t>(out) % 16 == 0) {
            slice_dequantize<type>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
            return;
        }
    }
    dequantize_rows_kernel<type, threads, dst_t>
        <<<dim3((k + QK_K - 1) / QK_K, rows, 1), threads, 0, stream>>>(
            static_cast<const char*>(weight), row_bytes, row_ids, out, out_row_stride, k);
    check(cudaGetLastError(), "dequantize launch");
}

template <typename dst_t>
void dispatch_dequantize(GgmlType type, const void* weight, std::int64_t row_bytes, int k,
                         const std::int32_t* row_ids, int rows, dst_t* out,
                         std::int64_t out_row_stride, cudaStream_t stream) {
    switch (type) {
    case GgmlType::Q8_0: return launch_dequantize<GGML_TYPE_Q8_0, 64>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::Q4_0:
        return launch_dequantize<GGML_TYPE_Q4_0, 64>(weight, row_bytes, k, row_ids, rows, out,
                                                     out_row_stride, stream);
    case GgmlType::Q5_0:
        return launch_dequantize<GGML_TYPE_Q5_0, 64>(weight, row_bytes, k, row_ids, rows, out,
                                                     out_row_stride, stream);
    case GgmlType::Q2_0:
        return launch_dequantize<GGML_TYPE_Q2_0, 64>(weight, row_bytes, k, row_ids, rows, out,
                                                     out_row_stride, stream);
    case GgmlType::Q2_K: return launch_dequantize<GGML_TYPE_Q2_K, 64>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::Q3_K: return launch_dequantize<GGML_TYPE_Q3_K, 64>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::Q4_K: return launch_dequantize<GGML_TYPE_Q4_K, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::Q5_K: return launch_dequantize<GGML_TYPE_Q5_K, 64>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::Q6_K: return launch_dequantize<GGML_TYPE_Q6_K, 64>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ2_XXS: return launch_dequantize<GGML_TYPE_IQ2_XXS, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ2_XS: return launch_dequantize<GGML_TYPE_IQ2_XS, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ2_S: return launch_dequantize<GGML_TYPE_IQ2_S, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ3_XXS: return launch_dequantize<GGML_TYPE_IQ3_XXS, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ3_S: return launch_dequantize<GGML_TYPE_IQ3_S, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ1_S: return launch_dequantize<GGML_TYPE_IQ1_S, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ1_M: return launch_dequantize<GGML_TYPE_IQ1_M, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ4_NL: return launch_dequantize<GGML_TYPE_IQ4_NL, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::IQ4_XS: return launch_dequantize<GGML_TYPE_IQ4_XS, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::MXFP4: return launch_dequantize<GGML_TYPE_MXFP4, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    case GgmlType::NVFP4: return launch_dequantize<GGML_TYPE_NVFP4, 32>(weight, row_bytes, k, row_ids, rows, out, out_row_stride, stream);
    }
    throw std::invalid_argument("gguf dequantize: unsupported block type");
}

} // namespace
} // namespace detail

BlockShape block_shape(GgmlType type) {
    switch (type) {
    case GgmlType::Q8_0: return {32, 34};
    case GgmlType::Q4_0:
        return {32, 18};
    case GgmlType::Q5_0:
        return {32, 22};
    case GgmlType::Q2_0:
        return {64, 18};
    case GgmlType::Q2_K: return {256, 84};
    case GgmlType::Q3_K: return {256, 110};
    case GgmlType::Q4_K: return {256, 144};
    case GgmlType::Q5_K: return {256, 176};
    case GgmlType::Q6_K: return {256, 210};
    case GgmlType::IQ2_XXS: return {256, 66};
    case GgmlType::IQ2_XS: return {256, 74};
    case GgmlType::IQ3_XXS: return {256, 98};
    case GgmlType::IQ1_S: return {256, 50};
    case GgmlType::IQ4_NL: return {32, 18};
    case GgmlType::IQ3_S: return {256, 110};
    case GgmlType::IQ2_S: return {256, 82};
    case GgmlType::IQ4_XS: return {256, 136};
    case GgmlType::IQ1_M: return {256, 56};
    case GgmlType::MXFP4: return {32, 17};
    case GgmlType::NVFP4: return {64, 36};
    }
    throw std::invalid_argument("gguf: unsupported block type");
}

bool has_matrix_kernel(GgmlType type) { return type != GgmlType::IQ1_M && type != GgmlType::MXFP4; }

int multiprocessor_count() { return detail::device_facts().sm_count; }

std::size_t vector_activation_scales_offset(int k, int columns) {
    return detail::vec_scales_offset(k, columns);
}

std::size_t vector_activation_bytes(int k, int columns) {
    return detail::vec_scales_offset(k, columns) + std::size_t(columns) * (k / QK8_1) * sizeof(half2);
}

void quantize_vector_activation(const __nv_bfloat16* x, int k, int columns,
                                const std::int32_t* input_columns, void* out,
                                cudaStream_t stream) {
    // Whole warps of 32 values: a warp past K leaves as one, before its reductions.
    if (k <= 0 || k % QK8_1 != 0 || columns <= 0 || columns > 65535) {
        throw std::invalid_argument("gguf vector activation: K must be whole 32-value blocks");
    }
    auto* base = static_cast<std::uint8_t*>(out);
    const dim3 blocks((k + detail::kQuantizeThreads - 1) / detail::kQuantizeThreads, columns, 1);
    detail::quantize_vector_kernel<<<blocks, detail::kQuantizeThreads, 0, stream>>>(
        x, input_columns, reinterpret_cast<std::int8_t*>(base),
        reinterpret_cast<half2*>(base + detail::vec_scales_offset(k, columns)), k);
    detail::check(cudaGetLastError(), "vector activation launch");
}

void quantize_vector_activation_f32(const float* x, int k, int columns, const std::int32_t* source,
                                    const float* scale, void* out, cudaStream_t stream) {
    if (k <= 0 || k % QK8_1 != 0 || columns <= 0 || columns > 65535) {
        throw std::invalid_argument("gguf vector activation: K must be whole 32-value blocks");
    }
    auto* base = static_cast<std::uint8_t*>(out);
    const dim3 blocks((k + detail::kQuantizeThreads - 1) / detail::kQuantizeThreads, columns, 1);
    detail::quantize_vector_f32_kernel<<<blocks, detail::kQuantizeThreads, 0, stream>>>(
        x, source, scale, reinterpret_cast<std::int8_t*>(base),
        reinterpret_cast<half2*>(base + detail::vec_scales_offset(k, columns)), k);
    detail::check(cudaGetLastError(), "vector activation launch");
}

void vector_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                    const void* activation, int columns, const VectorOutput& out,
                    cudaStream_t stream) {
    if ((out.bf16 == nullptr) == (out.f32 == nullptr) ||
        (out.epilogue == Epilogue::GateProduct && out.gate == nullptr)) {
        throw std::invalid_argument("gguf vector product: invalid output");
    }
    const detail::VecStore store{out.bf16,     out.f32,  out.column_stride,
                                 out.epilogue, out.gate, out.gate_column_stride,
                                 static_cast<int>(out.activation)};
    detail::vector_launch(type)(
        detail::vector_args(weight, row_bytes, rows, k, activation, columns, store), columns,
        false, stream);
}

void vector_swiglu(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                   const void* activation, int columns, __nv_bfloat16* out,
                   std::int64_t out_column_stride, cudaStream_t stream,
                   GateActivation gate_activation) {
    const detail::VecStore store{out,     nullptr, out_column_stride,
                                 Epilogue::Store, nullptr, 0,
                                 static_cast<int>(gate_activation)};
    detail::vector_launch(type)(
        detail::vector_args(weight, row_bytes, rows, k, activation, columns, store), columns,
        true, stream);
}

std::size_t moe_routing_bytes(int experts, int pairs) {
    return (std::size_t(experts) + 1 + std::size_t(pairs) + std::size_t(experts) + 1 +
            detail::kMoeTileWidths +
            std::size_t(detail::kMoeTileWidths) * detail::moe_tile_capacity(experts, pairs)) *
               sizeof(std::int32_t) +
           256;
}

namespace {

// moe_sort_routes, and with `vector_pairs` >= 0 the split's lists after its arrays.
MoeSplitRouting sort_routes(const std::int32_t* ids, int pairs, int experts, int vector_pairs,
                            void* workspace, cudaStream_t stream) {
    if (pairs <= 0 || experts <= 0 || experts > 8192 || workspace == nullptr) {
        throw std::invalid_argument("gguf moe routing: invalid pairs or experts");
    }
    auto* base                 = static_cast<std::int32_t*>(workspace);
    std::int32_t* bounds       = base;
    std::int32_t* sorted       = bounds + experts + 1;
    std::int32_t* active       = sorted + pairs;
    std::int32_t* active_count = active + experts;
    std::int32_t* tile_count   = active_count + 1;
    std::int32_t* tiles        = tile_count + detail::kMoeTileWidths;
    const int capacity         = detail::moe_tile_capacity(experts, pairs);
    const bool split           = vector_pairs >= 0;
    std::int32_t* small_active = split ? tiles + std::int64_t(detail::kMoeTileWidths) * capacity
                                       : nullptr;
    std::int32_t* small_count  = split ? small_active + experts : nullptr;
    std::int32_t* large_count  = split ? small_count + 1 : nullptr;
    std::int32_t* large_tiles  = split ? large_count + detail::kMoeTileWidths : nullptr;
    detail::moe_sort_kernel<<<1, detail::kSortThreads, 2 * experts * sizeof(int), stream>>>(
        ids, pairs, experts, bounds, sorted, active, active_count, tile_count, tiles, capacity,
        vector_pairs, small_active, small_count, large_count, large_tiles);
    detail::check(cudaGetLastError(), "moe routing launch");
    MoeSplitRouting out;
    out.all    = MoeRouting{bounds, sorted, active, active_count, tile_count, tiles, capacity};
    out.vector = out.all;
    out.matrix = out.all;
    if (split) {
        out.vector.active       = small_active;
        out.vector.active_count = small_count;
        out.matrix.tile_count   = large_count;
        out.matrix.tiles        = large_tiles;
    }
    return out;
}

} // namespace

MoeRouting moe_sort_routes(const std::int32_t* ids, int pairs, int experts, void* workspace,
                           cudaStream_t stream) {
    return sort_routes(ids, pairs, experts, -1, workspace, stream).all;
}

MoeRouting moe_sorted_routes(void* workspace, int pairs, int experts) {
    if (pairs <= 0 || experts <= 0 || experts > 8192 || workspace == nullptr) {
        throw std::invalid_argument("gguf moe routing: invalid pairs or experts");
    }
    auto* bounds       = static_cast<std::int32_t*>(workspace);
    auto* sorted       = bounds + experts + 1;
    auto* active       = sorted + pairs;
    auto* active_count = active + experts;
    auto* tile_count   = active_count + 1;
    auto* tiles        = tile_count + detail::kMoeTileWidths;
    return MoeRouting{bounds, sorted, active, active_count, tile_count, tiles,
                      detail::moe_tile_capacity(experts, pairs)};
}

std::size_t moe_split_routing_bytes(int experts, int pairs) {
    return moe_routing_bytes(experts, pairs) +
           (std::size_t(experts) + 1 + detail::kMoeTileWidths +
            std::size_t(detail::kMoeTileWidths) * detail::moe_tile_capacity(experts, pairs)) *
               sizeof(std::int32_t);
}

MoeSplitRouting moe_sort_routes_split(const std::int32_t* ids, int pairs, int experts,
                                      int vector_pairs, void* workspace, cudaStream_t stream) {
    if (vector_pairs < 0) { throw std::invalid_argument("gguf moe routing: negative split"); }
    return sort_routes(ids, pairs, experts, vector_pairs, workspace, stream);
}

namespace {

constexpr int kMaxFixedParts = 8;

struct FixedParts {
    const unsigned long long* part[kMaxFixedParts];
    int count;
};

__global__ void keep_experts_kernel(const std::int32_t* __restrict__ ids, int pairs, int begin,
                                    int end, int shift, std::int32_t* __restrict__ out) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pairs) { return; }
    const int e = ids[p];
    out[p]      = e >= begin && e < end ? e - shift : -1;
}

// Block x takes token x; each thread a run of its values over every pair of the token.
__global__ void weighted_rows_kernel(const float* __restrict__ rows,
                                     const std::int32_t* __restrict__ ids,
                                     const float* __restrict__ weights, int per_token,
                                     int row_values, unsigned long long* __restrict__ fixed) {
    const std::int64_t first = std::int64_t(blockIdx.x) * per_token;
    for (int r = threadIdx.x; r < row_values; r += blockDim.x) {
        unsigned long long sum = 0;
        for (int k = 0; k < per_token; ++k) {
            const std::int64_t p = first + k;
            if (ids[p] < 0) { continue; }
            sum += static_cast<unsigned long long>(
                fixed_product(rows[p * row_values + r], weights[p]));
        }
        fixed[std::int64_t(blockIdx.x) * row_values + r] = sum;
    }
}

__global__ void add_fixed_parts_kernel(unsigned long long* __restrict__ sums, FixedParts parts,
                                       std::int64_t n) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    unsigned long long sum = sums[i];
    for (int j = 0; j < parts.count; ++j) { sum += parts.part[j][i]; }
    sums[i] = sum;
}

} // namespace

void moe_keep_experts(const std::int32_t* ids, int pairs, int begin, int end, std::int32_t* out,
                      cudaStream_t stream) {
    if (pairs <= 0 || begin < 0 || end < begin) {
        throw std::invalid_argument("gguf moe_keep_experts: invalid pairs or range");
    }
    keep_experts_kernel<<<(pairs + 255) / 256, 256, 0, stream>>>(ids, pairs, begin, end, 0, out);
    detail::check(cudaGetLastError(), "moe_keep_experts launch");
}

void moe_local_experts(const std::int32_t* ids, int pairs, int begin, int end, std::int32_t* out,
                       cudaStream_t stream) {
    if (pairs <= 0 || begin < 0 || end < begin) {
        throw std::invalid_argument("gguf moe_local_experts: invalid pairs or range");
    }
    keep_experts_kernel<<<(pairs + 255) / 256, 256, 0, stream>>>(ids, pairs, begin, end, begin,
                                                                 out);
    detail::check(cudaGetLastError(), "moe_local_experts launch");
}

void moe_weighted_rows(const float* rows, const std::int32_t* ids, const float* weights, int tokens,
                       int per_token, int row_values, unsigned long long* fixed,
                       cudaStream_t stream) {
    if (tokens <= 0 || per_token <= 0 || row_values <= 0) {
        throw std::invalid_argument("gguf moe_weighted_rows: invalid shape");
    }
    weighted_rows_kernel<<<tokens, 256, 0, stream>>>(rows, ids, weights, per_token, row_values,
                                                     fixed);
    detail::check(cudaGetLastError(), "moe_weighted_rows launch");
}

void add_fixed_parts(unsigned long long* sums, const unsigned long long* const* parts, int count,
                     std::int64_t n, cudaStream_t stream) {
    if (count < 0 || count > kMaxFixedParts || n < 0) {
        throw std::invalid_argument("gguf add_fixed_parts: 0..8 parts");
    }
    if (count == 0 || n == 0) { return; }
    FixedParts packed{};
    packed.count = count;
    for (int j = 0; j < count; ++j) { packed.part[j] = parts[j]; }
    add_fixed_parts_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(sums, packed, n);
    detail::check(cudaGetLastError(), "add_fixed_parts launch");
}

namespace {

detail::MoeVecArgs moe_args(const MoeTable& first, const MoeTable* second,
                            const MoeRouting& routing, int column_group, int per_token,
                            const void* activation, int columns) {
    if (first.experts == nullptr || first.rows <= 0 || first.k <= 0 || column_group <= 0 ||
        per_token <= 0 || columns <= 0 || activation == nullptr || routing.bounds == nullptr ||
        (second != nullptr && (second->experts == nullptr || second->rows != first.rows ||
                               second->k != first.k || second->row_bytes != first.row_bytes))) {
        throw std::invalid_argument("gguf moe product: invalid tables or routing");
    }
    const auto* base = static_cast<const std::uint8_t*>(activation);
    detail::MoeVecArgs args{};
    args.first = reinterpret_cast<const std::uint8_t* const*>(first.experts);
    args.second =
        second != nullptr ? reinterpret_cast<const std::uint8_t* const*>(second->experts) : nullptr;
    args.row_bytes    = first.row_bytes;
    args.rows         = first.rows;
    args.k            = first.k;
    args.bounds       = routing.bounds;
    args.sorted       = routing.sorted;
    args.active       = routing.active;
    args.active_count = routing.active_count;
    args.column_group = column_group;
    args.per_token    = per_token;
    args.qs           = reinterpret_cast<const std::int8_t*>(base);
    args.ds = reinterpret_cast<const half2*>(base + detail::vec_scales_offset(first.k, columns));
    return args;
}

} // namespace

void moe_vector_product(GgmlType type, const MoeTable& table, const MoeRouting& routing,
                        int max_active, int column_group, int per_token, const void* activation,
                        int columns, const MoeOutput& out, int chunk, cudaStream_t stream) {
    const int targets =
        int(out.bf16 != nullptr) + int(out.f32 != nullptr) + int(out.weighted != nullptr);
    if (targets != 1 || (out.weighted != nullptr && out.weights == nullptr)) {
        throw std::invalid_argument("gguf moe product: invalid output");
    }
    auto args     = moe_args(table, nullptr, routing, column_group, per_token, activation, columns);
    args.out_bf16 = out.bf16;
    args.out_f32  = out.f32;
    args.weighted = out.weighted;
    args.weights    = out.weights;
    args.gate       = out.gate;
    args.activation = static_cast<int>(out.activation);
    detail::moe_launcher(type)(args, max_active, chunk, false, stream);
}

void moe_vector_swiglu(GgmlType type, const MoeTable& gate, const MoeTable& up,
                       const MoeRouting& routing, int max_active, int column_group,
                       const void* activation, int columns, __nv_bfloat16* out, int chunk,
                       cudaStream_t stream, GateActivation gate_activation) {
    if (out == nullptr) { throw std::invalid_argument("gguf moe swiglu: invalid output"); }
    auto args       = moe_args(gate, &up, routing, column_group, 1, activation, columns);
    args.out_bf16   = out;
    args.activation = static_cast<int>(gate_activation);
    detail::moe_launcher(type)(args, max_active, chunk, true, stream);
}

void moe_decode_up(GgmlType type, const MoeDecodeUpArgs& args, int chunk, cudaStream_t stream) {
    if (args.m == nullptr || args.gate == nullptr || args.up == nullptr || args.middle == nullptr) {
        throw std::invalid_argument("gguf moe decode: invalid up arguments");
    }
    detail::decode_launcher<detail::DecodeLaunch<MoeDecodeUpArgs>, detail::PickUp>(type)(args, chunk,
                                                                                      stream);
}

void moe_decode_down(GgmlType type, const MoeDecodeDownArgs& args, int chunk, cudaStream_t stream) {
    if (args.table_a == nullptr || args.middle == nullptr || args.weights == nullptr ||
        args.fixed == nullptr || (args.y != nullptr && args.counter == nullptr) ||
        (args.ids_b != nullptr && args.table_b == nullptr)) {
        throw std::invalid_argument("gguf moe decode: invalid down arguments");
    }
    detail::decode_launcher<detail::DecodeLaunch<MoeDecodeDownArgs>, detail::PickDown>(type)(
        args, chunk, stream);
}

void moe_decode_prep(GgmlType type, const MoeDecodePrepArgs& args, cudaStream_t stream) {
    if (args.m == nullptr || args.ids == nullptr || args.ds == nullptr || args.qsum == nullptr) {
        throw std::invalid_argument("gguf moe decode: invalid preparation arguments");
    }
    detail::decode_launcher<void (*)(const MoeDecodePrepArgs&, cudaStream_t), detail::PickPrep>(
        type)(args, stream);
}

bool moe_decode_transposed(GgmlType type) {
    // Decoder<type>::kTransposed of ggml_bridge_vec.cuh.
    return type == GgmlType::Q2_0;
}


int matrix_activation_layout(GgmlType type) {
    return static_cast<int>(mmq_get_q8_1_ds_layout(detail::to_ggml(type)));
}

bool matrix_activation_f16_sums(GgmlType type) {
    return mmq_get_q8_1_ds_layout(detail::to_ggml(type)) == MMQ_Q8_1_DS_LAYOUT_DS4;
}

bool vector_activation_f16_sums(GgmlType type) {
    return type == GgmlType::Q4_K || type == GgmlType::Q5_K || type == GgmlType::IQ1_S;
}

std::size_t matrix_activation_bytes(int k, int columns) {
    return std::size_t(columns) * (k / QK8_1_MMQ) * sizeof(block_q8_1_mmq) +
           detail::kMatrixActivationSlack;
}

namespace {

// k_padded is a whole number of 128-value blocks, so every warp of the grid is wholly in or out.
void quantize_matrix(GgmlType type, const __nv_bfloat16* x, int k, int k_padded, int columns,
                     const std::int32_t* input_columns, const std::int32_t* column_map,
                     int column_group, void* out, cudaStream_t stream) {
    const dim3 blocks(columns,
                      (k_padded + 4 * detail::kMatrixQuantizeThreads - 1) /
                          (4 * detail::kMatrixQuantizeThreads),
                      1);
    auto* y = static_cast<block_q8_1_mmq*>(out);
    switch (mmq_get_q8_1_ds_layout(detail::to_ggml(type))) {
    case MMQ_Q8_1_DS_LAYOUT_D4:
        detail::quantize_matrix_kernel<MMQ_Q8_1_DS_LAYOUT_D4>
            <<<blocks, detail::kMatrixQuantizeThreads, 0, stream>>>(
                x, input_columns, column_map, column_group, y, k, k_padded, columns);
        break;
    case MMQ_Q8_1_DS_LAYOUT_DS4:
        detail::quantize_matrix_kernel<MMQ_Q8_1_DS_LAYOUT_DS4>
            <<<blocks, detail::kMatrixQuantizeThreads, 0, stream>>>(
                x, input_columns, column_map, column_group, y, k, k_padded, columns);
        break;
    case MMQ_Q8_1_DS_LAYOUT_D2S6:
        detail::quantize_matrix_kernel<MMQ_Q8_1_DS_LAYOUT_D2S6>
            <<<blocks, detail::kMatrixQuantizeThreads, 0, stream>>>(
                x, input_columns, column_map, column_group, y, k, k_padded, columns);
        break;
    }
    detail::check(cudaGetLastError(), "matrix activation launch");
}

} // namespace

void quantize_matrix_activation(GgmlType type, const __nv_bfloat16* x, int k, int columns,
                                const std::int32_t* input_columns, void* out, cudaStream_t stream) {
    if (k <= 0 || k % (4 * detail::kMatrixQuantizeThreads) != 0 || columns <= 0) {
        throw std::invalid_argument("gguf matrix activation: K must be whole 512-value groups");
    }
    quantize_matrix(type, x, k, k, columns, input_columns, nullptr, 1, out, stream);
}

namespace {

// The matrix kernel's K step: an activation covers whole steps.
constexpr int kMatrixStep = 256;

int moe_padded_k(int k) { return (k + kMatrixStep - 1) / kMatrixStep * kMatrixStep; }

// Whether the MoE matrix product takes Volta's FP16 route (ggml_bridge_sm70.cuh) on the current
// device: wherever the device runs sm_70 code.
bool moe_matrix_sm70() { return current_device_volta(); }

// Volta's activation: an FP16 plane [pairs][k_padded], zero past k, then one FP32 factor per
// column. Column c holds x column sorted[c] / column_group times 2^-e and factor 2^e, e the least
// non-negative exponent that keeps the column's largest finite magnitude below 2^15 (zero for a
// column with an infinity or NaN, which then propagates as it would in BF16).
std::size_t moe_sm70_activation_bytes(int k_padded, int pairs) {
    return std::size_t(pairs) * k_padded * sizeof(__half) + std::size_t(pairs) * sizeof(float);
}

constexpr int kSm70ActivationThreads = 256;

// One column c of the plane from `load(i)`, values 2 i and 2 i + 1 of the column (i < k / 2).
template <typename Load>
__device__ __forceinline__ void moe_sm70_activation_column(Load load, int c, int k, int k_padded,
                                                           __half* __restrict__ plane,
                                                           float* __restrict__ factors) {
    __shared__ float warp_max[kSm70ActivationThreads / 32];
    float largest = 0.0f;
    for (int i = int(threadIdx.x); i < k / 2; i += kSm70ActivationThreads) {
        const float2 v = load(i);
        largest        = fmaxf(largest, fmaxf(fabsf(v.x), fabsf(v.y)));
        if (!isfinite(v.x) || !isfinite(v.y)) { largest = INFINITY; }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        largest = fmaxf(largest, __shfl_xor_sync(0xFFFFFFFFu, largest, offset));
    }
    if ((threadIdx.x & 31) == 0) { warp_max[threadIdx.x / 32] = largest; }
    __syncthreads();
    largest = warp_max[0];
#pragma unroll
    for (int w = 1; w < kSm70ActivationThreads / 32; ++w) { largest = fmaxf(largest, warp_max[w]); }
    // largest < 2^(ilogb + 1), so largest * 2^-(ilogb - 14) < 2^15.
    const int e      = isfinite(largest) && largest >= 32768.0f ? ilogbf(largest) - 14 : 0;
    const float down = ldexpf(1.0f, -e);
    __half2* out     = reinterpret_cast<__half2*>(plane + std::int64_t(c) * k_padded);
    for (int i = int(threadIdx.x); i < k_padded / 2; i += kSm70ActivationThreads) {
        float2 v = make_float2(0.0f, 0.0f);
        if (2 * i < k) {
            v = load(i);
        }
        out[i] = __floats2half2_rn(v.x * down, v.y * down);
    }
    if (threadIdx.x == 0) { factors[c] = ldexpf(1.0f, e); }
}

__global__ void __launch_bounds__(kSm70ActivationThreads)
    moe_sm70_activation_kernel(const __nv_bfloat16* __restrict__ x, int k, int k_padded,
                               const std::int32_t* __restrict__ sorted, int column_group,
                               __half* __restrict__ plane, float* __restrict__ factors) {
    const int c                  = int(blockIdx.x);
    const __nv_bfloat162* column = reinterpret_cast<const __nv_bfloat162*>(
        x + std::int64_t(sorted[c] / column_group) * k);
    moe_sm70_activation_column([column](int i) { return __bfloat1622float2(column[i]); }, c, k,
                               k_padded, plane, factors);
}

// The FP32 form: column c reads x column source[sorted[c]] (zeros for a negative entry), each value
// multiplied by scale[i] first when given.
__global__ void __launch_bounds__(kSm70ActivationThreads)
    moe_sm70_activation_f32_kernel(const float* __restrict__ x, int k, int k_padded,
                                   const std::int32_t* __restrict__ sorted,
                                   const std::int32_t* __restrict__ source,
                                   const float* __restrict__ scale, __half* __restrict__ plane,
                                   float* __restrict__ factors) {
    const int c      = int(blockIdx.x);
    const int pair   = sorted[c];
    const int from   = source != nullptr ? source[pair] : pair;
    const float2* column =
        from >= 0 ? reinterpret_cast<const float2*>(x + std::int64_t(from) * k) : nullptr;
    moe_sm70_activation_column(
        [column, scale](int i) {
            if (column == nullptr) { return make_float2(0.0f, 0.0f); }
            float2 v = column[i];
            if (scale != nullptr) {
                v = make_float2(__fmul_rn(v.x, scale[2 * i]), __fmul_rn(v.y, scale[2 * i + 1]));
            }
            return v;
        },
        c, k, k_padded, plane, factors);
}

} // namespace

std::size_t moe_matrix_activation_bytes(int k, int pairs) {
    if (moe_matrix_sm70()) { return moe_sm70_activation_bytes(moe_padded_k(k), pairs); }
    return matrix_activation_bytes(moe_padded_k(k), pairs);
}

int matrix_padded_k(int k) { return moe_padded_k(k); }

void quantize_matrix_activation_padded(GgmlType type, const __nv_bfloat16* x, int k, int columns,
                                       const std::int32_t* input_columns, void* out,
                                       cudaStream_t stream) {
    if (k <= 0 || k % 32 != 0 || columns <= 0) {
        throw std::invalid_argument("gguf padded matrix activation: K must be whole 32-value blocks");
    }
    quantize_matrix(type, x, k, moe_padded_k(k), columns, input_columns, nullptr, 1, out, stream);
}

void quantize_moe_matrix_activation(GgmlType type, const __nv_bfloat16* x, int k,
                                    const MoeRouting& routing, int pairs, int column_group,
                                    void* out, cudaStream_t stream) {
    if (routing.sorted == nullptr || column_group <= 0 || k <= 0 || k % 32 != 0 || pairs <= 0) {
        throw std::invalid_argument("gguf moe matrix activation: invalid routing or K");
    }
    if (moe_matrix_sm70()) {
        const int k_padded = moe_padded_k(k);
        auto* plane        = static_cast<__half*>(out);
        moe_sm70_activation_kernel<<<pairs, kSm70ActivationThreads, 0, stream>>>(
            x, k, k_padded, routing.sorted, column_group, plane,
            reinterpret_cast<float*>(plane + std::size_t(pairs) * k_padded));
        detail::check(cudaGetLastError(), "moe matrix activation (sm_70) launch");
        return;
    }
    quantize_matrix(type, x, k, moe_padded_k(k), pairs, nullptr, routing.sorted, column_group, out,
                    stream);
}

void quantize_moe_matrix_activation_f32(GgmlType type, const float* x, int k,
                                        const MoeRouting& routing, int pairs,
                                        const std::int32_t* source, const float* scale, void* out,
                                        cudaStream_t stream) {
    if (routing.sorted == nullptr || k <= 0 || k % 32 != 0 || pairs <= 0) {
        throw std::invalid_argument("gguf moe matrix activation: invalid routing or K");
    }
    const int k_padded = moe_padded_k(k);
    if (moe_matrix_sm70()) {
        auto* plane = static_cast<__half*>(out);
        moe_sm70_activation_f32_kernel<<<pairs, kSm70ActivationThreads, 0, stream>>>(
            x, k, k_padded, routing.sorted, source, scale, plane,
            reinterpret_cast<float*>(plane + std::size_t(pairs) * k_padded));
        detail::check(cudaGetLastError(), "moe matrix activation (sm_70) launch");
        return;
    }
    const dim3 blocks(pairs,
                      (k_padded + 4 * detail::kMatrixQuantizeThreads - 1) /
                          (4 * detail::kMatrixQuantizeThreads),
                      1);
    auto* y = static_cast<block_q8_1_mmq*>(out);
    switch (mmq_get_q8_1_ds_layout(detail::to_ggml(type))) {
    case MMQ_Q8_1_DS_LAYOUT_D4:
        detail::quantize_matrix_f32_kernel<MMQ_Q8_1_DS_LAYOUT_D4>
            <<<blocks, detail::kMatrixQuantizeThreads, 0, stream>>>(
                x, source, routing.sorted, scale, y, k, k_padded, pairs);
        break;
    case MMQ_Q8_1_DS_LAYOUT_DS4:
        detail::quantize_matrix_f32_kernel<MMQ_Q8_1_DS_LAYOUT_DS4>
            <<<blocks, detail::kMatrixQuantizeThreads, 0, stream>>>(
                x, source, routing.sorted, scale, y, k, k_padded, pairs);
        break;
    case MMQ_Q8_1_DS_LAYOUT_D2S6:
        detail::quantize_matrix_f32_kernel<MMQ_Q8_1_DS_LAYOUT_D2S6>
            <<<blocks, detail::kMatrixQuantizeThreads, 0, stream>>>(
                x, source, routing.sorted, scale, y, k, k_padded, pairs);
        break;
    }
    detail::check(cudaGetLastError(), "moe matrix activation launch");
}

std::size_t matrix_fixup_bytes(GgmlType type, int rows, int columns) {
    switch (type) {
    case GgmlType::Q8_0: return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q8_0>(rows, columns);
    case GgmlType::Q4_0:
        return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q4_0>(rows, columns);
    case GgmlType::Q5_0:
        return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q5_0>(rows, columns);
    case GgmlType::Q2_0:
        return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q2_0>(rows, columns);
    case GgmlType::Q2_K: return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q2_K>(rows, columns);
    case GgmlType::Q3_K: return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q3_K>(rows, columns);
    case GgmlType::Q4_K: return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q4_K>(rows, columns);
    case GgmlType::Q5_K: return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q5_K>(rows, columns);
    case GgmlType::Q6_K: return detail::matrix_fixup_bytes_impl<GGML_TYPE_Q6_K>(rows, columns);
    case GgmlType::IQ2_XXS: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ2_XXS>(rows, columns);
    case GgmlType::IQ2_XS: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ2_XS>(rows, columns);
    case GgmlType::IQ2_S: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ2_S>(rows, columns);
    case GgmlType::IQ3_XXS: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ3_XXS>(rows, columns);
    case GgmlType::IQ3_S: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ3_S>(rows, columns);
    case GgmlType::IQ1_S: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ1_S>(rows, columns);
    case GgmlType::IQ4_NL: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ4_NL>(rows, columns);
    case GgmlType::IQ4_XS: return detail::matrix_fixup_bytes_impl<GGML_TYPE_IQ4_XS>(rows, columns);
    case GgmlType::NVFP4: return detail::matrix_fixup_bytes_impl<GGML_TYPE_NVFP4>(rows, columns);
    case GgmlType::IQ1_M:
    case GgmlType::MXFP4: break;
    }
    throw std::invalid_argument("gguf matrix product: no integer kernel for this block type");
}

void matrix_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                    const void* activation, int columns, float* out,
                    std::int64_t out_column_stride, void* fixup, cudaStream_t stream, bool tail) {
#define NINFER_GGUF_MATRIX_CASE(NAME)                                                              \
    case GgmlType::NAME:                                                                           \
        detail::matrix_product_impl<GGML_TYPE_##NAME>(weight, row_bytes, rows, k, activation,      \
                                                      columns, out, out_column_stride, fixup,      \
                                                      stream, tail);                               \
        return
    switch (type) {
        NINFER_GGUF_MATRIX_CASE(Q8_0);
        NINFER_GGUF_MATRIX_CASE(Q4_0);
        NINFER_GGUF_MATRIX_CASE(Q5_0);
        NINFER_GGUF_MATRIX_CASE(Q2_0);
        NINFER_GGUF_MATRIX_CASE(Q2_K);
        NINFER_GGUF_MATRIX_CASE(Q3_K);
        NINFER_GGUF_MATRIX_CASE(Q4_K);
        NINFER_GGUF_MATRIX_CASE(Q5_K);
        NINFER_GGUF_MATRIX_CASE(Q6_K);
        NINFER_GGUF_MATRIX_CASE(IQ2_XXS);
        NINFER_GGUF_MATRIX_CASE(IQ2_XS);
        NINFER_GGUF_MATRIX_CASE(IQ2_S);
        NINFER_GGUF_MATRIX_CASE(IQ3_XXS);
        NINFER_GGUF_MATRIX_CASE(IQ3_S);
        NINFER_GGUF_MATRIX_CASE(IQ1_S);
        NINFER_GGUF_MATRIX_CASE(IQ4_NL);
        NINFER_GGUF_MATRIX_CASE(IQ4_XS);
        NINFER_GGUF_MATRIX_CASE(NVFP4);
    case GgmlType::IQ1_M:
    case GgmlType::MXFP4:
        break;
    }
#undef NINFER_GGUF_MATRIX_CASE
    throw std::invalid_argument("gguf matrix product: no integer kernel for this block type");
}

void matrix_product_bf16(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                         const void* activation, int columns, __nv_bfloat16* out,
                         std::int64_t out_column_stride, float* partial, void* fixup,
                         cudaStream_t stream, bool accumulate, bool tail) {
#define NINFER_GGUF_MATRIX_CASE(NAME)                                                              \
    case GgmlType::NAME:                                                                           \
        if (accumulate) {                                                                         \
            detail::matrix_product_impl<GGML_TYPE_##NAME, __nv_bfloat16, true>(                     \
                weight, row_bytes, rows, k, activation, columns, out, out_column_stride,           \
                fixup, stream, tail, partial);                                                     \
        } else {                                                                                  \
            detail::matrix_product_impl<GGML_TYPE_##NAME, __nv_bfloat16, false>(                    \
                weight, row_bytes, rows, k, activation, columns, out, out_column_stride,           \
                fixup, stream, tail, partial);                                                     \
        }                                                                                         \
        return
    switch (type) {
        NINFER_GGUF_MATRIX_CASE(Q8_0);
        NINFER_GGUF_MATRIX_CASE(Q4_0);
        NINFER_GGUF_MATRIX_CASE(Q5_0);
        NINFER_GGUF_MATRIX_CASE(Q2_0);
        NINFER_GGUF_MATRIX_CASE(Q2_K);
        NINFER_GGUF_MATRIX_CASE(Q3_K);
        NINFER_GGUF_MATRIX_CASE(Q4_K);
        NINFER_GGUF_MATRIX_CASE(Q5_K);
        NINFER_GGUF_MATRIX_CASE(Q6_K);
        NINFER_GGUF_MATRIX_CASE(IQ2_XXS);
        NINFER_GGUF_MATRIX_CASE(IQ2_XS);
        NINFER_GGUF_MATRIX_CASE(IQ2_S);
        NINFER_GGUF_MATRIX_CASE(IQ3_XXS);
        NINFER_GGUF_MATRIX_CASE(IQ3_S);
        NINFER_GGUF_MATRIX_CASE(IQ1_S);
        NINFER_GGUF_MATRIX_CASE(IQ4_NL);
        NINFER_GGUF_MATRIX_CASE(IQ4_XS);
        NINFER_GGUF_MATRIX_CASE(NVFP4);
    case GgmlType::IQ1_M:
    case GgmlType::MXFP4:
        break;
    }
#undef NINFER_GGUF_MATRIX_CASE
    throw std::invalid_argument("gguf matrix product: no integer kernel for this block type");
}

bool moe_matrix_supported(GgmlType type, int rows, int k, bool tail) {
    if (moe_matrix_sm70()) {
        return has_matrix_kernel(type) && rows > 0 && rows % 32 == 0 && k > 0 &&
               k % block_shape(type).elements == 0;
    }
#define NINFER_GGUF_MOE_FITS_CASE(NAME)                                                            \
    case GgmlType::NAME:                                                                           \
        return detail::moe_matrix_fits_impl<GGML_TYPE_##NAME>(rows, k, tail)
    switch (type) {
        NINFER_GGUF_MOE_FITS_CASE(Q8_0);
        NINFER_GGUF_MOE_FITS_CASE(Q4_0);
        NINFER_GGUF_MOE_FITS_CASE(Q5_0);
        NINFER_GGUF_MOE_FITS_CASE(Q2_0);
        NINFER_GGUF_MOE_FITS_CASE(Q2_K);
        NINFER_GGUF_MOE_FITS_CASE(Q3_K);
        NINFER_GGUF_MOE_FITS_CASE(Q4_K);
        NINFER_GGUF_MOE_FITS_CASE(Q5_K);
        NINFER_GGUF_MOE_FITS_CASE(Q6_K);
        NINFER_GGUF_MOE_FITS_CASE(IQ2_XXS);
        NINFER_GGUF_MOE_FITS_CASE(IQ2_XS);
        NINFER_GGUF_MOE_FITS_CASE(IQ2_S);
        NINFER_GGUF_MOE_FITS_CASE(IQ3_XXS);
        NINFER_GGUF_MOE_FITS_CASE(IQ3_S);
        NINFER_GGUF_MOE_FITS_CASE(IQ1_S);
        NINFER_GGUF_MOE_FITS_CASE(IQ1_M);
        NINFER_GGUF_MOE_FITS_CASE(IQ4_NL);
        NINFER_GGUF_MOE_FITS_CASE(IQ4_XS);
        NINFER_GGUF_MOE_FITS_CASE(NVFP4);
    case GgmlType::MXFP4:
        return false;
    }
#undef NINFER_GGUF_MOE_FITS_CASE
    return false;
}

void moe_matrix_product(GgmlType type, const MoeTable& table, const MoeRouting& routing,
                        int max_active, int pairs, bool tail, const void* activation, float* out,
                        cudaStream_t stream) {
    if (table.experts == nullptr || routing.bounds == nullptr || routing.tiles == nullptr ||
        activation == nullptr || out == nullptr) {
        throw std::invalid_argument("gguf moe matrix product: invalid tables or routing");
    }
    if (moe_matrix_sm70()) {
#define NINFER_GGUF_MOE_SM70_CASE(NAME)                                                            \
    case GgmlType::NAME:                                                                           \
        detail::moe_matrix_sm70_launch<GGML_TYPE_##NAME>(table.experts, table.row_bytes,           \
                                                         table.rows, table.k,                      \
                                                         moe_padded_k(table.k), routing,           \
                                                         max_active, pairs, activation, out,       \
                                                         stream);                                  \
        return
        switch (type) {
            NINFER_GGUF_MOE_SM70_CASE(Q8_0);
            NINFER_GGUF_MOE_SM70_CASE(Q4_0);
            NINFER_GGUF_MOE_SM70_CASE(Q5_0);
            NINFER_GGUF_MOE_SM70_CASE(Q2_0);
            NINFER_GGUF_MOE_SM70_CASE(Q2_K);
            NINFER_GGUF_MOE_SM70_CASE(Q3_K);
            NINFER_GGUF_MOE_SM70_CASE(Q4_K);
            NINFER_GGUF_MOE_SM70_CASE(Q5_K);
            NINFER_GGUF_MOE_SM70_CASE(Q6_K);
            NINFER_GGUF_MOE_SM70_CASE(IQ2_XXS);
            NINFER_GGUF_MOE_SM70_CASE(IQ2_XS);
            NINFER_GGUF_MOE_SM70_CASE(IQ2_S);
            NINFER_GGUF_MOE_SM70_CASE(IQ3_XXS);
            NINFER_GGUF_MOE_SM70_CASE(IQ3_S);
            NINFER_GGUF_MOE_SM70_CASE(IQ1_S);
            NINFER_GGUF_MOE_SM70_CASE(IQ4_NL);
            NINFER_GGUF_MOE_SM70_CASE(IQ4_XS);
            NINFER_GGUF_MOE_SM70_CASE(NVFP4);
        case GgmlType::IQ1_M:
        case GgmlType::MXFP4:
            break;
        }
#undef NINFER_GGUF_MOE_SM70_CASE
        throw std::invalid_argument("gguf moe matrix product: no matrix kernel for this block type");
    }
#define NINFER_GGUF_MOE_MATRIX_CASE(NAME)                                                          \
    case GgmlType::NAME:                                                                           \
        detail::moe_matrix_product_impl<GGML_TYPE_##NAME>(table.experts, table.row_bytes,          \
                                                          table.rows, table.k, routing,            \
                                                          max_active, pairs, tail,                 \
                                                          table.aligned16, activation,             \
                                                          out, stream);                            \
        return
    switch (type) {
        NINFER_GGUF_MOE_MATRIX_CASE(Q8_0);
        NINFER_GGUF_MOE_MATRIX_CASE(Q4_0);
        NINFER_GGUF_MOE_MATRIX_CASE(Q5_0);
        NINFER_GGUF_MOE_MATRIX_CASE(Q2_0);
        NINFER_GGUF_MOE_MATRIX_CASE(Q2_K);
        NINFER_GGUF_MOE_MATRIX_CASE(Q3_K);
        NINFER_GGUF_MOE_MATRIX_CASE(Q4_K);
        NINFER_GGUF_MOE_MATRIX_CASE(Q5_K);
        NINFER_GGUF_MOE_MATRIX_CASE(Q6_K);
        NINFER_GGUF_MOE_MATRIX_CASE(IQ2_XXS);
        NINFER_GGUF_MOE_MATRIX_CASE(IQ2_XS);
        NINFER_GGUF_MOE_MATRIX_CASE(IQ2_S);
        NINFER_GGUF_MOE_MATRIX_CASE(IQ3_XXS);
        NINFER_GGUF_MOE_MATRIX_CASE(IQ3_S);
        NINFER_GGUF_MOE_MATRIX_CASE(IQ1_S);
        NINFER_GGUF_MOE_MATRIX_CASE(IQ1_M);
        NINFER_GGUF_MOE_MATRIX_CASE(IQ4_NL);
        NINFER_GGUF_MOE_MATRIX_CASE(IQ4_XS);
        NINFER_GGUF_MOE_MATRIX_CASE(NVFP4);
    case GgmlType::MXFP4:
        break;
    }
#undef NINFER_GGUF_MOE_MATRIX_CASE
    throw std::invalid_argument("gguf moe matrix product: no integer kernel for this block type");
}

void store_plane(const float* in, std::int64_t in_column_stride, int rows, int columns,
                 const VectorOutput& out, cudaStream_t stream) {
    if ((out.bf16 == nullptr) == (out.f32 == nullptr) ||
        (out.epilogue == Epilogue::GateProduct && out.gate == nullptr) || rows <= 0 ||
        columns <= 0) {
        throw std::invalid_argument("gguf store plane: invalid output");
    }
    const detail::VecStore store{out.bf16,     out.f32,  out.column_stride,
                                 out.epilogue, out.gate, out.gate_column_stride,
                                 static_cast<int>(out.activation)};
    detail::store_plane_kernel<<<dim3((rows + 255) / 256, columns, 1), 256, 0, stream>>>(
        in, in_column_stride, rows, store);
    detail::check(cudaGetLastError(), "store plane launch");
}

void gather_columns(const __nv_bfloat16* x, int k, int columns, const std::int32_t* input_columns,
                    __nv_bfloat16* out, cudaStream_t stream) {
    detail::gather_columns_kernel<<<dim3((k + 255) / 256, columns, 1), 256, 0, stream>>>(
        x, k, input_columns, out);
    detail::check(cudaGetLastError(), "gather columns launch");
}

int dequantized_rows_per_pass(int k, std::size_t scratch_bytes) {
    const std::size_t row = std::size_t(k) * sizeof(__nv_bfloat16);
    return static_cast<int>(std::min<std::size_t>(scratch_bytes / row, 1U << 20) / 128 * 128);
}

void dequantized_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows, int k,
                         const __nv_bfloat16* x, int columns, float* out,
                         std::int64_t out_column_stride, void* scratch, std::size_t scratch_bytes,
                         cudaStream_t stream) {
    const int pass = dequantized_rows_per_pass(k, scratch_bytes);
    if (pass <= 0) { throw std::invalid_argument("gguf dequantized product: scratch too small"); }
    cublasHandle_t blas = detail::blas_handle();
    if (cublasSetStream(blas, stream) != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("gguf: cublasSetStream failed");
    }
    auto* rows_bf16    = static_cast<__nv_bfloat16*>(scratch);
    const float alpha  = 1.0f;
    const float beta   = 0.0f;
    for (int first = 0; first < rows; first += pass) {
        const int count = std::min(pass, rows - first);
        dequantize_rows(type, static_cast<const char*>(weight) + first * row_bytes, row_bytes, k,
                        nullptr, count, rows_bf16, k, stream);
        const cublasStatus_t status = cublasGemmEx(
            blas, CUBLAS_OP_T, CUBLAS_OP_N, count, columns, k, &alpha, rows_bf16, CUDA_R_16BF, k,
            x, CUDA_R_16BF, k, &beta, out + first, CUDA_R_32F,
            static_cast<int>(out_column_stride), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("gguf dequantized product: cublasGemmEx failed");
        }
    }
}

void dequantize_rows(GgmlType type, const void* weight, std::int64_t row_bytes, int k,
                     const std::int32_t* row_ids, int rows, __nv_bfloat16* out,
                     std::int64_t out_row_stride, cudaStream_t stream) {
    detail::dispatch_dequantize(type, weight, row_bytes, k, row_ids, rows, out, out_row_stride,
                                stream);
}

void dequantize_rows(GgmlType type, const void* weight, std::int64_t row_bytes, int k,
                     const std::int32_t* row_ids, int rows, float* out,
                     std::int64_t out_row_stride, cudaStream_t stream) {
    detail::dispatch_dequantize(type, weight, row_bytes, k, row_ids, rows, out, out_row_stride,
                                stream);
}

void dequantize_rows(GgmlType type, const void* weight, std::int64_t row_bytes, int k,
                     const std::int32_t* row_ids, int rows, __half* out,
                     std::int64_t out_row_stride, cudaStream_t stream) {
    detail::dispatch_dequantize(type, weight, row_bytes, k, row_ids, rows, out, out_row_stride,
                                stream);
}

} // namespace ninfer::ops::gguf
