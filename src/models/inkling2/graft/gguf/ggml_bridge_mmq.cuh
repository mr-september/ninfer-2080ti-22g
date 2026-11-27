#pragma once

// Host side of ggml's integer tensor-core matrix kernel (mmq.cuh launch_mul_mat_q), without ggml's
// backend context: the caller owns the activation, the FP32 output and the stream-k fixup plane.

#include "ggml_bridge_mmq_launch.cuh"
#include "mmq_limits.h"
#include "../../common/device_route_lookup.h"

#include <climits>
#include <cstdint>
#include <mutex>
#include <type_traits>

namespace ninfer::ops::gguf::detail {

struct MatrixLaunch {
    int J                   = 0;
    int I                   = 0;
    int threads             = 0;
    std::size_t shared      = 0;
    int row_tiles           = 0;
    int column_tiles        = 0;
    int k_step              = 0;
    bool stream_k           = false;
    int blocks              = 0;
    bool fixup              = false;
    int async               = 0; // the staging copy's width in bytes; 0 for the compiled loop
};

// The device-route keys of the asynchronous tile loop (mmq.cuh ggml_cuda_mmq_async_row_bytes): a
// format's dense and MoE products take it where the profile's band says `async`.
template <ggml_type type>
constexpr const char* async_matrix_key() {
    switch (type) {
    case GGML_TYPE_Q2_0: return "gguf_matrix/q2_0";
    case GGML_TYPE_Q4_0: return "gguf_matrix/q4_0";
    case GGML_TYPE_Q5_0: return "gguf_matrix/q5_0";
    case GGML_TYPE_Q8_0: return "gguf_matrix/q8_0";
    case GGML_TYPE_IQ1_M: return "gguf_matrix/iq1_m";
    case GGML_TYPE_IQ4_XS: return "gguf_matrix/iq4_xs";
    case GGML_TYPE_IQ4_NL: return "gguf_matrix/iq4_nl";
    case GGML_TYPE_Q4_K: return "gguf_matrix/q4_k";
    case GGML_TYPE_Q5_K: return "gguf_matrix/q5_k";
    default: return nullptr;
    }
}

template <ggml_type type>
constexpr const char* async_moe_matrix_key() {
    switch (type) {
    case GGML_TYPE_Q2_0: return "gguf_moe_matrix/q2_0";
    case GGML_TYPE_Q4_0: return "gguf_moe_matrix/q4_0";
    case GGML_TYPE_Q5_0: return "gguf_moe_matrix/q5_0";
    case GGML_TYPE_Q8_0: return "gguf_moe_matrix/q8_0";
    case GGML_TYPE_IQ1_M: return "gguf_moe_matrix/iq1_m";
    case GGML_TYPE_IQ4_XS: return "gguf_moe_matrix/iq4_xs";
    case GGML_TYPE_IQ4_NL: return "gguf_moe_matrix/iq4_nl";
    case GGML_TYPE_Q4_K: return "gguf_moe_matrix/q4_k";
    case GGML_TYPE_Q5_K: return "gguf_moe_matrix/q5_k";
    default: return nullptr;
    }
}

template <ggml_type type>
constexpr bool async_matrix_built() {
    return async_matrix_key<type>() != nullptr && ggml_cuda_mmq_async_tiles_supported(type, 128, false);
}

// The widest staging copy rows of `row_bytes` from `base` allow: the format's word (8 or 16 bytes),
// else 4 bytes (rows whose value count is not a whole number of 256-value steps); 0 for none.
template <ggml_type type>
constexpr int async_copy_width(std::int64_t row_bytes, std::uintptr_t base) {
    constexpr int word = ggml_cuda_mmq_async_word(type);
    if (row_bytes % word == 0 && base % word == 0) { return word; }
    return row_bytes % 4 == 0 && base % 4 == 0 ? 4 : 0;
}

template <ggml_type type>
MatrixLaunch select_matrix_launch(int rows, int columns,
                                  int copy = ggml_cuda_mmq_async_word(type)) {
    if (rows <= 0 || columns <= 0) {
        throw std::invalid_argument("gguf matrix product: nonpositive geometry");
    }
    const DeviceFacts& device = device_facts();
    MatrixLaunch best;
    int best_tiles = INT_MAX;
    for (const int J : kMatrixColumnTiles) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, false, device.cc);
        if (config.type == GGML_TYPE_COUNT) { continue; }
        const std::size_t shared = mmq_get_nbytes_shared(config, device.cc);
        if (shared > device.shared_per_block_optin) { continue; }
        const int tiles = 1 + (columns - 1) / J;
        if (tiles < best_tiles) {
            best_tiles        = tiles;
            best.J            = J;
            best.I            = config.I;
            best.threads      = config.nthreads;
            best.shared       = shared;
            best.stream_k     = config.stream_k;
            best.k_step       = config.K_vram / ggml_cuda_type_traits<type>::qk;
        }
        if (tiles == 1) { break; }
    }
    if (best.J == 0) { throw std::invalid_argument("gguf matrix product: no tile fits this device"); }
    if constexpr (async_matrix_built<type>()) {
        if (ggml_cuda_mmq_async_tiles_supported(type, best.J, false) && copy != 0 &&
            rows % best.I == 0 &&
            ggml_cuda_mmq_async_tiles_available(device.cc) &&
            device_route_schedule(async_matrix_key<type>(), columns) == "async") {
            const std::size_t async_shared = mmq_get_nbytes_shared(
                ggml_cuda_mmq_get_config(type, best.J, false, device.cc), device.cc, true);
            if (async_shared <= device.shared_per_block_optin) {
                best.async  = copy;
                best.shared = async_shared;
            }
        }
    }
    best.row_tiles    = 1 + (rows - 1) / best.I;
    best.column_tiles = 1 + (columns - 1) / best.J;
    const int tiles   = matrix_work_blocks(best.row_tiles, best.column_tiles, 1);
    if (best.stream_k) {
        const int waves      = 1 + (tiles - 1) / device.sm_count;
        const int efficiency = int(100LL * tiles / (std::int64_t(device.sm_count) * waves));
        best.blocks          = efficiency >= 90 ? tiles : device.sm_count;
        best.fixup           = tiles % best.blocks != 0;
    } else {
        best.blocks = tiles;
    }
    return best;
}

template <ggml_type type>
std::size_t matrix_fixup_bytes_impl(int rows, int columns) {
    const MatrixLaunch launch = select_matrix_launch<type>(rows, columns);
    return launch.fixup ? std::size_t(launch.blocks) * launch.J * launch.I * sizeof(float) : 0;
}

template <ggml_type type, int J, typename Dst, bool accumulate, int async = 0>
void launch_matrix(const MatrixLaunch& launch, const char* weight, int stride_row_blocks, int rows,
                   int k, const int* activation, int columns, Dst* out,
                   std::int64_t out_column_stride, float* fixup, cudaStream_t stream, float* partial) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    const std::int64_t work = matrix_work_blocks(launch.row_tiles, launch.column_tiles, k / qk);
    static std::once_flag raised[16];
    int device = 0;
    check(cudaGetDevice(&device), "device");
    std::call_once(raised[device & 15], [&] {
        check(cudaFuncSetAttribute(mul_mat_q<type, J, false, GGML_PREC_Q8, Dst, accumulate, async>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(launch.shared)),
              "matrix shared memory");
        if constexpr (!std::is_same_v<Dst, float>) {
            check(cudaFuncSetAttribute(mul_mat_q<type, J, false, GGML_PREC_Q8, float, false, async>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(launch.shared)),
                  "matrix partial shared memory");
        }
    });
    const uint3 blocks_fd   = init_fastdiv_values(k / qk);
    const uint3 one_fd      = init_fastdiv_values(1);
    const uint3 ntx_fd      = init_fastdiv_values(launch.column_tiles);
    const dim3 block_dims(WARP_SIZE, launch.threads / WARP_SIZE, 1);
    const int stride        = static_cast<int>(out_column_stride);
    if (!launch.stream_k) {
        mul_mat_q<type, J, false, GGML_PREC_Q8, Dst, accumulate, async>
            <<<dim3(launch.row_tiles, launch.column_tiles, 1), block_dims, launch.shared, stream>>>(
                weight, activation, nullptr, nullptr, out, nullptr, nullptr, blocks_fd, rows,
                columns, stride_row_blocks, columns, stride, one_fd, one_fd, 0, 0, 0, one_fd,
                one_fd, 0, 0, 0, ntx_fd, partial, rows);
        check(cudaGetLastError(), "matrix product launch");
        return;
    }
    if (launch.fixup && (fixup == nullptr ||
                         (!std::is_same_v<Dst, float> && partial == nullptr))) {
        throw std::invalid_argument("gguf matrix product: missing FP32 stream-k storage");
    }
    bool partials_only = false;
    if constexpr (!std::is_same_v<Dst, float>) {
        const std::int64_t segment = (work + launch.blocks - 1) / launch.blocks + launch.k_step - 1;
        partials_only = launch.fixup && segment < k / qk;
    }
    if (partials_only) {
        mul_mat_q<type, J, false, GGML_PREC_Q8, float, false, async>
            <<<dim3(launch.blocks, 1, 1), block_dims, launch.shared, stream>>>(
            weight, activation, nullptr, nullptr, partial, fixup, nullptr,
            blocks_fd, rows, columns, stride_row_blocks, columns, rows, one_fd, one_fd, 0, 0, 0,
            one_fd, one_fd, 0, 0, 0, ntx_fd);
    } else {
        mul_mat_q<type, J, false, GGML_PREC_Q8, Dst, accumulate, async>
            <<<dim3(launch.blocks, 1, 1), block_dims, launch.shared, stream>>>(
            weight, activation, nullptr, nullptr, out, launch.fixup ? fixup : nullptr, nullptr,
            blocks_fd, rows, columns, stride_row_blocks, columns, stride, one_fd, one_fd, 0, 0, 0,
            one_fd, one_fd, 0, 0, 0, ntx_fd, partial, rows);
    }
    check(cudaGetLastError(), "matrix product launch");
    if (!launch.fixup) { return; }
    const dim3 fixup_blocks(launch.blocks, launch.I / WARP_SIZE, 1);
    const dim3 fixup_dims(block_dims.x, block_dims.y / 2, 1);
    mul_mat_q_stream_k_fixup<type, J, false, GGML_PREC_Q8, Dst, accumulate>
        <<<fixup_blocks, fixup_dims, 0, stream>>>(
        nullptr, nullptr, out, fixup, blocks_fd, rows, columns, stride, one_fd, 0, one_fd, 0,
        ntx_fd, partial, rows);
    check(cudaGetLastError(), "matrix fixup launch");
}

template <ggml_type type, typename Dst, bool accumulate>
void matrix_product_impl(const void* weight, std::int64_t row_bytes, int rows, int k,
                         const void* activation, int columns, Dst* out,
                         std::int64_t out_column_stride, void* fixup, cudaStream_t stream,
                         bool tail, float* partial) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int block_bytes = ggml_cuda_type_traits<type>::bs;
    // A row that is not a whole number of 256-value steps is read past its end (into the next row,
    // and past the last one into the caller's zeros); its activation is zero there.
    const bool steps = k % 256 == 0 || (tail && k % qk == 0 && k % 32 == 0);
    if (k <= 0 || !steps || rows % 128 != 0 || row_bytes % block_bytes != 0 ||
        row_bytes < std::int64_t(k / qk) * block_bytes) {
        throw std::invalid_argument("gguf matrix product: unsupported geometry");
    }
    // The asynchronous loop copies whole words (4, 8 or 16 bytes) of each row's 256-value steps.
    const MatrixLaunch launch = select_matrix_launch<type>(
        rows, columns, async_copy_width<type>(row_bytes, reinterpret_cast<std::uintptr_t>(weight)));
    const auto* x             = static_cast<const char*>(weight);
    const auto* y             = static_cast<const int*>(activation);
    auto* f                   = static_cast<float*>(fixup);
    const int stride_row      = static_cast<int>(row_bytes / block_bytes);
    const auto run = [&](auto tile) {
        constexpr int J = decltype(tile)::value;
        if constexpr (async_matrix_built<type>()) {
            constexpr int word = ggml_cuda_mmq_async_word(type);
            if (launch.async == word) {
                launch_matrix<type, J, Dst, accumulate, word>(launch, x, stride_row, rows, k, y,
                                                              columns, out, out_column_stride, f,
                                                              stream, partial);
                return;
            }
            if (launch.async == 4) {
                launch_matrix<type, J, Dst, accumulate, 4>(launch, x, stride_row, rows, k, y,
                                                           columns, out, out_column_stride, f,
                                                           stream, partial);
                return;
            }
        }
        launch_matrix<type, J, Dst, accumulate>(launch, x, stride_row, rows, k, y, columns, out,
                                                out_column_stride, f, stream, partial);
    };
    switch (launch.J) {
    case 16: return run(std::integral_constant<int, 16>{});
    case 32: return run(std::integral_constant<int, 32>{});
    case 64: return run(std::integral_constant<int, 64>{});
    case 128: return run(std::integral_constant<int, 128>{});
    default:
        break;
    }
    throw std::invalid_argument("gguf matrix product: unexpected column tile");
}

// ggml's tile loop over the experts a router selected: one CTA per row tile and listed (expert,
// column tile), the expert's rows reached through a table of base pointers (a bank, a cache slot
// or mapped host memory) instead of a fixed stride. Column c of the activation is the routing's
// c-th pair, and pair p's result row lands at out[p * rows]. The grid's column dimension bounds
// the list; the CTAs past its length leave at once.
template <ggml_type type, int J, int async = 0>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, false),
                  ggml_cuda_mmq_get_occupancy(type, J, false)) static __global__
    void moe_mul_mat_q(const void* const* __restrict__ experts, const int* __restrict__ y,
                       const std::int32_t* __restrict__ sorted,
                       const std::int32_t* __restrict__ bounds,
                       const std::int32_t* __restrict__ tiles,
                       const std::int32_t* __restrict__ tile_count, float* __restrict__ dst,
                       const uint3 blocks_per_row, int rows, int stride_row_x, int ncols_y) {
    if (ggml_cuda_mmq_get_config(type, J, false).type == GGML_TYPE_COUNT) {
        NO_DEVICE_CODE;
        return;
    }
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, false) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, false);
    if (int(blockIdx.y) >= *tile_count) { return; }
    const int tile     = tiles[blockIdx.y];
    const int expert   = tile & 0xFFFF;
    const int jt       = tile >> 16;
    const int col_low  = bounds[expert];
    const int col_diff = bounds[expert + 1] - col_low;
    const int it       = blockIdx.x;
    // The tile's destination pairs head the dynamic shared memory, where the tile loop reads them.
    extern __shared__ int ids_dst_shared[];
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps * warp_size) {
        const int j = j0 + threadIdx.y * warp_size + threadIdx.x;
        if (j0 + nwarps * warp_size > J && j >= J) { break; }
        ids_dst_shared[j] = jt * J + j < col_diff ? sorted[col_low + jt * J + j] : 0;
    }
    __syncthreads();
    const int offset_y     = (col_low + jt * J) * int(sizeof(block_q8_1_mmq) / sizeof(int));
    const int tile_x_max_i = rows - it * I - 1;
    const int tile_y_max_j = col_diff - jt * J - 1;
    mul_mat_q_process_tile<type, J, false, false, GGML_PREC_Q8, float, false, async>(
        static_cast<const char*>(experts[expert]), it * I * stride_row_x, y + offset_y,
        ids_dst_shared, dst + it * I, nullptr, nullptr, stride_row_x, ncols_y, rows, tile_x_max_i,
        tile_y_max_j, 0, blocks_per_row.z);
}

// The column tile for products whose experts take `columns` pairs on average: the narrowest tile
// that holds them, or the widest available. Zero when no tile of this type fits the geometry (the
// rows a whole number of row tiles; k a whole number of the kernel's K steps, or of blocks over a
// 256-value step when zeros follow the bank) or the device.
template <ggml_type type>
MatrixLaunch select_moe_launch(int rows, int k, int columns, bool tail, int copy = 0) {
    const DeviceFacts& device = device_facts();
    constexpr int qk          = ggml_cuda_type_traits<type>::qk;
    MatrixLaunch best;
    for (const int J : kMatrixColumnTiles) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, false, device.cc);
        const bool steps = k % config.K_vram == 0 ||
                           (tail && config.K_vram == MMQ_ITER_K && k % qk == 0 && k % 32 == 0);
        if (config.type == GGML_TYPE_COUNT || rows % config.I != 0 || !steps) { continue; }
        const std::size_t shared = mmq_get_nbytes_shared(config, device.cc);
        if (shared > device.shared_per_block_optin) { continue; }
        best.J       = J;
        best.I       = config.I;
        best.threads = config.nthreads;
        best.shared  = shared;
        if (J >= columns) { break; }
    }
    if constexpr (async_matrix_built<type>()) {
        if (ggml_cuda_mmq_async_tiles_supported(type, best.J, false) && copy != 0 &&
            ggml_cuda_mmq_async_tiles_available(device.cc) &&
            device_route_schedule(async_moe_matrix_key<type>(), columns) == "async") {
            const std::size_t async_shared = mmq_get_nbytes_shared(
                ggml_cuda_mmq_get_config(type, best.J, false, device.cc), device.cc, true);
            if (async_shared <= device.shared_per_block_optin) {
                best.async  = copy;
                best.shared = async_shared;
            }
        }
    }
    return best;
}

template <ggml_type type>
bool moe_matrix_fits_impl(int rows, int k, bool tail) {
    return select_moe_launch<type>(rows, k, 1, tail).J != 0;
}

template <ggml_type type, int J, int async = 0>
void launch_moe_matrix(const MatrixLaunch& launch, const void* const* experts, int stride_row,
                       int rows, int k, const MoeRouting& routing, int max_active, int pairs,
                       const int* activation, float* out, cudaStream_t stream) {
    static std::once_flag raised[16];
    int device = 0;
    check(cudaGetDevice(&device), "device");
    std::call_once(raised[device & 15], [&] {
        check(cudaFuncSetAttribute(moe_mul_mat_q<type, J, async>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(launch.shared)),
              "moe matrix shared memory");
    });
    constexpr int qk    = ggml_cuda_type_traits<type>::qk;
    constexpr int width = J == 16 ? 0 : J == 32 ? 1 : J == 64 ? 2 : 3;
    static_assert(kMatrixColumnTiles[width] == J, "MoeRouting lists tiles of width 16 << w");
    // Each active expert's tiles hold its pairs, the last one partly: no more than pairs / J
    // whole tiles and one partial tile per active expert.
    const int listed = (pairs + J - 1) / J + max_active;
    if (listed > routing.tile_capacity || listed > 65535) {
        throw std::invalid_argument("gguf moe matrix product: the routing lists fewer tiles");
    }
    const dim3 grid(rows / launch.I, listed, 1);
    const dim3 block(WARP_SIZE, launch.threads / WARP_SIZE, 1);
    moe_mul_mat_q<type, J, async><<<grid, block, launch.shared, stream>>>(
        experts, activation, routing.sorted, routing.bounds,
        routing.tiles + std::int64_t(width) * routing.tile_capacity, routing.tile_count + width,
        out, init_fastdiv_values(k / qk), rows, stride_row, pairs);
    check(cudaGetLastError(), "moe matrix product launch");
}

template <ggml_type type>
void moe_matrix_product_impl(const void* const* experts, std::int64_t row_bytes, int rows, int k,
                             const MoeRouting& routing, int max_active, int pairs, bool tail,
                             bool aligned, const void* activation, float* out,
                             cudaStream_t stream) {
    constexpr int block_bytes = ggml_cuda_type_traits<type>::bs;
    constexpr int qk          = ggml_cuda_type_traits<type>::qk;
    if (max_active <= 0 || pairs <= 0 || row_bytes % block_bytes != 0 ||
        row_bytes < std::int64_t(k / qk) * block_bytes) {
        throw std::invalid_argument("gguf moe matrix product: invalid geometry");
    }
    // `aligned`: every expert's base pointer is 16-byte aligned; the row stride sets the copy.
    const MatrixLaunch launch = select_moe_launch<type>(
        rows, k, (pairs + max_active - 1) / max_active, tail,
        aligned ? async_copy_width<type>(row_bytes, 0) : 0);
    if (launch.J == 0) {
        throw std::invalid_argument("gguf moe matrix product: no tile fits this geometry");
    }
    const auto* y        = static_cast<const int*>(activation);
    const int stride_row = static_cast<int>(row_bytes / block_bytes);
    const auto run = [&](auto tile) {
        constexpr int J = decltype(tile)::value;
        if constexpr (async_matrix_built<type>()) {
            constexpr int word = ggml_cuda_mmq_async_word(type);
            if (launch.async == word) {
                return launch_moe_matrix<type, J, word>(launch, experts, stride_row, rows, k,
                                                        routing, max_active, pairs, y, out, stream);
            }
            if (launch.async == 4) {
                return launch_moe_matrix<type, J, 4>(launch, experts, stride_row, rows, k,
                                                     routing, max_active, pairs, y, out, stream);
            }
        }
        launch_moe_matrix<type, J>(launch, experts, stride_row, rows, k, routing, max_active,
                                   pairs, y, out, stream);
    };
    switch (launch.J) {
    case 16: return run(std::integral_constant<int, 16>{});
    case 32: return run(std::integral_constant<int, 32>{});
    case 64: return run(std::integral_constant<int, 64>{});
    case 128: return run(std::integral_constant<int, 128>{});
    default:
        break;
    }
    throw std::invalid_argument("gguf moe matrix product: unexpected column tile");
}

} // namespace ninfer::ops::gguf::detail

// The routed experts' matrix product alone, for a type whose dense products take a BF16 GEMM.
#define NINFER_GGUF_MOE_MATRIX_INSTANCE(TYPE)                                                      \
    template bool ninfer::ops::gguf::detail::moe_matrix_fits_impl<TYPE>(int, int, bool);           \
    template void ninfer::ops::gguf::detail::moe_matrix_product_impl<TYPE>(                        \
        const void* const*, std::int64_t, int, int, const ninfer::ops::gguf::MoeRouting&, int,     \
        int, bool, bool, const void*, float*, cudaStream_t)

#define NINFER_GGUF_MATRIX_INSTANCE(TYPE)                                                          \
    template std::size_t ninfer::ops::gguf::detail::matrix_fixup_bytes_impl<TYPE>(int, int);       \
    template void ninfer::ops::gguf::detail::matrix_product_impl<TYPE>(                            \
        const void*, std::int64_t, int, int, const void*, int, float*, std::int64_t, void*,        \
        cudaStream_t, bool, float*);                                                              \
    template void ninfer::ops::gguf::detail::matrix_product_impl<TYPE, __nv_bfloat16, false>(    \
        const void*, std::int64_t, int, int, const void*, int, __nv_bfloat16*, std::int64_t, void*, \
        cudaStream_t, bool, float*);                                                              \
    template void ninfer::ops::gguf::detail::matrix_product_impl<TYPE, __nv_bfloat16, true>(     \
        const void*, std::int64_t, int, int, const void*, int, __nv_bfloat16*, std::int64_t, void*, \
        cudaStream_t, bool, float*);                                                              \
    NINFER_GGUF_MOE_MATRIX_INSTANCE(TYPE)
