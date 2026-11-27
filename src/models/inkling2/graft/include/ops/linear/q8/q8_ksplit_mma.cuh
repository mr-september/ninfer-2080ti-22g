#pragma once

// Q8G32 RowSplit K-split MMA contraction.
//
// Geometry and scheduling are compile-time values; columns may be exact or capacity-bounded.
// K-split warps
// cooperatively own one 16-row output tile; each warp evaluates a disjoint 64-wide K slice, then
// the CTA reduces FP32 partials in shared memory. Output owns physical row/token addressing; an
// optional caller epilogue may instead consume the FP32 tile.

#include "ops/common/mma.cuh"
#include "core/pdl.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/q8/q8_ksplit_config.h"
#include "ops/linear/q8/q8_rowsplit_output.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

struct Q8KSplitStoreEpilogue {};

struct Q8KSplitResidualEpilogue {};

struct Q8KSplitIdentityRows {
    static constexpr int kOutputRowsPerCta = 16;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + local_row;
    }
};

__device__ __forceinline__ int q8_ksplit_swizzle_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

union Q8KSplitBf16PairBits {
    __nv_bfloat162 pair;
    unsigned bits;
};

__device__ __forceinline__ unsigned q8_ksplit_bf16_pair_from_s8(unsigned values) {
    Q8KSplitBf16PairBits biased;
    biased.bits          = __byte_perm(values, 0x43004300u, 0x7150) & 0xff7fff7fu;
    const unsigned signs = (values & 0x80u) | ((values & 0x8000u) << 8);
    Q8KSplitBf16PairBits bias;
    bias.bits = 0x43004300u | signs;
    Q8KSplitBf16PairBits result;
    result.pair = __hsub2_rn(biased.pair, bias.pair);
    return result.bits;
}

#if defined(NINFER_SM75_FORMS)
// Turing has neither BF16 MMA nor BF16 arithmetic: the pair above runs __hsub2_rn in software and
// mma_bf16 converts all six operands on every call. The codes are integers, exact in FP16 as in
// BF16, so a Turing build forms the FP16 pair directly: code + 128 in the mantissa of 1024.0, then
// 1024 + 128 subtracted, natively and exactly.
__device__ __forceinline__ unsigned q8_ksplit_f16_pair_from_s8(unsigned values) {
    const unsigned biased = __byte_perm(values ^ 0x8080u, 0x64646464u, 0x5140);
    const __half2 code    = __hsub2(*reinterpret_cast<const __half2*>(&biased),
                                    __half2half2(__ushort_as_half(0x6480u)));
    return *reinterpret_cast<const unsigned*>(&code);
}
#endif

// The contraction owns the shared layout; tiled launchers use the same type for opt-in capacity.
template <class Schedule>
union alignas(16) Q8KSplitSharedStorage {
    struct {
        std::uint8_t codes[Schedule::kRowsPerCta][Schedule::kGroupK];
        __nv_bfloat16 activations[Schedule::kKWarps]
                                 [Schedule::kTileTokens * Schedule::kTileKPerWarp];
        std::uint8_t scales[Schedule::kRowsPerCta]
                           [Schedule::kScaleAccess == Q8KSplitScaleAccess::Shared
                                ? Schedule::kScaleBytesPerRow
                                : 1];
    } staging;

    float partial[Schedule::kKWarps * (Schedule::kTileTokens / 8) * 32 * 4];
};

struct Q8KSplitIdentityColumns {
    __device__ __forceinline__ int operator()(int column) const { return column; }
};

template <class Geometry, int ActiveCols, class Schedule, class Output,
          class Epilogue = Q8KSplitStoreEpilogue, class RowPolicy = Q8KSplitIdentityRows,
          bool DirectPairEpilogue = false, bool TiledColumns = false,
          class ColumnPolicy = Q8KSplitIdentityColumns>
__device__ __forceinline__ void
q8_ksplit_mma(const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
              const std::uint8_t* __restrict__ scales, Output output, Epilogue epilogue = {},
              RowPolicy row_policy = {}, std::int32_t columns = ActiveCols,
              ColumnPolicy column_policy = {}) {
    const int column_offset = TiledColumns ? static_cast<int>(blockIdx.y) * ActiveCols : 0;
    const int live_columns  = TiledColumns ? min(ActiveCols, columns - column_offset) : ActiveCols;
    constexpr int kHidden   = Geometry::kInputRows;
    constexpr int kTileK    = Schedule::kTileKPerWarp;
    constexpr int kWarps    = Schedule::kKWarps;
    constexpr int kMmaRows  = Schedule::kRowsPerCta;
    constexpr int kRowsPerCta = Schedule::kRowsPerCta;
    constexpr int kGroupK     = Schedule::kGroupK;
    constexpr int kGroups     = kHidden / kGroupK;
    constexpr int kTileCols   = Schedule::kTileTokens;
    constexpr bool kRuntimeActive =
        Schedule::kActivationStage == Q8KSplitActivationStage::RuntimeActive;
    static_assert((kHidden % kGroupK) == 0);
    static_assert(ActiveCols >= 1 && ActiveCols <= kTileCols);
    static_assert(RowPolicy::kOutputRowsPerCta <= kRowsPerCta);
    static_assert(!kRuntimeActive || TiledColumns,
                  "runtime-active staging requires a runtime column extent");
    constexpr int kNt        = kTileCols / 8;
    constexpr unsigned kMask = 0xffffffffu;

    using SharedStorage = Q8KSplitSharedStorage<Schedule>;

    constexpr bool kDynamicShared = TiledColumns && ActiveCols > 64;
    __shared__ __align__(
        16) unsigned char static_shared[kDynamicShared ? 1 : sizeof(SharedStorage)];
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    auto& shared =
        *reinterpret_cast<SharedStorage*>(kDynamicShared ? dynamic_shared : static_shared);
    auto& code_shared  = shared.staging.codes;
    auto& b_shared     = shared.staging.activations;
    auto& scale_shared = shared.staging.scales;

    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int gid     = lane >> 2;
    const int lid     = lane & 3;
    const int k_split = warp;

    const int cta_row0 = static_cast<int>(blockIdx.x) * RowPolicy::kOutputRowsPerCta;

    const auto stage_x = [&](int group_k0) {
        constexpr bool kPaddedStage =
            Schedule::kActivationStage == Q8KSplitActivationStage::PaddedZero;
        constexpr int kStageCols  = kPaddedStage ? kTileCols : ActiveCols;
        const int stage_columns   = kRuntimeActive ? live_columns : kStageCols;
        const int items_per_split = stage_columns * (kTileK / 8);
        for (int item = lane; item < items_per_split; item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            auto* dst     = &b_shared[warp][col * kTileK + q8_ksplit_swizzle_64(col, k8 * 8)];
            if constexpr (TiledColumns && !kRuntimeActive) {
                const int source_col = col < live_columns ? col : 0;
                cp_async_zfill<16, Schedule::kActivationCache>(
                    dst,
                    &x[static_cast<std::int64_t>(column_policy(column_offset + source_col)) *
                           kHidden +
                       group_k0 + warp * kTileK + k8 * 8],
                    col < live_columns ? 16 : 0);
            } else if constexpr (kRuntimeActive || !kPaddedStage || ActiveCols == kTileCols) {
                cp_async<16, Schedule::kActivationCache>(
                    dst,
                    &x[static_cast<std::int64_t>(column_policy(column_offset + col)) * kHidden +
                       group_k0 + warp * kTileK + k8 * 8]);
            } else {
                const int source_col = col < ActiveCols ? col : 0;
                cp_async_zfill<16, Schedule::kActivationCache>(
                    dst,
                    &x[static_cast<std::int64_t>(column_policy(column_offset + source_col)) *
                           kHidden +
                       group_k0 + warp * kTileK + k8 * 8],
                    col < ActiveCols ? 16 : 0);
            }
        }
    };

    const auto stage_codes = [&](int group_k0) {
#pragma unroll
        for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
            const int row        = warp * Schedule::kRowsPerLoaderWarp + row_item;
            const int weight_row = row_policy.weight_row(cta_row0, row);
            for (int chunk = lane; chunk < kGroupK / 16; chunk += 32) {
                const int swizzled_chunk = chunk ^ (row & 7);
                cp_async<16, Schedule::kWeightCache>(
                    &code_shared[row][swizzled_chunk * 16],
                    codes + static_cast<std::int64_t>(weight_row) * kHidden + group_k0 +
                        chunk * 16);
            }
        }
        if constexpr (Schedule::kScaleAccess == Q8KSplitScaleAccess::Shared) {
            constexpr int kScaleChunksPerRow = Schedule::kScaleBytesPerRow / 16;
            for (int item = tid; item < kMmaRows * kScaleChunksPerRow; item += kWarps * 32) {
                const int row        = item / kScaleChunksPerRow;
                const int chunk      = item - row * kScaleChunksPerRow;
                const int weight_row = row_policy.weight_row(cta_row0, row);
                cp_async<16, Schedule::kWeightCache>(
                    &scale_shared[row][chunk * 16],
                    scales + (static_cast<std::int64_t>(weight_row) * Geometry::kGroupsPerRow +
                              group_k0 / 32 + chunk * 8) *
                                 2);
            }
        }
    };

    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_koff = k_split * kTileK;
    float acc[kNt][4];
#pragma unroll
    for (int ni = 0; ni < kNt; ++ni) {
        acc[ni][0] = 0.0f;
        acc[ni][1] = 0.0f;
        acc[ni][2] = 0.0f;
        acc[ni][3] = 0.0f;
    }

    stage_codes(0);
    pdl::enter_streaming();
    stage_x(0);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

#if defined(NINFER_SM75_DEVICE)
    // Turing copies synchronously (ops/common/memory.cuh), so a stage issued between this group's
    // barriers would wait out its whole load latency there. The next group's codes and scales, and
    // its activations when they take at most two 16-byte loads a lane, are loaded into registers
    // before this group's MMAs and stored after the barrier that ends its reads. A wider
    // activation tile is staged at that barrier as before: holding it in registers costs more
    // occupancy than the latency it hides.
    constexpr bool kSharedScales = Schedule::kScaleAccess == Q8KSplitScaleAccess::Shared;
    constexpr bool kPaddedStage = Schedule::kActivationStage == Q8KSplitActivationStage::PaddedZero;
    constexpr int kCodeChunks   = kGroupK / 16;
    constexpr int kCodeLoads    = (kCodeChunks + 31) / 32;
    constexpr int kScaleChunksPerRow = Schedule::kScaleBytesPerRow / 16;
    constexpr int kScaleChunks       = kMmaRows * kScaleChunksPerRow;
    constexpr int kScaleLoads =
        kSharedScales ? (kScaleChunks + kWarps * 32 - 1) / (kWarps * 32) : 1;
    constexpr int kXLoadsMax  = ((kPaddedStage ? kTileCols : ActiveCols) * (kTileK / 8) + 31) / 32;
    constexpr bool kPrefetchX = kXLoadsMax <= 2;
    constexpr int kXLoads     = kPrefetchX ? kXLoadsMax : 1;
    uint4 next_codes[Schedule::kRowsPerLoaderWarp][kCodeLoads];
    uint4 next_scales[kScaleLoads];
    uint4 next_x[kXLoads];
    unsigned direct_scale_pair      = 0;
    unsigned next_direct_scale_pair = 0;

    const auto load_direct_scale_pair = [&](int group_k0) {
        const int scale_row = row_policy.weight_row(cta_row0, gid + lid * 8);
        return *reinterpret_cast<const unsigned*>(
            scales + (static_cast<std::int64_t>(scale_row) * Geometry::kGroupsPerRow +
                      group_k0 / 32 + warp_koff / 32) *
                         2);
    };
    // Activation item `item` of this warp's slice as stage_x copies it: its column and 8-element
    // K offset, and whether that column is read from x (zero-filled otherwise).
    const int x_items =
        (kRuntimeActive ? live_columns : (kPaddedStage ? kTileCols : ActiveCols)) * (kTileK / 8);
    const auto x_item = [&](int item, int& col, int& k8) {
        col = item / (kTileK / 8);
        k8  = item - col * (kTileK / 8);
        if constexpr (TiledColumns && !kRuntimeActive) {
            return col < live_columns;
        } else if constexpr (kRuntimeActive || !kPaddedStage || ActiveCols == kTileCols) {
            return true;
        } else {
            return col < ActiveCols;
        }
    };
    const auto load_next = [&](int group_k0) {
#    pragma unroll
        for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
            const int weight_row =
                row_policy.weight_row(cta_row0, warp * Schedule::kRowsPerLoaderWarp + row_item);
#    pragma unroll
            for (int load = 0; load < kCodeLoads; ++load) {
                const int chunk = lane + 32 * load;
                if (chunk < kCodeChunks) {
                    next_codes[row_item][load] = load_global<uint4, Schedule::kWeightCache>(
                        codes + static_cast<std::int64_t>(weight_row) * kHidden + group_k0 +
                        chunk * 16);
                }
            }
        }
        if constexpr (kPrefetchX) {
#    pragma unroll
            for (int load = 0; load < kXLoads; ++load) {
                const int item = lane + 32 * load;
                int col, k8;
                if (item < x_items) {
                    next_x[load] =
                        x_item(item, col, k8)
                            ? load_global<uint4, Schedule::kActivationCache>(
                                  &x[static_cast<std::int64_t>(column_policy(column_offset + col)) *
                                         kHidden +
                                     group_k0 + warp * kTileK + k8 * 8])
                            : make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        if constexpr (kSharedScales) {
#    pragma unroll
            for (int load = 0; load < kScaleLoads; ++load) {
                const int item = tid + load * kWarps * 32;
                if (item < kScaleChunks) {
                    const int row        = item / kScaleChunksPerRow;
                    const int chunk      = item - row * kScaleChunksPerRow;
                    const int weight_row = row_policy.weight_row(cta_row0, row);
                    next_scales[load]    = load_global<uint4, Schedule::kWeightCache>(
                        scales + (static_cast<std::int64_t>(weight_row) * Geometry::kGroupsPerRow +
                                  group_k0 / 32 + chunk * 8) *
                                     2);
                }
            }
        } else if (lid < 2) {
            next_direct_scale_pair = load_direct_scale_pair(group_k0);
        }
    };
    const auto store_next = [&](int group_k0) {
#    pragma unroll
        for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
            const int row = warp * Schedule::kRowsPerLoaderWarp + row_item;
#    pragma unroll
            for (int load = 0; load < kCodeLoads; ++load) {
                const int chunk = lane + 32 * load;
                if (chunk < kCodeChunks) {
                    *reinterpret_cast<uint4*>(&code_shared[row][(chunk ^ (row & 7)) * 16]) =
                        next_codes[row_item][load];
                }
            }
        }
        if constexpr (kPrefetchX) {
#    pragma unroll
            for (int load = 0; load < kXLoads; ++load) {
                const int item = lane + 32 * load;
                int col, k8;
                if (item < x_items) {
                    (void)x_item(item, col, k8);
                    *reinterpret_cast<uint4*>(
                        &b_shared[warp][col * kTileK + q8_ksplit_swizzle_64(col, k8 * 8)]) =
                        next_x[load];
                }
            }
        } else {
            stage_x(group_k0);
        }
        if constexpr (kSharedScales) {
#    pragma unroll
            for (int load = 0; load < kScaleLoads; ++load) {
                const int item = tid + load * kWarps * 32;
                if (item < kScaleChunks) {
                    const int row   = item / kScaleChunksPerRow;
                    const int chunk = item - row * kScaleChunksPerRow;
                    *reinterpret_cast<uint4*>(&scale_shared[row][chunk * 16]) = next_scales[load];
                }
            }
        } else {
            direct_scale_pair = next_direct_scale_pair;
        }
    };
    if constexpr (!kSharedScales) {
        if (lid < 2) { direct_scale_pair = load_direct_scale_pair(0); }
    }
#endif

    constexpr int kGroupUnroll = kHidden <= 6144 ? kGroups : 12;
#pragma unroll kGroupUnroll
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int group_k0 = group_index * kGroupK;
#if defined(NINFER_SM75_DEVICE)
        if (group_index + 1 < kGroups) { load_next(group_k0 + kGroupK); }
#endif

        unsigned lane_scale_pair = 0;
        if (lid < 2) {
            if constexpr (Schedule::kScaleAccess == Q8KSplitScaleAccess::Shared) {
                const int scale_row = gid + lid * 8;
                lane_scale_pair =
                    *reinterpret_cast<const unsigned*>(&scale_shared[scale_row][warp_koff / 16]);
            } else {
#if defined(NINFER_SM75_DEVICE)
                lane_scale_pair = direct_scale_pair;
#else
                const int scale_row = row_policy.weight_row(cta_row0, gid + lid * 8);
                lane_scale_pair     = *reinterpret_cast<const unsigned*>(
                    scales + (static_cast<std::int64_t>(scale_row) * Geometry::kGroupsPerRow +
                              group_k0 / 32 + warp_koff / 32) *
                                 2);
#endif
            }
        }
        const unsigned top_scale_pair = __shfl_sync(kMask, lane_scale_pair, lane & ~3);
        const unsigned bot_scale_pair = __shfl_sync(kMask, lane_scale_pair, (lane & ~3) + 1);

#pragma unroll
        for (int group = 0; group < 2; ++group) {
            float group_acc[kNt][4];
#pragma unroll
            for (int ni = 0; ni < kNt; ++ni) {
                group_acc[ni][0] = 0.0f;
                group_acc[ni][1] = 0.0f;
                group_acc[ni][2] = 0.0f;
                group_acc[ni][3] = 0.0f;
            }
#pragma unroll
            for (int ki = 0; ki < 2; ++ki) {
                const int ks              = group * 2 + ki;
                const int code_col        = ks * 16 + lid * 2;
                const auto load_code_pair = [&](int code_row, int col) {
                    const int chunk  = (warp_koff + col) >> 4;
                    const int offset = (chunk ^ (code_row & 7)) * 16 + (col & 15);
                    return static_cast<unsigned>(
                        *reinterpret_cast<const unsigned short*>(&code_shared[code_row][offset]));
                };
#if defined(NINFER_SM75_DEVICE)
                // The FP16 codes and, when every live activation of the tile is an FP16 value, the
                // converted activations are the operands mma_bf16 would form, issued in its order,
                // so the bits are the same; a tile holding any other live activation runs
                // mma_bf16, which settles each output on its own operands. A lane's activations
                // are column ni * 8 + gid; a column past the live extent is never staged and holds
                // whatever the shared memory held before, and its outputs are discarded, so it
                // does not choose the path.
                const unsigned code0 = load_code_pair(gid, code_col);
                const unsigned code1 = load_code_pair(gid + 8, code_col);
                const unsigned code2 = load_code_pair(gid, code_col + 8);
                const unsigned code3 = load_code_pair(gid + 8, code_col + 8);
                const unsigned hf0   = q8_ksplit_f16_pair_from_s8(code0);
                const unsigned hf1   = q8_ksplit_f16_pair_from_s8(code1);
                const unsigned hf2   = q8_ksplit_f16_pair_from_s8(code2);
                const unsigned hf3   = q8_ksplit_f16_pair_from_s8(code3);
#    pragma unroll
                for (int ni = 0; ni < kNt; ++ni) {
                    unsigned bf0, bf1;
                    const int br = ni * 8 + b_rin;
                    ldmatrix_x2(
                        bf0, bf1,
                        smem_addr(&b_shared[k_split][br * kTileK +
                                                     q8_ksplit_swizzle_64(br, ks * 16 + b_koff)]));
                    const bool dead = ni * 8 + gid >= live_columns;
                    if (__all_sync(kMask, dead || (sm75::bf16x2_exact_in_fp16(bf0) &&
                                                   sm75::bf16x2_exact_in_fp16(bf1)))) {
                        mma_f16(group_acc[ni][0], group_acc[ni][1], group_acc[ni][2],
                                group_acc[ni][3], hf0, hf1, hf2, hf3, sm75::bf16x2_to_half2(bf0),
                                sm75::bf16x2_to_half2(bf1));
                    } else {
                        mma_bf16(group_acc[ni][0], group_acc[ni][1], group_acc[ni][2],
                                 group_acc[ni][3], q8_ksplit_bf16_pair_from_s8(code0),
                                 q8_ksplit_bf16_pair_from_s8(code1),
                                 q8_ksplit_bf16_pair_from_s8(code2),
                                 q8_ksplit_bf16_pair_from_s8(code3), bf0, bf1);
                    }
                }
#else
                const unsigned af0 = q8_ksplit_bf16_pair_from_s8(load_code_pair(gid, code_col));
                const unsigned af1 = q8_ksplit_bf16_pair_from_s8(load_code_pair(gid + 8, code_col));
                const unsigned af2 = q8_ksplit_bf16_pair_from_s8(load_code_pair(gid, code_col + 8));
                const unsigned af3 =
                    q8_ksplit_bf16_pair_from_s8(load_code_pair(gid + 8, code_col + 8));
#    pragma unroll
                for (int ni = 0; ni < kNt; ++ni) {
                    unsigned bf0, bf1;
                    const int br = ni * 8 + b_rin;
                    ldmatrix_x2(
                        bf0, bf1,
                        smem_addr(&b_shared[k_split][br * kTileK +
                                                     q8_ksplit_swizzle_64(br, ks * 16 + b_koff)]));
                    mma_bf16(group_acc[ni][0], group_acc[ni][1], group_acc[ni][2], group_acc[ni][3],
                             af0, af1, af2, af3, bf0, bf1);
                }
#endif
            }
            const unsigned top_bits = group == 0 ? top_scale_pair & 0xffffu : top_scale_pair >> 16;
            const unsigned bot_bits = group == 0 ? bot_scale_pair & 0xffffu : bot_scale_pair >> 16;
            const float top_scale   = __half2float(__ushort_as_half(top_bits));
            const float bot_scale   = __half2float(__ushort_as_half(bot_bits));
#pragma unroll
            for (int ni = 0; ni < kNt; ++ni) {
                acc[ni][0] = fmaf(group_acc[ni][0], top_scale, acc[ni][0]);
                acc[ni][1] = fmaf(group_acc[ni][1], top_scale, acc[ni][1]);
                acc[ni][2] = fmaf(group_acc[ni][2], bot_scale, acc[ni][2]);
                acc[ni][3] = fmaf(group_acc[ni][3], bot_scale, acc[ni][3]);
            }
        }

        if (group_index + 1 < kGroups) {
            __syncthreads();
#if defined(NINFER_SM75_DEVICE)
            store_next(group_k0 + kGroupK);
#else
            stage_codes(group_k0 + kGroupK);
            stage_x(group_k0 + kGroupK);
            cp_commit();
            cp_wait<0>();
#endif
            __syncthreads();
        }
    }

    pdl::trigger_dependents();
    __syncthreads();
    auto* partial = shared.partial;
    if ((k_split & 1) != 0) {
#pragma unroll
        for (int ni = 0; ni < kNt; ++ni) {
            store_vec(partial + ((warp * kNt + ni) * 32 + lane) * 4,
                      make_float4(acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]));
        }
    }
    __syncthreads();

    if ((k_split & 1) == 0) {
#pragma unroll
        for (int ni = 0; ni < kNt; ++ni) {
            const float4 partner =
                load_vec<float4>(partial + (((warp + 1) * kNt + ni) * 32 + lane) * 4);
            acc[ni][0] += partner.x;
            acc[ni][1] += partner.y;
            acc[ni][2] += partner.z;
            acc[ni][3] += partner.w;
            if (k_split != 0) {
                store_vec(partial + ((warp * kNt + ni) * 32 + lane) * 4,
                          make_float4(acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]));
            }
        }
    }
    __syncthreads();

    if (k_split == 0) {
        float* projected = partial;
#pragma unroll
        for (int ni = 0; ni < kNt; ++ni) {
            float4 sum = make_float4(acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]);
#pragma unroll
            for (int split = 2; split < kWarps; split += 2) {
                const float4 value =
                    load_vec<float4>(partial + ((split * kNt + ni) * 32 + lane) * 4);
                sum.x += value.x;
                sum.y += value.y;
                sum.z += value.z;
                sum.w += value.w;
            }
            const int col0 = ni * 8 + 2 * lid;
            if constexpr (std::is_same_v<Epilogue, Q8KSplitStoreEpilogue> ||
                          std::is_same_v<Epilogue, Q8KSplitResidualEpilogue>) {
                const auto store = [&](int row, int col, float value) {
                    if constexpr (TiledColumns) {
                        if (col >= live_columns) return;
                    }
                    __nv_bfloat16* destination = output.tile(cta_row0).at(row, col + column_offset);
                    if constexpr (std::is_same_v<Epilogue, Q8KSplitResidualEpilogue>) {
                        value += __bfloat162float(*destination);
                    }
                    *destination = __float2bfloat16_rn(value);
                };
                if (col0 < ActiveCols) {
                    store(cta_row0 + gid, col0, sum.x);
                    store(cta_row0 + gid + 8, col0, sum.z);
                }
                if (col0 + 1 < ActiveCols) {
                    store(cta_row0 + gid, col0 + 1, sum.y);
                    store(cta_row0 + gid + 8, col0 + 1, sum.w);
                }
            } else if constexpr (DirectPairEpilogue) {
                epilogue.store_pair(cta_row0 + gid, col0 + column_offset, sum,
                                    TiledColumns ? columns : ActiveCols);
            } else {
                if (col0 < ActiveCols) {
                    projected[gid * kTileCols + col0]       = sum.x;
                    projected[(gid + 8) * kTileCols + col0] = sum.z;
                }
                if (col0 + 1 < ActiveCols) {
                    projected[gid * kTileCols + col0 + 1]       = sum.y;
                    projected[(gid + 8) * kTileCols + col0 + 1] = sum.w;
                }
            }
        }
        if constexpr (!std::is_same_v<Epilogue, Q8KSplitStoreEpilogue> &&
                      !std::is_same_v<Epilogue, Q8KSplitResidualEpilogue> && !DirectPairEpilogue) {
            __syncwarp();
            if (lane < kRowsPerCta) {
                float row_values[ActiveCols];
#pragma unroll
                for (int token = 0; token < ActiveCols; ++token) {
                    row_values[token] = projected[lane * kTileCols + token];
                }
                epilogue.store(cta_row0 + lane, row_values);
            }
        }
    }
}

// Standard projection entry. Multi-layer fused Ops call the same contraction after selecting
// their independent weight views and provide a closed FP32 epilogue.
template <class Geometry, int ActiveCols, class Schedule, class Output,
          class Epilogue = Q8KSplitStoreEpilogue, class RowPolicy = Q8KSplitIdentityRows,
          bool DirectPairEpilogue = false, bool TiledColumns = false>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void q8_ksplit_mma_kernel(
    const __nv_bfloat16* x, const std::uint8_t* codes, const std::uint8_t* scales, Output output,
    Epilogue epilogue = {}, RowPolicy row_policy = {}, std::int32_t columns = ActiveCols) {
    q8_ksplit_mma<Geometry, ActiveCols, Schedule, Output, Epilogue, RowPolicy, DirectPairEpilogue,
                  TiledColumns>(x, codes, scales, output, epilogue, row_policy, columns);
}

} // namespace ninfer::ops::detail
