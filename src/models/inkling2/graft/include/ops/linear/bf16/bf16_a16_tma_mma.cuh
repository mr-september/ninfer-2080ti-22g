#pragma once

#include "core/tma_descriptor_staging.cuh"
#include "ops/common/device_tier.h"
#include "ops/common/mbarrier.cuh"
#include "ops/linear/bf16/bf16_mma_common.cuh"
#include "ops/linear/bf16/bf16_operands.h"
#include "ops/common/token_slices.h"
#include "ops/common/math.h"
#include <cuda.h>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::unified {

// Passed by value as a __grid_constant__ kernel parameter (through device memory where
// NINFER_TMA_STAGED_DESCRIPTORS is set: MSVC cannot pass an over-aligned parameter by value,
// C2719). TMA requires 64-byte descriptor alignment, stated here because cuda.h keys CUtensorMap's
// own alignas(128) on __cplusplus, which MSVC reports as 199711L unless /Zc:__cplusplus is given.
// It is never stated below CUtensorMap's own alignment: a weaker alignas than a member's is
// ill-formed, and nvcc's MSVC front end rejects it.
struct alignas(alignof(CUtensorMap) > 64 ? alignof(CUtensorMap) : 64) Bf16TmaDescriptors {
    CUtensorMap weight;
    CUtensorMap activation;
};
static_assert(alignof(Bf16TmaDescriptors) % 64 == 0);

#ifdef NINFER_TMA_STAGED_DESCRIPTORS
// The single staged-descriptor (Windows) staging for Bf16TmaDescriptors. See
// core/tma_descriptor_staging.cuh for the invariants its device buffer relies on.
inline TmaDescriptorStaging<Bf16TmaDescriptors>& bf16_tma_descriptor_staging() {
    static TmaDescriptorStaging<Bf16TmaDescriptors> staging;
    return staging;
}
#endif

inline CUtensorMap bf16_tma_map(const __nv_bfloat16* pointer, int rows, int k, int block_rows,
                                int block_k) {
    // Factor K into 64-element sectors. The contiguous 128-byte dimension matches the
    // hardware swizzle while the next box dimension permits larger K tiles without repacking.
    CUtensorMap result{};
    const std::uint64_t dimensions[]{64, static_cast<std::uint64_t>(k / 64),
                                     static_cast<std::uint64_t>(rows)};
    const std::uint64_t strides[]{128, static_cast<std::uint64_t>(k) * 2};
    const std::uint32_t box[]{64, static_cast<std::uint32_t>(block_k / 64),
                              static_cast<std::uint32_t>(block_rows)};
    const std::uint32_t steps[]{1, 1, 1};
    const auto status = cuTensorMapEncodeTiled(
        &result, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<__nv_bfloat16*>(pointer),
        dimensions, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("BF16 TMA descriptor: ") +
                                 (name ? name : "CUDA error"));
    }
    return result;
}

template <class Schedule>
Bf16TmaDescriptors make_bf16_tma_descriptors(const Bf16A16Operands& p) {
    validate_bf16_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK)
        throw std::invalid_argument("BF16 TMA requires complete row/K tiles");
    return {bf16_tma_map(p.weight, p.rows, p.k, Schedule::kBlockRows, Schedule::kBlockK),
            bf16_tma_map(p.x, p.tokens, p.k, Schedule::kBlockTokens, Schedule::kBlockK)};
}

__device__ __forceinline__ void bf16_tma_load(void* destination, const CUtensorMap* map,
                                              int k_sector, int row, std::uint64_t* barrier) {
#if defined(NINFER_PRE_BLACKWELL_DEVICE)
    __trap();
#else
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4}], [%5];"
                 :
                 : "r"(smem_addr(destination)), "l"(map), "r"(0), "r"(k_sector), "r"(row),
                   "r"(smem_addr(barrier))
                 : "memory");
#endif
}

template <class Schedule, class Epilogue>
inline constexpr int bf16_tma_scratch_bytes =
    ((Schedule::kTensorBytes > bf16_epilogue_bytes<Schedule, Epilogue>
          ? Schedule::kTensorBytes
          : bf16_epilogue_bytes<Schedule, Epilogue>)+127) /
    128 * 128;

template <class Schedule, bool FullTokens, class Output, class Epilogue>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void bf16_a16_tma_mma_kernel(
#ifdef NINFER_TMA_STAGED_DESCRIPTORS
    // MSVC cannot pass the over-aligned (alignas(128)) CUtensorMap struct by value as a
    // __grid_constant__ parameter (C2719); on Windows the descriptors come from the staging buffer.
    const Bf16TmaDescriptors* descriptors_pointer,
#else
    const __grid_constant__ Bf16TmaDescriptors descriptors,
#endif
    Output output, Epilogue epilogue, int rows, int input_rows, int token_offset, int count) {
#ifdef NINFER_TMA_STAGED_DESCRIPTORS
    const Bf16TmaDescriptors& descriptors = *descriptors_pointer;
#endif
    constexpr int BR = Schedule::kBlockRows, BT = Schedule::kBlockTokens, BK = Schedule::kBlockK;
    constexpr int S   = Schedule::kStages;
    const int K       = Schedule::kStaticK ? Schedule::kStaticK : input_rows;
    const int tiles_r = rows / BR, tiles_t = (count + BT - 1) / BT;
    int tile_r, tile_t;
    bf16_mma_tile_coordinates<Schedule>(blockIdx.x, tiles_r, tiles_t, tile_r, tile_t);
    const int row_begin = tile_r * BR, token_begin = token_offset + tile_t * BT;
    extern __shared__ __align__(128) unsigned char storage[];
    auto* a = reinterpret_cast<__nv_bfloat16*>(storage);
    auto* b = a + S * BR * BK;
    auto* full =
        reinterpret_cast<std::uint64_t*>(storage + bf16_tma_scratch_bytes<Schedule, Epilogue>);
    auto* empty = full + S;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < S; ++stage) {
            cta_mbarrier_init(full + stage, 1);
            cta_mbarrier_init(empty + stage, Schedule::kConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();
    const int tiles_k = K / BK;
    if (threadIdx.x < Schedule::kProducerThreads) {
        if (threadIdx.x == 0) {
#ifdef NINFER_TMA_STAGED_DESCRIPTORS
            // Each staged 128-byte map needs its own acquire for the TMA (tensormap) proxy.
            acquire_staged_tensor_map(&descriptors.weight);
            acquire_staged_tensor_map(&descriptors.activation);
#endif
            for (int kt = 0; kt < tiles_k; ++kt) {
                const int stage = kt % S;
                cta_mbarrier_wait(empty + stage, 1U ^ ((kt / S) & 1U));
                cta_mbarrier_arrive_expect_tx(full + stage, (BR + BT) * BK * 2);
                bf16_tma_load(a + stage * BR * BK, &descriptors.weight, kt * (BK / 64), row_begin,
                              full + stage);
                bf16_tma_load(b + stage * BT * BK, &descriptors.activation, kt * (BK / 64),
                              token_begin, full + stage);
            }
        }
        return;
    }
    const int tid  = threadIdx.x - Schedule::kProducerThreads;
    const int warp = tid / 32, lane = tid & 31;
    float accum[Schedule::kMmaRows][Schedule::kMmaTokens][4] = {};
    for (int kt = 0; kt < tiles_k; ++kt) {
        const int stage = kt % S;
        cta_mbarrier_wait(full + stage, (kt / S) & 1U);
        bf16_mma_compute_stage<Schedule>(a + stage * BR * BK, b + stage * BT * BK, accum, warp,
                                         lane);
        cta_mbarrier_release_stage(empty + stage, lane);
    }
    bf16_finish_mma_tile<Schedule, FullTokens>(output, epilogue, storage, accum, row_begin,
                                               token_begin, rows, token_offset + count, warp, lane);
}

template <class Schedule, class Output, class Epilogue>
void launch_bf16_a16_mma(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                         cudaStream_t stream);

// Cards without a tensor memory accelerator (sm_8x, and every card but a Blackwell one in a build
// of several architectures) run the same tile over cp.async staging.
template <class S>
using Bf16TmaFallbackSchedule = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<S::kBlockRows, S::kBlockTokens, S::kBlockK, S::kWarpRows, S::kWarpTokens,
                       S::kStages, S::kMinBlocksPerSm, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, S::kRaster, Bf16MmaSwizzle::Xor64,
                       S::kRasterGroupRows>,
    S::kStaticK>;

template <class Schedule, class Output, class Epilogue>
void launch_bf16_a16_tma_mma(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                             cudaStream_t stream) {
#if defined(NINFER_SM8X_COMPAT) && !defined(NINFER_SM120_NVFP4)
    launch_bf16_a16_mma<Bf16TmaFallbackSchedule<Schedule>>(p, output, epilogue, stream);
#else
    if constexpr (kBuildTiers > 1) {
        if (!current_device_blackwell()) {
            launch_bf16_a16_mma<Bf16TmaFallbackSchedule<Schedule>>(p, output, epilogue, stream);
            return;
        }
    }
    // Descriptors are launch-owned values, copied into kernel parameters during Graph capture.
    const auto descriptors = make_bf16_tma_descriptors<Schedule>(p);
#ifdef NINFER_TMA_STAGED_DESCRIPTORS
    // Staged once: every token-slice launch below reads the same copy, in stream order.
    const Bf16TmaDescriptors* staged = bf16_tma_descriptor_staging().stage(descriptors, stream);
#endif
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const auto blocks = static_cast<std::int64_t>(p.rows / Schedule::kBlockRows) *
                            div_up(count, Schedule::kBlockTokens);
        if (blocks > 2147483647LL)
            throw std::invalid_argument("BF16 TMA grid exceeds CUDA grid.x capacity");
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = bf16_a16_tma_mma_kernel<Schedule, Full, Output, Epilogue>;
            constexpr int bytes =
                bf16_tma_scratch_bytes<Schedule, Epilogue> + Schedule::kBarrierBytes;
            bf16_prepare_shared<bytes, kernel>();
#ifdef NINFER_TMA_STAGED_DESCRIPTORS
            kernel<<<static_cast<unsigned>(blocks), Schedule::kThreads, bytes, stream>>>(
                staged, output, epilogue, p.rows, p.k, offset, count);
#else
            kernel<<<static_cast<unsigned>(blocks), Schedule::kThreads, bytes, stream>>>(
                descriptors, output, epilogue, p.rows, p.k, offset, count);
#endif
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
#endif
}

} // namespace ninfer::ops::detail::unified
