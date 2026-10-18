#pragma once

#include "ops/common/cuda/memory.cuh"
#include "ops/common/quant/integer.cuh"

namespace ninfer::ops::detail {

template <int BlockRows, int BlockTokens, int WarpsPerRow, int GroupsPerWarpStage, int Stages,
          Cache CodeCache, int MinBlocksPerSm, bool Predicated = false>
struct Q8A16SimtSchedule {
    static_assert(BlockRows > 0 && BlockRows <= 32);
    static_assert(BlockTokens > 0 && BlockTokens <= 8);
    static_assert(WarpsPerRow == 1 || WarpsPerRow == 2 || WarpsPerRow == 4 || WarpsPerRow == 8);
    static_assert(GroupsPerWarpStage > 0 && GroupsPerWarpStage % 2 == 0);
    static_assert(Stages >= 2 && Stages <= 8);
    static_assert(MinBlocksPerSm > 0);
    static constexpr bool kPredicated        = Predicated;
    static constexpr int kBlockRows          = BlockRows;
    static constexpr int kBlockTokens        = BlockTokens;
    static constexpr int kWarpsPerRow        = WarpsPerRow;
    static constexpr int kGroupsPerWarpStage = GroupsPerWarpStage;
    static constexpr int kStages             = Stages;
    static constexpr Cache kCodeCache        = CodeCache;
    static constexpr int kMinBlocksPerSm     = MinBlocksPerSm;
    static constexpr int kWarps              = BlockRows * WarpsPerRow;
    static constexpr int kThreads            = kWarps * 32;
    static constexpr int kStageK = GroupsPerWarpStage * common::Q8RowSplitStorage::kGroupK;
    static constexpr int kCodeVecsPerStage   = GroupsPerWarpStage * 2;
    static constexpr int kScalePairsPerStage = GroupsPerWarpStage / 2;
    static constexpr int kCodePhases         = (GroupsPerWarpStage + 7) / 8;
    static constexpr int kStagingBytes       = kWarps * Stages * GroupsPerWarpStage *
                                         (common::Q8RowSplitStorage::kCodeBytesPerGroup +
                                          common::Q8RowSplitStorage::kScaleBytesPerGroup);
    static constexpr int kPartialBytes = WarpsPerRow > 1 ? kWarps* BlockTokens * sizeof(float) : 0;
    static constexpr int kSharedBytes =
        ((kStagingBytes > kPartialBytes ? kStagingBytes : kPartialBytes) + 15) / 16 * 16;
    static_assert(kThreads <= 1024);
    static_assert(kSharedBytes <= 48 * 1024, "Q8 SIMT shared memory exceeds 48 KiB");
};

template <int BlockRows, int WarpsPerRow, int MinBlocksPerSm, int StaticK = 0, int Unroll = 8>
struct Q8A16GemvSchedule {
    static_assert(BlockRows > 0 && WarpsPerRow > 0);
    static_assert(WarpsPerRow == 1 || WarpsPerRow == 2 || WarpsPerRow == 4 || WarpsPerRow == 8);
    static_assert(MinBlocksPerSm > 0 && Unroll > 0);
    static_assert(StaticK == 0 || (StaticK > 0 && StaticK % 256 == 0));
    static constexpr int kBlockRows      = BlockRows;
    static constexpr int kBlockTokens    = 1;
    static constexpr int kWarpsPerRow    = WarpsPerRow;
    static constexpr int kThreads        = BlockRows * WarpsPerRow * 32;
    static constexpr int kMinBlocksPerSm = MinBlocksPerSm;
    static constexpr int kStaticK        = StaticK;
    static constexpr int kUnroll         = Unroll;
    static_assert(kThreads <= 1024);
};

enum class Q8MmaFragmentPipeline { Serial, PingPong };

// Quant codes are prefetched while MMA consumes the decoded BF16 weight tile.
// Eight G32 scales are cached per row. Activations may use one or two buffers.
// Padding-free problems can fix K and specialize the outer-loop unroll factor.
template <int BlockRows, int BlockTokens, int BlockK, int WarpRows, int WarpTokens,
          int ActivationStages, int MinBlocksPerSm,
          Q8MmaFragmentPipeline FragmentPipeline = Q8MmaFragmentPipeline::PingPong,
          Cache WeightCache = Cache::cg, Cache ActivationCache = Cache::cg,
          Cache PredicatedCache = Cache::ca, bool Predicated = false, int StaticK = 0,
          int KLoopUnroll = 4>
struct Q8A16MmaSchedule {
    static constexpr int kBlockRows         = BlockRows;
    static constexpr int kBlockTokens       = BlockTokens;
    static constexpr int kBlockK            = BlockK;
    static constexpr int kWarpRows          = WarpRows;
    static constexpr int kWarpTokens        = WarpTokens;
    static constexpr int kActivationStages  = ActivationStages;
    static constexpr int kStages            = 2;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr auto kFragmentPipeline = FragmentPipeline;
    static constexpr auto kWeightCache      = WeightCache;
    static constexpr auto kActivationCache  = ActivationCache;
    static constexpr auto kPredicatedCache  = PredicatedCache;
    static constexpr bool kPredicated       = Predicated;
    static constexpr int kStaticK           = StaticK;
    static constexpr int kKLoopUnroll       = KLoopUnroll;
    static constexpr int kWarpGridRows      = BlockRows / WarpRows;
    static constexpr int kWarpGridTokens    = BlockTokens / WarpTokens;
    static constexpr int kWarps             = kWarpGridRows * kWarpGridTokens;
    static constexpr int kThreads           = kWarps * 32;
    static constexpr int kMmaRows           = WarpRows / 16;
    static constexpr int kMmaTokens         = WarpTokens / 8;
    static constexpr int kMmaKSteps         = BlockK / 16;
    static constexpr int kScaleCacheBytes   = 16;
    static constexpr int kSharedBytes       = BlockRows * BlockK * 3 +
                                        ActivationStages * BlockTokens * BlockK * 2 +
                                        BlockRows * kScaleCacheBytes;
    static_assert(BlockRows > 0 && BlockTokens > 0 && WarpRows > 0 && WarpTokens > 0);
    static_assert(BlockRows % WarpRows == 0 && BlockTokens % WarpTokens == 0);
    static_assert(WarpRows % 16 == 0 && WarpTokens % 8 == 0);
    static_assert(BlockK == 64 || BlockK == 128);
    static_assert(StaticK == 0 || (StaticK > 0 && StaticK % BlockK == 0));
    static_assert(KLoopUnroll > 0);
    static_assert(ActivationStages == 1 || ActivationStages == 2);
    static_assert(kThreads <= 1024 && MinBlocksPerSm > 0);
    static_assert(kSharedBytes <= core::kMaxCtaSharedBytes);
};

// Stage and K-warp counts are tuned for the 99 KiB Blackwell CTA. A smaller shared-memory limit
// first single-buffers the K ring and then halves the K-warp count (each K warp stages its own
// 64-deep K slice of the weights and of every active token), instead of rejecting the schedule.
constexpr int q8_fit_stages(int stages, int bytes_per_stage) {
    return stages * bytes_per_stage <= core::kMaxCtaSharedBytes ? stages : 1;
}

constexpr int q8_fit_k_warps(int k_warps, int bytes_per_k_warp_stage) {
    while (k_warps > 2 && k_warps * bytes_per_k_warp_stage > core::kMaxCtaSharedBytes) {
        k_warps /= 2;
    }
    return k_warps;
}

enum class Q8ScaleAccess : std::uint8_t { Direct, Shared };
enum class Q8ActivationStage : std::uint8_t {
    ActiveOnly,    // Stage the compile-time capacity, zero filling inactive tokens.
    PaddedZero,    // Also initialize MMA columns beyond the capacity.
    RuntimeActive, // Stage only live columns; the epilogue discards inactive columns.
};

template <int BlockTokens, int KWarps, int Stages, int MinBlocksPerSm, Q8ScaleAccess ScaleAccess,
          Cache ActivationCache = Cache::ca, Cache WeightCache = Cache::cg,
          Q8ActivationStage ActivationStage = Q8ActivationStage::ActiveOnly, int StaticK = 0,
          int TokenCapacity = BlockTokens, bool ExactTokens = false>
struct Q8A16SlicedKMmaSchedule {
    static_assert(KWarps == 2 || KWarps == 4 || KWarps == 8 || KWarps == 16);
    static_assert(BlockTokens >= 8 && BlockTokens % 8 == 0);
    static_assert(Stages == 1 || Stages == 2);
    static_assert(MinBlocksPerSm > 0);
    static_assert(TokenCapacity > 0 && TokenCapacity <= BlockTokens);
    static_assert(StaticK == 0 || (StaticK > 0 && StaticK % (KWarps * 64) == 0));
    template <int K, int Capacity, bool Exact = false>
    using with_problem =
        Q8A16SlicedKMmaSchedule<BlockTokens, KWarps, Stages, MinBlocksPerSm, ScaleAccess,
                                ActivationCache, WeightCache, ActivationStage, K, Capacity, Exact>;
    static constexpr int kBlockRows         = 16;
    static constexpr int kBlockTokens       = BlockTokens;
    static constexpr int kTokenCapacity     = TokenCapacity;
    static constexpr int kStaticK           = StaticK;
    static constexpr bool kExactTokens      = ExactTokens;
    // Single-stage staging per K warp: weights, active-token activations and shared scales.
    static constexpr int kKWarps = q8_fit_k_warps(
        KWarps, 16 * 64 + BlockTokens * 64 * 2 +
                    16 * (ScaleAccess == Q8ScaleAccess::Shared ? 64 / 16 : 1));
    static constexpr int kStages = q8_fit_stages(
        Stages, 16 * kKWarps * 64 + kKWarps * BlockTokens * 64 * 2 +
                    16 * (ScaleAccess == Q8ScaleAccess::Shared ? kKWarps * 64 / 16 : 1));
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr auto kScaleAccess      = ScaleAccess;
    static constexpr auto kActivationCache  = ActivationCache;
    static constexpr auto kWeightCache      = WeightCache;
    static constexpr auto kActivationStage  = ActivationStage;
    static constexpr int kThreads           = kKWarps * 32;
    static constexpr int kWarpK             = 64;
    static constexpr int kBlockK            = kKWarps * kWarpK;
    static constexpr int kRowsPerLoaderWarp = kBlockRows / kKWarps;
    static constexpr int kScaleBytesPerRow  = kBlockK / 16;
    static constexpr int kStagingBytes =
        kStages * (kBlockRows * kBlockK + kKWarps * BlockTokens * kWarpK * 2 +
                  kBlockRows * (ScaleAccess == Q8ScaleAccess::Shared ? kScaleBytesPerRow : 1));
    static constexpr int kPartialBytes = kKWarps * (BlockTokens / 8) * 32 * 16;
    static constexpr int kSharedBytes =
        kStagingBytes > kPartialBytes ? kStagingBytes : kPartialBytes;
    static_assert(kSharedBytes <= core::kMaxCtaSharedBytes);
};

template <int BlockTokens, int KWarps, int TokenGroups, int Stages, int MinBlocksPerSm,
          int StaticK = 0, Cache WeightCache = Cache::cg, Cache ActivationCache = Cache::cg,
          bool TiledTokens = false, bool ReadOnlyScales = true>
struct Q8A16GroupedSlicedKMmaSchedule {
    static_assert(KWarps == 2 || KWarps == 4 || KWarps == 8);
    static_assert(TokenGroups > 0 && BlockTokens > 0 && BlockTokens % TokenGroups == 0);
    static_assert((BlockTokens / TokenGroups) % 8 == 0);
    static_assert(Stages == 1 || Stages == 2);
    static_assert(MinBlocksPerSm > 0);
    static_assert(StaticK == 0 || (StaticK > 0 && StaticK % (KWarps * 64) == 0));
    static constexpr int kBlockRows        = 16;
    static constexpr int kBlockTokens      = BlockTokens;
    static constexpr int kKWarps =
        q8_fit_k_warps(KWarps, 16 * 64 + BlockTokens * 64 * 2);
    static constexpr int kTokenGroups      = TokenGroups;
    static constexpr int kWarpTokens       = BlockTokens / TokenGroups;
    static constexpr int kWarps            = kKWarps * TokenGroups;
    static constexpr int kThreads          = kWarps * 32;
    static constexpr int kBlockK           = kKWarps * 64;
    static constexpr int kStages =
        q8_fit_stages(Stages, 16 * kKWarps * 64 + kKWarps * BlockTokens * 64 * 2);
    static constexpr int kMinBlocksPerSm   = MinBlocksPerSm;
    static constexpr int kStaticK          = StaticK;
    static constexpr auto kWeightCache     = WeightCache;
    static constexpr auto kActivationCache = ActivationCache;
    static constexpr bool kTiledTokens     = TiledTokens;
    static constexpr bool kReadOnlyScales  = ReadOnlyScales;
    static constexpr int kSharedBytes = kStages * (16 * kBlockK + kWarps * kWarpTokens * 64 * 2);
    static_assert(kThreads <= 1024 && kSharedBytes <= core::kMaxCtaSharedBytes);
};

} // namespace ninfer::ops::detail
