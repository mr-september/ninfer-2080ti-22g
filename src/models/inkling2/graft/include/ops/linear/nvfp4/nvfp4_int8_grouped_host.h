#pragma once

// Host-side arithmetic for the NVFP4 Tier-2 int8-grouped route (see nvfp4_int8_grouped_mma.cuh
// for the device side and the performance model). Kept separate from the .cuh so plan and
// dispatch translation units -- which the MSVC host compiler builds -- can query admission and
// workspace sizes without pulling in device intrinsics.
//
// Width is not an admission question: the family answers every positive width, and both
// wide mainloops serve it by padding their last token tile (nvfp4_int8_padded_tokens) instead of
// refusing it. Which wide mainloop that is belongs to the route table (nvfp4_int8_route.h);
// admission looks only at whether the family is registered for the call.
//
// The token tile is itself a function of the call width now. Every mainloop here issues one MMA per
// (16 output rows, 8 tokens, 16 k) tile, so with a fixed 128-token tile a single-token call pays for
// 128 tokens' worth of tensor-core work while reading the same weight bytes once -- and if the tile
// were merely relabelled without changing how many of them a call launches, a small J would instead
// re-read the weights once per tile column. nvfp4_int8_select_j therefore picks the largest admitted
// tile that still covers the width, and both grid and padding follow it, so every call reads each
// weight element exactly once no matter how narrow it is; surplus tile columns are computed and
// discarded by the epilogue guard rather than launched.

#include "ops/linear/nvfp4/nvfp4_int8_route.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::unified {

// Shared launch geometry. BM=BN=128, BK=64 (exactly four G16 weight groups per k-tile), 16 bytes
// of smem row padding to keep every lds128 word-aligned without a 128-multiple stride. The CTA is
// 256 threads / 8 warps (ggml-MMQ shaped): 256 * 255 regs stays under the SM's 64K register file
// with room to hoist the A operands for the whole k-tile, which the old 512-thread CTA (512 * 128
// = the whole file) could not.
namespace detail {

constexpr int kInt8TileRows    = 128; // BM
constexpr int kInt8TileTokens  = 128; // BN
constexpr int kInt8TileK       = 64;  // BK, exactly four G16 groups
constexpr int kInt8SRow        = kInt8TileK + 16;
constexpr int kInt8Threads     = 256;
constexpr int kInt8ScaleGroups = 4; // weight group scales per row per k-tile
constexpr int kInt8XScales     = 2; // per-32 activation scales per row per k-tile

} // namespace detail

constexpr int nvfp4_int8_grouped_smem_bytes() {
    // Two buffer sets: the mainloop stages tile t+1 while it computes tile t, which is what buys the
    // single barrier per tile. The activation half is sized for BM rows although a stage only holds
    // BN <= BM of them, so one launch size covers every slide of the token-tile ladder.
    return 2 * detail::kInt8TileRows * detail::kInt8SRow * 2 +                 // sa + sb, double buffered
           2 * detail::kInt8TileRows * detail::kInt8ScaleGroups * sizeof(float) + // sws
           2 * detail::kInt8TileRows * detail::kInt8XScales * sizeof(float) +     // sxs
           256 * sizeof(float);                                                   // E4M3 scale table
}

// Token-tile ladder. Both mainloops are templated on J from this set, which is ggml's Ampere NVFP4
// column set intersected with ours (mmq-config-ampere.cuh's fallback=false row, plus the multiples of
// eight in between that its rows_per_warp rule also covers). 8 is the floor because an MMA column
// tile is 8 tokens wide -- below it there is nothing smaller to buy, and below it the MMVQ-shaped
// kernel takes over (nvfp4_mmvq_host.h).
// ggml's Ampere NVFP4 column set (mmq-config-ampere.cuh's non-fallback row), ascending. It is not
// "every multiple of eight": 56, 72, 88, 104 and 120 have no row upstream, and a stage that no
// translation unit instantiates is a launch that throws. The set is what both dispatch switches are
// written over, so adding one here means adding it there.
constexpr int kInt8TileStages[] = {8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128};
constexpr int kInt8TileStageCount = sizeof(kInt8TileStages) / sizeof(kInt8TileStages[0]);
constexpr int kInt8MinTileTokens = kInt8TileStages[0];
constexpr int kInt8MaxTileTokens = kInt8TileStages[kInt8TileStageCount - 1];

constexpr bool nvfp4_int8_j_admitted(int j) {
    for (int stage : kInt8TileStages) {
        if (stage == j) { return true; }
    }
    return false;
}

// The token tile a call of `tokens` columns stages: the smallest stage covering the width, capped at
// 128 -- above the cap a call is served by several whole tiles.
constexpr int nvfp4_int8_select_j(std::int32_t tokens) {
    if (tokens <= 0) { return 0; }
    const int capped = tokens > kInt8MaxTileTokens ? kInt8MaxTileTokens : static_cast<int>(tokens);
    for (int stage : kInt8TileStages) {
        if (stage >= capped) { return stage; }
    }
    return kInt8MaxTileTokens;
}

// The token count a padded launch actually runs. The mainloop has no predicated tail -- its grid is
// one CTA per whole J-token tile and every one of its J activation rows is read -- so a width that is
// not a whole tile is served by running the tile full and discarding the surplus: the quantiser
// zeroes the padding rows' planes instead of reading activations that do not exist, and the epilogue
// stores only the tokens the caller asked for.
constexpr std::int32_t nvfp4_int8_padded_tokens(std::int32_t tokens) {
    const int j = nvfp4_int8_select_j(tokens);
    return tokens <= 0 ? 0
                       : ((tokens + j - 1) / j) * j;
}

// Width admission of the int8 *family*: every positive width is answered -- the MMA mainloops pad
// rather than refuse above the crossover, and the MMVQ-shaped row-form kernel takes T <= 8 on top
// of whichever wide mainloop the table selected (mmvq::applies). This is a family question, not a
// mainloop question: the label a plan declares is decided here, so it stays `Int8Grouped` at narrow
// widths even though the row-form kernel is what runs there. Whether the family is registered at
// all is the model layer's decision (--nvfp4-w4a8-int), not a property of a width.
inline bool nvfp4_int8_route_admits(std::int32_t tokens) { return tokens >= 1; }

// The same question for a [min,max] planning interval: can any width in it reach the int8 kernels?
inline bool nvfp4_int8_route_interval_admits(std::int32_t min_tokens, std::int32_t max_tokens) {
    return min_tokens >= 1 && max_tokens >= min_tokens;
}

// s8 codes [T][K] plus the FP32 per-(token,32) scale plane [T][K/32], each 256-aligned by the arena,
// at a width that is already the width the launch addresses. The tile-padded route reaches it through
// `nvfp4_int8_grouped_workspace_bytes` below; the narrow band's sizing calls it directly, because for
// it the call's own width IS the width the kernel addresses.
//
// The MMQ mainloop reads the same quantisation interleaved instead -- per token, per 128-column
// window, four FP32 scales then 128 s8 codes, 144 bytes -- which is the same K + K/8 bytes per
// token as one contiguous span rather than two 256-aligned ones, so it never exceeds this envelope.
// mmq::activation_bytes() is what that route allocates, and it is checked against this function
// for every T in nvfp4_int8's workspace-envelope test.
constexpr std::size_t nvfp4_int8_plane_bytes(std::int32_t addressed_tokens, std::int32_t k) {
    const std::size_t t      = static_cast<std::size_t>(addressed_tokens);
    const std::size_t groups = static_cast<std::size_t>(k) / 32;
    return ((t * static_cast<std::size_t>(k) + 255) / 256) * 256 +
           ((t * groups * sizeof(float) + 255) / 256) * 256;
}

// The same two planes for a tile-padded launch: sized for the PADDED token count, because that is
// what the launch addresses. An exact whole-tile width is unchanged by this, and a ragged one is
// covered by the pad rather than by the count of its own tokens.
constexpr std::size_t nvfp4_int8_grouped_workspace_bytes(std::int32_t tokens, std::int32_t k) {
    return nvfp4_int8_plane_bytes(nvfp4_int8_padded_tokens(tokens), k);
}

} // namespace ninfer::ops::detail::unified
