#pragma once

// Host-side geometry for the unified chain's narrow-band mainloop -- the same row-form shape
// own/legacy runs through the MMVQ-shaped kernel (nvfp4_mmvq_host.h), rebuilt here so the unified
// route stops depending on a transplanted kernel and can name the widths it actually serves.
//
// Two things differ from that kernel, and both are why this file exists:
//
//   * The column ladder is every width from one to eight, not the powers of two plus six. MMVQ
//     keeps its NC a power of two because each stage is a full instantiation of the mainloop per
//     adapter and llama.cpp's table only ever answers a handful of widths; the unified chain owns its
//     instantiation list and can spend it, so a five-token call runs five columns instead of six and
//     a seven-token call runs seven instead of eight. Nothing about the arithmetic moves: NC only
//     sizes the accumulator array, and the store is already guarded by the caller's real width.
//   * The activation planes are sized to the call, not padded to eight. The width the kernel reads is
//     the width the caller asked for, so the rows a padded plane would have held are rows nobody
//     reads -- which also means the quantiser launches `tokens` blocks instead of `padded`. Both
//     halves of that have to change together; see the note in nvfp4_int8_narrow_launch.cuh.
//
// The crossover itself is shared, not restated: `applies` is `mmvq::applies`, so the two chains cannot
// drift apart into different narrow bands while the device profile calibrates a third number.

#include "ops/linear/nvfp4/nvfp4_mmvq_host.h"

#include <cstdint>

namespace ninfer::ops::detail::narrow {

// Same warp group as the MMVQ-shaped kernel, so the reduction tree has the same shape and the same
// number of partials to absorb. Bit-for-bit equality with it below the crossover is a construction,
// not a hope: identical trip order, identical per-group rescale expression, identical butterfly
// offsets and identical absorption order.
constexpr int kWarps      = mmvq::kWarps;
constexpr int kThreads    = mmvq::kThreads;
constexpr int kWarpSize   = mmvq::kWarpSize;
constexpr int kGroupCodes = mmvq::kGroupCodes;
constexpr int kGroupBytes = mmvq::kGroupBytes;
constexpr int kMaxTokens  = mmvq::kMaxTokens;

// The widest row-sharing factor instantiated. Four is what the MMVQ-shaped kernel's constant allows;
// which widths here launch it follows from rows_per_block, and the two are separate because at an
// exact column count the register cost lands somewhere different from the padded case.
constexpr int kMaxRowsPerBlock = 4;

// The row-sharing factor for one column stage. R folds several output rows into the same activation
// reads, and MMVQ applies it at every stage it has, so a stage that dropped it would give up shared
// activation reads for nothing: R starts at two columns and stops at one, where there is no second
// row to share with.
//
// R=4 pays only at the two widest exact stages: the narrower stages cannot amortise the doubled
// weight traffic the fourth row brings, the two widest can. NC=7 gets its own stage here, where the
// padded MMVQ ladder folds it into the eight-column one -- this ladder runs the caller's real width.
constexpr int rows_per_block(int nc) { return nc >= 7 ? 4 : (nc >= 2 ? 2 : 1); }

/// The column count this kernel runs: the caller's width exactly. Every stage from one to eight is
/// instantiated, so there is nothing to round up to.
inline int columns(std::int32_t tokens) { return static_cast<int>(tokens); }

/// Whether this call is answered by the narrow kernel: below the crossover and on the unified table.
/// The test seam is honoured through `nvfp4_int8_mainloop`, so a case that pins a mainloop gets the
/// narrow band that mainloop would really take.
inline bool select(std::int32_t tokens, Nvfp4Int8Mainloop table) {
    return mmvq::applies(tokens) && nvfp4_int8_mainloop(table) == Nvfp4Int8Mainloop::Grouped;
}

/// The width a call's activation planes are sized to on this table: the narrow band addresses the
/// columns the caller asked for, every other answer addresses the tile envelope. The launch and the
/// capacity queries all ask this one question, because a plane sized for a width the quantiser never
/// filled is read back by the next projection.
inline std::int32_t addressed_tokens(std::int32_t tokens, Nvfp4Int8Mainloop table) {
    return select(tokens, table) ? tokens : unified::nvfp4_int8_padded_tokens(tokens);
}

/// The two activation planes a call on this table reserves for a row count of `k`: the same two
/// 256-aligned spans the launch takes out of the caller's arena.
inline std::size_t workspace_bytes(std::int32_t tokens, std::int32_t k, Nvfp4Int8Mainloop table) {
    return unified::nvfp4_int8_plane_bytes(addressed_tokens(tokens, table), k);
}

} // namespace ninfer::ops::detail::narrow
