#pragma once

// Host-side geometry for the NVFP4 route whose mainloop follows ggml's MMVQ -- the lattice llama.cpp
// uses for every width below its tensor-core crossover, which is also where ninfer's int8 mainloops
// are weakest: they tile tokens eight at a time, so a single-token call pays eight tokens' worth of
// `mma.m16n8k16.s8` for one column of useful output. MMVQ has no token tile at all. One warp group
// owns one output row outright, walks its K again in sixteen-column weight groups and accumulates in
// `dp4a`, which is the other integer accelerator sm_86 has and the one that can retire a single
// column without buying the fifteen neighbours.
//
// Ported from ggml/src/ggml-cuda/mmvq.cu (mul_mat_vec_q) and vecdotq.cuh (vec_dot_nvfp4_q8_1), with
// three differences forced by the formats either side of that port:
//
//   * Activation quantisation is ninfer's existing per-(token, 32 columns) pass, not llama's
//     block_q8_1 vocabulary: one FP32 scale per 32 columns and the same s8 codes, which is exactly
//     what `d = ue4m3_to_fp32(bq4->d) * __low2float(bq8->ds)` consumes -- llama's vec_dot reads only
//     the scale out of its block and never the `sum` term, so the two planes agree byte for byte on
//     everything either side reads. Only the addressing differs, and there is no second copy.
//   * The weights are read straight from the ninfer NVFP4 planes -- row-major E2M1 nibbles plus the
//     BlockScaleK16M128x4 scale plane -- so a thread's sixteen-column group is one contiguous eight
//     byte run of the code plane and one E4M3 byte, rather than llama's block_nvfp4 and its own
//     packing order. What is copied is the shape of the loop and of the reduction.
//   * Fusion stays with the caller's epilogue adapter instead of llama's `has_fusion` template: the
//     gate half of gate_up writes an FP32 plane and the up half reads it back, which is what keeps
//     "no BF16 rounding before the SiLU" true without introducing a fused-GLU contract of our own.
//     Two launches over one row range cost no extra weight traffic.
//
// NC is the number of columns the kernel retires, rounded up to one of the instantiated column
// stages so the accumulator stays a register array rather than becoming local memory under a runtime
// trip count. Widths above
// kMaxTokens are not a refusal -- they are the MMA mainloops' business, mmq's or grouped's by route,
// exactly the division of labour llama.cpp makes at `ggml_cuda_should_use_mmvq` (MMVQ below the
// crossover, MMA above it).

#include "ops/linear/nvfp4/nvfp4_int8_grouped_host.h"

#include <cstdint>

namespace ninfer::ops::detail::mmvq {

// MMVQ_MAX_BATCH_SIZE (mmvq.cuh): the widest ncols_dst MMVQ instantiates, and where llama.cpp hands
// the same contraction to MMQ instead.
constexpr int kMaxTokens = 8;

// A block owns its output rows outright, so every weight byte of a row is read exactly once no matter
// how narrow the call; four warps share each row's K and meet in shared memory at the end -- llama's
// calc_nwarps for the generic table at ncols_dst = 1.
constexpr int kWarps   = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kWarpSize = 32;

// The atomic unit of an NVFP4 row: sixteen E2M1 codes sharing one per-16 E4M3 scale, eight bytes of
// the code plane. It is also the widest group a single activation scale can cover, since a 32-column
// activation group is a whole number of them -- so one group contributes one rescale product per
// column and no group ever straddles two activation scales.
constexpr int kGroupCodes = 16;
constexpr int kGroupBytes = kGroupCodes / 2;

// The widest row-sharing factor the kernel is written for. R is not a tile -- it is how many output
// rows one block folds into the *same* activation reads, and on this shape that stream is NC * k
// bytes per row against k/2 bytes of weight, so it is the whole lever on the L1 path.
constexpr int kMaxRowsPerBlock = 4;

// The row-sharing factor this build launches. Two: four halves the activation stream again and does
// raise DRAM throughput, but its register cost drops block residency, and below that occupancy the
// narrow widths lose more than the widest stage gains -- and the six-column stage's sign moves between
// a cold and a warm sweep, so it is not a stage to buy with residency.
constexpr int kRowsPerBlock = 2;

/// The MMVQ-shaped kernel answers this call: llama.cpp's own crossover
/// (`MMVQ_MAX_BATCH_SIZE`), applied to **every** int8 call regardless of which wide mainloop the
/// route table selected. Neither wide mainloop answers T <= 8 -- the row-form one does, so the decode
/// band is a single implementation instead of three, while the wide band stays whatever the table
/// names. This is the one place the band is stated; `mmq::select` refuses through it, so the two
/// cannot drift.
inline bool applies(std::int32_t tokens) { return tokens >= 1 && tokens <= kMaxTokens; }

/// The column count the kernel instantiates for a width: the smallest stage at or above it, so an
/// accumulator indexed by column never leaves registers. The stages are the powers of two plus six,
/// and six is there because the widths that come here are the ones a speculative round verifies at:
/// the per-block activation traffic is NC * k bytes and the profile puts this kernel on the L1 path,
/// so padding a six-column call to eight buys a third again of activation rows the caller asked for
/// and never stored. Staying in the plane is free either way -- the activation buffer is sized for
/// the padded token tile, whose floor is eight columns.
inline int padded_columns(std::int32_t tokens) {
    return tokens <= 1 ? 1 : tokens <= 2 ? 2 : tokens <= 4 ? 4 : tokens <= 6 ? 6 : 8;
}

} // namespace ninfer::ops::detail::mmvq
