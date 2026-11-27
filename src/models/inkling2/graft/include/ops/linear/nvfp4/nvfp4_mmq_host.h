#pragma once

// Host-side geometry for the NVFP4 route whose mainloop follows ggml's MMQ instead of the
// int8-grouped staging (see nvfp4_mmq_mma.cuh for the device side and the mapping onto mmq.cuh).
// Mirrors the Ampere NVFP4 row of mmq-config-ampere.cuh: 256 threads / 8 warps at occupancy 1,
// I = 128 rows of src0 (our output rows), J = tokens per block, K_vram = MMQ_ITER_K = 256.
//
// The activation plane this route consumes is the int8-grouped route's quantisation in a different
// order: per token, per 128-column window, the window's four FP32 scales first (one per 32-column
// group) and then its 128 s8 codes -- one block_q8_1_mmq row, 144 bytes. It costs K + K/8 bytes
// per token, byte-for-byte what the two-plane form costs, so plan, dispatch and the workspace
// envelope are shared with the grouped route; only the quantiser's addressing differs.

#include "ops/linear/nvfp4/nvfp4_int8_grouped_host.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::mmq {

constexpr int kThreads     = 256; // MMQ_NWARPS * warp size
constexpr int kRows        = 128; // I: rows of the weight per block
constexpr int kIterK       = 256; // MMQ_ITER_K: logical k per x tile
constexpr int kTileNeK     = 32;  // MMQ_TILE_NE_K: int words per half of the x tile
constexpr int kTileYK      = kTileNeK + 4;            // MMQ_TILE_Y_K: 4 scales then 32 words
constexpr int kSramStride  = 2 * kTileNeK + kTileNeK / 2 + 4; // 84 ints, NVFP4 SRAM layout
constexpr int kThreadsPerRow = kIterK / 64;           // 4 staging threads per weight row
constexpr int kRowsPerWarp = 32 / kThreadsPerRow;     // 8 rows per warp during x staging
constexpr int kNwarps      = kThreads / 32;

// ggml's rows_per_warp: J >= 48 && J % 16 == 0 ? 32 : 16, and ntx = rows_per_warp / 16. Both keep
// eight warps over exactly the block's 128 rows and every warp's J slice a whole tile_C wide.
__host__ __device__ constexpr int rows_per_warp(int j) { return j >= 48 && j % 16 == 0 ? 32 : 16; }
__host__ __device__ constexpr int ntx(int j) { return rows_per_warp(j) / 16; }
__host__ __device__ constexpr int j_steps(int j) { return j / (ntx(j) * 8); }

constexpr bool j_admitted(int j) { return j >= 8 && j <= 128 && j % 8 == 0; }

// One activation tile row: the 128-column window's four FP32 scales, then its 128 s8 codes. Equal
// to kTileYK * sizeof(int) -- the row the mainloop's tile_y holds -- and a multiple of 16, so every
// row of every token is 16-byte aligned in the workspace.
constexpr int kActivationScaleBytes = 4 * static_cast<int>(sizeof(float));
constexpr int kActivationRowBytes   = kActivationScaleBytes + 128;
static_assert(kActivationRowBytes == kTileYK * static_cast<int>(sizeof(int)),
              "an activation row is one tile_y row");

// The route admits a call only when the whole K partition lands on ITER_K tiles. The token count
// need not land on a whole J tile: the launch pads one and discards the surplus columns. K is a
// weight-plane property, with nothing to pad against.
//
// Bytes the interleaved activation plane needs, sized for the PADDED token count -- the layout is
// window-major, [window][token][row], so the stride itself depends on how many tokens a launch
// addresses, and the writer and the reader have to agree on the padded number. Never more than the
// two-plane envelope the grouped route reserves for the same token count (which rounds up the same
// way), so a route switch cannot outgrow the workspace the plan already declared.
constexpr std::size_t activation_bytes(std::int32_t tokens, std::int32_t k) {
    return static_cast<std::size_t>(
               ninfer::ops::detail::unified::nvfp4_int8_padded_tokens(tokens)) *
           static_cast<std::size_t>(k) / 128 * static_cast<std::size_t>(kActivationRowBytes);
}

// Shared budget: a J-row activation tile plus the 128-row x tile; withJ = 128 it is 60 KB, which
// needs the opt-in dynamic limit (sm_86 allows 100 KB), so every launcher sets that attribute.
constexpr std::size_t smem_bytes(int j) {
    return (static_cast<std::size_t>(j) * kTileYK + static_cast<std::size_t>(kRows) * kSramStride) *
           sizeof(int);
}

constexpr bool k_admitted(std::int32_t k) { return k > 0 && k % kIterK == 0; }

} // namespace ninfer::ops::detail::mmq
