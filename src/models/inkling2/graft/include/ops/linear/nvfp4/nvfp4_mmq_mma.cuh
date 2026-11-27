#pragma once

// NVFP4 weights on the integer tensor cores, second implementation: the mainloop follows ggml's
// MMQ (ggml/src/ggml-cuda/mmq.cuh, mmq-load-tiles.cuh, mmq-vec-dot.cuh) rather than the
// int8-grouped staging of nvfp4_int8_grouped_mma.cuh. Same arithmetic destination -- E2M1 codes
// widened to s8 = 2*e2m1, one NVFP4 G16 group per mma.m16n8k16.s8, the per-16 E4M3 scales
// rescaling the int32 partials in FP32, per-32 FP32 activation scales, FP32 accumulators and one
// BF16 rounding at the store -- so the two routes are interchangeable and PPL-comparable.
//
// What is ported from MMQ, and what each piece costs or buys against the grouped route:
//
//   * I = 128 rows x J tokens per CTA, 256 threads / 8 warps, occupancy 1 -- mmq-config-ampere
//     gives exactly this row for NVFP4 on every Turing+ part. Same CTA as the grouped route.
//   * K_vram = ITER_K = 256: one x tile carries 256 logical k per row and is staged once, against
//     the grouped route's four BK=64 tiles for the same span. Each staging thread therefore owns
//     one whole 64-element NVFP4 block (32 bytes of codes + its 4 scale bytes) and writes it as
//     four STS.128 -- MMQ's natural row order -- instead of eight scattered STS.32.
//   * Two vec_dot calls per x tile, each reading the other half of the x words, separated by two
//     differently sourced activation tiles: MMQ interleaves "stage x -> stage y half 0 -> mma ->
//     stage y half 1 -> mma" inside one loop step, four barriers per 256 k where the grouped route
//     spends two per 64 k.
//   * Aoperands for one 128-k window live in registers across the whole vec_dot (A[ntx][8] of
//     tile<16,4,int>, gathered with ldmatrix from the naturally ordered tile), B values stream one
//     128-k tile at a time in the MMQ block_q8_1_mmq arrangement: four FP32 scales at the head of
//     each token row. The grouped route re-reads A from shared per 64-k tile instead.
//
// Weights are read straight from the ninfer NVFP4 planes -- row-major E2M1 nibbles plus the
// BlockScaleK16M128x4 scale plane -- with no repack: the tile loader below is what changes.
//
// Activations are the grouped route's quantisation -- one FP32 scale per (token, 32 columns), the
// s8 codes the same pass produces -- emitted by a quantiser of this route directly in the tile's
// row order (four scales, then 128 codes per 128-column window), so the staging below is a copy
// instead of a reshape. The layout is produced at quantisation time on a scratch plane the caller
// owns for one launch; nothing persistent is rewritten and the weight planes are never touched.

#include "ops/linear/common/output.cuh"
#include "ops/linear/nvfp4/nvfp4_codec_unified.cuh"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_mma.cuh" // e2m1_word, the E4M3 LUT, mma_s8_k16
#include "ops/linear/nvfp4/nvfp4_mmq_host.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail::mmq {

// The codec helpers (nibble -> s8, the E4M3 table, the scale-plane address) are the ones the
// int8-grouped route already relies on; including its header reuses them instead of restating the
// table. Nothing of its mainloop is instantiated from here.
namespace g = ninfer::ops::detail::unified;
namespace d = ninfer::ops::detail::unified::detail;

// ---------------------------------------------------------------------------
// Activation quantiser for this route: the same per-(token, 32 columns) s8 codes and FP32 scale the
// grouped route computes, written as the tile rows the mainloop consumes -- per token, per
// 128-column window, the window's four FP32 scales (16 B) followed by its 128 s8 codes, i.e. one
// block_q8_1_mmq row of kActivationRowBytes = 144 B. K + K/8 bytes per token: byte-for-byte the
// cost of the two-plane form, so the workspace envelope is unchanged and only the addressing moves.
//
// The plane is window-major -- [window][token][row] -- so the rows one staging pass copies are
// contiguous: every output row block re-reads the same activations, and it is that repeated read
// (112 row blocks over the same 896 tokens on the production geometry), not this one-shot write,
// that decides the layout. One thread still owns one 32-column group -- the same parallelism as the
// two-plane pass. It writes its 32 codes at the group's slot inside the row and its own scale at the
// group's float slot; the four groups of a row are four consecutive threads, so the four scale
// stores coalesce into the row's 16-byte head and no exchange between lanes is needed. The codes and
// the scale come from nvfp4_int8_quantize_group, the helper the two-plane pass uses, so the two
// layouts cannot disagree about a value.
//
// This is activation scratch, produced once per launch and consumed once per launch: it is not the
// runtime weight repack AGENTS.md forbids, and no persistent plane is written.
// ---------------------------------------------------------------------------
static __global__ void quantize_nvfp4_int8_mmq(const __nv_bfloat16* __restrict__ x,
                                               std::int32_t tokens, std::int32_t tile_tokens,
                                               std::int32_t k, std::int8_t* __restrict__ tile) {
    const std::int32_t token = blockIdx.x;
    const std::int32_t groups = k >> 5; // one thread per 32-column group
    const std::size_t row_bytes = kActivationRowBytes;
    // The stride is the padded count: the reader computes it the same way, so writer and mainloop
    // cannot disagree about where a window begins.
    const std::size_t window_bytes = static_cast<std::size_t>(tile_tokens) * row_bytes;
    for (std::int32_t grp = threadIdx.x; grp < groups; grp += blockDim.x) {
        float scale  = 0.0F;
        uint4 packed[2];
        if (token < tokens) {
            const uint4* source = reinterpret_cast<const uint4*>(
                x + static_cast<std::size_t>(token) * k + static_cast<std::size_t>(grp) * 32);
            g::nvfp4_int8_quantize_group(source, scale, packed);
        } else {
            // A padding row of a forced narrow run: the same slot, written zero. e2m1 code 0 is the
            // value 0 and the group scale is 0 too, so the column contributes nothing -- and x is
            // not touched, which is the whole point, since its rows stop at the real token count.
            packed[0] = make_uint4(0u, 0u, 0u, 0u);
            packed[1] = make_uint4(0u, 0u, 0u, 0u);
        }
        const std::int32_t slot = grp & 3; // the group's place inside its 128-column window
        std::int8_t* row        = tile + static_cast<std::size_t>(grp >> 2) * window_bytes +
                           static_cast<std::size_t>(token) * row_bytes;
        reinterpret_cast<float*>(row)[slot] = scale;
        auto* codes = reinterpret_cast<uint4*>(row + kActivationScaleBytes);
        codes[slot * 2 + 0] = packed[0];
        codes[slot * 2 + 1] = packed[1];
    }
}

__device__ __forceinline__ void mmq_ldmatrix_x4(unsigned (&out)[4], const int* base, int stride,
                                                unsigned lane) {
    // Threads 0-7/8-15/16-23/24-31 supply the eight row addresses of the four 8x8 b16 matrices,
    // which here are the four column blocks of a 16-row x 32-byte window of tile_x.
    const int* p = base + (lane % 16) * stride + (lane / 16) * 4;
#if defined(NINFER_SM70_DEVICE)
    sm70::ldmatrix<4>(out, static_cast<unsigned>(__cvta_generic_to_shared(p)));
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "l"(p));
#endif
}

// One block covers kRows output rows x J tokens; J must be whole and every shape predicate is the
// caller's (see nvfp4_mmq_host.h). `Output` is the same adapter family the int8-grouped mainloop
// takes, so every projection route reuses its epilogue unchanged.
//
// Two token counts, deliberately: `tokens` is the PADDED width the launch addresses -- the plane is
// window-major [window][token][row], so the stride the reader computes has to be the stride the
// writer used -- while `valid_tokens` is what the caller asked for, and only those columns are
// stored. The grouped mainloop needs no such split because its planes are token-major with a
// k-sized stride, independent of how many tokens a call carries.
template <int J, class Output>
__global__ __launch_bounds__(kThreads) void nvfp4_mmq_kernel(
    const std::uint8_t* __restrict__ weight_codes, const std::uint8_t* __restrict__ weight_scales,
    const std::int8_t* __restrict__ x_tile, float alpha_half, Output output, std::int32_t k,
    std::int32_t tokens, std::int32_t valid_tokens) {
    constexpr int I       = kRows;
    constexpr int TYK     = kTileYK;
    constexpr int SS      = kSramStride;
    constexpr int rows_warp = rows_per_warp(J);
    constexpr int NTX     = ntx(J);
    constexpr int JSTEPS  = j_steps(J);
    static_assert(I == (kThreads / 32 / NTX) * rows_warp, "eight warps must cover the row tile");
    static_assert(JSTEPS * NTX * 8 == J, "J must tile into warp-wide tile_C columns");

    extern __shared__ __align__(16) int smem[];
    int* tile_y = smem;                 // [J][TYK]: 4 FP32 scales then 32 words of codes
    int* tile_x = tile_y + J * TYK;     // [I][SS]: 64 words of codes then 16 FP32 scales

    const unsigned tid   = threadIdx.x;
    const unsigned lane  = tid & 31u;
    const int      warp   = static_cast<int>(tid >> 5);
    const int      lane_i = static_cast<int>(lane >> 2); // row within a 16-row tile
    const int      lane_k = static_cast<int>(lane & 3u); // word within a 16-byte k chunk

    // Same transpose as the grouped mainloop's: the J token tiles of ONE 128-row weight tile are
    // consecutive blocks, so the plane those blocks share is fetched once into L2 instead of once
    // per tile column (8x at T=1024). Row/column arithmetic per block is untouched.
    const int row_block = static_cast<int>(blockIdx.y) * I;
    const int col_block = static_cast<int>(blockIdx.x) * J;
    const int row_base  = (warp / NTX) * rows_warp;      // this warp's 16/32-row band
    const int col_base  = (warp % NTX) * 8;              // this warp's 8-token slice of the pair

    const auto epilogue = linear_output_tile<I>(output, row_block);

    const int code_row_bytes = k / 2;
    const int groups         = k / kIterK; // whole ITER_K tiles

    // Activation staging is a flat copy: the quantiser emitted the plane window-major, so the J
    // rows of one 128-k window are one contiguous run of J * 144 bytes in exactly the order tile_y
    // wants them -- four FP32 scales then the 128 s8 codes per token. No gather, no second plane,
    // no index arithmetic: one 16-byte store per word down a linear run, and cp.async names no
    // register, so the copy leaves the load/store pipe as one request instead of a dependent
    // load-then-store pair, and it stays in flight while the weight tile is decoded.
    constexpr int kRowWords = kTileYK / 4; // 9 uint4 per tile row
    const std::size_t y_window_bytes =
        static_cast<std::size_t>(tokens) * static_cast<std::size_t>(kActivationRowBytes);
    const std::int8_t* y_window0 =
        x_tile + static_cast<std::size_t>(col_block) * kActivationRowBytes;
    const auto stage_y = [&](int kbase) {
        const uint4* src = reinterpret_cast<const uint4*>(
            y_window0 + static_cast<std::size_t>(kbase / 128) * y_window_bytes);
        for (int idx = static_cast<int>(tid); idx < J * kRowWords; idx += kThreads) {
            cp_async<16, Cache::cg>(reinterpret_cast<uint4*>(tile_y + idx * 4), src + idx);
        }
        cp_commit();
    };

    // Weight staging, one whole ITER_K span per pass: each thread owns one 64-element NVFP4 block
    // of one row and lays it down in natural k order so ldmatrix can lift the A operand directly.
    //
    // Split into a load and a store so the loads for the next span can be issued while the current
    // span is still on the MMA pipes: the pass mapping is identical, and the buffer holds values
    // this thread will consume itself, so no barrier is involved in the handoff.
    constexpr int kPassStride = kRowsPerWarp * kNwarps;
    constexpr int kXPasses    = I / kPassStride;
    static_assert(I % kPassStride == 0, "staging sweep must cover the whole row tile");
    struct XChunk {
        uint4 w0, w1;
        unsigned ws;
    };
    const int kbx         = static_cast<int>(lane % kThreadsPerRow);
    const int row_in_warp = static_cast<int>(lane / kThreadsPerRow);
    const auto load_x = [&](int kt, XChunk (&buf)[kXPasses]) {
#pragma unroll
        for (int p = 0; p < kXPasses; ++p) {
            const int i    = p * kPassStride + warp * kRowsPerWarp + row_in_warp;
            const int grow = row_block + i;
            const std::uint8_t* src =
                weight_codes + static_cast<std::size_t>(grow) * code_row_bytes + kt * 128 + kbx * 32;
            buf[p].w0 = *reinterpret_cast<const uint4*>(src);
            buf[p].w1 = *reinterpret_cast<const uint4*>(src + 16);
            buf[p].ws = *reinterpret_cast<const unsigned*>(
                weight_scales + g::nvfp4_scale_byte_offset(grow, (kt * 4 + kbx) * 4, k));
        }
    };
    const auto store_x = [&](const XChunk (&buf)[kXPasses]) {
#pragma unroll
        for (int p = 0; p < kXPasses; ++p) {
            const int i = p * kPassStride + warp * kRowsPerWarp + row_in_warp;
            unsigned q[16];
            g::e2m1_word(buf[p].w0.x, q[0], q[1]);
            g::e2m1_word(buf[p].w0.y, q[2], q[3]);
            g::e2m1_word(buf[p].w0.z, q[4], q[5]);
            g::e2m1_word(buf[p].w0.w, q[6], q[7]);
            g::e2m1_word(buf[p].w1.x, q[8], q[9]);
            g::e2m1_word(buf[p].w1.y, q[10], q[11]);
            g::e2m1_word(buf[p].w1.z, q[12], q[13]);
            g::e2m1_word(buf[p].w1.w, q[14], q[15]);
            int* dst = tile_x + i * SS + kbx * 16;
            reinterpret_cast<uint4*>(dst)[0] = make_uint4(q[0], q[1], q[2], q[3]);
            reinterpret_cast<uint4*>(dst)[1] = make_uint4(q[4], q[5], q[6], q[7]);
            reinterpret_cast<uint4*>(dst)[2] = make_uint4(q[8], q[9], q[10], q[11]);
            reinterpret_cast<uint4*>(dst)[3] = make_uint4(q[12], q[13], q[14], q[15]);
            float4 decoded;
            decoded.x = g::kNvfp4E4M3ScaleLut[buf[p].ws & 0xFFu];
            decoded.y = g::kNvfp4E4M3ScaleLut[(buf[p].ws >> 8) & 0xFFu];
            decoded.z = g::kNvfp4E4M3ScaleLut[(buf[p].ws >> 16) & 0xFFu];
            decoded.w = g::kNvfp4E4M3ScaleLut[buf[p].ws >> 24];
            *reinterpret_cast<float4*>(tile_x + i * SS + kTileNeK * 2 + kbx * 4) = decoded;
        }
    };

    float sum[NTX][JSTEPS][4];
#pragma unroll
    for (int n = 0; n < NTX; ++n)
#pragma unroll
        for (int t = 0; t < JSTEPS; ++t)
#pragma unroll
            for (int l = 0; l < 4; ++l) sum[n][t][l] = 0.0F;

    // One vec_dot over a kTileNeK-word window of the x tile (k00 == 0 the low half of ITER_K,
    // k00 == kTileNeK the high half) against the freshly staged activation tile.
    const auto vec_dot = [&](int k00) {
        unsigned aval[NTX][8][2];      // [n][16-byte k chunk][low row / high row]
        float    aval_s[NTX][2][8];    // [n][low/high row][chunk]
#pragma unroll
        for (int n = 0; n < NTX; ++n) {
            const int rb = row_base + n * 16;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                unsigned regs[4];
                mmq_ldmatrix_x4(regs, tile_x + rb * SS + k00 + c * 8, SS, lane);
                aval[n][2 * c + 0][0] = regs[0];
                aval[n][2 * c + 0][1] = regs[1];
                aval[n][2 * c + 1][0] = regs[2];
                aval[n][2 * c + 1][1] = regs[3];
            }
            const float* dlo =
                reinterpret_cast<const float*>(tile_x + (rb + lane_i) * SS + kTileNeK * 2);
            const float* dhi =
                reinterpret_cast<const float*>(tile_x + (rb + lane_i + 8) * SS + kTileNeK * 2);
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                aval_s[n][0][t] = dlo[k00 / 4 + t];
                aval_s[n][1][t] = dhi[k00 / 4 + t];
            }
        }
        // The k step is the outer loop and the token tile the inner one, deliberately: the JSTEPS
        // accumulators a thread owns for one n are independent registers, so walking them between
        // two fmas hides the fixed latency the jt-outer order exposes as a serial chain on
        // sum[n][jt]. Per accumulator the additions arrive in ascending k, which is what makes the
        // interchange bit-exact rather than a reassociation.
#pragma unroll
        for (int k01 = 0; k01 < kTileNeK; k01 += 8) {
#pragma unroll
            for (int jt = 0; jt < JSTEPS; ++jt) {
                // The B values a lane supplies sit on the activation row of its own column (lane_i of
                // the 8-column tile); the per-32 scale a result needs sits on the row of the token the
                // result is stored to, which for register l is the even/odd pair lane_k*2 + (l & 1) --
                // two different rows, so two scales per k-step pair, exactly MMQ's dB[l%2].
                const int* yrow = tile_y + (col_base + jt * NTX * 8 + lane_i) * TYK;
                const float* dcol = reinterpret_cast<const float*>(
                    tile_y + (col_base + jt * NTX * 8 + lane_k * 2) * TYK);
                const unsigned b0 = yrow[4 + k01 + lane_k];
                const unsigned b1 = yrow[4 + k01 + 4 + lane_k];
                const float    db[2] = {dcol[k01 >> 3], dcol[TYK + (k01 >> 3)]};
#pragma unroll
                for (int n = 0; n < NTX; ++n) {
                    // Register l of a tile_C sits on row lane_i + 8*(l >> 1) -- so the weight (row)
                    // scale is dA[l/2] -- and on token 2*lane_k + (l & 1) -- so the activation
                    // scale is dB[l%2]. Exactly ggml's
                    //   sum += dB[l%2]*(C0*dA[n][l/2][k01/4] + C1*dA[n][l/2][k01/4+1]).
                    // The four accumulators are named, never indexed through &c0: taking their
                    // address would push the tile_C out of registers into local memory.
                    const int m = k01 / 4;
                    const float s0 = aval_s[n][0][m] * db[0];
                    const float s1 = aval_s[n][0][m] * db[1];
                    const float s2 = aval_s[n][1][m] * db[0];
                    const float s3 = aval_s[n][1][m] * db[1];
                    const float t0 = aval_s[n][0][m + 1] * db[0];
                    const float t1 = aval_s[n][0][m + 1] * db[1];
                    const float t2 = aval_s[n][1][m + 1] * db[0];
                    const float t3 = aval_s[n][1][m + 1] * db[1];
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                    int e0 = 0, e1 = 0, e2 = 0, e3 = 0;
                    d::mma_s8_k16(c0, c1, c2, c3, aval[n][m][0], aval[n][m][1], b0);
                    d::mma_s8_k16(e0, e1, e2, e3, aval[n][m + 1][0], aval[n][m + 1][1], b1);
                    sum[n][jt][0] = fmaf(static_cast<float>(c0), s0, sum[n][jt][0]);
                    sum[n][jt][1] = fmaf(static_cast<float>(c1), s1, sum[n][jt][1]);
                    sum[n][jt][2] = fmaf(static_cast<float>(c2), s2, sum[n][jt][2]);
                    sum[n][jt][3] = fmaf(static_cast<float>(c3), s3, sum[n][jt][3]);
                    sum[n][jt][0] = fmaf(static_cast<float>(e0), t0, sum[n][jt][0]);
                    sum[n][jt][1] = fmaf(static_cast<float>(e1), t1, sum[n][jt][1]);
                    sum[n][jt][2] = fmaf(static_cast<float>(e2), t2, sum[n][jt][2]);
                    sum[n][jt][3] = fmaf(static_cast<float>(e3), t3, sum[n][jt][3]);
                }
            }
        }
    };

    // The activation copy is issued FIRST so its global latency overlaps the weight decode, and it
    // is waited on only after that decode: wait_group 0 is per-thread bookkeeping for its own
    // copies, and the barrier that follows is the same publication point the load/store form used,
    // so no thread reads a byte another thread has not published.
    //
    // On top of that the span's weight loads are hoisted a whole iteration back (P2a′): xb holds
    // span kt, which is stored while span kt-1's MMAs are done, and span kt+1's load is issued right
    // after that store, so its global latency is covered by this span's tensor-core work.
    XChunk xb[kXPasses];
    load_x(0, xb);
    for (int kt = 0; kt < groups; ++kt) {
        stage_y(kt * kIterK);
        store_x(xb);
        if (kt + 1 < groups) { load_x(kt + 1, xb); }
        cp_wait<0>();
        __syncthreads();
        vec_dot(0);
        __syncthreads();
        stage_y(kt * kIterK + 128);
        cp_wait<0>();
        __syncthreads();
        vec_dot(kTileNeK);
        __syncthreads();
    }

    // Accumulator layout carries over from the int8-grouped mainloop verbatim: register l of a
    // tile_C sits on row lane_i + 8*(l >> 1) of its 16-row tile and token 2*lane_k + (l & 1) of
    // its 8-token tile, which is also how MMQ's write_back_mma indexes get_i/get_j -- so the
    // epilogue adapter is shared across both routes without change.
    //
    // The padding columns of a forced narrow run are computed and discarded here, exactly as in the
    // grouped mainloop, and for the same two reasons: linear_add's store reads back the residual it
    // is about to update, and the swiglu gate plane would hand uninitialised columns to the up
    // projection's silu. t0 rises with jt for a fixed thread, so one break retires the band.
#pragma unroll
    for (int n = 0; n < NTX; ++n) {
        const int r0 = row_base + n * 16 + lane_i;
#pragma unroll
        for (int jt = 0; jt < JSTEPS; ++jt) {
            const int t0 = col_block + col_base + jt * NTX * 8 + lane_k * 2;
            if (t0 >= valid_tokens) { break; }
            epilogue.store(row_block + r0, t0, sum[n][jt][0] * alpha_half);
            epilogue.store(row_block + r0 + 8, t0, sum[n][jt][2] * alpha_half);
            if (t0 + 1 < valid_tokens) {
                epilogue.store(row_block + r0, t0 + 1, sum[n][jt][1] * alpha_half);
                epilogue.store(row_block + r0 + 8, t0 + 1, sum[n][jt][3] * alpha_half);
            }
        }
    }
}

} // namespace ninfer::ops::detail::mmq
