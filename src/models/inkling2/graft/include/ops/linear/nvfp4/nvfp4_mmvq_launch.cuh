#pragma once

// Launch front end for the MMVQ-shaped NVFP4 mainloop (nvfp4_mmvq_mma.cuh). Width, and only width,
// decides whether a call comes here -- llama.cpp's own crossover: `ggml_cuda_should_use_mmvq` answers
// for ncols_dst <= 8 and everything wider is MMQ's or grouped's business. The unified chain answers
// the same band with its own narrow kernel (nvfp4_int8_narrow_launch.cuh) at the call's exact width;
// this port is the legacy chain's answer, which is the shape its bands were calibrated against.
//
// The activation planes this kernel reads are the grouped route's two -- s8 codes [T][K] plus one
// FP32 scale per (token, 32 columns) -- so unlike mmq it needs no separate quantisation pass and no
// separate workspace accounting: whatever the grouped mainloop would have been handed is what this
// kernel takes.

#include "ops/linear/nvfp4/nvfp4_int8_route.h"
#include "ops/linear/nvfp4/nvfp4_mmvq_host.h"
#include "ops/linear/nvfp4/nvfp4_mmvq_mma.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::mmvq {

/// Whether this call answered by the MMVQ-shaped kernel. Route name and width, nothing else: the k
/// constraints are this route's existing 64-column tile and are enforced upstream by the same
/// alignment gate every NVFP4 route goes through.
inline bool select(std::int32_t tokens) { return applies(tokens); }

template <int NC, int R, class Output>
inline void launch_nc(const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                      const std::int8_t* x_codes, const float* x_scales, float alpha_half,
                      Output output, std::int32_t k, std::int32_t tokens, std::int32_t rows,
                      cudaStream_t stream) {
    // Rows are never ragged here: every NVFP4 route admits a weight with n % 128 == 0, so a block
    // taking R consecutive rows always finds R rows to take.
    nvfp4_mmvq_kernel<NC, R, Output><<<rows / R, kThreads, 0, stream>>>(
        weight_codes, weight_scales, x_codes, x_scales, alpha_half, output, k, tokens);
}

/// R output rows per CUDA block, NC the smallest instantiated column stage at or above the caller's
/// width. The stage set is the powers of two plus six (see `padded_columns`): NC indexes the
/// accumulator array, so every additional stage is another full instantiation of the mainloop per
/// projection route.
template <class Output>
inline void launch(const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                   const std::int8_t* x_codes, const float* x_scales, float alpha_half,
                   Output output, std::int32_t k, std::int32_t tokens, std::int32_t rows,
                   cudaStream_t stream) {
    switch (padded_columns(tokens)) {
        case 1:
            launch_nc<1, kRowsPerBlock, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                alpha_half, output, k, tokens, rows, stream);
            return;
        case 2:
            launch_nc<2, kRowsPerBlock, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                alpha_half, output, k, tokens, rows, stream);
            return;
        case 4:
            launch_nc<4, kRowsPerBlock, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                 alpha_half, output, k, tokens, rows, stream);
            return;
        case 6:
            launch_nc<6, kRowsPerBlock, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                 alpha_half, output, k, tokens, rows, stream);
            return;
        // NC=8 is the one stage where the extra load sharing pays: it is the widest column array, so
        // the activation re-read that R normally costs is amortised over eight columns instead of a
        // narrower tile's few, and the mio_throttle the T=7/T=8 band sits on drops. Every stage below
        // eight was measured to LOSE from R=4 (see EXECUTION-LOG R3-4), so they keep the default two.
        case 8:
            launch_nc<8, 4, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                    alpha_half, output, k, tokens, rows, stream);
            return;
        default:
            break;
    }
    throw std::invalid_argument("nvfp4 mmvq: no column stage for T=" + std::to_string(tokens));
}

} // namespace ninfer::ops::detail::mmvq
