#pragma once

// Launch front end for the MMQ-structured NVFP4 mainloop (nvfp4_mmq_mma.cuh). One header so every
// projection route -- Linear, LinearAdd, both halves of gate_up, attn/gdn input projections --
// switches with the same three lines and the same predicate.
//
// The mainloop comes from the route table, not from a name: a call the own (legacy) table serves
// runs MMQ, a call the unified table serves runs the grouped ladder -- see nvfp4_int8_route.h.
// `select` is the entry point a route calls; it refuses on K rather than falling back, because a
// weight plane cannot be padded the way an activation tile can.
//
// Both mainloops quantise the same activations and the two layouts cost the same bytes per token
// (K + K/8), so a switch costs no extra scratch; only the quantiser that fills the plane differs.

#include "ops/linear/nvfp4/nvfp4_int8_route.h"
#include "ops/linear/nvfp4/nvfp4_mmq_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_mmvq_host.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::mmq {

// Whether this call runs the MMQ mainloop. It serves every token count **above the crossover** by
// padding its last J-tile, so only K can refuse it -- and K has nothing to pad against, so a
// geometry that fails on K is an error rather than a quiet fallback to the grouped kernel. A16
// never reaches either int8 mainloop; the family is registered or it is not.
//
// The narrow band belongs to the int8 family rather than to a mainloop: the MMVQ-shaped kernel
// takes T <= 8 whichever wide mainloop the table selected, on the two-plane activation form, so
// those widths answer false here. The refusal goes through `mmvq::applies` rather than a copy of
// its test, because a width claimed by both would make the site allocate the interleaved plane for
// a kernel that reads two.
inline bool select(std::int32_t k, std::int32_t tokens, Nvfp4Int8Mainloop table) {
    if (nvfp4_int8_mainloop(table) != Nvfp4Int8Mainloop::Mmq) { return false; }
    if (mmvq::applies(tokens)) { return false; }
    if (!k_admitted(k)) {
        throw std::invalid_argument(
            "nvfp4 mmq mainloop: K=" + std::to_string(k) +
            " is not a whole ITER_K=256 partition; the grouped mainloop has no such requirement and "
            "a weight plane cannot be padded the way an activation tile can");
    }
    return true;
}

// Quantise the activations into this route's plane: per token, per 128-column window, four FP32
// scales then 128 s8 codes (see quantize_nvfp4_int8_mmq). Callers that fail applies() must keep the
// two-plane form and the grouped mainloop.
inline void quantize_activations(const void* x, std::int32_t tokens, std::int32_t k, void* tile,
                                 cudaStream_t stream) {
    // One block per token of the PADDED width: the surplus rows are zeroed, not skipped, because
    // the mainloop reads every row of the tile it is given.
    quantize_nvfp4_int8_mmq<<<unified::nvfp4_int8_padded_tokens(tokens), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), tokens, unified::nvfp4_int8_padded_tokens(tokens), k,
        static_cast<std::int8_t*>(tile));
}

// One block per (128 output rows, J tokens). J is not fixed at 128: it is the stage of the token
// ladder the width selects, and both the grid and the plane's stride follow it -- which is
// the difference between "small J buys less tensor-core work" and "small J re-reads the weights once
// per tile column". Only a simultaneous change to the pad basis buys the former; see
// nvfp4_int8_select_j.
template <int J, class Output>
inline void launch_j(const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                     const std::int8_t* x_tile, float alpha_half, Output output, std::int32_t k,
                     std::int32_t tokens, std::int32_t rows, cudaStream_t stream) {
    static const bool smem_opted_in = [] {
        cudaFuncSetAttribute(reinterpret_cast<const void*>(&nvfp4_mmq_kernel<J, Output>),
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem_bytes(J)));
        return true;
    }();
    (void)smem_opted_in;
    // The grid and the plane layout run over the padded width; the kernel stores only the tokens
    // the caller asked for. Passing one number for both is what would make a narrow forced run
    // either write out of range or silently drop columns.
    const std::int32_t tile_tokens = unified::nvfp4_int8_padded_tokens(tokens);
    // Token tiles on the fast axis, row tiles on the slow one: see the kernel's block-index note.
    const dim3 grid(tile_tokens / J, rows / kRows);
    nvfp4_mmq_kernel<J, Output><<<grid, kThreads, smem_bytes(J), stream>>>(
        weight_codes, weight_scales, x_tile, alpha_half, output, k, tile_tokens, tokens);
}

// The entry point a route calls: same ladder, same widths, same result, whichever stage the width
// selects.
template <class Output>
inline void launch(const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                   const std::int8_t* x_tile, float alpha_half, Output output, std::int32_t k,
                   std::int32_t tokens, std::int32_t rows, cudaStream_t stream) {
#define NINFER_MMQ_LAUNCH_STAGE(STAGE)                                                    \
    case STAGE:                                                                           \
        launch_j<STAGE, Output>(weight_codes, weight_scales, x_tile, alpha_half, output, k, \
                                tokens, rows, stream);                                    \
        return;
    switch (unified::nvfp4_int8_select_j(tokens)) {
        NINFER_MMQ_LAUNCH_STAGE(8)
        NINFER_MMQ_LAUNCH_STAGE(16)
        NINFER_MMQ_LAUNCH_STAGE(24)
        NINFER_MMQ_LAUNCH_STAGE(32)
        NINFER_MMQ_LAUNCH_STAGE(40)
        NINFER_MMQ_LAUNCH_STAGE(48)
        NINFER_MMQ_LAUNCH_STAGE(64)
        NINFER_MMQ_LAUNCH_STAGE(80)
        NINFER_MMQ_LAUNCH_STAGE(96)
        NINFER_MMQ_LAUNCH_STAGE(112)
        NINFER_MMQ_LAUNCH_STAGE(128)
        default:
            break;
    }
#undef NINFER_MMQ_LAUNCH_STAGE
    throw std::invalid_argument("nvfp4 mmq: no admissible J tile for T=" + std::to_string(tokens));
}

} // namespace ninfer::ops::detail::mmq
