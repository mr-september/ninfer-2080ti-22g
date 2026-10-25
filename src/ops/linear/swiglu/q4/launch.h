#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Turing: single-stage 128-token folded tiles, two resident per SM.
void launch_q4_swiglu_mma_turing_prefill(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void launch_q4_swiglu_small_t_tiled(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream);

void launch_q4_swiglu_materialized_projection(const Tensor&, const Weight&, Tensor&, cudaStream_t);
} // namespace ninfer::ops::detail
