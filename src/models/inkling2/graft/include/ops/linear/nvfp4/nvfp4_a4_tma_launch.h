#pragma once
#include "ops/linear/nvfp4/nvfp4_operands.h"

namespace ninfer::ops::detail::unified {
void launch_nvfp4_a4_tma_linear(Nvfp4GeometryId problem, const Nvfp4A4Operands& p,
                                __nv_bfloat16* output, cudaStream_t stream);

// A run-time shape, such as a tensor-parallel rank's share of a projection: rows and K from the
// operands, the token tile from their activation-scale layout, into BF16 or an FP32 partial sum.
void launch_nvfp4_a4_tma_general(const Nvfp4A4Operands& p, const Tensor& out, cudaStream_t stream);
} // namespace ninfer::ops::detail::unified
