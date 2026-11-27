#pragma once

// Host-facing entry points for the fused [RMSNorm -> int8 quantise] edge of the NVFP4 int8 chain.
//
// The device half lives in nvfp4_int8_fused_norm_quant.cuh; this header exists so a host
// translation unit (a route launcher, a test) can call the pass without pulling warp intrinsics
// into a C++ compile. See the .cuh for the bit-for-bit contract this preserves.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::unified {

/// Fill the separated int8 activation planes (codes, scales) with the RMSNorm of `residual`
/// quantised per 32 columns, in one launch. `addressed` is the width the consuming mainloop
/// addresses -- the padded token tile above the crossover, the call's own width in the narrow band
/// -- because a plane sized for columns the quantiser never filled is read back uninitialised.
/// Requires `residual.ne[0] == 5120` and rows/planes sized for `addressed`.
void launch_rmsnorm_int8_quantize(const Tensor& residual, const Tensor& norm_weight, float eps,
                                  std::int8_t* codes, float* scales, std::int32_t addressed,
                                  cudaStream_t stream);

/// The GDN output projection's edge: a per-head SiLU-gated RMSNorm (the shape
/// `gated_rmsnorm` produces for the 27B: 48 head rows of 128) followed by the same per-32 int8
/// quantise, in one launch, without the BF16 `on` plane ever going to global memory. Same
/// pass above. Instantiated for the (6144, 128) geometry only; anything else throws rather than
/// routing through a kernel that was not sized for it. `source` and `gate` may be the model layer's
/// 3-D per-head view {128, 48, T} or the flat {6144, T}; either way their memory must be the token's
/// K-wide row, so the caller names the token count instead of it being read off `ne[1]`.
void launch_gated_rmsnorm_int8_quantize(const Tensor& source, const Tensor& norm_weight,
                                        const Tensor& gate, float eps, std::int8_t* codes,
                                        float* scales, std::int32_t tokens, std::int32_t addressed,
                                        cudaStream_t stream);

/// The standalone per-32 quantiser over an already-normalised BF16 row, host-callable. This is the
/// second half of the two launches the fused pass replaces, kept reachable so the fused route can
/// be byte-compared against the composition it is claiming to equal.
void launch_nvfp4_int8_quantize32(const void* x, std::int32_t tokens, std::int32_t k,
                                  std::int8_t* codes, float* scales, std::int32_t addressed,
                                  cudaStream_t stream);

} // namespace ninfer::ops::detail::unified
