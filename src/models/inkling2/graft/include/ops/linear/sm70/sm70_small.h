#pragma once

// Small-T products on sm_70 (Tesla V100): every dense weight format (sm70_dequantize.h) times up
// to the wide route's gate of columns, on the tensor cores and without the emulated Ampere warp
// primitives (sm70_small.cuh). Decode and verification projections of the integer, BF16, FP8 and
// NVFP4 formats take it. Up to kSmallDirectTokens columns the kernel reads the BF16 activations
// and converts them as it loads them, without workspace; wider calls convert them to FP16 once.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/linear/sm70/sm70_project.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops::detail::sm70 {

// Whether a product of `tokens` columns takes the small-T route in this build: from one column
// up to the wide route's gate.
[[nodiscard]] bool small_takes(std::int32_t tokens);

inline constexpr std::int32_t kSmallDirectTokens = 8;

// The FP16 activation plane of a call wider than kSmallDirectTokens; none up to it.
[[nodiscard]] std::size_t small_workspace_bytes(std::int32_t k, std::int32_t tokens);
[[nodiscard]] Activation small_prepare(const Tensor& x, WorkspaceArena& workspace,
                                       cudaStream_t stream);
// small_prepare of a call no wider than kSmallDirectTokens.
[[nodiscard]] Activation small_direct(const Tensor& x);

void small_project(const Activation& x, const Weight& w, std::span<const Destination> destinations,
                   cudaStream_t stream);
void small_swiglu(const Activation& x, const Weight& gate, const Weight* up, Tensor& out,
                  cudaStream_t stream);

// The smallest K a small-T product takes (whole 64-value chunks).
[[nodiscard]] bool small_shape(const Weight& w);

} // namespace ninfer::ops::detail::sm70
