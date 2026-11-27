#pragma once

// The integer small-T route of an sm_70 build: T2 weights times int8 activations by dp4a, at the
// decode and verify widths where the other targets' T2 route quantizes the activations too
// (t2_a8.h, under the same LinearPolicy). Volta has no integer tensor cores. At one or two columns
// its FP16 tensor-core route (sm70_small.cuh) spends most of its instructions widening codes to
// FP16 and on HMMA steps for columns it does not use; dp4a takes four ternary products per
// instruction and leaves the projection bound by the weight stream. Up to eight columns (a verify
// round of up to seven drafts) dp4a still beats that route, by 10-35% (sm70_integer.cu).
//
// The activations are quantized once per call into the workspace, one FP16-rounded scale per
// (column, 64-value group) as t2_small_t_i8_quantize_kernel does, with the codes of every
// sixteen-value block stored in the byte order of a T2 code word's 2-bit fields
// (integer_prepare).

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear_policy.h"
#include "ops/linear/sm70/sm70_project.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops::detail::sm70 {

inline constexpr std::int32_t kIntegerMaxTokens = 8;

// Whether a call of `tokens` columns over weights of this format and input width, under `policy`,
// takes the integer route in this build.
[[nodiscard]] bool integer_format(QType format, std::int32_t k, std::int32_t tokens,
                                  LinearPolicy policy);
[[nodiscard]] bool integer_takes(const Weight& w, std::int32_t tokens, LinearPolicy policy);

// The quantized activations of `tokens` columns of K values.
[[nodiscard]] std::size_t integer_bytes(std::int32_t k, std::int32_t tokens);

// Quantizes x (BF16 [K, T], T <= kIntegerMaxTokens) into the workspace and records it in `x_out`.
void integer_prepare(const Tensor& x, WorkspaceArena& workspace, cudaStream_t stream,
                     Activation& x_out);

void integer_project(const Activation& x, const Weight& w,
                     std::span<const Destination> destinations, cudaStream_t stream);
void integer_swiglu(const Activation& x, const Weight& gate, const Weight* up, Tensor& out,
                    cudaStream_t stream);

} // namespace ninfer::ops::detail::sm70
