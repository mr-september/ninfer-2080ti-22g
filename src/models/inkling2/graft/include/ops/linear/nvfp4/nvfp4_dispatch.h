#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear_policy.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                                std::int32_t input_rows,
                                                                LinearPolicy policy,
                                                                std::int32_t min_tokens,
                                                                std::int32_t max_tokens);

// Whether a table names [N,K]. Every other shape, and every FP32 result, runs a runtime-shape route
// over whole tiles of `kNvfp4ShareUnits`: A4 (nvfp4_general_a4.cu) for a shape no table names where
// the policy allows it and nvfp4_general_uses_a4 holds, A16 (nvfp4_general.cu) otherwise. A
// registered shape's FP32 result stays A16, so its table's workspace covers every call.
[[nodiscard]] bool nvfp4_registered(std::int32_t output_rows, std::int32_t input_rows);
inline constexpr LinearShareUnits kNvfp4ShareUnits{128, 256};
void nvfp4_general_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

// The runtime-shape A4 route's K is two 256-column tiles at least, its MMA schedules' stages.
inline constexpr std::int32_t kNvfp4GeneralA4MinK = 512;
// Up to 8 tokens the A16 route ties or beats A4 on three of the four shares measured on the RTX
// 5090 (bench/ops/linear_bench.cu over Qwen3.8-27B's MLP at two and four ranks) and trails it by
// one 2 us timer tick on the fourth. At 9 tokens A4 is 2 to 6 us faster on three of them and ties
// on the fourth, the four-rank down projection, which it passes at 16.
inline constexpr std::int32_t kNvfp4GeneralA4MinTokens = 9;

[[nodiscard]] constexpr bool nvfp4_general_uses_a4(std::int32_t input_rows, std::int32_t tokens) {
    return input_rows >= kNvfp4GeneralA4MinK && tokens >= kNvfp4GeneralA4MinTokens;
}

void nvfp4_general_a4(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
                      cudaStream_t stream);

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
