#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear_policy.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              LinearPolicy policy,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

// Whether a table names [N,K]. Every other shape, and an FP32 result where the selected table has
// no FP32 form (all but the vocabulary head's unified table), runs a runtime-shape route over whole
// tiles of `kFp8ShareUnits`: A8 (fp8_general_a8.cu) for a shape no table names where the policy
// allows it and fp8_general_uses_a8 holds, A16 (fp8_general.cu) otherwise. A registered shape's
// FP32 result stays A16, so its table's workspace covers every call.
[[nodiscard]] bool fp8_registered(std::int32_t output_rows, std::int32_t input_rows);
inline constexpr LinearShareUnits kFp8ShareUnits{64, 512};
void fp8_general_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

// Up to 16 tokens the A16 route beats A8 by up to 6 us on the small shares measured on the RTX 5090
// (bench/ops/linear_bench.cu over Qwen3.8-27B's projections at two and four ranks) and trails it
// by at most one 2 us timer tick on the large ones; from 20, the next width measured, A8 ties or
// beats it on every share.
inline constexpr std::int32_t kFp8GeneralA8MinTokens = 17;

[[nodiscard]] constexpr bool fp8_general_uses_a8(std::int32_t tokens) {
    return tokens >= kFp8GeneralA8MinTokens;
}

void fp8_general_a8(const Tensor& x, const Weight& weight, Tensor& out, WorkspaceArena& workspace,
                    cudaStream_t stream);

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
