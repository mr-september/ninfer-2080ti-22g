#pragma once

#include "ninfer/ops/linear_policy.h"
#include "ops/linear/t2/t2_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

// K granularity of the routes at shapes outside the registered table.
inline constexpr std::int32_t kT2RuntimeShapeK = 512;

[[nodiscard]] bool t2_runtime_shape(std::int32_t n, std::int32_t k);
T2Launch select_t2_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t);
// The routes that write an FP32 result (the small-T v2 kernel and the MMA bands), at any
// runtime shape.
T2Launch select_t2_fp32_launch(std::int32_t n, std::int32_t k, std::int32_t t);
T2Launch select_t2_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy);

void t2_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream);

} // namespace ninfer::ops::detail
