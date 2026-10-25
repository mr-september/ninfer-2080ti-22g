#pragma once

#include "ops/moe/kernels/cuda/routing/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/launch.h"

#include <span>

namespace ninfer::ops::detail::moe_complete {

inline constexpr int kSliceTokens = 4096;

namespace kernel = moe_cuda;
using kernel::routing::RouterAlgorithm;

struct Plan {
    kernel::routing::RouterPlan router;
    kernel::block_scaled::Plan block_scaled;
};

struct ShapePolicy {
    Plan (*select)(int tokens);
    std::span<const int> allocation_boundaries;
};

const ShapePolicy& shape_dispatch(const MoeProfile& profile);
const ShapePolicy& fp8_h2560_i640_e512_k10();

} // namespace ninfer::ops::detail::moe_complete
