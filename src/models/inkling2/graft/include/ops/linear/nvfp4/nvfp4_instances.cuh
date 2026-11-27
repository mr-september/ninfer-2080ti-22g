#pragma once
#include "ops/common/device_limits.h"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"

namespace ninfer::ops::detail::unified {
template <int Tokens, int Warps, int Stages>
using Nvfp4SlicedInstance = Nvfp4A16SlicedKMmaSchedule<
    Warps, Tokens,
    4,
    Cache::ca, Cache::cg, Nvfp4ActivationStage::PaddedZero, Stages>;
} // namespace ninfer::ops::detail::unified
