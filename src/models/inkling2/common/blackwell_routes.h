#pragma once

// The FP8 A8 and NVFP4 A4 routes run Blackwell instructions (block-scaled FP8 and FP4 MMA, TMA).
// A build without 120a links fp8_sm86_stubs.cpp and nvfp4_sm86_stubs.cpp in their place, which
// refuse every call. A build of several architectures with 120a carries the routes; their entry
// points refuse, with the same messages, a call on a card that does not run the 120a code, before
// any of their kernels (which trap in that card's image) is launched.

#include "ops/common/device_tier.h"

#include <stdexcept>

namespace ninfer::ops::detail {

inline void require_fp8_a8_card() {
    if constexpr (kBuildTiers > 1) {
        if (!current_device_blackwell()) {
            throw std::runtime_error("FP8 A8 execution requires an sm_100a or sm_120a GPU");
        }
    }
}

inline void require_nvfp4_a4_card() {
    if constexpr (kBuildTiers > 1) {
        if (!current_device_blackwell()) {
            throw std::runtime_error("NVFP4 A4 execution requires an sm_120a GPU");
        }
    }
}

} // namespace ninfer::ops::detail
