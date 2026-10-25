#include "ops/moe/routed_swiglu/cuda/dispatch.h"
#include "ops/moe/operands.h"

namespace ninfer::ops::detail::moe_routed {

const ShapePolicy& shape_dispatch(const RoutedSwiGLUProfile& profile) {
    validate_expert_profile(profile);
    switch (profile.ffn.down_type) {
    case QType::FP8_E4M3FN_BLOCK128X128_BF16:
        return fp8_h2560_i640_e512_k10();
    default:
        throw std::invalid_argument("unsupported MoE expert format");
    }
}

} // namespace ninfer::ops::detail::moe_routed
