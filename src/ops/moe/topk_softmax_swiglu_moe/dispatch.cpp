#include "ops/moe/topk_softmax_swiglu_moe/dispatch.h"
#include "ops/moe/operands.h"

namespace ninfer::ops::detail::moe_complete {

const ShapePolicy& shape_dispatch(const MoeProfile& profile) {
    validate_moe_profile(profile);
    switch (profile.routed.ffn.down_type) {
    case QType::FP8_E4M3FN_BLOCK128X128_BF16:
        return fp8_h2560_i640_e512_k10();
    default:
        throw std::invalid_argument("unsupported MoE expert format");
    }
}
} // namespace ninfer::ops::detail::moe_complete
