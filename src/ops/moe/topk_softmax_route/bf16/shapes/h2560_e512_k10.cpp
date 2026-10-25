#include "ops/moe/topk_softmax_route/dispatch.h"

#include <array>

namespace ninfer::ops::detail::moe_route {
namespace {
constexpr std::array kBoundaries{2};
} // namespace

RouterPlan select_h2560_e512_k10(int tokens) {
    return {tokens == 1 ? RouterAlgorithm::Vector : RouterAlgorithm::Mma};
}

std::span<const int> allocation_boundaries() { return kBoundaries; }
} // namespace ninfer::ops::detail::moe_route
