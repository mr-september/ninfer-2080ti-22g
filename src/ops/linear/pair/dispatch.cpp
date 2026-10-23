#include "ops/linear/pair/dispatch.h"
#include "ops/common/host/operands.h"
#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
void require_weight(const Weight& w, int k) {
    const std::uint64_t payload_bytes =
        std::uint64_t(1024) * k + std::uint64_t(1024) * (k / 32) * 2;
    if (w.qtype != QType::Q8_G32_FP16 || w.layout != QuantLayout::RowSplit ||
        w.scale_dtype != DType::FP16 || w.group != 32 || w.group_size != 32 || w.ndim != 2 ||
        w.n != 1024 || w.k != k || w.shape[0] != 1024 || w.shape[1] != k ||
        w.padded_shape[0] != 1024 || w.padded_shape[1] != k || w.payload_bytes < payload_bytes ||
        w.qhigh != nullptr || w.high_plane_bytes != 0 || !common::operand::aligned(w.qdata, 16) ||
        !common::operand::aligned(w.scales, 16))
        throw std::invalid_argument(
            "linear_pair: invalid Q8 row-split weight representation or alignment");
}
} // namespace

LinearPairLaunch select_linear_pair(int k, int tokens) {
    if (tokens > 0) {
        if (k == 5120) return select_q8_pair_n1024_k5120(tokens);
    }
    throw std::invalid_argument("linear_pair: unsupported shape or token extent");
}

void validate_pair_weights(const Weight& first, const Weight& second, int max_tokens) {
    (void)select_linear_pair(first.k, max_tokens);
    require_weight(first, first.k);
    require_weight(second, first.k);
}
} // namespace ninfer::ops::detail
