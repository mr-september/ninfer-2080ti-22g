#include "ops/linear/swiglu/dispatch.h"
#include <stdexcept>

namespace ninfer::ops::detail {
void validate_swiglu_geometry(QType type, int rows, int k) {
    if (type == QType::Q8_G32_FP16 && rows == 12288 && k == 2048) return;
    if (rows == 34816 && k == 5120 &&
        (type == QType::Q4_G64_FP16 || type == QType::Q8_G32_FP16 ||
         type == QType::FP8_E4M3FN_ROW_BF16))
        return;
    throw std::invalid_argument("linear_swiglu: unsupported weight geometry");
}

LinearSwiGluLaunch select_q8_linear_swiglu(int rows, int k, int tokens) {
    if (tokens > 0) {
        if (rows == 12288 && k == 2048) return select_q8_swiglu_n12288_k2048(tokens);
        if (rows == 34816 && k == 5120) return select_q8_swiglu_n34816_k5120(tokens);
    }
    throw std::invalid_argument("q8 linear_swiglu: unsupported shape or token extent");
}
} // namespace ninfer::ops::detail
