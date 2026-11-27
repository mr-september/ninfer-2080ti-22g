#pragma once

#include "ops/linear/q4/q4_launch.h"

namespace ninfer::ops::detail {

// Any shape no table below names, writing BF16 or FP32 (q4_general.cu).
[[nodiscard]] Q4Launch select_q4_general_launch(std::int32_t rows, std::int32_t k,
                                                std::int32_t tokens, DType out);

[[nodiscard]] Q4Launch select_q4_n1024_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n4096_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n5120_k4096(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n5120_k6144(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n5120_k17408(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n5120_k25600(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n6144_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n7168_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n34816_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n131072_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n131072_k2048(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n248320_k5120(std::int32_t tokens);
// The vocabulary head's FP32 logits on its GEMV and small-T widths; null beyond them.
[[nodiscard]] Q4Launch select_q4_n248320_k5120_fp32(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n3456_k1152(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n4304_k1152(std::int32_t tokens);

[[nodiscard]] Q4Launch select_q4_n1024_k5120_unified(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n4096_k5120_unified(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n5120_k6144_unified(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n6144_k5120_unified(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n7168_k5120_unified(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n34816_k5120_unified(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n131072_k5120_unified(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n131072_k2048_unified(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n3456_k1152_unified(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n4304_k1152_unified(std::int32_t tokens);

} // namespace ninfer::ops::detail
