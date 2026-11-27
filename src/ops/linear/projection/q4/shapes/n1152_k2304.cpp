#include "ops/linear/projection/q4/q4_shapes.h"

namespace ninfer::ops::detail {
Q4Launch select_q4_n1152_k2304(std::int32_t tokens) {
    // inkling2 MoE gate/up route: correctness-first buckets from shape-generic
    // launchers (M3); per-shape route tuning is deferred to the M4 autotuner pass.
    if (tokens <= 1) return launch_q4_a16_gemv_r4_w1_direct;
    if (tokens <= 8) return launch_q4_a16_simt_r4_t4_w2_g8_s2;
    if (tokens <= 32) return launch_q4_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r16_t32_w4_s2;
    return launch_q4_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
