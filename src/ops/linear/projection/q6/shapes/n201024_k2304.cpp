#include "ops/linear/projection/q6/q6_shapes.h"

namespace ninfer::ops::detail {

Q6Launch select_q6_n201024_k2304(std::int32_t tokens) {
    // inkling2 head route: capacity buckets mirrored from the n248320_k5120
    // anchor (shape-generic launchers; M3 correctness-first, M4 retunes).
    if (tokens <= 2) return launch_q6_a16_simt_r8_t4;
    if (tokens <= 8) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q6_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 32) return launch_q6_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 48) return launch_q6_a16_mma_r64_t48_k128;
    if (tokens <= 56) return launch_q6_a16_mma_r64_t56_k128;
    if (tokens <= 64) return launch_q6_a16_mma_r64_t64_k128;
    if (tokens <= 80) return launch_q6_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q6_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q6_a16_mma_r64_t112;
    return launch_q6_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
