#include "ops/linear/projection/q5/q5_shapes.h"

namespace ninfer::ops::detail {
Q5Launch select_q5_n2304_k1152(std::int32_t tokens) {
    // inkling2 MoE down route: T buckets mirrored from n1024_k5120 (generic
    // launchers); a K=1152 T=1 direct instance was added for this shape.
    // M3 correctness-first; M4 autotuner retunes the prefill tail.
    if (tokens <= 1) return launch_q5_a16_direct_r1_t1_w4_k1152;
    if (tokens <= 8) return launch_q5_a16_direct_r2_t4_w2_g8_b4;
    if (tokens <= 16) return launch_q5_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 32) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 64) return launch_q5_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 80) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 112) return launch_q5_a16_sliced_r32_t24_w4_s2_pairwise;
    if (tokens <= 160) return launch_q5_a16_sliced_r32_t32_w4_s2;
    static constexpr TokenRoute<Q5Launch> kTuringPrefill[] = {
        {480, launch_q5_a16_sliced_r32_t32_w4_s1},
        {768, launch_q5_a16_mma_r64_t64_a1},
        {kAnyTokens, launch_q5_a16_mma_r128_t64_s1_a1},
    };
    return select_token_route(kTuringPrefill, tokens);
}
} // namespace ninfer::ops::detail
