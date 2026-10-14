#include "ops/linear/q5/q5_shapes.h"

namespace ninfer::ops::detail {
Q5Launch select_q5_n1024_k5120(std::int32_t tokens) {
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 1) return launch_q5_a16_direct_r1_t1_w4_k5120;
    if (tokens <= 8) return launch_q5_a16_direct_r2_t4_w2_g8_b4;
    if (tokens <= 16) return launch_q5_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 32) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 64) return launch_q5_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 80) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 112) return launch_q5_a16_sliced_r32_t24_w4_s2_pairwise;
    if (tokens <= 160) return launch_q5_a16_sliced_r32_t32_w4_s2;
#if defined(NINFER_SM75)
    // Turing prefill: two single-buffered CTAs per SM overlap one tile's MMA with
    // another's loads; tuned per shape with the 2080 Ti route autotuner.
    static constexpr TokenRoute<Q5Launch> kTuringPrefill[] = {
        {480, launch_q5_a16_sliced_r32_t32_w4_s1},
        {768, launch_q5_a16_mma_r64_t64_a1},
        {kAnyTokens, launch_q5_a16_mma_r128_t64_s1_a1},
    };
    return select_token_route(kTuringPrefill, tokens);
#else
    if (tokens <= 480) return launch_q5_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 640) return launch_q5_a16_sliced_r32_t64_w2_s2;
    if (tokens <= 768) return launch_q5_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 1280) return launch_q5_a16_mma_r32_t128;
    if (tokens <= 1344) return launch_q5_a16_sliced_r32_t64_w2_s1;
    return launch_q5_a16_mma_r64_t128;
#endif
}
} // namespace ninfer::ops::detail
