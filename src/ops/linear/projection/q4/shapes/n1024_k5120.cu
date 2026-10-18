#include "ops/linear/projection/q4/q4_shapes.h"

namespace ninfer::ops::detail {
Q4Launch select_q4_n1024_k5120(std::int32_t tokens) {
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 1) return launch_q4_a16_gemv_r1_w8_direct;
    if (tokens <= 8) return launch_q4_a16_simt_r4_t4_w2_g8_s2;
    if (tokens <= 32) return launch_q4_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 80) return launch_q4_a16_sliced_r16_t16_w4_s2;
#if defined(NINFER_SM75)
    // Turing prefill: two single-buffered CTAs per SM overlap one tile's MMA with another's loads;
    // tuned per shape with the 2080 Ti route autotuner.
    static constexpr TokenRoute<Q4Launch> kTuringPrefill[] = {
        {128, launch_q4_a16_sliced_r32_t32_w4_s1},
        {384, launch_q4_a16_mma_r64_t64_k128_s2_a1},
        {768, launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2},
        {1536, launch_q4_a16_mma_r64_t128_a1},
        {kAnyTokens, launch_q4_a16_mma_r128_t64_s1_a1},
    };
    return select_token_route(kTuringPrefill, tokens);
#else
    if (tokens <= 320) return launch_q4_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 576) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 768) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    if (tokens <= 1280) return launch_q4_a16_mma_r32_t128_k64_s2_a2;
    if (tokens <= 1344) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    return launch_q4_a16_mma_r64_t128;
#endif
}
} // namespace ninfer::ops::detail
