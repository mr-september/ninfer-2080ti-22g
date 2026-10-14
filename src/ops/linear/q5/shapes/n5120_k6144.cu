#include "ops/linear/q5/q5_shapes.h"

namespace ninfer::ops::detail {
Q5Launch select_q5_n5120_k6144(std::int32_t tokens) {
#if defined(NINFER_SM75)
    // Turing: the SIMT/direct decode kernels beat the tensor-core small-T routes (tuned on
    // RTX 5090) at the MTP verify widths, measured per shape with the 2080 Ti.
    if (tokens == 4) return launch_q5_a16_direct_r2_t4_w2_g8_b4;
#endif
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 1) return launch_q5_a16_direct_r1_t1_w4_k6144;
    if (tokens <= 2) return launch_q5_a16_direct_r1_t2_w2_k6144;
    if (tokens <= 3) return launch_q5_a16_direct_r1_t3_w2_k6144;
    if (tokens <= 4) return launch_q5_a16_sliced_r16_t8_capacity4;
    if (tokens <= 8) return launch_q5_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 32) return launch_q5_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 48) return launch_q5_a16_sliced_r32_t24_w4_s2_pairwise;
    if (tokens <= 64) return launch_q5_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 128) return launch_q5_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 160) return launch_q5_a16_sliced_r32_t64_w2_s1;
#if defined(NINFER_SM75)
    // Turing prefill: two single-buffered CTAs per SM overlap one tile's MMA with
    // another's loads; tuned per shape with the 2080 Ti route autotuner.
    static constexpr TokenRoute<Q5Launch> kTuringPrefill[] = {
        {256, launch_q5_a16_sliced_r32_t32_w4_s1},
        {384, launch_q5_a16_mma_r64_t128_a1},
        {1536, launch_q5_a16_mma_r128_t64_s1_a1},
        {kAnyTokens, launch_q5_a16_mma_r64_t128_a1},
    };
    return select_token_route(kTuringPrefill, tokens);
#else
    if (tokens <= 256) return launch_q5_a16_mma_r32_t128;
    return launch_q5_a16_mma_r64_t128;
#endif
}
} // namespace ninfer::ops::detail
