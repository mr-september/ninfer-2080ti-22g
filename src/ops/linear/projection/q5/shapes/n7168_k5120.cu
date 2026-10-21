#include "ops/linear/projection/q5/q5_shapes.h"

namespace ninfer::ops::detail {
Q5Launch select_q5_n7168_k5120(std::int32_t tokens) {
    // Turing: the SIMT/direct decode kernels beat the tensor-core small-T routes (tuned on
    // RTX 5090) at the MTP verify widths, measured per shape with the 2080 Ti.
    if (tokens == 3) return launch_q5_a16_direct_r1_t3_w4_k5120;
    if (tokens == 4) return launch_q5_a16_direct_r2_t4_w2_g8_b4;
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 1) return launch_q5_a16_direct_r1_t1_w4_k5120;
    if (tokens <= 2) return launch_q5_a16_direct_r1_t2_w4_k5120;
    if (tokens <= 4) return launch_q5_a16_sliced_r16_t8_capacity4;
    if (tokens <= 8) return launch_q5_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 24) return launch_q5_a16_sliced_r16_t24_w4_s2;
    if (tokens <= 32) return launch_q5_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 96) return launch_q5_a16_sliced_r32_t32_w4_s1;
    // Turing prefill: two single-buffered CTAs per SM overlap one tile's MMA with
    // another's loads; tuned per shape with the 2080 Ti route autotuner.
    static constexpr TokenRoute<Q5Launch> kTuringPrefill[] = {
        {192, launch_q5_a16_mma_r64_t128_s1_a1},
        {kAnyTokens, launch_q5_a16_mma_r128_t64_s1_a1},
    };
    return select_token_route(kTuringPrefill, tokens);
}
} // namespace ninfer::ops::detail
