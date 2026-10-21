#include "ops/linear/projection/q4/q4_shapes.h"

namespace ninfer::ops::detail {
Q4Launch select_q4_n6144_k5120(std::int32_t tokens) {
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 1) return launch_q4_a16_gemv_r1_w8_direct;
    if (tokens <= 4) return launch_q4_a16_sliced_r16_t8_capacity4;
    if (tokens <= 8) return launch_q4_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_k5120_t16;
    if (tokens <= 24) return launch_q4_a16_sliced_k5120_t24;
    if (tokens <= 48) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 64) return launch_q4_a16_mma_r32_t32_k128_s2_a2;
    if (tokens <= 96) return launch_q4_a16_sliced_r32_t32_w4_s1;
    // Turing prefill: two single-buffered CTAs per SM overlap one tile's MMA with another's loads;
    // tuned per shape with the 2080 Ti route autotuner.
    static constexpr TokenRoute<Q4Launch> kTuringPrefill[] = {
        {1536, launch_q4_a16_mma_r64_t64_a1},
        {kAnyTokens, launch_q4_a16_mma_r64_t128_a1},
    };
    return select_token_route(kTuringPrefill, tokens);
}
} // namespace ninfer::ops::detail
