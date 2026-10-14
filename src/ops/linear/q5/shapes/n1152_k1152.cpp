#include "ops/linear/q5/q5_shapes.h"
#include <stdexcept>

namespace ninfer::ops::detail {
Q5Launch select_q5_n1152_k1152(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("q5 linear: T must be a multiple of 4 in [4,131072]");
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 8) return launch_q5_a16_direct_r2_t4_w2_g8_b4;
    if (tokens <= 64) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 128) return launch_q5_a16_sliced_r16_t16_w2_s2;
#if defined(NINFER_SM75)
    // Turing prefill: two single-buffered CTAs per SM overlap one tile's MMA with
    // another's loads; tuned per shape with the 2080 Ti route autotuner.
    static constexpr TokenRoute<Q5Launch> kTuringPrefill[] = {
        {576, launch_q5_a16_sliced_r32_t32_w2_s2},
        {1536, launch_q5_a16_sliced_r32_t64_w2_s1},
        {kAnyTokens, launch_q5_a16_mma_r128_t64_s1_a1},
    };
    return select_token_route(kTuringPrefill, tokens);
#else
    if (tokens <= 576) return launch_q5_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 1152) return launch_q5_a16_mma_r32_t128;
    return launch_q5_a16_mma_r64_t128;
#endif
}
} // namespace ninfer::ops::detail
