#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {
Q4Launch select_q4_n4096_k5120(std::int32_t tokens) {
#if defined(NINFER_SM75)
    // Turing: the SIMT/direct decode kernels beat the tensor-core small-T routes (tuned on
    // RTX 5090) at the MTP verify widths, measured per shape with the 2080 Ti.
    if (tokens <= 1) return launch_q4_a16_simt_r4_t1_w2_g8_s2;
    if (tokens <= 4) return launch_q4_a16_simt_r4_t4_w2_g8_s2;
#endif
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 1) return launch_q4_a16_gemv_r1_w8_direct;
    if (tokens <= 8) return launch_q4_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 96) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 128) return launch_q4_a16_sliced_r32_t32_w2_s2;
#if defined(NINFER_SM75)
    // Turing prefill: two single-buffered CTAs per SM overlap one tile's MMA with another's loads;
    // tuned per shape with the 2080 Ti route autotuner.
    static constexpr TokenRoute<Q4Launch> kTuringPrefill[] = {
        {1536, launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2},
        {kAnyTokens, launch_q4_a16_mma_r64_t64_a1},
    };
    return select_token_route(kTuringPrefill, tokens);
#else
    if (tokens <= 320) return launch_q4_a16_mma_r32_t32_k64_wr16_wt16_s3_a3_b2;
    return launch_q4_a16_mma_r64_t128;
#endif
}
} // namespace ninfer::ops::detail
