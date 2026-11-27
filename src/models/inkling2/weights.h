#pragma once

#include "models/common/weights.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer::models::inkling2 {

// One physical layer. Every field defaults to "absent" (WeightId{}); which names exist per
// layer is the artifact's `bindings` table (verified 2026-11-25): controller/* on the 12
// looped layers (6-17); *_convolution_projection on looped full-attention layers 11,17;
// *_input_norm on the four full-attention layers 5,11,17,23 (which lack input_norm);
// residual_query absent on layer 0.
struct AttentionLayerWeights {
    WeightId input_norm;            // block entry norm (absent where the mixer reuses one)
    WeightId residual_query;        // block attention-residual softmax query (absent on layer 0)

    WeightId query, key, value, output;
    WeightId gate;                  // sigmoid output gate
    WeightId query_input_norm, key_input_norm, value_input_norm;  // full-attention layers
    WeightId query_clip, relative_clip;    // per-layer clip scales (learned)
    WeightId relative_bias, relative_projection, sink_projection; // learned rel-bias + per-head sink
    WeightId key_convolution, value_convolution, post_convolution;  // causal tap weights (K4)
    WeightId key_convolution_projection, value_convolution_projection;  // looped layers

    WeightId post_attention_norm;
    WeightId ffn_residual_query, ffn_post_convolution;

    struct Moe {
        WeightId router, router_bias;
        struct Expert { WeightId gate, up, down; };
        std::vector<Expert> experts;    // top-k sigmoid routing over all experts
    } moe;

    // Cross-pass loop controller (rank-`controller_rank` gates; looped layers only).
    struct Controller {
        WeightId message_query;
        WeightId q_gain, q_forget, q_down;
        WeightId k_gain, k_forget, k_down;
        WeightId v_gain, v_forget, v_down;
        WeightId ffn_gain, ffn_forget, ffn_down;
    };
    // optional (not a pointer into *this): layers live in a vector and must stay movable.
    std::optional<Controller> controller;
};

struct TextWeights {
    WeightId token_embedding, embedding_norm;
    WeightId final_norm, final_residual_query;
    WeightId output_head;
    std::vector<AttentionLayerWeights> layers;
};

// Handles into the frozen weight array; no artifact-id lookup during execution.
struct ModelWeights {
    TextWeights text;
};

} // namespace ninfer::models::inkling2
