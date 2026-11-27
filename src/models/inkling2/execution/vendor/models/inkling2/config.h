#pragma once

// The configs of Inkling 2 (Inkling2LoopForConditionalGeneration) as tools/convert/inkling2.py
// writes them into the artifact.
//
// Text: 24 stored layers run as 48 visits, the prelude [0, core_start), the core [core_start,
// core_end) num_passes times and the coda after it; every visit has its own keys, values and
// short-convolution history. Attention has no rotation: grouped-query heads of 128 with a 16-wide
// relative feature per query head that scores a learned table of distances, a sink logit per
// query and head, Q and K clipped to an L2 norm, V RMS-normed without a weight, 4-tap causal
// convolutions over K, V and both residual updates (dynamic per-head taps over K and V in the
// full-attention layers), and a sigmoid gate over the heads. Each layer reads its inputs through
// a depth attention over the residual blocks of block_size layers; a core layer's passes after the
// first take messages from the pass before through rank-gate_rank controllers. The MoE routes
// top_k of num_experts experts by sigmoid score plus a selection bias and runs SiTU over each
// expert's [gate; up] rows; there is no shared expert. The output is the final mix, RMS-normed,
// projected, soft-capped and cut to the unpadded vocabulary.

#include "artifact/schema.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace ninfer::models::inkling2 {

// The RMS epsilon of the depth mixes and the controllers' inputs (the release's constant, not
// rms_norm_eps).
inline constexpr float kMixEps = 1e-6f;

struct TextConfig {
    std::uint32_t hidden_size             = 0;
    std::uint32_t vocab_size              = 0; // embedding and head rows (padded)
    std::uint32_t unpadded_vocab_size     = 0; // the tokens the head scores
    std::uint32_t num_hidden_layers       = 0; // stored layers
    std::uint32_t max_position_embeddings = 0;
    float rms_norm_eps                    = 0;
    std::uint32_t num_attention_heads     = 0;
    std::uint32_t num_key_value_heads     = 0;
    std::uint32_t head_dim                = 0;
    // Per stored layer: true for full attention, false for a sliding window of sliding_window
    // positions (the query's own included).
    std::vector<bool> full_attention;
    std::uint32_t sliding_window = 0;
    std::uint32_t relative_width = 0;  // d_rel
    std::uint32_t relative_extent = 0; // distances a full-attention table scores
    float qk_clip       = 0;
    float relative_clip = 0;
    // Full-attention layers scale Q and the relative feature by 1 + alpha log(max(1, (p + 1) /
    // floor)).
    float log_scale_alpha       = 0;
    std::uint32_t log_scale_floor = 0;
    std::uint32_t conv_taps     = 0;
    std::uint32_t block_size    = 0; // layers per residual block
    std::uint32_t num_experts   = 0;
    std::uint32_t top_k_experts = 0;
    std::uint32_t moe_intermediate_size = 0;
    float gate_cap    = 0; // SiTU: gate_cap tanh(g / gate_cap) sigmoid(g)
    float up_cap      = 0; //       times up_cap tanh(u / up_cap)
    float route_scale = 0;
    float logit_softcap = 0;
    // The loop: [core_start, core_end) runs num_passes times; gate_rank is its controllers' rank.
    std::uint32_t core_start = 0, core_end = 0, num_passes = 1, gate_rank = 0;

    [[nodiscard]] bool core(std::uint32_t layer) const noexcept {
        return num_passes > 1 && layer >= core_start && layer < core_end;
    }
    // The visits in order: (stored layer, pass).
    [[nodiscard]] std::vector<std::pair<std::uint32_t, std::uint32_t>> visits() const;
    [[nodiscard]] std::uint32_t visit_count() const noexcept {
        return num_hidden_layers + (num_passes - 1) * (core_end - core_start);
    }
};

// The release's vision tower (Inkling2VisionModel): a ViT of num_hidden_layers pre-norm blocks
// (LayerNorm with biases, attention with biases and 2D-rotated heads, a GELU MLP, a learned scale
// after each half), a class row, learned position rows of a posemb_grid x posemb_grid grid
// resized to each image, then 2 x 2 patch merging and a two-layer GELU projector to the text
// width.
struct VisionConfig {
    std::uint32_t hidden_size        = 0;
    std::uint32_t num_hidden_layers  = 0;
    std::uint32_t num_attention_heads = 0;
    std::uint32_t intermediate_size  = 0;
    std::uint32_t patch_size         = 0;
    std::uint32_t merge_size         = 0;
    std::uint32_t projection_hidden  = 0;
    std::uint32_t posemb_grid        = 0;
    std::uint32_t text_hidden_size   = 0;
    std::uint32_t max_image_tokens   = 0;
    std::uint32_t image_token_id     = 0;
    float layer_norm_eps             = 0;
    float rope_theta                 = 0;

    [[nodiscard]] std::uint32_t head_dim() const noexcept {
        return hidden_size / num_attention_heads;
    }
    [[nodiscard]] std::uint64_t patch_features() const noexcept {
        return std::uint64_t(3) * patch_size * patch_size;
    }
    [[nodiscard]] std::uint64_t merged_features() const noexcept {
        return std::uint64_t(hidden_size) * merge_size * merge_size;
    }
};

// The release's DSpark drafter: num_hidden_layers non-causal blocks over a block of block_size
// rows (the anchor, then mask rows), each with its own projections of the target's tapped hidden
// states into its keys and values, then a recurrent Markov head of rank markov_rank that corrects
// the target head's logits row by row.
struct DSparkConfig {
    std::uint32_t num_hidden_layers   = 0;
    std::uint32_t num_attention_heads = 0;
    std::uint32_t num_key_value_heads = 0;
    std::uint32_t head_dim            = 0;
    std::uint32_t intermediate_size   = 0;
    std::uint32_t block_size          = 0;
    std::uint32_t markov_rank         = 0;
    std::uint32_t sliding_window      = 0;
    float rope_theta                  = 0;
    float rms_norm_eps                = 0;
    std::vector<bool> full_attention;
    // The visits whose input the drafter taps (the last: the final hidden state, after every
    // visit).
    std::vector<std::uint32_t> target_visits;
};

// Each throws artifact::ArtifactError naming the field that is missing, malformed or outside the
// implemented geometry.
[[nodiscard]] TextConfig parse_text_config(const artifact::Json& value);
[[nodiscard]] VisionConfig parse_vision_config(const artifact::Json& value, const TextConfig& text);
[[nodiscard]] DSparkConfig parse_dspark_config(const artifact::Json& value, const TextConfig& text);

} // namespace ninfer::models::inkling2
