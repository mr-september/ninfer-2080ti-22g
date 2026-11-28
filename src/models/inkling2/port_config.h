#pragma once

#include "models/registry.h"

#include <cstdint>
#include <vector>

namespace ninfer::artifact {
struct Directory;
}

namespace ninfer::models::inkling2 {

// Learned relative attention over the last `relative_extent` positions (no RoPE).
struct AttentionConfig {
    std::uint32_t relative_taps   = 0;  // learned bias taps per head
    std::uint32_t relative_extent = 0;  // positions covered by the relative bias
    float qk_clip                 = 0;  // abs clip on post-norm Q/K
    float relative_clip           = 0;  // clip on the relative-bias table lookup
    float log_scale_floor         = 0;  // distance-log scaling floor
    float log_scale_alpha         = 0;  // distance-log scaling blend
};

// Shared-weight loop: layers [start, end) run `count` passes with rank-`controller_rank`
// cross-pass gates. Visits per token = (end - start) * (count - 1) + (L - (end - start)).
struct LoopConfig {
    std::uint32_t start           = 0;
    std::uint32_t end             = 0;  // exclusive
    std::uint32_t count           = 0;  // passes per looped layer
    std::uint32_t controller_rank = 0;

    [[nodiscard]] std::uint32_t visits_per_layer(std::uint32_t layer,
                                                 std::uint32_t layers) const noexcept {
        if (layer >= start && layer < end) return count;
        return 1;
    }

    [[nodiscard]] std::uint64_t visits_per_token(std::uint32_t layers) const noexcept {
        std::uint64_t total = 0;
        for (std::uint32_t layer = 0; layer < layers; ++layer) total += visits_per_layer(layer, layers);
        return total;
    }

    [[nodiscard]] bool loops(std::uint32_t layer) const noexcept {
        return layer >= start && layer < end;
    }
};

// Sigmoid top-k sparse MoE with SitU activations and softcapped router weights.
struct MoeConfig {
    std::uint32_t num_experts         = 0;
    std::uint32_t num_experts_per_tok = 0;
    std::uint32_t intermediate_size   = 0;
    float router_scale                = 0;
    float situ_beta                   = 0;
    float situ_linear_beta            = 0;
};

struct Config {
    Architecture architecture = Architecture::Inkling2;
    std::uint32_t hidden_size             = 0;
    std::uint32_t vocab_size              = 0;
    std::uint32_t unpadded_vocab_size     = 0;   // the rows the head scores; default = vocab_size
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t max_position_embeddings = 0;
    std::uint32_t num_attention_heads     = 0;
    std::uint32_t num_key_value_heads     = 0;
    std::uint32_t head_dim                = 0;
    std::uint32_t sliding_window          = 0;   // local layers attend to the last N tokens
    std::uint32_t conv_taps               = 0;   // causal conv kernel size (K/V, attn-out, FFN-out)
    std::uint32_t attn_res_block_size     = 0;   // block attention-residual mix period
    std::uint32_t eos_token_id            = 0;
    float rms_norm_eps              = 0;
    float final_logit_softcapping   = 0;

    // Per-layer mixer: true = full attention, false = sliding-window local attention.
    std::vector<bool> full_attention_layer;

    AttentionConfig attention;
    LoopConfig loop;
    MoeConfig moe;

    [[nodiscard]] std::uint64_t query_width() const noexcept {
        return std::uint64_t(num_attention_heads) * head_dim;
    }
    [[nodiscard]] std::uint64_t key_width() const noexcept {
        return std::uint64_t(num_key_value_heads) * head_dim;
    }
    [[nodiscard]] bool is_full_attention(std::uint32_t layer) const noexcept {
        return layer < num_hidden_layers && full_attention_layer[layer];
    }
};

// Reads components.text.config from an artifact directory written by tools/convert (inkling2 recipe).
[[nodiscard]] Config config_from_directory(const artifact::Directory&);

} // namespace ninfer::models::inkling2
