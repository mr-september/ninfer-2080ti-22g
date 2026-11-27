#pragma once

#include "core/qtype.h"
#include "models/dspine/config.h"
#include "models/load_options.h"
#include "models/registry.h"

#include <array>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace ninfer::artifact {
struct Directory;
}

namespace ninfer::models::qwen3_5 {

enum class MixerKind { FullAttention, LinearAttention };
enum class DraftAttentionKind { FullAttention, SlidingAttention };

struct AttentionConfig {
    std::uint32_t num_attention_heads = 0;
    std::uint32_t num_key_value_heads = 0;
    std::uint32_t head_dim            = 0;

    [[nodiscard]] std::uint64_t query_width() const noexcept {
        return std::uint64_t(num_attention_heads) * head_dim;
    }

    [[nodiscard]] std::uint64_t key_width() const noexcept {
        return std::uint64_t(num_key_value_heads) * head_dim;
    }
};

struct RopeConfig {
    float rope_theta            = 0;
    float partial_rotary_factor = 0;
    std::array<std::uint32_t, 3> mrope_section{};
    std::uint32_t rotary_dim = 0;
    std::vector<std::uint8_t> pair_axes;
};

struct GdnConfig {
    std::uint32_t linear_num_key_heads   = 0;
    std::uint32_t linear_key_head_dim    = 0;
    std::uint32_t linear_num_value_heads = 0;
    std::uint32_t linear_value_head_dim  = 0;
    std::uint32_t linear_conv_kernel_dim = 0;

    [[nodiscard]] std::uint64_t key_width() const noexcept {
        return std::uint64_t(linear_num_key_heads) * linear_key_head_dim;
    }

    [[nodiscard]] std::uint64_t value_width() const noexcept {
        return std::uint64_t(linear_num_value_heads) * linear_value_head_dim;
    }

    [[nodiscard]] std::uint64_t conv_channels() const;
};

struct DenseConfig {
    std::uint32_t intermediate_size = 0;
};

struct MoeConfig {
    std::uint32_t num_experts                     = 0;
    std::uint32_t num_experts_per_tok             = 0;
    std::uint32_t moe_intermediate_size           = 0;
    std::uint32_t shared_expert_intermediate_size = 0;
};

struct TextConfig {
    Architecture architecture             = Architecture::Qwen3_5;
    std::uint32_t hidden_size             = 0;
    std::uint32_t vocab_size              = 0;
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t max_position_embeddings = 0;
    bool tie_word_embeddings              = false;
    float rms_norm_eps                    = 0;
    std::vector<MixerKind> layer_types;
    std::vector<std::uint32_t> compact_layer_indices;
    std::uint32_t full_attention_layers   = 0;
    std::uint32_t linear_attention_layers = 0;
    std::optional<AttentionConfig> attention;
    std::optional<RopeConfig> rope_parameters;
    std::optional<GdnConfig> gdn;
    std::variant<DenseConfig, MoeConfig> ffn;
};

struct VisionConfig {
    std::uint32_t depth                   = 0;
    std::uint32_t hidden_size             = 0;
    std::uint32_t intermediate_size       = 0;
    std::uint32_t num_heads               = 0;
    std::uint32_t patch_size              = 0;
    std::uint32_t temporal_patch_size     = 0;
    std::uint32_t spatial_merge_size      = 0;
    std::uint32_t num_position_embeddings = 0;
    std::uint32_t position_grid_side      = 0;
    // Norms, biases and position embeddings; projection matrices retain their own formats.
    QType direct_format = QType::BF16;
    [[nodiscard]] std::uint64_t patch_width() const;
    [[nodiscard]] std::uint64_t merger_width() const;
};

struct DFlash2Config {
    std::uint32_t conv_kernel_size = 0;
    std::uint32_t conv_group_size  = 0;
    std::uint32_t selector_rank    = 0;
    std::uint32_t selector_top_k   = 0;
};

struct DraftConfig {
    AttentionConfig attention;
    std::uint32_t intermediate_size       = 0;
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t max_position_embeddings = 0;
    float rms_norm_eps                    = 0;
    float rope_theta                      = 0;
    std::vector<DraftAttentionKind> layer_types;
    std::optional<std::uint32_t> sliding_window;
    std::vector<std::uint32_t> target_layer_ids;
    std::uint32_t mask_token_id = 0;
    std::optional<DFlash2Config> dflash2;
    // --spec dspine: the standalone drafter's config, which the fields above restate (every layer
    // sliding, the trained window); its weights are not the model artifact's.
    std::optional<dspine::Config> dspine;

    [[nodiscard]] std::uint32_t local_layer_count() const noexcept {
        return static_cast<std::uint32_t>(std::count(layer_types.begin(), layer_types.end(),
                                                     DraftAttentionKind::SlidingAttention));
    }

    [[nodiscard]] std::uint32_t full_layer_count() const noexcept {
        return num_hidden_layers - local_layer_count();
    }

    [[nodiscard]] std::uint32_t compact_layer_index(std::uint32_t layer) const {
        const auto kind = layer_types.at(layer);
        return static_cast<std::uint32_t>(
            std::count(layer_types.begin(), layer_types.begin() + layer, kind));
    }

    // Whether a DFlash2 drafter of this config over a target of width `hidden_size` has the
    // geometry the fused DFlash2 Ops implement (context_kv_materialize and the two
    // dynamic-convolution Ops: the official 27B drafter's). A DFlash2 drafter of another geometry
    // runs the same mathematics composed from the runtime-shape Ops.
    [[nodiscard]] bool fused_dflash2(std::uint32_t hidden_size) const noexcept {
        return dflash2.has_value() && hidden_size == 5120 && num_hidden_layers == 5 &&
               local_layer_count() == 5 && intermediate_size == 17408 &&
               attention.num_attention_heads == 32 && attention.num_key_value_heads == 8 &&
               attention.head_dim == 128 && dflash2->conv_kernel_size == 2 &&
               dflash2->conv_group_size == 16;
    }
};

struct Config {
    TextConfig text;
    std::optional<VisionConfig> vision;
    bool mtp = false;
    std::optional<DraftConfig> draft;
};

[[nodiscard]] Config parse_config(const artifact::Directory& directory, const LoadOptions& options);
// The config of an artifact's `vision` component, which another family that reuses this tower
// parses too.
[[nodiscard]] VisionConfig parse_vision_config(const artifact::Directory& directory);

// Tensor parallelism: the part of every layer one rank runs. GDN key heads [key_head_begin,
// key_head_end) with the value heads that read them; attention KV heads [kv_head_begin,
// kv_head_end) with the query heads that read them and their KV cache; a dense FFN's rows
// [ffn_begin, ffn_end); and of a MoE layer either the routed experts [expert_begin, expert_end)
// (expert parallelism: every rank routes, and the shared expert is rank 0's), or, where
// ffn_end > ffn_begin, every expert's intermediate columns [ffn_begin, ffn_end), routed and shared
// (expert_begin 0, expert_end all). The ranks' shares are contiguous, in rank order, and together
// cover every head, row, expert and column. A model has fewer KV heads than GDN heads, so the
// ranks past its KV-head count hold no attention -- unless the ranks divide attention by positions
// (attention_positions, --tensor-attention positions): then every rank projects every query and KV
// head (kv_head_begin 0, kv_head_end all), holds every KV head of the cache pages p with
// p % ranks == its rank, attends over those keys, and gates and projects the merged output of
// query heads [query_head_begin, query_head_end), the ranks' runs covering every query head.
struct TensorShare {
    std::uint32_t key_head_begin = 0;
    std::uint32_t key_head_end   = 0;
    std::uint32_t kv_head_begin  = 0;
    std::uint32_t kv_head_end    = 0;
    std::uint32_t ffn_begin      = 0;
    std::uint32_t ffn_end        = 0;
    std::uint32_t expert_begin   = 0;
    std::uint32_t expert_end     = 0;
    bool attention_positions       = false;
    std::uint32_t query_head_begin = 0;
    std::uint32_t query_head_end   = 0;

    bool operator==(const TensorShare&) const = default;
};

// The text model as `share` sees it: its GDN key and value heads, its attention heads (none for a
// rank without KV heads) and dense FFN rows; the router and every other width stay whole. Throws
// std::invalid_argument unless the model has GDN layers and the share lies inside its heads and
// its FFN rows or experts.
[[nodiscard]] TextConfig tensor_parallel_share(const TextConfig& text, const TensorShare& share);

} // namespace ninfer::models::qwen3_5
