#pragma once

// The `dspine` component of a standalone DSpine drafter artifact as tools/convert/dspine.py
// writes it (docs/maintainer/dspine.md): Qwen3-style sliding-window layers, causal inside the
// drafted block, over a ring of the target's tapped hidden states; a code bridge that writes into
// the residual after every layer; and a serial refinement of the last write that chains the
// block's candidates. The target's token embedding and output head stay in the target's artifact.

#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace ninfer::artifact {
class Reader;
}

namespace ninfer::models::dspine {

struct Config {
    std::uint32_t hidden_size             = 0;
    std::uint32_t intermediate_size       = 0;
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t num_attention_heads     = 0;
    std::uint32_t num_key_value_heads     = 0;
    std::uint32_t head_dim                = 0;
    std::uint32_t max_position_embeddings = 0;
    std::uint32_t vocab_size              = 0;
    // The target's layer count the taps index into.
    std::uint32_t num_target_layers = 0;
    // The trained block: the anchor and block_size - 1 masks, the most drafts a round takes.
    std::uint32_t block_size     = 0;
    std::uint32_t sliding_window = 0;
    float rms_norm_eps           = 0;
    float rope_theta             = 0;
    // The target layers whose outputs, side by side, are the drafter's context features.
    std::vector<std::uint32_t> target_layer_ids;
    std::uint32_t mask_token_id = 0;
    // The code bridge: code width C, message rank R, its normalization epsilon, and the head's
    // candidates per draft row the refinement chains (16).
    std::uint32_t code_size  = 0;
    std::uint32_t rank       = 0;
    std::uint32_t head_top_k = 0;
    float bridge_eps         = 0;

    [[nodiscard]] std::uint32_t taps() const noexcept {
        return static_cast<std::uint32_t>(target_layer_ids.size());
    }
    [[nodiscard]] std::uint64_t query_width() const noexcept {
        return std::uint64_t(num_attention_heads) * head_dim;
    }
    [[nodiscard]] std::uint64_t key_width() const noexcept {
        return std::uint64_t(num_key_value_heads) * head_dim;
    }
};

// The config of the standalone DSpine drafter `reader` opened at `path`. Throws
// artifact::ArtifactError when the artifact is not one (its only component must be `dspine`) or
// names the field that is missing, malformed or outside what the engine implements.
[[nodiscard]] Config read_config(const artifact::Reader& reader,
                                 const std::filesystem::path& path);

// The drafter artifact's config, read from its directory without materializing a weight.
[[nodiscard]] Config read_config(const std::filesystem::path& path);

// Refuses a target the drafter was not trained for: another hidden width, vocabulary or layer
// count, or a tap outside its layers. `family` names the target in the message.
void check_target(const Config& config, std::uint64_t hidden_size, std::uint64_t vocab_size,
                  std::uint64_t layers, std::string_view family);

// The ring capacity and attention window a context of up to `max_context` positions needs: the
// smallest of 2048, 4096 and 8192 (the attention Op's windows) that holds every position a
// drafted block can see, at most the trained window. A narrower window is exact, since no
// visible key is farther away than it.
[[nodiscard]] std::uint32_t ring_capacity(const Config& config, std::uint32_t max_context);

// Device bytes of one sequence's context ring at `capacity` (every layer's K and V, BF16).
[[nodiscard]] std::uint64_t ring_bytes(const Config& config, std::uint32_t capacity);

// Device bytes of the head-write table a Drafter derives from its target's head, BF16 [R, V].
[[nodiscard]] std::uint64_t head_write_bytes(const Config& config);

// Device bytes of the drafter's weights: its artifact's payload, which it materializes whole.
[[nodiscard]] std::uint64_t weight_bytes(const std::filesystem::path& path);

} // namespace ninfer::models::dspine
