#include "models/inkling2/config.h"

#include <cmath>
#include <limits>
#include <string>

namespace ninfer::models::inkling2 {
namespace {

using artifact::ArtifactError;
using artifact::Json;

std::uint32_t integer(const Json& config, const char* name, const char* what = "Inkling 2 config") {
    if (!config.contains(name)) { throw ArtifactError(std::string(what) + ": missing " + name); }
    const auto n = artifact::require_u64(config.at(name), name, true);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(name) + ": exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

float real(const Json& config, const char* name, bool zero = false,
           const char* what = "Inkling 2 config") {
    if (!config.contains(name) || !config.at(name).is_number()) {
        throw ArtifactError(std::string(what) + ": " + name + " must be a real value");
    }
    const float value = config.at(name).get<float>();
    if (!std::isfinite(value) || value < 0 || (!zero && value == 0)) {
        throw ArtifactError(std::string(what) + ": " + name + " is out of range");
    }
    return value;
}

void architecture(const Json& value, const char* name, const char* model_type) {
    const auto& architectures = value.at("architectures");
    if (!architectures.is_array() || architectures.size() != 1 || architectures[0] != name ||
        value.value("model_type", "") != model_type) {
        throw ArtifactError(std::string(name) + " config: architecture/model_type mismatch");
    }
}

std::vector<bool> layer_kinds(const Json& value, std::uint32_t count, const char* what) {
    const auto& layers = value.at("layer_types");
    if (!layers.is_array() || layers.size() != count) {
        throw ArtifactError(std::string(what) + ": layer_types must name every layer");
    }
    std::vector<bool> full;
    for (const auto& kind : layers) {
        if (kind == "full_attention") {
            full.push_back(true);
        } else if (kind == "sliding_attention") {
            full.push_back(false);
        } else {
            throw ArtifactError(std::string(what) + ": unknown layer type");
        }
    }
    return full;
}

} // namespace

std::vector<std::pair<std::uint32_t, std::uint32_t>> TextConfig::visits() const {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> out;
    if (num_passes <= 1) {
        for (std::uint32_t i = 0; i < num_hidden_layers; ++i) { out.emplace_back(i, 0); }
        return out;
    }
    for (std::uint32_t i = 0; i < core_start; ++i) { out.emplace_back(i, 0); }
    for (std::uint32_t k = 0; k < num_passes; ++k) {
        for (std::uint32_t i = core_start; i < core_end; ++i) { out.emplace_back(i, k); }
    }
    for (std::uint32_t i = core_end; i < num_hidden_layers; ++i) { out.emplace_back(i, 0); }
    return out;
}

TextConfig parse_text_config(const Json& value) {
    architecture(value, "Inkling2ForCausalLM", "inkling2");
    TextConfig out;
    out.hidden_size             = integer(value, "hidden_size");
    out.vocab_size              = integer(value, "vocab_size");
    out.unpadded_vocab_size     = integer(value, "unpadded_vocab_size");
    out.num_hidden_layers       = integer(value, "num_hidden_layers");
    out.max_position_embeddings = integer(value, "max_position_embeddings");
    out.rms_norm_eps            = real(value, "rms_norm_eps");
    out.num_attention_heads     = integer(value, "num_attention_heads");
    out.num_key_value_heads     = integer(value, "num_key_value_heads");
    out.head_dim                = integer(value, "head_dim");
    out.sliding_window          = integer(value, "sliding_window_size");
    out.relative_width          = integer(value, "d_rel");
    out.relative_extent         = integer(value, "rel_extent");
    out.qk_clip                 = real(value, "qk_l2_clip_threshold");
    out.relative_clip           = real(value, "rel_l2_clip_threshold");
    out.log_scale_alpha         = real(value, "log_scaling_alpha", true);
    out.log_scale_floor         = integer(value, "log_scaling_n_floor");
    out.conv_taps               = integer(value, "sconv_kernel_size");
    out.block_size              = integer(value, "n_layers_per_attn_res_block");
    out.num_experts             = integer(value, "n_routed_experts");
    out.top_k_experts           = integer(value, "num_experts_per_tok");
    out.moe_intermediate_size   = integer(value, "intermediate_size");
    out.gate_cap                = real(value, "mlp_gate_softcap");
    out.up_cap                  = real(value, "mlp_up_softcap");
    out.route_scale             = real(value, "route_scale");
    out.logit_softcap           = real(value, "final_logit_softcapping");
    out.full_attention          = layer_kinds(value, out.num_hidden_layers, "Inkling 2 config");
    if (!value.contains("loop") || !value.at("loop").is_object()) {
        throw ArtifactError("Inkling 2 config: missing loop");
    }
    const Json& loop = value.at("loop");
    out.num_passes   = integer(loop, "num_passes");
    if (out.num_passes > 1) {
        out.core_start = integer(loop, "core_start");
        out.core_end   = integer(loop, "core_end");
        out.gate_rank  = integer(loop, "gate_rank");
        if (out.core_start == 0 || out.core_start >= out.core_end ||
            out.core_end > out.num_hidden_layers || out.core_start % out.block_size != 0 ||
            (out.core_end - out.core_start) % out.block_size != 0) {
            throw ArtifactError("Inkling 2 config: the looped core must start after the first "
                                "block and span whole residual blocks");
        }
    }
    if (out.unpadded_vocab_size == 0 || out.unpadded_vocab_size > out.vocab_size) {
        throw ArtifactError("Inkling 2 config: the unpadded vocabulary exceeds the embeddings");
    }
    // The attention kernels implement heads of 128 values with at most 4 query heads per key/value
    // head, a relative feature of 16 and tables of at most 4096 distances; the convolution 4 taps;
    // the row kernels whole float4s of at most 4096 values and q8_1 blocks of 32; the routing at
    // most 8 of 256 experts; the depth mixes at most 8 states.
    const std::uint32_t group = out.num_key_value_heads != 0
                                    ? out.num_attention_heads / out.num_key_value_heads
                                    : 0;
    // A pass of the core starts again from the prelude's blocks: a mix sees at most one block per
    // block_size stored layers, and the current state.
    const std::uint32_t blocks =
        out.block_size != 0 ? (out.num_hidden_layers + out.block_size - 1) / out.block_size : 0;
    if (out.head_dim != 128 || group == 0 || group > 4 ||
        out.num_attention_heads != group * out.num_key_value_heads || out.relative_width != 16 ||
        out.relative_extent == 0 || out.relative_extent > 4096 || out.sliding_window == 0 ||
        out.sliding_window > out.relative_extent || out.conv_taps != 4 ||
        out.hidden_size % 256 != 0 || out.hidden_size > 4096 || out.num_experts > 256 ||
        out.top_k_experts == 0 || out.top_k_experts > 8 || out.top_k_experts > out.num_experts ||
        out.moe_intermediate_size % 32 != 0 || out.moe_intermediate_size > 4096 ||
        out.block_size == 0 || blocks + 1 > 8 || (out.num_passes > 1 && out.gate_rank > 256)) {
        throw ArtifactError("Inkling 2 config: geometry outside the implemented kernels");
    }
    return out;
}

VisionConfig parse_vision_config(const Json& value, const TextConfig& text) {
    architecture(value, "Inkling2VisionModel", "inkling2_vision");
    constexpr const char* what = "Inkling 2 vision config";
    VisionConfig out;
    out.hidden_size         = integer(value, "hidden_size", what);
    out.num_hidden_layers   = integer(value, "num_hidden_layers", what);
    out.num_attention_heads = integer(value, "num_attention_heads", what);
    out.intermediate_size   = integer(value, "intermediate_size", what);
    out.patch_size          = integer(value, "native_patch_size", what);
    out.merge_size          = integer(value, "adapter_merge_size", what);
    out.projection_hidden   = integer(value, "projection_hidden_dim", what);
    out.posemb_grid         = integer(value, "posemb_grid_size", what);
    out.text_hidden_size    = integer(value, "text_hidden_size", what);
    out.max_image_tokens    = integer(value, "max_image_tokens", what);
    out.image_token_id      = integer(value, "image_token_id", what);
    out.layer_norm_eps      = real(value, "layer_norm_eps", false, what);
    out.rope_theta          = real(value, "rope_theta", false, what);
    if (out.text_hidden_size != text.hidden_size || out.num_attention_heads == 0 ||
        out.hidden_size % out.num_attention_heads != 0 || out.head_dim() % 4 != 0 ||
        out.head_dim() > 128 || out.hidden_size % 4 != 0 || out.hidden_size > 4096 ||
        out.merge_size == 0 || out.max_image_tokens == 0 ||
        out.image_token_id >= text.unpadded_vocab_size) {
        throw ArtifactError("Inkling 2 vision config: geometry outside the implemented kernels");
    }
    return out;
}

DSparkConfig parse_dspark_config(const Json& value, const TextConfig& text) {
    architecture(value, "Inkling2DSpark", "inkling2_dspark");
    constexpr const char* what = "Inkling 2 DSpark config";
    DSparkConfig out;
    out.num_hidden_layers   = integer(value, "num_hidden_layers", what);
    out.num_attention_heads = integer(value, "num_attention_heads", what);
    out.num_key_value_heads = integer(value, "num_key_value_heads", what);
    out.head_dim            = integer(value, "head_dim", what);
    out.intermediate_size   = integer(value, "intermediate_size", what);
    out.block_size          = integer(value, "block_size", what);
    out.markov_rank         = integer(value, "markov_rank", what);
    out.sliding_window      = integer(value, "sliding_window_size", what);
    out.rope_theta          = real(value, "rope_theta", false, what);
    out.rms_norm_eps        = real(value, "rms_norm_eps", false, what);
    out.full_attention      = layer_kinds(value, out.num_hidden_layers, what);
    const auto& targets     = value.at("target_layer_ids");
    if (!targets.is_array() || targets.size() < 2) {
        throw ArtifactError("Inkling 2 DSpark config: target_layer_ids must name its taps");
    }
    for (const auto& target : targets) {
        out.target_visits.push_back(
            static_cast<std::uint32_t>(artifact::require_u64(target, "target_layer_ids", false)));
    }
    const std::uint32_t group = out.num_key_value_heads != 0
                                    ? out.num_attention_heads / out.num_key_value_heads
                                    : 0;
    if (out.target_visits.back() != text.visit_count() || out.head_dim != 128 || group == 0 ||
        group > 4 || out.num_attention_heads != group * out.num_key_value_heads ||
        out.block_size < 2 || out.block_size > 8 || out.markov_rank != 256 ||
        out.intermediate_size % 32 != 0 || out.target_visits.size() > 8) {
        throw ArtifactError("Inkling 2 DSpark config: geometry outside the implemented kernels");
    }
    for (std::size_t i = 1; i < out.target_visits.size(); ++i) {
        if (out.target_visits[i] <= out.target_visits[i - 1]) {
            throw ArtifactError("Inkling 2 DSpark config: target_layer_ids must ascend");
        }
    }
    return out;
}

} // namespace ninfer::models::inkling2
