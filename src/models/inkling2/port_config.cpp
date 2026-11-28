#include "models/inkling2/port_config.h"

#include "artifact/schema.h"

#include <cmath>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>

namespace ninfer::models::inkling2 {
namespace {

using artifact::ArtifactError;
using artifact::Json;
using artifact::require_members;

std::uint32_t integer(const Json& value, std::string_view label, bool positive = true) {
    const auto n = artifact::require_u64(value, label, positive);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(label) + ": config dimension exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

std::uint32_t dimension(const Json& config, const char* name) {
    return integer(config.at(name), name);
}

float positive_float(const Json& config, const char* name) {
    const auto& value = config.at(name);
    if (!value.is_number()) { throw ArtifactError(std::string(name) + " must be a real value"); }
    const float result = value.get<float>();
    if (!std::isfinite(result) || result <= 0) {
        throw ArtifactError(std::string(name) + " must be positive finite FP32");
    }
    return result;
}

// The v3 artifact carries donor-native key names; the M1 conversion uses the port names.
// Values are identical, so every lookup accepts either spelling.
const Json* find_any(const Json& config, std::initializer_list<const char*> names) {
    for (const char* name : names) {
        if (config.is_object() && config.contains(name)) { return &config.at(name); }
    }
    return nullptr;
}

std::uint32_t dimension_any(const Json& config, std::initializer_list<const char*> names) {
    const Json* value = find_any(config, names);
    if (!value) { throw ArtifactError(std::string("missing config key: ") + *names.begin()); }
    return integer(*value, *names.begin());
}

float positive_float_any(const Json& config, std::initializer_list<const char*> names) {
    const Json* value = find_any(config, names);
    if (!value) { throw ArtifactError(std::string("missing config key: ") + *names.begin()); }
    if (!value->is_number()) { throw ArtifactError(std::string(*names.begin()) + " must be a real value"); }
    const float result = value->get<float>();
    if (!std::isfinite(result) || result <= 0) {
        throw ArtifactError(std::string(*names.begin()) + " must be positive finite FP32");
    }
    return result;
}

AttentionConfig attention(const Json& value) {
    const Json& table = value.contains("attention") ? value.at("attention") : value;
    AttentionConfig out;
    out.relative_taps   = dimension_any(table, {"relative_taps", "d_rel"});
    out.relative_extent = dimension_any(table, {"relative_extent", "rel_extent"});
    out.qk_clip         = positive_float_any(table, {"qk_clip", "qk_l2_clip_threshold"});
    out.relative_clip   = positive_float_any(table, {"relative_clip", "rel_l2_clip_threshold"});
    out.log_scale_floor = positive_float_any(table, {"log_scale_floor", "log_scaling_n_floor"});
    out.log_scale_alpha = positive_float_any(table, {"log_scale_alpha", "log_scaling_alpha"});
    return out;
}

LoopConfig loop(const Json& value) {
    LoopConfig out;
    const Json* start = find_any(value, {"start", "core_start"});
    if (!start) { throw ArtifactError("missing config key: loop.start"); }
    out.start = integer(*start, "loop.start", /*positive=*/false);
    out.end             = dimension_any(value, {"end", "core_end"});
    out.count           = dimension_any(value, {"count", "num_passes"});
    out.controller_rank = dimension_any(value, {"controller_rank", "gate_rank"});
    if (out.end <= out.start || out.count < 2) {
        throw ArtifactError("loop requires end > start and count >= 2");
    }
    return out;
}

MoeConfig moe(const Json& value) {
    const Json& table = value.contains("moe") ? value.at("moe") : value;
    MoeConfig out;
    out.num_experts         = dimension_any(table, {"num_experts", "n_routed_experts"});
    out.num_experts_per_tok = dimension_any(table, {"num_experts_per_tok"});
    out.intermediate_size   = dimension_any(table, {"moe_intermediate_size", "intermediate_size"});
    out.router_scale        = positive_float_any(table, {"router_scale", "route_scale"});
    out.situ_beta           = positive_float_any(table, {"situ_beta", "mlp_gate_softcap"});
    out.situ_linear_beta    = positive_float_any(table, {"situ_linear_beta", "mlp_up_softcap"});
    if (out.num_experts_per_tok > out.num_experts) {
        throw ArtifactError("num_experts_per_tok exceeds num_experts");
    }
    return out;
}

} // namespace

Config config_from_directory(const artifact::Directory& directory) {
    const auto& config = directory.component("text").config;
    // Lenient on unknown members: artifact config objects may carry extra fields.
    for (std::string_view name :
         {"hidden_size", "vocab_size", "num_hidden_layers", "max_position_embeddings",
          "num_attention_heads", "num_key_value_heads", "head_dim", "layer_types",
          "rms_norm_eps", "loop", "final_logit_softcapping", "eos_token_id"}) {
        if (!config.contains(name)) {
            throw ArtifactError(std::string("text config: missing member ") + std::string(name));
        }
    }

    Config out;
    out.architecture            = Architecture::Inkling2;
    out.hidden_size             = dimension(config, "hidden_size");
    out.vocab_size              = dimension(config, "vocab_size");
    // Artifacts converted before the head-cut fix carry no unpadded member: fall back to the
    // padded vocab (uncut head — the historical behavior), never to zero.
    out.unpadded_vocab_size     = config.contains("unpadded_vocab_size")
                                      ? dimension(config, "unpadded_vocab_size")
                                      : out.vocab_size;
    if (out.unpadded_vocab_size == 0 || out.unpadded_vocab_size > out.vocab_size) {
        throw ArtifactError("text config: unpadded_vocab_size must be in [1, vocab_size]");
    }
    out.num_hidden_layers       = dimension(config, "num_hidden_layers");
    out.max_position_embeddings = dimension(config, "max_position_embeddings");
    out.num_attention_heads     = dimension(config, "num_attention_heads");
    out.num_key_value_heads     = dimension(config, "num_key_value_heads");
    out.head_dim                = dimension(config, "head_dim");
    out.sliding_window          = dimension_any(config, {"sliding_window", "sliding_window_size"});
    out.conv_taps               = dimension_any(config, {"conv_taps", "sconv_kernel_size"});
    out.attn_res_block_size     = dimension_any(config, {"attn_res_block_size", "n_layers_per_attn_res_block"});
    out.eos_token_id            = integer(config.at("eos_token_id"), "eos_token_id", false);
    out.rms_norm_eps            = positive_float(config, "rms_norm_eps");
    out.final_logit_softcapping = positive_float(config, "final_logit_softcapping");
    out.attention               = attention(config);
    out.loop                    = loop(config.at("loop"));
    out.moe                     = moe(config);

    const auto& types = config.at("layer_types");
    if (!types.is_array() || types.size() != out.num_hidden_layers) {
        throw ArtifactError("layer_types must name every layer");
    }
    out.full_attention_layer.reserve(out.num_hidden_layers);
    for (const auto& entry : types) {
        const auto name = artifact::require_id(entry, "layer_types entry");
        if (name == "full_attention") {
            out.full_attention_layer.push_back(true);
        } else if (name == "sliding_attention" || name == "windowed_attention") {
            out.full_attention_layer.push_back(false);
        } else {
            throw ArtifactError("layer_types entry must be full_attention or sliding_attention");
        }
    }

    if (out.loop.end > out.num_hidden_layers) {
        throw ArtifactError("loop.end exceeds num_hidden_layers");
    }
    if (out.query_width() % out.num_attention_heads != 0) {
        throw ArtifactError("query_width must divide by heads");
    }
    return out;
}

} // namespace ninfer::models::inkling2
