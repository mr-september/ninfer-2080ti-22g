#include "models/inkling2/model.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "core/qtype.h"
#include "core/tensor_split.h"
#include "models/expert_parallel.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/load/bindings.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::inkling2 {
namespace {

using qwen3_5::loading::Bindings;

// Zero bytes after a matrix whose rows are not whole 256-value steps: the integer matrix kernel
// reads past the last row's end (gguf_project, moe_matrix_product).
constexpr std::uint64_t kRowTail = 512;

FrontendResources bind_resources(artifact::Binder& binder, const TextConfig& config, bool vision) {
    const auto resource = [&](std::string_view role, std::string_view component = "text") {
        const auto bytes = binder.host_object(binder.resource(component, role));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    FrontendResources out;
    out.tokenizer_json         = resource("tokenizer.json");
    out.tokenizer_config_json  = resource("tokenizer_config.json");
    out.chat_template_jinja    = resource("chat_template.jinja");
    out.generation_config_json = resource("generation_config.json");
    if (vision) { out.processor_config_json = resource("processor_config.json", "vision"); }
    out.tokenizer =
        std::make_shared<const qwen3_5::frontend::Tokenizer>(qwen3_5::frontend::TokenizerResources{
            out.tokenizer_json, out.tokenizer_config_json, out.generation_config_json});
    const auto count = out.tokenizer->vocab_size();
    if (!count || count > config.unpadded_vocab_size ||
        !out.tokenizer->has_exact_token_domain(count)) {
        throw artifact::ArtifactError(
            "tokenizer must expose a contiguous public domain within the head's rows");
    }
    out.public_token_count = static_cast<std::uint32_t>(count);
    return out;
}

void tail(Bindings& b, WeightId id) {
    for (const auto& part : b.at(id).reference.binding.parts) {
        if (is_gguf(b.binder.geometry(part.object).format)) {
            b.binder.device_tail(part.object, kRowTail);
        }
    }
}

// Resolve stored GGUF rows directly. Disk mode validates without materializing the bank.
// Host mode borrows the materializer's pinned parent. Neither path repacks model weights.
std::vector<ExpertBlob> expert_blobs(const artifact::Reader& reader,
                                    const artifact::ParameterReference& reference,
                                    const artifact::MaterializedArtifact& backing,
                                    std::uint64_t experts, std::uint64_t rows, std::uint64_t k,
                                    QType& format, std::uint64_t& row_bytes) {
    std::vector<ExpertBlob> result;
    const auto& parts = reference.binding.parts;
    if (parts.size() != 1) {
        throw artifact::ArtifactError(reference.name + ": expert bank must be one stored region");
    }
    const auto& part = parts.front();
    const auto& geometry = reader.geometry(part.object);
    const bool bf16 = geometry.format == QType::BF16;
    if ((!is_gguf(geometry.format) && !bf16) || geometry.shape.size() != 2 ||
        geometry.shape[1] != k || part.begin % k || part.end - part.begin != experts * rows * k) {
        throw artifact::ArtifactError(reference.name +
                                      ": experts must be stored GGUF block rows or BF16 rows");
    }
    format = geometry.format;
    if (bf16) {
        row_bytes = k * 2;
    } else {
        const auto block = gguf_block_shape(format);
        if (k % block.elements) {
            throw artifact::ArtifactError(reference.name + ": partial stored quantization block");
        }
        row_bytes = k / block.elements * block.bytes;
    }
    reader.validate_object(part.object);
    const auto start = part.begin / k * row_bytes;
    for (std::uint64_t e = 0; e < experts; ++e) {
        ExpertBlob blob;
        blob.bytes = rows * row_bytes;
        blob.zero_tail = kRowTail;
        const auto offset = start + e * blob.bytes;
        // A host expert the pinned block could not hold is read from the files, as on disk.
        if (reference.residency == artifact::Residency::PinnedOrFiles &&
            backing.pinned(part.object)) {
            const auto* parent = backing.pinned_parent(part.object).data;
            blob.pinned = parent + offset;
            blob.pinned_region = reinterpret_cast<std::uintptr_t>(parent);
        } else {
            const auto logical = reader.directory().tensor(part.object).offset + offset;
            for (const auto& segment : reader.segments(logical, blob.bytes)) {
                blob.extents.push_back({segment.file_index, segment.file_offset, segment.bytes});
            }
        }
        result.push_back(std::move(blob));
    }
    return result;
}

bool component_is(const artifact::Reader& reader, std::string_view component,
                  std::string_view architecture) {
    const auto& components = reader.directory().components;
    const auto found       = components.find(std::string(component));
    if (found == components.end()) { return false; }
    const auto& config = found->second.config;
    return config.is_object() && config.contains("architectures") &&
           config.at("architectures").is_array() && config.at("architectures").size() == 1 &&
           config.at("architectures")[0].is_string() &&
           config.at("architectures")[0].get_ref<const std::string&>() == architecture;
}

} // namespace

Model::Model(TextConfig config, TextWeights weights, std::vector<BoundWeight> bound,
             FrontendResources resources, InstanceInfo info, artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), weights_(std::move(weights)),
      bound_(std::move(bound)), resources_(std::move(resources)), info_(std::move(info)) {}

Model::~Model() = default;

ops::WeightInput Model::input(WeightId id) const {
    const auto& parameter = weight(id);
    if (parameter.uses.size() != 1) {
        throw std::invalid_argument(parameter.name + ": a weight input needs exactly one use");
    }
    const auto& use = parameter.uses.front();
    return ops::WeightInput{parameter.view, use.policy, use.activation_input_divisor};
}

std::pair<std::uint32_t, std::uint32_t> Model::expert_range(std::size_t rank) const {
    if (expert_split_.empty()) { return {0, rank == 0 ? config_.num_experts : 0}; }
    const auto ends =
        split_shares(config_.num_experts, unit_positions(config_.num_experts, 1), expert_split_);
    return {std::uint32_t(ends.at(rank)), std::uint32_t(ends.at(rank + 1))};
}

std::vector<double> default_expert_split(const artifact::Reader& reader,
                                         std::span<const std::uint64_t> free_bytes,
                                         std::span<const double> speed,
                                         const std::filesystem::path& dspine) {
    std::vector<std::uint64_t> rank_free(free_bytes.begin(), free_bytes.end());
    if (!dspine.empty() && !rank_free.empty()) {
        const std::uint64_t drafter = dspine::weight_bytes(dspine) +
                                      dspine::head_write_bytes(dspine::read_config(dspine));
        rank_free.front() -= std::min(rank_free.front(), drafter);
    }
    return models::default_expert_split(
        reader, [](std::string_view name) { return name.starts_with("text/"); },
        [](std::string_view name) {
            return name.ends_with("/moe/gate_up") || name.ends_with("/moe/down");
        },
        rank_free, speed);
}

bool is_inkling2(const artifact::Reader& reader) {
    return component_is(reader, "text", "Inkling2ForCausalLM");
}

bool has_vision(const artifact::Reader& reader) {
    return is_inkling2(reader) && component_is(reader, "vision", "Inkling2VisionModel");
}

bool has_dspark(const artifact::Reader& reader) {
    return is_inkling2(reader) && component_is(reader, "dspark", "Inkling2DSpark");
}

std::unique_ptr<Model> load_model(const artifact::Reader& reader, const LoadOptions& options,
                                  DeviceContext& device, const StartupObserver* observer) {
    if (!is_inkling2(reader)) {
        throw artifact::ArtifactError("the artifact's text component is not Inkling2ForCausalLM");
    }
    TextConfig config = parse_text_config(reader.directory().component("text").config);
    if (!options.dspine.empty()) {
        if (options.dspark) {
            throw std::invalid_argument("Inkling 2 drafts with one drafter: its DSpark drafter or "
                                        "a DSpine drafter");
        }
        dspine::check_target(dspine::read_config(options.dspine), config.hidden_size,
                             config.vocab_size, config.visit_count(), "Inkling 2");
    }
    const bool vision = options.vision != VisionPlacement::None;
    if (vision && !has_vision(reader)) {
        throw artifact::ArtifactError("the artifact carries no Inkling 2 vision tower (convert "
                                      "with --components text,vision)");
    }
    if (options.dspark && !has_dspark(reader)) {
        throw artifact::ArtifactError("the artifact carries no Inkling 2 DSpark drafter (convert "
                                      "with --components text,dspark)");
    }
    artifact::Binder binder(reader);
    FrontendResources resources = bind_resources(binder, config, vision);
    Bindings b(binder);
    const bool offload = options.experts != ExpertResidency::Device;
    // Expert parallelism: rank r holds experts [ends[r], ends[r + 1]) of every layer.
    std::vector<std::uint64_t> ends;
    if (!options.expert_split.empty()) {
        if (offload) {
            throw std::invalid_argument("--tensor-parallel divides Inkling 2's experts over the "
                                        "devices: it needs --expert-residency device");
        }
        if (options.expert_split.size() != device.size() || device.size() < 2) {
            throw std::invalid_argument("expert parallelism needs one share per device rank");
        }
        ends = split_shares(config.num_experts, unit_positions(config.num_experts, 1),
                            options.expert_split);
    } else if (device.size() != 1) {
        throw std::invalid_argument("Inkling 2 runs on one device, or divides its experts over "
                                    "several with --tensor-parallel");
    }
    std::vector<std::array<artifact::ParameterReference, 2>> expert_references;
    const std::uint64_t h    = config.hidden_size;
    const std::uint64_t d    = config.head_dim;
    const std::uint64_t kv   = config.num_key_value_heads;
    const std::uint64_t q    = std::uint64_t(config.num_attention_heads) * d;
    const std::uint64_t taps = config.conv_taps;
    const std::uint64_t rel  = config.relative_width;
    const std::uint64_t e = config.num_experts, w = config.moe_intermediate_size;
    const auto vector = [&](const std::string& name, artifact::Shape shape) {
        return b.direct(name, std::move(shape), QType::FP32);
    };
    TextWeights weights;
    weights.token_embedding = b.parameter("text/token_embedding", {config.vocab_size, h});
    weights.embedding_norm  = vector("text/embedding_norm", {h});
    weights.output_head     = b.parameter("text/output_head", {config.vocab_size, h},
                                          {"text/final_hidden"});
    weights.final_norm      = vector("text/final_norm", {h});
    weights.output_mix      = vector("text/output_mix", {h});
    tail(b, weights.output_head);
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const std::string p = "text/layers/" + std::to_string(i) + "/";
        const std::string a = p + "attention/", m = p + "moe/";
        const bool full     = config.full_attention[i];
        LayerWeights layer;
        AttentionWeights& at = layer.attention;
        at.query    = b.parameter(a + "query", {q, h}, {a + "query_input"});
        at.gate     = b.parameter(a + "gate", {q, h}, {a + "query_input"});
        at.relative = b.parameter(a + "relative", {config.num_attention_heads * rel, h},
                                  {a + "query_input"});
        at.key      = b.parameter(a + "key", {kv * d, h}, {a + "key_input"});
        at.value    = b.parameter(a + "value", {kv * d, h}, {a + "value_input"});
        at.output   = b.parameter(a + "output", {h, q}, {a + "gated_output"});
        tail(b, at.output);
        at.sinks         = vector(a + "sinks", {config.num_attention_heads, h});
        at.query_clip    = vector(a + "query_clip", {config.num_attention_heads, d});
        at.relative_clip = vector(a + "relative_clip", {config.num_attention_heads, rel});
        at.distances     = vector(a + "distances",
                                  {full ? config.relative_extent : config.sliding_window, rel});
        at.key_conv      = vector(a + "key_conv", {kv * d, taps});
        at.value_conv    = vector(a + "value_conv", {kv * d, taps});
        at.output_conv   = vector(a + "output_conv", {h, taps});
        if (full) {
            at.query_norm = vector(a + "query_norm", {h});
            at.key_norm   = vector(a + "key_norm", {h});
            at.value_norm = vector(a + "value_norm", {h});
            at.key_taps   = vector(a + "key_taps", {kv * taps, h});
            at.value_taps = vector(a + "value_taps", {kv * taps, h});
        } else {
            at.input_norm = vector(a + "input_norm", {h});
        }
        if (i > 0) { layer.attention_mix = vector(p + "attention_mix", {full ? 3u : 1u, h}); }
        layer.ffn_mix  = vector(p + "ffn_mix", {h});
        layer.ffn_norm = vector(p + "ffn_norm", {h});
        MoeWeights& moe = layer.moe;
        moe.router      = vector(m + "router", {e, h});
        moe.router_bias = vector(m + "router_bias", {e});
        moe.output_conv = vector(m + "output_conv", {h, taps});
        if (binder.contains(m + "gate_up_scale")) {
            moe.gate_up_scale = vector(m + "gate_up_scale", {e});
        }
        if (binder.contains(m + "down_scale")) { moe.down_scale = vector(m + "down_scale", {e}); }
        if (offload) {
            const auto residency = options.experts == ExpertResidency::Host
                                       ? artifact::Residency::PinnedOrFiles
                                           : artifact::Residency::Values;
            expert_references.push_back({
                binder.parameter(m + "gate_up", {e * 2 * w, h}, residency),
                binder.parameter(m + "down", {e * h, w}, residency)});
            // Uses still belong to the family contract, even without a resident operand.
            for (const auto& use : {std::pair{m + "gate_up", m + "input"},
                                    std::pair{m + "down", m + "product"}}) {
                const auto& contract = binder.use(use.first, use.second);
                if (!contract.activation_policy || !contract.auxiliaries.empty()) {
                    throw artifact::ArtifactError(use.first +
                        ": streamed GGUF experts require an activation policy and no auxiliaries");
                }
            }
        } else if (!ends.empty()) {
            using Piece = qwen3_5::loading::ParameterPiece;
            for (std::size_t r = 0; r + 1 < ends.size(); ++r) {
                const Piece gate_up{Piece::Cut::Rows, {{ends[r] * 2 * w, ends[r + 1] * 2 * w}}};
                const Piece down{Piece::Cut::Rows, {{ends[r] * h, ends[r + 1] * h}}};
                moe.gate_up_shares.push_back(
                    b.piece(m + "gate_up", {e * 2 * w, h}, {m + "input"}, {}, r, gate_up));
                moe.down_shares.push_back(
                    b.piece(m + "down", {e * h, w}, {m + "product"}, {}, r, down));
                tail(b, moe.down_shares.back());
                tail(b, moe.gate_up_shares.back());
            }
        } else {
            moe.gate_up = b.parameter(m + "gate_up", {e * 2 * w, h}, {m + "input"});
            moe.down    = b.parameter(m + "down", {e * h, w}, {m + "product"});
            tail(b, *moe.down);
            tail(b, *moe.gate_up);
        }
        if (config.core(i)) {
            LoopWeights loop;
            loop.queries = vector(p + "loop/queries", {4, h});
            constexpr std::array<const char*, 4> kNames{"query", "key", "value", "ffn"};
            const std::uint64_t r = config.gate_rank;
            for (std::size_t c = 0; c < kNames.size(); ++c) {
                const std::string s = p + "loop/" + kNames[c] + "/";
                loop.controllers[c] = {vector(s + "down", {r, 2 * h}), vector(s + "gain", {h, r}),
                                       vector(s + "forget", {h, r})};
            }
            layer.loop = loop;
        }
        weights.layers.push_back(std::move(layer));
    }
    std::optional<VisionConfig> vision_config;
    std::optional<VisionWeights> vision_weights;
    if (vision) {
        const VisionConfig& vc = vision_config.emplace(
            parse_vision_config(reader.directory().component("vision").config, config));
        const auto residency = options.vision == VisionPlacement::Pinned
                                   ? artifact::Residency::Pinned : artifact::Residency::Device;
        const std::uint64_t vh = vc.hidden_size, vw = vc.intermediate_size;
        // BF16 matrices, or GGUF block rows as a llama.cpp mmproj stores them (Q8_0); the tower
        // checks which.
        const auto dense = [&](const std::string& name, artifact::Shape shape,
                               const std::string& input) {
            return b.parameter(name, std::move(shape), {input}, std::nullopt, residency);
        };
        VisionWeights& v   = vision_weights.emplace();
        v.patch_embedding  = dense("vision/patch_embedding", {vh, vc.patch_features()},
                                   "vision/patches");
        v.class_embedding  = vector("vision/class_embedding", {vh});
        v.class_position   = vector("vision/class_position", {vh});
        v.positions        = vector("vision/positions",
                                    {std::uint64_t(vc.posemb_grid) * vc.posemb_grid, vh});
        v.pre_norm         = vector("vision/pre_norm", {vh});
        v.pre_norm_bias    = vector("vision/pre_norm_bias", {vh});
        v.projector_up     = dense("vision/projector/up", {vc.projection_hidden, vc.merged_features()},
                                   "vision/merged");
        v.projector_out    = dense("vision/projector/out", {h, vc.projection_hidden},
                                   "vision/projector_hidden");
        for (std::uint32_t i = 0; i < vc.num_hidden_layers; ++i) {
            const std::string p = "vision/layers/" + std::to_string(i) + "/";
            const std::string a = p + "attention/";
            VisionLayerWeights layer;
            layer.attention_norm      = vector(p + "attention_norm", {vh});
            layer.attention_norm_bias = vector(p + "attention_norm_bias", {vh});
            layer.query       = dense(a + "query", {vh, vh}, p + "attention_input");
            layer.query_bias  = vector(a + "query_bias", {vh});
            layer.key         = dense(a + "key", {vh, vh}, p + "attention_input");
            layer.key_bias    = vector(a + "key_bias", {vh});
            layer.value       = dense(a + "value", {vh, vh}, p + "attention_input");
            layer.value_bias  = vector(a + "value_bias", {vh});
            layer.output      = dense(a + "output", {vh, vh}, a + "output_input");
            layer.output_bias = vector(a + "output_bias", {vh});
            layer.attention_scale = vector(p + "attention_scale", {vh});
            layer.mlp_norm        = vector(p + "mlp_norm", {vh});
            layer.mlp_norm_bias   = vector(p + "mlp_norm_bias", {vh});
            layer.up        = dense(p + "mlp/up", {vw, vh}, p + "mlp_input");
            layer.up_bias   = vector(p + "mlp/up_bias", {vw});
            layer.down      = dense(p + "mlp/down", {vh, vw}, p + "mlp_product");
            layer.down_bias = vector(p + "mlp/down_bias", {vh});
            layer.mlp_scale = vector(p + "mlp_scale", {vh});
            v.layers.push_back(layer);
        }
    }
    std::optional<DSparkConfig> dspark_config;
    std::optional<DSparkWeights> dspark;
    if (options.dspark) {
        const DSparkConfig& dc = dspark_config.emplace(
            parse_dspark_config(reader.directory().component("dspark").config, config));
        const std::uint64_t dq  = std::uint64_t(dc.num_attention_heads) * dc.head_dim;
        const std::uint64_t dkv = std::uint64_t(dc.num_key_value_heads) * dc.head_dim;
        const std::uint64_t rank = dc.markov_rank;
        const std::uint64_t width = dc.intermediate_size;
        DSparkWeights& dw   = dspark.emplace();
        dw.fuse            = vector("dspark/fuse", {dc.num_hidden_layers + 1u,
                                                    std::uint64_t(dc.target_visits.size())});
        dw.context_norm    = vector("dspark/context_norm", {h});
        dw.final_norm      = vector("dspark/final_norm", {h});
        dw.mask_embedding  = vector("dspark/mask_embedding", {h});
        dw.markov_embed    = b.direct("dspark/markov_embed", {config.vocab_size, rank}, QType::BF16);
        dw.joint           = vector("dspark/joint", {3 * rank, 2 * rank + 2 * h});
        dw.joint_bias      = vector("dspark/joint_bias", {3 * rank});
        dw.hidden_out      = vector("dspark/hidden_out", {h, rank});
        dw.hidden_out_bias = vector("dspark/hidden_out_bias", {h});
        for (std::uint32_t i = 0; i < dc.num_hidden_layers; ++i) {
            const std::string p = "dspark/layers/" + std::to_string(i) + "/";
            const std::string a = p + "attention/";
            DSparkLayerWeights layer;
            layer.context_norm   = vector(p + "context_norm", {h});
            layer.attention_norm = vector(p + "attention_norm", {h});
            layer.ffn_norm       = vector(p + "ffn_norm", {h});
            layer.query  = b.parameter(a + "query", {dq, h}, {p + "attention_input"});
            layer.key    = b.parameter(a + "key", {dkv, h}, {p + "attention_input"});
            layer.value  = b.parameter(a + "value", {dkv, h}, {p + "attention_input"});
            layer.output = b.parameter(a + "output", {h, dq}, {a + "output_input"});
            layer.context_key   = b.parameter(a + "context_key", {dkv, h}, {p + "context_input"});
            layer.context_value = b.parameter(a + "context_value", {dkv, h}, {p + "context_input"});
            layer.query_norm       = vector(a + "query_norm", {dc.head_dim});
            layer.key_norm         = vector(a + "key_norm", {dc.head_dim});
            layer.context_key_norm = vector(a + "context_key_norm", {dc.head_dim});
            layer.gate = b.parameter(p + "mlp/gate", {width, h}, {p + "mlp_input"});
            layer.up   = b.parameter(p + "mlp/up", {width, h}, {p + "mlp_input"});
            layer.down = b.parameter(p + "mlp/down", {h, width}, {p + "mlp_product"});
            for (const WeightId id : {layer.output, layer.down}) { tail(b, id); }
            dw.layers.push_back(layer);
        }
    }
    InstanceInfo info;
    info.name            = reader.directory().metadata.value("name", std::string("Inkling-2-Mini"));
    info.metadata_json   = reader.directory().metadata.dump();
    info.provenance_json = reader.directory().provenance.dump();
    info.artifact_id     = reader.artifact_id();
    auto pending         = std::move(b.weights);
    auto materialization = std::move(binder).finish();
    auto backing = artifact::materialize(reader, std::move(materialization), device, observer,
                                         nullptr, {.files_after_bytes = options.expert_host_bytes});
    auto bound   = qwen3_5::loading::resolve_weights(std::move(pending), backing);
    std::vector<ExpertBanks> expert_banks;
    for (const auto& reference : expert_references) {
        ExpertBanks bank;
        auto gu = expert_blobs(reader, reference[0], backing, e, 2 * w, h, bank.gate_up_type,
                               bank.gate_up_row_bytes);
        auto down = expert_blobs(reader, reference[1], backing, e, h, w, bank.down_type,
                                 bank.down_row_bytes);
        for (std::size_t x = 0; x < gu.size(); ++x) {
            bank.delivery.experts.push_back({std::move(gu[x]), std::move(down[x])});
        }
        // Gate rows, up rows, the [gate; up] parent, down rows: the executor's expert tables.
        bank.delivery.aliases = {{0, 0}, {0, w * bank.gate_up_row_bytes}, {0, 0}, {1, 0}};
        expert_banks.push_back(std::move(bank));
    }
    auto model = std::unique_ptr<Model>(new Model(std::move(config), std::move(weights),
                                                  std::move(bound), std::move(resources),
                                                  std::move(info), std::move(backing)));
    model->expert_residency_ = options.experts;
    model->expert_split_     = options.expert_split;
    model->expert_banks_     = std::move(expert_banks);
    model->vision_config_    = std::move(vision_config);
    model->vision_           = std::move(vision_weights);
    model->vision_placement_ = options.vision;
    model->dspark_config_    = std::move(dspark_config);
    model->dspark_           = std::move(dspark);
    if (options.experts == ExpertResidency::Disk || model->expert_file_bytes() > 0) {
        for (const auto& file : reader.directory().files) {
            model->expert_files_.push_back(file.path ? options.artifact.parent_path() / *file.path
                                                     : options.artifact);
        }
    }
    if (!options.dspine.empty()) {
        model->dspine_ = dspine::load_model(options.dspine, device, observer);
    }
    return model;
}

} // namespace ninfer::models::inkling2
