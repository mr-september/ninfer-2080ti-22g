#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "models/qwen3_5/load.h"

#include <algorithm>
#include <array>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace q35 = ninfer::models::qwen3_5;
using namespace ninfer::test::artifact_fixture;

Json added_token(int id, const std::string& content) {
    return {{"id", id},        {"content", content}, {"special", true},    {"single_word", false},
            {"lstrip", false}, {"rstrip", false},    {"normalized", false}};
}

struct ModelFixture {
    Fixture file;

    void tensor(const std::string& id, std::vector<std::uint64_t> shape, QType format = QType::BF16,
                QuantLayout layout = QuantLayout::Contiguous) {
        const auto geometry = weight_geometry(format, layout, shape);
        const auto offset   = (file.payload.size() + 255) / 256 * 256;
        file.payload.resize(offset + geometry.bytes);
        file.root["objects"].push_back({{"id", id},
                                        {"kind", "tensor"},
                                        {"shape", shape},
                                        {"format", artifact::format_name(format)},
                                        {"layout", artifact::layout_name(layout)},
                                        {"offset", offset},
                                        {"bytes", geometry.bytes}});
    }

    void resource(const std::string& role, const std::string& bytes) {
        const auto offset = file.payload.size();
        for (const auto byte : bytes) { file.payload.push_back(std::byte(byte)); }
        file.root["objects"].push_back({{"id", role},
                                        {"kind", "resource"},
                                        {"encoding", "raw_bytes_v1"},
                                        {"offset", offset},
                                        {"bytes", bytes.size()}});
        file.root["components"]["text"]["resources"][role] = role;
    }

    void use(const std::string& name, const std::string& input, std::optional<int> divisor = {}) {
        Json value{{"parameter", name}, {"input", input}, {"activation_policy", "A16Only"}};
        if (divisor) {
            value["auxiliaries"] = {
                {"activation_input_divisor",
                 {{"parts", Json::array({{{"object", "calibration"},
                                          {"range", {*divisor, *divisor + 1}}}})}}}};
        }
        file.root["uses"].push_back(value);
    }

    void parameter(const std::string& name, std::vector<std::uint64_t> shape,
                   const std::string& input = {}) {
        tensor(name, shape);
        file.root["bindings"][name] = {{"object", name}};
        if (!input.empty()) { use(name, input); }
    }

    ModelFixture() {
        file.payload.clear();
        file.root = {{"components",
                      {{"text",
                        {{"config",
                          {{"architectures", {"Qwen3_5ForCausalLM"}},
                           {"model_type", "qwen3_5_text"},
                           {"hidden_size", 128},
                           {"vocab_size", 272},
                           {"num_hidden_layers", 1},
                           {"max_position_embeddings", 128},
                           {"tie_word_embeddings", true},
                           {"rms_norm_eps", 1e-6},
                           {"layer_types", {"full_attention"}},
                           {"num_attention_heads", 2},
                           {"num_key_value_heads", 1},
                           {"head_dim", 8},
                           {"intermediate_size", 24},
                           {"rope_parameters",
                            {{"rope_theta", 10000},
                             {"partial_rotary_factor", 0.5},
                             {"mrope_section", {1, 1, 0}}}}}}}}}},
                     {"objects", Json::array()},
                     {"bindings", Json::object()},
                     {"uses", Json::array()},
                     {"metadata", {{"name", "a-user-trained-model"}}}};
        tensor("embedding-storage", {272, 128});
        file.root["bindings"]["text/token_embedding"] = {{"object", "embedding-storage"}};
        file.root["bindings"]["text/output_head"]     = {{"object", "embedding-storage"}};
        use("text/output_head", "text/final_hidden");
        parameter("text/final_norm", {128});
        const std::string p = "text/layers/0/";
        parameter(p + "input_norm", {128});
        parameter(p + "post_attention_norm", {128});
        tensor("mixer-storage", {128, 128}, QType::FP8_E4M3FN_BLOCK128X128_BF16,
               QuantLayout::BlockScaleM128K128);
        tensor("calibration", {2}, QType::FP32);
        const auto calibration = file.root["objects"].back()["offset"].get<std::size_t>();
        put_word(file.payload, calibration, 0x40000000, 4);
        put_word(file.payload, calibration + 4, 0x40400000, 4);
        const std::array roles = {"query", "key", "gate", "value"};
        const std::array rows  = {0, 16, 24, 40, 48};
        for (std::size_t i = 0; i < roles.size(); ++i) {
            const auto name             = p + "attention/" + roles[i];
            file.root["bindings"][name] = {
                {"parts", Json::array({{{"object", "mixer-storage"},
                                        {"range", {rows[i] * 128, rows[i + 1] * 128}}}})}};
            use(name, p + "mixer_input", i < 2 ? 0 : 1);
        }
        parameter(p + "attention/query_norm", {8});
        parameter(p + "attention/key_norm", {8});
        parameter(p + "attention/output", {128, 16}, p + "attention/gated_output");
        parameter(p + "mlp/gate", {24, 128}, p + "ffn_input");
        parameter(p + "mlp/up", {24, 128}, p + "ffn_input");
        tensor("down-storage", {128, 24}, QType::Q6_G64_FP16, QuantLayout::RowSplit);
        file.root["bindings"][p + "mlp/down"] = {{"object", "down-storage"}};
        use(p + "mlp/down", p + "mlp/product");

        Json vocab    = Json::object();
        unsigned next = 256;
        for (unsigned byte = 0; byte < 256; ++byte) {
            const bool visible =
                (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174;
            const unsigned cp = visible ? byte : next++;
            std::string token;
            if (cp < 128) {
                token.push_back(static_cast<char>(cp));
            } else {
                token.push_back(static_cast<char>(0xc0 | (cp >> 6)));
                token.push_back(static_cast<char>(0x80 | (cp & 63)));
            }
            vocab[token] = byte;
        }
        Json added                = Json::array();
        Json decoder              = Json::object();
        const std::array specials = {"<|endoftext|>",  "<|im_start|>",  "<|im_end|>",
                                     "<think>",        "</think>",      "<|vision_start|>",
                                     "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"};
        for (std::size_t i = 0; i < specials.size(); ++i) {
            const auto token = added_token(static_cast<int>(256 + i), specials[i]);
            if (i < 8) { added.push_back(token); }
            decoder[std::to_string(256 + i)] = token;
        }
        resource("tokenizer.json",
                 Json{{"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", Json::array()}}},
                      {"added_tokens", added}}
                     .dump());
        resource("tokenizer_config.json", Json{{"added_tokens_decoder", decoder}}.dump());
        resource("generation_config.json", Json{{"eos_token_id", {256, 258}}}.dump());
        resource("chat_template.jinja", "a user-provided template, preserved as data");
        file.root["files"] =
            Json::array({{{"path", nullptr}, {"payload_bytes", file.payload.size()}}});
    }
};

void logical_data_and_instances() {
    ModelFixture fixture;
    fixture.file.root["components"]["vision"] = {{"config", Json::object()}, {"target", "text"}};
    fixture.file.root["components"]["text"]["config"]["rope_parameters"]["partial_rotary_factor"] =
        0.4999999999;
    fixture.file.write();
    artifact::Reader reader(fixture.file.entry);
    auto plan = q35::plan_load(reader);
    require(plan.config().text.rope_parameters->partial_rotary_factor == 0.5F &&
                plan.config().text.rope_parameters->rotary_dim == 4,
            "rotary width was derived before normalizing the PositiveF32 config value");
    require(plan.config().text.architecture == Architecture::Qwen3_5 &&
                plan.config().text.hidden_size == 128,
            "binding chose a checkpoint-specific geometry");
    require(plan.resources().public_token_count == 265 &&
                plan.resources().tokenizer->encode("<|video_pad|>") == std::vector<int>{264},
            "tokenizer_config added tokens were not included in the public domain");
    const auto& attention = std::get<q35::AttentionWeights>(plan.weights().text.layers[0].mixer);
    const auto& query     = plan.parameter(attention.query);
    const auto& gate      = plan.parameter(attention.gate);
    require(query.binding.parts[0].object == gate.binding.parts[0].object &&
                query.binding.parts[0].begin == 0 && gate.binding.parts[0].begin == 24 * 128,
            "Q/gate logical row correspondence changed");
    require(plan.uses(attention.query)[0].activation_input_divisor == 2.0F &&
                plan.uses(attention.gate)[0].activation_input_divisor == 3.0F,
            "shared-parent Uses contaminated one another");
    require(plan.parameter(plan.weights().text.token_embedding).binding.parts[0].object ==
                plan.parameter(plan.weights().text.output_head).binding.parts[0].object,
            "explicit shared embedding/head did not bind the same parent");
    require(!plan.weights().vision && plan.resources().preprocessor_config_json.empty(),
            "unused Vision was required");
    rejects([&] { (void)q35::plan_load(reader, {.vision = true}); },
            "incomplete selected Vision was accepted");
    rejects([&] { (void)q35::plan_load(reader, {.speculative = SpeculativeBackend::Mtp}); },
            "absent MTP was accepted");

    const auto resolved                   = q35::resolve_load(std::move(plan), Device::cuda(0));
    const auto capacity                   = resolved.materialization().device_capacity_bytes;
    fixture.file.root["metadata"]["name"] = "another-training-run-and-recipe";
    fixture.file.root["components"]["text"]["config"]["num_hidden_layers"] = 2;
    fixture.file.root["components"]["text"]["config"]["layer_types"]       = {"full_attention",
                                                                              "full_attention"};
    const auto original_bindings = fixture.file.root["bindings"];
    for (const auto& [name, binding] : original_bindings.items()) {
        if (name.starts_with("text/layers/0/")) {
            auto second = name;
            second.replace(12, 1, "1");
            fixture.file.root["bindings"][second] = binding;
        }
    }
    const auto original_uses = fixture.file.root["uses"];
    for (auto use : original_uses) {
        auto parameter = use["parameter"].get<std::string>();
        if (parameter.starts_with("text/layers/0/")) {
            auto input = use["input"].get<std::string>();
            parameter.replace(12, 1, "1");
            input.replace(12, 1, "1");
            use["parameter"] = parameter;
            use["input"]     = input;
            fixture.file.root["uses"].push_back(use);
        }
    }
    fixture.file.write();
    artifact::Reader changed(fixture.file.entry);
    auto expanded = q35::plan_load(changed);
    require(expanded.weights().text.layers.size() == 2 &&
                expanded.config().text.compact_layer_indices == std::vector<std::uint32_t>{0, 1},
            "config did not drive layer geometry and compact indices");
    require(q35::resolve_load(std::move(expanded), Device::cuda(0))
                    .materialization()
                    .device_capacity_bytes == capacity,
            "cross-layer sharing duplicated parent storage");
}

void native_uses() {
    const std::array<std::uint64_t, 2> shape{256, 128};
    const auto geometry = weight_geometry(QType::FP8_E4M3FN_BLOCK128X128_BF16,
                                          QuantLayout::BlockScaleM128K128, shape);
    std::vector<std::byte> bytes(geometry.bytes);
    const WeightParent parent{geometry, Device::cpu(), bytes.data(), {}};
    const WeightView gate{{128, 128}, {{&parent, 0, 16384}}};
    const WeightView up{{128, 128}, {{&parent, 16384, 32768}}};
    const auto joined = ops::prepare_linear_swiglu_weight({gate, ops::LinearPolicy::AllowA8},
                                                          {up, ops::LinearPolicy::AllowA8});
    require(joined.policy == ops::LinearPolicy::AllowA8 && joined.weight.payload == bytes.data() &&
                joined.weight.n == 256,
            "shared quantization did not keep its Use policy or preserve the complete parent");
    const WeightView full{{256, 128}, {{&parent, 0, 32768}}};
    for (const auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8}) {
        const auto single = ops::prepare_linear_weight({full, policy});
        require(single.policy == policy && single.weight.payload == bytes.data(),
                "native use altered its policy or lost the complete parent");
        const auto mixed = ops::prepare_linear_swiglu_weight({gate, ops::LinearPolicy::AllowA8},
                                                             {up, policy});
        require(mixed.policy == policy, "combined use did not intersect the Use policies");
    }
    rejects<std::invalid_argument>(
        [&] {
            (void)ops::prepare_linear_swiglu_weight({up, ops::LinearPolicy::A16Only},
                                                    {gate, ops::LinearPolicy::A16Only});
        },
        "reordered parent regions silently changed fused gate/up row order");
    const WeightView short_gate{{128, 128}, {{&parent, 0, 16320}}};
    const WeightView long_up{{128, 128}, {{&parent, 16320, 32768}}};
    rejects<std::invalid_argument>(
        [&] {
            (void)ops::prepare_linear_swiglu_weight({short_gate, ops::LinearPolicy::A16Only},
                                                    {long_up, ops::LinearPolicy::A16Only});
        },
        "combined coverage hid an incorrect gate/up boundary");
}

void invalid_model_data() {
    ModelFixture fixture;
    const Json original = fixture.file.root;
    const auto bad      = [&](auto mutate) {
        fixture.file.root = original;
        mutate(fixture.file.root);
        fixture.file.write();
        rejects(
            [&] {
                artifact::Reader reader(fixture.file.entry);
                (void)q35::plan_load(reader);
            },
            "invalid model data was accepted");
    };
    bad([](Json& root) { root["components"]["text"]["config"]["layer_types"] = Json::array(); });
    bad([](Json& root) { root["components"]["text"]["config"]["num_key_value_heads"] = 3; });
    bad([](Json& root) {
        root["components"]["text"]["config"]["rope_parameters"]["mrope_section"] = {0, 2, 0};
    });
    bad([](Json& root) {
        root["bindings"]["text/layers/0/attention/query"]["parts"][0]["range"] = {0, 8 * 128};
    });
    bad([](Json& root) { root["uses"][1].erase("activation_policy"); });
    bad([](Json& root) { root["components"]["text"]["config"]["vocab_size"] = 264; });
}

} // namespace

int main() {
    try {
        logical_data_and_instances();
        native_uses();
        invalid_model_data();
        std::cout << "model config, binding, resources and Use checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
