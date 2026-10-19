// Scores fixed token sequences through the public Engine (CausalScoring, the ninfer-perplexity path)
// so two builds can be compared at the log-probability level.
//
//   ninfer-logit-compare <model.ninfer> --messages <file.json> [--ids "1 2 3"] [--candidates We,Let]
//                        [--context N] [--kv-dtype bf16|int8] [--device N]
//
// The prompt is the single user message of the messages file rendered with the default chat
// template and thinking enabled; the rendering is checked against Engine::count_tokens.
// Prints one JSON object: candidate first-token log-probabilities after the prompt and the
// log-probabilities of --ids (the continuation) given the prompt.
#include "core/os.h"
#include "ninfer/engine.h"
#include "product/defaults.h"
#include "product/prompt_input/prompt_input.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

std::vector<std::string> split(const std::string& text, char separator) {
    std::vector<std::string> out;
    std::stringstream stream(text);
    for (std::string item; std::getline(stream, item, separator);) { out.push_back(item); }
    return out;
}

int run(int argc, char** argv) {
    if (argc < 2) { throw std::invalid_argument("artifact path is required"); }
    std::filesystem::path artifact = argv[1];
    std::filesystem::path messages_path;
    std::string ids_text;
    std::string candidates_text;
    std::uint32_t context = 8192;
    int device            = 0;
    auto kv               = ninfer::KvCacheStorage::BFloat16;
    for (int i = 2; i < argc; ++i) {
        const std::string option = argv[i];
        const auto value         = [&] {
            if (++i >= argc) { throw std::invalid_argument(option + " requires a value"); }
            return std::string(argv[i]);
        };
        if (option == "--messages") {
            messages_path = value();
        } else if (option == "--ids") {
            ids_text = value();
        } else if (option == "--candidates") {
            candidates_text = value();
        } else if (option == "--context") {
            context = static_cast<std::uint32_t>(std::stoul(value()));
        } else if (option == "--device") {
            device = std::stoi(value());
        } else if (option == "--kv-dtype") {
            const std::string dtype = value();
            if (dtype == "int8") {
                kv = ninfer::KvCacheStorage::Int8Group64;
            } else if (dtype != "bf16") {
                throw std::invalid_argument("--kv-dtype must be bf16 or int8");
            }
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }

    const json messages = json::parse(std::ifstream(messages_path, std::ios::binary));
    const bool has_system = messages.is_array() && messages.size() == 2 && messages[0].at("role") == "system";
    if (!messages.is_array() || messages.size() != (has_system ? 2U : 1U) ||
        messages.back().at("role") != "user" || !messages.back().at("content").is_string() ||
        (has_system && !messages[0].at("content").is_string())) {
        throw std::invalid_argument(
            "messages must hold one user message, optionally after one system message, with string content");
    }
    // Default template (builtin sharp v22.5) for one user message with an optional system message,
    // thinking on, medium effort, no tools: the "terse" system text is always emitted. The result is checked against
    // Engine::count_tokens below.
    const std::string rendered =
        std::string("<|im_start|>system\n") +
        (has_system ? messages[0].at("content").get<std::string>() + "\n\n" : std::string()) +
        std::string("Answer directly, after thinking. Lead with the answer, then only what it needs "
                    "to be correct and usable.\n"
                    "Never: open with preamble or pleasantries; restate the question; add filler "
                    "transitions; hedge with niceties; or repeat a point you've already made.\n"
                    "Always: keep essential steps, caveats, uncertainties, and specifics \xe2\x80\x94 "
                    "never drop correctness or a needed warning for brevity. Keep the final answer "
                    "lean. Use the least structure that conveys it (plain prose when short; lists or "
                    "code only when they earn their place). If genuinely uncertain, say so and "
                    "explain why \xe2\x80\x94 never omit uncertainty for the sake of brevity.\n"
                    "If a user request is genuinely ambiguous, ask a sharp question, don't "
                    "guess.<|im_end|>\n<|im_start|>user\n") +
        messages.back().at("content").get<std::string>() +
        "<|im_end|>\n<|im_start|>assistant\n<think>\n";

    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.purpose       = ninfer::EnginePurpose::CausalScoring;
    options.device        = device;
    options.max_context   = context;
    options.kv_cache      = kv;
    ninfer::product::select_chat_template(options, {}, false);
    ninfer::Engine engine(std::move(options));

    const std::vector<ninfer::TokenId> prompt = engine.tokenize_text(rendered);
    const std::uint32_t engine_count = engine.count_tokens(
        ninfer::product::prompt_from_messages(messages_path, std::nullopt, false));
    if (engine_count != prompt.size()) {
        throw std::runtime_error("rendered prompt has " + std::to_string(prompt.size()) +
                                 " tokens but the engine's template gives " +
                                 std::to_string(engine_count));
    }

    json report{{"prompt_tokens", prompt.size()}, {"engine_prompt_tokens", engine_count}};

    json candidates = json::array();
    for (const std::string& text : split(candidates_text, ',')) {
        const std::vector<ninfer::TokenId> ids = engine.tokenize_text(text);
        if (ids.size() != 1) { throw std::runtime_error("candidate is not one token: " + text); }
        std::vector<ninfer::TokenId> input = prompt;
        input.push_back(ids[0]);
        const std::vector<float> lp =
            engine.score_tokens(std::move(input), static_cast<std::uint32_t>(prompt.size()));
        candidates.push_back({{"text", text}, {"id", ids[0]}, {"logprob", lp.at(0)}});
    }
    report["candidates"] = std::move(candidates);

    std::vector<ninfer::TokenId> continuation;
    for (const std::string& item : split(ids_text, ' ')) {
        if (!item.empty()) { continuation.push_back(static_cast<ninfer::TokenId>(std::stoul(item))); }
    }
    if (!continuation.empty()) {
        std::vector<ninfer::TokenId> input = prompt;
        input.insert(input.end(), continuation.begin(), continuation.end());
        report["continuation_ids"]      = continuation;
        report["continuation_logprobs"] =
            engine.score_tokens(std::move(input), static_cast<std::uint32_t>(prompt.size()));
    }
    std::cout << report.dump() << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    ninfer::os::init_console();
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "ninfer-logit-compare: " << error.what() << '\n';
        return 1;
    }
}
