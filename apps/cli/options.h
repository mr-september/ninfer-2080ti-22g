#pragma once

#include "ninfer/types.h"
#include "product/defaults.h"
#include "product/logging/logging.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::cli {

struct Options {
    bool help_requested = false;

    std::filesystem::path artifact_path;
    std::filesystem::path chat_template_path;
    bool artifact_chat_template = false;
    std::filesystem::path grammar_path;
    std::filesystem::path json_schema_path;
    std::optional<std::string> regex;
    std::vector<std::string> choices;
    bool json_object = false;
    std::string prompt;
    std::filesystem::path messages_path;

    std::uint32_t max_new        = 128;
    std::uint32_t max_context    = product::kDefaultCliMaxContext;
    bool max_context_explicit    = false; // --max-context was given; never lowered to fit
    bool kv_capacity_explicit    = false; // --kv-capacity was given
    KvCapacityPolicy kv_capacity =
        KvCapacityPolicy::explicit_capacity(product::kDefaultCliMaxContext);
    std::uint32_t prefill_chunk  = 0;
    int device                   = 0;

    KvCacheStorage kv_cache        = product::default_kv_cache();
    SpeculativeOptions speculative = product::default_speculative();
    bool enable_vision  = false;
    std::uint32_t vision_max_item_tokens = kMaximumVisionItemTokens;
    bool use_cuda_graph = true;

    bool raw_output      = false;
    bool print_token_ids = false;
    std::optional<bool> enable_thinking;
    std::optional<std::uint32_t> thinking_budget;
    std::optional<ReasoningEffort> reasoning_effort;

    std::vector<TokenId> stop_token_ids;
    std::vector<StopString> stop_strings;

    // Omitted fields are resolved from the loaded model and rendered prompt mode by Engine.
    SamplingOverrides sampling;
    bool greedy                 = false;
    product::LogLevel log_level = product::LogLevel::Info;
};

[[nodiscard]] Options parse_options(int argc, char** argv);
[[nodiscard]] std::string usage_text(const char* argv0);

} // namespace ninfer::cli
