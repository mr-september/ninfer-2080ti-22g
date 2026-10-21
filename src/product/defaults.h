#pragma once

// Defaults of the shipped executables (ninfer, ninfer-serve, ninfer-perplexity). They select the
// best-tested configuration for the GPU the binary is built for. Each one is an ordinary flag, so
// an ablation names the option it changes; the README lists the full-argument equivalent.

#include "ninfer/types.h"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <string>
#include <string_view>

namespace ninfer::product {

// One request's logical ceiling. The KV pool defaults to the same size. Turing builds raise it to
// 64k; other targets keep the upstream values.
inline constexpr std::uint32_t kDefaultCliMaxContext   = 65536;
inline constexpr std::uint32_t kDefaultServeMaxContext = 65536;

// The executables render prompts with the vendored Sharp template, embedded in the binary at
// configure time, unless --chat-template names another file or --artifact-chat-template selects
// the template stored in the artifact.
inline constexpr unsigned char kDefaultChatTemplateBytes[] = {
#include "product/default_chat_template.inc"
};
inline constexpr const char* kDefaultChatTemplateName = "builtin:qwen3_8_sharp_v22_5.jinja";

[[nodiscard]] inline std::string_view default_chat_template_source() noexcept {
    return {reinterpret_cast<const char*>(kDefaultChatTemplateBytes),
            sizeof(kDefaultChatTemplateBytes)};
}

// Checks the template flags while parsing, so a typo fails before any weights load.
inline void validate_chat_template_selection(const std::filesystem::path& requested,
                                             bool artifact_template) {
    if (artifact_template) {
        if (!requested.empty()) {
            throw std::invalid_argument(
                "--artifact-chat-template cannot be combined with --chat-template");
        }
        return;
    }
    if (requested.empty()) { return; }
    std::error_code error;
    const auto size = std::filesystem::file_size(requested, error);
    if (error || !std::filesystem::is_regular_file(requested)) {
        throw std::invalid_argument("cannot read --chat-template: " + requested.string());
    }
    if (size == 0 || size > 16ULL * 1024 * 1024) {
        throw std::invalid_argument(
            "--chat-template must be a nonempty UTF-8 source file up to 16 MiB: " +
            requested.string());
    }
}

// Applies the validated selection: the artifact's template when requested, otherwise the named
// file, otherwise the embedded default.
inline void select_chat_template(EngineOptions& engine, const std::filesystem::path& requested,
                                 bool artifact_template) {
    engine.chat_template_path.clear();
    engine.chat_template_source.clear();
    engine.chat_template_source_name.clear();
    if (artifact_template) { return; }
    if (!requested.empty()) {
        engine.chat_template_path = requested;
        return;
    }
    engine.chat_template_source      = std::string(default_chat_template_source());
    engine.chat_template_source_name = kDefaultChatTemplateName;
}

// MTP draft window: best for code (acceptance 79% at 3); prose prefers 1 to 2.
inline constexpr std::uint32_t kDefaultDraftTokens = 3;

[[nodiscard]] constexpr KvCacheStorage default_kv_cache() noexcept {
    // Turing has no FP8 KV route; INT8 group-64 halves BF16 with no measured quality change.
    return KvCacheStorage::Int8Group64;
}

// ninfer-perplexity scores with FP8 KV on other targets, where Turing has no FP8 cache.
[[nodiscard]] constexpr KvCacheStorage default_perplexity_kv_cache() noexcept {
    return KvCacheStorage::Int8Group64;
}

// Turing builds speculate with MTP and the optimized proposal head when the artifact carries one;
// other targets do not speculate unless asked.
[[nodiscard]] constexpr SpeculativeOptions default_speculative() noexcept {
    return {SpeculativeBackend::Mtp, kDefaultDraftTokens, ProposalHead::Auto};
}

// Help-text line naming the Turing defaults, with its own indent and newline; empty on other
// targets, whose defaults match upstream's documented ones.
[[nodiscard]] inline std::string defaults_help(std::uint32_t max_context, std::string_view indent) {
    return std::string(indent) + "defaults: --max-context " + std::to_string(max_context) +
           " (KV capacity the same), --kv-dtype int8, --spec mtp --draft-tokens " +
           std::to_string(kDefaultDraftTokens) +
           " with the optimized proposal head when the artifact has one (--spec none and "
           "--no-lm-head-draft switch them off)\n";
}

} // namespace ninfer::product
