#pragma once

// Defaults of the shipped executables (ninfer, ninfer-serve, ninfer-perplexity). They select the
// best-tested configuration for the GPU the binary is built for. Each one is an ordinary flag, so
// an ablation names the option it changes; the README lists the full-argument equivalent.

#include "ninfer/types.h"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

// One request's logical ceiling. The KV pool defaults to the same size. Turing builds raise it to
// 64k; other targets keep the upstream values.
#if defined(NINFER_SM75)
inline constexpr std::uint32_t kDefaultCliMaxContext   = 65536;
inline constexpr std::uint32_t kDefaultServeMaxContext = 65536;
#else
inline constexpr std::uint32_t kDefaultCliMaxContext   = 2048;
inline constexpr std::uint32_t kDefaultServeMaxContext = 8192;
#endif

// The executables render prompts with the vendored Sharp template unless --chat-template names
// another file or --artifact-chat-template selects the template stored in the artifact.
[[nodiscard]] inline std::filesystem::path resolve_chat_template(std::filesystem::path requested,
                                                                 bool artifact_template) {
    if (artifact_template) {
        if (!requested.empty()) {
            throw std::invalid_argument(
                "--artifact-chat-template cannot be combined with --chat-template");
        }
        return {};
    }
    return requested.empty() ? std::filesystem::path(NINFER_DEFAULT_CHAT_TEMPLATE) : requested;
}

// MTP draft window: best for code (acceptance 79% at 3); prose prefers 1 to 2.
inline constexpr std::uint32_t kDefaultDraftTokens = 3;

[[nodiscard]] constexpr KvCacheStorage default_kv_cache() noexcept {
#if defined(NINFER_SM75)
    // Turing has no FP8/NVFP4 KV; INT8 group-64 halves BF16 with no measured quality change.
    return KvCacheStorage::Int8Group64;
#else
    return KvCacheStorage::BFloat16;
#endif
}

// ninfer-perplexity scores with FP8 KV on other targets, where Turing has no FP8 cache.
[[nodiscard]] constexpr KvCacheStorage default_perplexity_kv_cache() noexcept {
#if defined(NINFER_SM75)
    return KvCacheStorage::Int8Group64;
#else
    return KvCacheStorage::Fp8E4M3Row256;
#endif
}

// Turing builds speculate with MTP and the optimized proposal head when the artifact carries one;
// other targets do not speculate unless asked.
[[nodiscard]] constexpr SpeculativeOptions default_speculative() noexcept {
#if defined(NINFER_SM75)
    return {SpeculativeBackend::Mtp, kDefaultDraftTokens, ProposalHead::Auto};
#else
    return {};
#endif
}

// Help-text line naming the Turing defaults, with its own indent and newline; empty on other
// targets, whose defaults match upstream's documented ones.
[[nodiscard]] inline std::string defaults_help(std::uint32_t max_context, std::string_view indent) {
#if defined(NINFER_SM75)
    return std::string(indent) + "defaults: --max-context " + std::to_string(max_context) +
           " (KV capacity the same), --kv-dtype int8, --spec mtp --draft-tokens " +
           std::to_string(kDefaultDraftTokens) +
           " with the optimized proposal head when the artifact has one (--spec none and "
           "--no-lm-head-draft switch them off)\n";
#else
    static_cast<void>(max_context);
    static_cast<void>(indent);
    return {};
#endif
}

} // namespace ninfer::product
