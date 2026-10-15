#pragma once

// Defaults of the shipped executables (ninfer, ninfer-serve, ninfer-perplexity). They select the
// best-tested configuration for the GPU the binary is built for. Each one is an ordinary flag, so an
// ablation names the option it changes; the README lists the full-argument equivalent.

#include "ninfer/types.h"

#include <cstdint>
#include <filesystem>
#include <stdexcept>

namespace ninfer::product {

// One request's logical ceiling. The KV pool defaults to the same size.
inline constexpr std::uint32_t kDefaultMaxContext = 65536;

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

// MTP speculation with the optimized proposal head when the artifact carries one.
[[nodiscard]] constexpr SpeculativeOptions default_speculative() noexcept {
    return {SpeculativeBackend::Mtp, kDefaultDraftTokens, ProposalHead::Auto};
}

} // namespace ninfer::product
