#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "none") { return SpeculativeBackend::None; }
    if (value == "mtp") { return SpeculativeBackend::Mtp; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    if (value == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    }
    return "unknown";
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options);

// Settles the speculative options after every flag is parsed. The executables start from
// default_speculative() (MTP, 3 draft tokens, Auto proposal head), so "--spec none" clears the window
// and DFlash needs its own explicit --draft-tokens.
inline void finalize_speculative_cli_options(SpeculativeOptions& options,
                                             bool draft_tokens_explicit) {
    if (options.backend == SpeculativeBackend::None) {
        if (draft_tokens_explicit) {
            throw std::invalid_argument("--draft-tokens requires --spec mtp|dflash|dflash2");
        }
        options.draft_tokens = 0;
        if (options.proposal_head == ProposalHead::Auto) { options.proposal_head = ProposalHead::Full; }
    } else if (options.backend != SpeculativeBackend::Mtp && !draft_tokens_explicit) {
        throw std::invalid_argument("--spec dflash|dflash2 requires --draft-tokens");
    }
    validate_speculative_cli_options(options);
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    switch (options.backend) {
    case SpeculativeBackend::None:
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "--draft-tokens and --lm-head-draft require --spec mtp|dflash|dflash2");
        }
        return;
    case SpeculativeBackend::Mtp:
        if (options.draft_tokens == 0 || options.draft_tokens > 5) {
            throw std::invalid_argument("--spec mtp requires --draft-tokens in [1,5]");
        }
        return;
    case SpeculativeBackend::DFlash:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
        }
        return;
    case SpeculativeBackend::DFlash2:
        if (options.draft_tokens == 0 || options.draft_tokens > 15) {
            throw std::invalid_argument("--spec dflash2 requires --draft-tokens in [1,15]");
        }
        return;
    }
    throw std::invalid_argument("invalid speculative backend");
}

} // namespace ninfer::product
