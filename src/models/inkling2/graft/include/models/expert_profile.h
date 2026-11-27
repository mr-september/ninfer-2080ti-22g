#pragma once

// Routing profiles of MoE models with host or disk experts: how often each layer's experts were
// routed, which fills a device expert cache at startup and sizes each layer's share of its slots.
// A recorded profile (--expert-profile-out) names its artifact and bank widths, so it cannot rank
// another model's experts. The built-in profiles were recorded on a family's release and serve
// every artifact of the same family and bank geometry: a profile only seeds the cache, whose tier
// then follows the conversation, so it never changes a result.

#include "artifact/framing.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::models {

using ExpertRouteCounts = std::vector<std::vector<std::uint64_t>>;

// Counts address the family's expert layers in order (Flash-Next: the text layers, then the MTP
// block's). An artifact identity and exact bank widths prevent a profile from silently ranking
// another model's experts. A run that leaves out trailing layers (Flash-Next without its MTP
// block: no drafts, DFlash2, perplexity) reads the leading layers' counts of a profile recorded
// with them.
ExpertRouteCounts read_expert_profile(const std::filesystem::path& path,
    const artifact::ArtifactId& artifact, std::span<const std::size_t> experts);
void write_expert_profile(const std::filesystem::path& path,
    const artifact::ArtifactId& artifact, const ExpertRouteCounts& counts);

// The built-in profile of `family` ("qwen3.8-flash-next", ...) for layers of these bank widths:
// one recorded on exactly these layers, else the leading layers of one recorded with more, or
// nothing when none was recorded for that geometry.
std::optional<ExpertRouteCounts> builtin_expert_profile(std::string_view family,
                                                        std::span<const std::size_t> experts);

} // namespace ninfer::models
