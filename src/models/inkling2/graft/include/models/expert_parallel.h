#pragma once

#include "artifact/reader.h"

#include <cstdint>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::models {

// Expert parallelism's split when --tensor-split gives none: each rank's share of the routed
// experts in proportion to its speed, except that a rank takes no more experts than fit in its free
// memory beside what it holds anyway (rank 0 every other parameter the load binds) and a margin for
// state, workspace and the exchange. `loaded` picks the logical parameters a load binds to the
// device and `expert` those of the routed experts, by the artifact's parameter names.
[[nodiscard]] std::vector<double>
default_expert_split(const artifact::Reader& reader,
                     const std::function<bool(std::string_view)>& loaded,
                     const std::function<bool(std::string_view)>& expert,
                     std::span<const std::uint64_t> free_bytes, std::span<const double> speed);

} // namespace ninfer::models
