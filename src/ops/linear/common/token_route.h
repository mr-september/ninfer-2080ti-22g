#pragma once

// Token-count route tables for the per-shape linear selectors. A table lists launches in ascending
// order of the largest token count each one serves; the last entry is kAnyTokens.

#include <cstddef>
#include <cstdint>
#include <limits>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kAnyTokens = std::numeric_limits<std::int32_t>::max();

template <class Launch>
struct TokenRoute {
    std::int32_t max_tokens;
    Launch launch;
};

template <class Launch, std::size_t N>
[[nodiscard]] constexpr Launch select_token_route(const TokenRoute<Launch> (&routes)[N],
                                                  std::int32_t tokens) noexcept {
    static_assert(N > 0);
    for (const TokenRoute<Launch>& route : routes) {
        if (tokens <= route.max_tokens) { return route.launch; }
    }
    return routes[N - 1].launch;
}

} // namespace ninfer::ops::detail
