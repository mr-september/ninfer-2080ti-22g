#pragma once

#include "ops/common/device_multiprocessors.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace ninfer::ops::detail {

// A runtime-shape route's candidate tile: the tokens and rows one CTA covers, and the measured time
// of one wave of its CTAs, in a unit common to the candidates.
struct WaveTile {
    std::int32_t tokens;
    std::int32_t rows;
    std::int32_t wave_cost;
};

// The candidate whose CTAs over `rows` by `tokens` fill the fewest whole waves of the current
// device's multiprocessors, each wave weighted by its cost, and of those the one with the most
// CTAs; the earlier one on a full tie. A candidate whose rows do not divide `rows` is skipped.
inline std::size_t cheapest_wave_tile(std::span<const WaveTile> tiles, std::int32_t rows,
                                      std::int32_t tokens) {
    const std::int64_t multiprocessors = current_device_multiprocessors(1);
    std::size_t best                   = tiles.size();
    std::int64_t best_cost             = std::numeric_limits<std::int64_t>::max();
    std::int64_t best_ctas             = 0;
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        const WaveTile& tile = tiles[i];
        if (rows % tile.rows != 0) continue;
        const std::int64_t ctas =
            static_cast<std::int64_t>((tokens + tile.tokens - 1) / tile.tokens) *
            (rows / tile.rows);
        const std::int64_t cost = (ctas + multiprocessors - 1) / multiprocessors * tile.wave_cost;
        if (cost < best_cost || (cost == best_cost && ctas > best_ctas)) {
            best      = i;
            best_cost = cost;
            best_ctas = ctas;
        }
    }
    if (best == tiles.size()) throw std::logic_error("wave tiles: no candidate divides the rows");
    return best;
}

} // namespace ninfer::ops::detail
