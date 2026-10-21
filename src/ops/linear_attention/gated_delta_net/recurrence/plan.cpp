#include "ops/linear_attention/gated_delta_net/recurrence/plan.h"
#include <climits>
#include <stdexcept>

namespace ninfer::ops::detail::gated_delta_net {
RecurrenceRoute select_recurrence_route(int tokens) {
    return tokens < 16 ? RecurrenceRoute::Recurrent : RecurrenceRoute::Chunked;
}

chunked::WorkspaceLayout recurrence_workspace_layout(int qk_heads, int value_heads, int tokens) {
    return select_recurrence_route(tokens) == RecurrenceRoute::Recurrent
               ? chunked::WorkspaceLayout{}
               : chunked::workspace_layout(qk_heads, value_heads, tokens);
}

RecurrencePlan select_recurrence_plan(int qk_heads, int value_heads, int tokens, int sm_count) {
    const auto route = select_recurrence_route(tokens);
    return {route,
            route == RecurrenceRoute::Chunked ? select_value_tile(value_heads, tokens, sm_count)
                                              : 0,
            recurrence_workspace_layout(qk_heads, value_heads, tokens)};
}

int select_value_tile(int value_heads, int /*tokens*/, int multiprocessor_count) {
    // Small grids need the narrowest stripe. Otherwise use fixed relative tile costs
    // to weight the SM-wave count. These are launch-heuristic weights, not measured latency.
    // Wider stripes reduce replicated packet reads but retain more state per CTA.
    if (static_cast<std::int64_t>(value_heads) * 16 <= multiprocessor_count) return 8;
    int best               = 64;
    std::int64_t best_work = INT64_MAX;

    struct Choice {
        int tile, cost;
    };

    for (const auto [tile, cost] : {Choice{64, 34}, Choice{32, 20}, Choice{16, 13}}) {
        const auto blocks = static_cast<std::int64_t>(value_heads) * (kStateDim / tile);
        const auto waves  = (blocks + multiprocessor_count - 1) / multiprocessor_count;
        const auto work   = waves * cost;
        if (work < best_work) {
            best      = tile;
            best_work = work;
        }
    }
    return best;
}
} // namespace ninfer::ops::detail::gated_delta_net
