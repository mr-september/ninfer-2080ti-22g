#pragma once

#include "ops/softmax_attention/kernels/causal/int8/schedule.cuh"

namespace ninfer::ops::detail {

template <class G, int Tokens>
struct Int8KvGroupedInstance {
    static_assert(Tokens > 0 && Tokens * G::GroupSize <= 64);
    static constexpr int kRowTiles  = (Tokens * G::GroupSize + 15) / 16;
    // The 64-key arena (4 * 64 * 256 B) plus Q state exceeds a Turing CTA's 64 KiB.
    static constexpr bool kWideKeys = false;
    using Schedule                  = Int8KvGroupedMmaSchedule<Tokens,
                                              kWideKeys        ? 4 * kRowTiles
                                                               : kRowTiles <= 2 ? 8
                                                                                : 2 * kRowTiles,
                                              kWideKeys ? 64 : 32, kWideKeys ? 1 : 2>;
    using Merge                     = Int8KvMergeSchedule<G::QHeads == 24 ? 256 : 64>;
};

// 64x64 stages 92 KiB, too large for a Turing CTA; 64x32 stages 54 KiB and doubles the warps per
// CTA (and the reuse of each Q tile) over 32x32, which stages 44 KiB.
using Int8KvTiledInstance = Int8KvTiledMmaSchedule<64, 32, 128>;

} // namespace ninfer::ops::detail
