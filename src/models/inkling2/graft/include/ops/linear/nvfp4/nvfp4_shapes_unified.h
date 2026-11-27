#pragma once
#include "core/arena.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

namespace ninfer::ops::detail::unified {

// The plain Linear int8-grouped route (AllowNvfp4Int8): the per-(token,32) quantised activation
// planes in the caller workspace, addressed a whole 128-token tile at a time (a width that is not
// one is padded rather than refused, whenever the route admits it).
void nvfp4_linear_int8_grouped_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                      cudaStream_t);
// Token admission of the int8 route over a [min,max] interval, independent of the policy (the
// dispatcher combines it with allows_nvfp4_int8).
bool nvfp4_linear_int8_tokens_supported(std::int32_t min_tokens, std::int32_t max_tokens);

struct Nvfp4LinearShape {
    std::int32_t n, k;
    Nvfp4Launch a16;
    void (*a4)(const Tensor&, const Weight&, Tensor&, Nvfp4A4Workspace, cudaStream_t);
    bool (*uses_a4)(std::int32_t min_tokens, std::int32_t max_tokens);
    // The integer-activation route; null where a geometry does not register it.
    void (*int8)(const Tensor&, const Weight&, Tensor&, WorkspaceArena&, cudaStream_t);
    bool (*uses_int8)(std::int32_t min_tokens, std::int32_t max_tokens);
};

extern const Nvfp4LinearShape kNvfp4N14336K5120;
extern const Nvfp4LinearShape kNvfp4N16384K5120;
extern const Nvfp4LinearShape kNvfp4N34816K5120;
extern const Nvfp4LinearShape kNvfp4N5120K6144;
extern const Nvfp4LinearShape kNvfp4N5120K17408;
// A geometry no model in the tree carries: it exists so the op tests can contrast a registered
// table row against the geometry-only admission, which has no A16 composition to fall back on.
extern const Nvfp4LinearShape kNvfp4N8192K4096;

// The *geometry* admission the int8 route enforces, exposed so dispatch can serve a geometry that
// has no table row: N is a whole number of 128-row tiles and K a whole number of the 64-column
// k-tiles whose four G16 weight groups a tile carries. Width is not part of this contract -- the
// token tile is padded to 128 and its surplus columns discarded -- so a ragged or narrow T is
// admissible whenever the int8 family is registered at all (nvfp4_int8_route_admits), which the
// model layer decides with --nvfp4-w4a8-int.
constexpr bool nvfp4_shape_tile_aligned(std::int32_t n, std::int32_t k) {
    return n > 0 && k > 0 && n % 128 == 0 && k % 64 == 0;
}
} // namespace ninfer::ops::detail::unified
