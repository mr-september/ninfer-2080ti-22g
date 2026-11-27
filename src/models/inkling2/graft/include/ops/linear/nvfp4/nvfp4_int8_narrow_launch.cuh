#pragma once

// Launch front end for the unified chain's narrow-band mainloop. Selection is width alone, through
// the same crossover MMVQ uses (`mmvq::applies`), and the caller additionally has to be on the
// unified table -- below the crossover the legacy chain stays on the MMVQ kernel it was calibrated
// with, so one crossover serves both chains.
//
// The instantiation set is a whitelist, written out one case at a time: one column stage per width
// from one to eight, each at the row factor that width's own stage declares (NC1 x R1,
// NC{2,3,4,5,6,7,8} x R2). It is a list rather than a derived ladder because a ladder over {NC} x {R}
// would be thirty-two -- every one of them a full instantiation of the mainloop in every projection
// route that takes this kernel. The switch below is the other half of the same discipline: adding a
// stage is a deliberate line, not a loop bound.
//
// Plane sizing is the caller's contract and it moves with this kernel: the column count is the call's
// width, so the activation planes are `tokens` rows, not padded to eight. Allocating the padded plane
// and quantising `tokens` (or the reverse) leaves columns a later read picks up uninitialised, and on
// the swiglu pair the gate half's plane feeds the up projection's SiLU -- which is why the five call
// sites change both numbers in the same breath.

#include "ops/linear/nvfp4/nvfp4_int8_narrow_host.h"
#include "ops/linear/nvfp4/nvfp4_int8_narrow_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_int8_route.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::narrow {

/// Rows one block owns at a whitelisted column stage. `StageRows` is the same answer as a template
/// argument so the unused row stages are never instantiated rather than compiled and jumped over.
template <int NC>
struct StageRows {
    static constexpr int value = rows_per_block(NC);
};

template <int NC, int R, class Output>
inline void launch_nc(const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                      const std::int8_t* x_codes, const float* x_scales, float alpha_half,
                      Output output, std::int32_t k, std::int32_t tokens, std::int32_t rows,
                      cudaStream_t stream) {
    static_assert(NC >= 1 && NC <= kMaxTokens, "column stage must be on the whitelist");
    static_assert(R >= 1 && R <= kMaxRowsPerBlock, "row stage must be on the whitelist");
    // Rows are never ragged: every NVFP4 route admits a weight with n % 128 == 0, and R divides 128.
    nvfp4_int8_narrow_kernel<NC, R, Output><<<rows / R, kThreads, 0, stream>>>(
        weight_codes, weight_scales, x_codes, x_scales, alpha_half, output, k, tokens);
}

/// NC is the call's width exactly; R is that stage's own row factor. Everything above `kMaxTokens` is
/// the wide mainloop's business and never reaches this function.
template <class Output>
inline void launch(const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
                   const std::int8_t* x_codes, const float* x_scales, float alpha_half,
                   Output output, std::int32_t k, std::int32_t tokens, std::int32_t rows,
                   cudaStream_t stream) {
    switch (columns(tokens)) {
        case 1:
            launch_nc<1, StageRows<1>::value, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                      alpha_half, output, k, tokens, rows, stream);
            return;
        case 2:
            launch_nc<2, StageRows<2>::value, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                      alpha_half, output, k, tokens, rows, stream);
            return;
        case 3:
            launch_nc<3, StageRows<3>::value, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                      alpha_half, output, k, tokens, rows, stream);
            return;
        case 4:
            launch_nc<4, StageRows<4>::value, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                      alpha_half, output, k, tokens, rows, stream);
            return;
        case 5:
            launch_nc<5, StageRows<5>::value, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                      alpha_half, output, k, tokens, rows, stream);
            return;
        case 6:
            launch_nc<6, StageRows<6>::value, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                      alpha_half, output, k, tokens, rows, stream);
            return;
        case 7:
            launch_nc<7, StageRows<7>::value, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                      alpha_half, output, k, tokens, rows, stream);
            return;
        case 8:
            launch_nc<8, StageRows<8>::value, Output>(weight_codes, weight_scales, x_codes, x_scales,
                                                      alpha_half, output, k, tokens, rows, stream);
            return;
        default:
            break;
    }
    throw std::invalid_argument("nvfp4 int8 narrow: no column stage for T=" +
                                std::to_string(tokens));
}

} // namespace ninfer::ops::detail::narrow
