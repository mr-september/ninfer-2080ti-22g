#pragma once

// The activation-compute policies of the linear projections, their width gates and the share units
// of tensor-parallel matrices, which the route code of every quantised format reads; the Op entry
// points are in ninfer/ops/linear.h, which includes this header.

// graft shim: the donor tree defines LinearPolicy, valid_linear_policy and allows_a8 in this
// header. ninfer owns that name, so ninfer/ops/linear.h carries the donor's full policy set
// and this header adds only the donor-only helpers and constants the route code reads.

#include "ninfer/ops/linear.h"

namespace ninfer::ops {

[[nodiscard]] constexpr bool allows_a4(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowA4;
}

/// Integer-A8 (s8 activation, groupwise-int weight) profiles. Never implies the FP8 A8 path.
[[nodiscard]] constexpr bool allows_a8_int(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowA8Int || policy == LinearPolicy::AllowA8IntDecode ||
           policy == LinearPolicy::AllowPrefillCublas;
}

/// The NVFP4 int8-grouped route: the same activation trade as allows_a8_int, but for block-scaled
/// NVFP4 weights against their stored E2M1/E4M3 planes, and never for any other weight format.
[[nodiscard]] constexpr bool allows_nvfp4_int8(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowNvfp4Int8;
}

/// The cuBLAS prefill route. Only ever admitted past a width gate, because its dequantise pass is
/// weight-sized and has to be amortised over the tokens in the call.
[[nodiscard]] constexpr bool allows_cublas_prefill(LinearPolicy policy) noexcept {
    return policy == LinearPolicy::AllowPrefillCublas;
}

/// The route's width gate where the device profile names none: from here it beats the integer
/// mainloop it replaces on every card measured. The dequantise pass costs the same whatever the
/// token count, so a narrow call pays for it without the GEMM to amortise it; where the narrower
/// gate lies depends on the card and on whether the width is a multiple of the integer route's
/// 128-column tile, so a calibrated profile places it per Op (`w4_cublas/...` keys).
inline constexpr std::int32_t kCublasPrefillMinTokens = 512;

/// No profile moves the gate to this width or below it: a decode or verify round runs up to
/// kMaximumConcurrency lanes of a 16-token window, 8 x 16 = 128 columns, through the same Ops and
/// policies, and those rounds stay off the route.
inline constexpr std::int32_t kCublasPrefillFloorTokens = 128;

// The rows and K columns that a share of a matrix (a tensor-parallel rank's piece) is a whole
// number of in its format, so that a runtime-shape route takes it, beyond the stored layout's units:
// T2 runs 32-row tiles over 512-column K slabs, FP8 64 rows over 512 and NVFP4 128 over 256; every
// other format adds none.
struct LinearShareUnits {
    std::uint64_t rows    = 1;
    std::uint64_t columns = 1;
};

} // namespace ninfer::ops
