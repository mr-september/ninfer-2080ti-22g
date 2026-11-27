#pragma once

#include "ops/linear/common/output.cuh"

namespace ninfer::ops::detail {

// Thread-local operations over a fully reduced accumulator. The contraction
// owns predicates and synchronization; epilogues only receive valid coordinates.
//
// apply_scaled(row, token, value, scale) is apply(row, token, value * scale) with the rounding
// stated by the epilogue. A kernel that scales under a live-column predicate calls it, so the
// compiler's MUL+ADD contraction choice cannot differ between full and partial tiles and a
// column's result does not depend on how many tokens share its tile.
struct LinearIdentityEpilogue {
    __device__ __forceinline__ float apply(int, int, float value) const { return value; }

    __device__ __forceinline__ float apply_scaled(int, int, float value, float scale) const {
        return value * scale;
    }
};

struct LinearResidualAddEpilogue {
    LinearBf16InputView residual;

    __device__ __forceinline__ float apply(int row, int token, float value) const {
        return value + residual.load(row, token);
    }

    // One rounding for the scaled residual update.
    __device__ __forceinline__ float apply_scaled(int row, int token, float value,
                                                  float scale) const {
        return __fmaf_rn(value, scale, residual.load(row, token));
    }
};

// Row consumers run in the single thread owning the fully reduced row. They
// may publish fused Op state, but do not introduce warp/CTA synchronization.
template <class Output, class Epilogue, int Tokens>
__device__ __forceinline__ void
linear_finish_row(const Output& output, const Epilogue& epilogue, int row, int token_begin,
                  const float (&values)[Tokens], int active_tokens) {
    if constexpr (requires {
                      epilogue.apply_row(output, row, token_begin, values, active_tokens);
                  }) {
        epilogue.apply_row(output, row, token_begin, values, active_tokens);
    } else {
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            if (token < active_tokens) {
                const int global_token = token_begin + token;
                output.store(row, global_token, epilogue.apply(row, global_token, values[token]));
            }
        }
    }
}

} // namespace ninfer::ops::detail
