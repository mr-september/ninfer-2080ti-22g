#pragma once

#include <cfloat>

namespace ninfer::ops {

// A router score as expert selection orders it: NaN and -inf become the lowest finite score, which
// still ranks above the -inf that pads a block past its experts. A token whose hidden state
// overflowed then selects distinct experts in range, and its NaN reaches the logits through the
// residual, where the Engine ends that request alone; without it a NaN breaks the order and the
// selection names experts past the bank. Finite scores and +inf are unchanged.
__device__ __forceinline__ float route_order(float score) {
    return score >= -FLT_MAX ? score : -FLT_MAX;
}

} // namespace ninfer::ops
