#include "ops/linear/swiglu/q4/launch.h"
#include "ops/linear/projection/q4/q4_shapes.h"

namespace ninfer::ops::detail {

void launch_q4_swiglu_materialized_projection(const Tensor& x, const Weight& w, Tensor& out,
                                              cudaStream_t stream) {
    // Turing: the tuned plain-projection routes (FP16-mantissa GEMV at T=1, SIMT up to T=4, the
    // single-buffered MMA tiles for prefill) are used directly.
    select_q4_n34816_k5120(x.ne[1])(x, w, out, stream);
}
} // namespace ninfer::ops::detail
