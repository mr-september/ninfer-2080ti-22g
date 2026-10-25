#include "ops/linear_attention/gated_delta_net/input_conv/q8/launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {

void q8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                           cudaStream_t stream) {
    if (weight.k != 2560) throw std::invalid_argument("GDN input: unsupported Q8 geometry");
    q8_gdn_projection_n16384_k2560(x, weight, qkv, z, stream);
}
} // namespace ninfer::ops::detail
