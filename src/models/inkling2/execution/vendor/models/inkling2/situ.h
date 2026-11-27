#pragma once

// Inkling 2's expert activation, SiTU: a tanh(g / a) sigmoid(g) times b tanh(u / b) of a pair's
// gate and up values (a = mlp_gate_softcap, b = mlp_up_softcap), each first multiplied by the
// expert's scale of the [gate; up] product. The GPU's kernels and the CPU's expert products
// (cpu_experts.cpp) compute it from correctly rounded operations only, as ops/common/silu.h does
// SiLU, so both give the same bits: e^y as silu.h's, sigmoid 1 / (1 + e^-g), and tanh as Cephes'
// odd polynomial below 0.625 and 1 - 2 / (e^(2|x|) + 1) above, its sign restored. Host translation
// units that include this header build with floating-point contraction off.

#include "ops/common/silu.h"

#include <cmath>

#if defined(__CUDACC__)
#    define NINFER_SITU_HD __host__ __device__ __forceinline__
#else
#    define NINFER_SITU_HD inline
#endif

namespace ninfer::models::inkling2 {

NINFER_SITU_HD float portable_div(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}

NINFER_SITU_HD float portable_tanh(float x) {
    namespace s = ops::silu_detail;
    const float a = x < 0.0F ? -x : x;
    if (a != a) { return x; }
    if (a < 0.625F) {
        const float z = s::mul(x, x);
        float p       = s::fma(-5.70498872745e-3F, z, 2.06390887954e-2F);
        p             = s::fma(p, z, -5.37397155531e-2F);
        p             = s::fma(p, z, 1.33314422036e-1F);
        p             = s::fma(p, z, -3.33332819422e-1F);
        return s::fma(s::mul(p, z), x, x);
    }
    // e^(2|x|) is +inf from |x| near 44.4, where the quotient is 0 and tanh 1.
    const float y = s::add(1.0F, -portable_div(2.0F, s::add(s::exp(s::mul(2.0F, a)), 1.0F)));
    return x < 0.0F ? -y : y;
}

NINFER_SITU_HD float portable_sigmoid(float x) {
    return portable_div(1.0F, ops::silu_detail::add(1.0F, ops::silu_detail::exp(-x)));
}

// SiTU of the gate and up values g and u, after the expert's scale.
NINFER_SITU_HD float situ(float g, float u, float gate_cap, float up_cap) {
    namespace s = ops::silu_detail;
    const float gate = s::mul(s::mul(gate_cap, portable_tanh(portable_div(g, gate_cap))),
                              portable_sigmoid(g));
    return s::mul(gate, s::mul(up_cap, portable_tanh(portable_div(u, up_cap))));
}

} // namespace ninfer::models::inkling2

#undef NINFER_SITU_HD
