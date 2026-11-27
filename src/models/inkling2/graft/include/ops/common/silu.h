#pragma once

// SiLU x / (1 + e^-x) (portable_silu, beside math.cuh's device silu) and the tanh-approximated
// GELU from correctly rounded operations only (multiply, add, divide, fused multiply-add, round to
// integer and exact power-of-two scaling), so that the GPU's expert kernels and the CPU's expert
// products (gguf_expert_cpu, sparse_moe streamed_cpu) compute the same bits; CUDA's expf
// goes through an approximate hardware exp2. e^y: n = rint(y log2 e), r = y - n ln2 with ln2 split
// in two parts, e^r by Cephes' degree-6 polynomial (|r| <= ln2 / 2, within about one unit in the
// last place), times 2^n. Every step is spelled out, so no compiler fuses a multiply and an add on
// one side only: device code uses the _rn intrinsics, and host translation units that include this
// header build with floating-point contraction off.

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#    define NINFER_SILU_HD __host__ __device__ __forceinline__
#else
#    define NINFER_SILU_HD inline
#endif

namespace ninfer::ops::silu_detail {

NINFER_SILU_HD float mul(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}

NINFER_SILU_HD float add(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}

NINFER_SILU_HD float fma(float a, float b, float c) {
#if defined(__CUDA_ARCH__)
    return __fmaf_rn(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

NINFER_SILU_HD float from_bits(std::uint32_t bits) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(bits);
#else
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

// 2^k for k in [-126, 127].
NINFER_SILU_HD float power_of_two(int k) { return from_bits(std::uint32_t(k + 127) << 23); }

// e^y; 0 below the smallest normal float, +inf past the largest.
NINFER_SILU_HD float exp(float y) {
    if (y != y) { return y; }
    if (y >= 88.7228394F) { return from_bits(0x7F800000U); }
    if (y < -87.3365479F) { return 0.0F; }
#if defined(__CUDA_ARCH__)
    const float n = rintf(mul(y, 1.44269504F));
#else
    const float n = std::rint(mul(y, 1.44269504F));
#endif
    float r = fma(n, -0.693145751953125F, y);
    r       = fma(n, -1.42860677e-6F, r);
    float p = fma(1.9875691500e-4F, r, 1.3981999507e-3F);
    p       = fma(p, r, 8.3334519073e-3F);
    p       = fma(p, r, 4.1665795894e-2F);
    p       = fma(p, r, 1.6666665459e-1F);
    p       = fma(p, r, 5.0000001201e-1F);
    p       = fma(p, mul(r, r), r);
    p       = add(p, 1.0F);
    // n is -126 .. 128: 2^128 is two steps.
    const int k = static_cast<int>(n);
    return mul(mul(p, power_of_two(k > 127 ? 127 : k)), k > 127 ? 2.0F : 1.0F);
}

} // namespace ninfer::ops::silu_detail

namespace ninfer::ops {

NINFER_SILU_HD float portable_silu(float x) {
#if defined(__CUDA_ARCH__)
    return __fdiv_rn(x, silu_detail::add(1.0F, silu_detail::exp(-x)));
#else
    return x / silu_detail::add(1.0F, silu_detail::exp(-x));
#endif
}

// The tanh-approximated GELU 0.5 x (1 + tanh(u)), u = sqrt(2 / pi) (x + 0.044715 x^3), in its
// equal form x / (1 + e^(-2u)), from the same correctly rounded operations: the cube as
// ((0.044715 x) x) x, then 2u = 2 sqrt(2 / pi) (x + cube).
NINFER_SILU_HD float gelu_tanh(float x) {
    using silu_detail::add;
    using silu_detail::mul;
    const float cube  = mul(mul(mul(0.044715F, x), x), x);
    const float twice = mul(1.5957691216057308F, add(x, cube));
#if defined(__CUDA_ARCH__)
    return __fdiv_rn(x, add(1.0F, silu_detail::exp(-twice)));
#else
    return x / add(1.0F, silu_detail::exp(-twice));
#endif
}

} // namespace ninfer::ops

#undef NINFER_SILU_HD
