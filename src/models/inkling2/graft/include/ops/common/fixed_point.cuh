#pragma once

// round(value * weight * 2^32) to the nearest int64, ties to even, clamped to +-4e18: the 2^-32
// fixed-point contribution the MoE expert sums accumulate. It equals
//
//   __double2ll_rn(fmin(fmax(double(value) * double(weight) * 2^32, -4e18), 4e18))
//
// bit for bit (the FP64 product of two FP32 values is exact, so is the power-of-two scale, and a
// NaN product clamps to -4e18 as fmax/fmin do), in integer arithmetic: GeForce parts run FP64 at
// 1/64 of the FP32 rate, which made the FP64 form the cost of a wide call's accumulation.

#include <cstdint>
#include <cstring>

namespace ninfer::ops {

inline constexpr long long kFixedClamp = 4000000000000000000LL;

__host__ __device__ __forceinline__ long long fixed_product(float value, float weight) {
#if defined(__CUDA_ARCH__)
    const std::uint32_t ua = __float_as_uint(value), ub = __float_as_uint(weight);
#else
    std::uint32_t ua, ub;
    std::memcpy(&ua, &value, 4);
    std::memcpy(&ub, &weight, 4);
#endif
    const bool negative = ((ua ^ ub) >> 31) != 0;
    const int ea = int((ua >> 23) & 0xFF), eb = int((ub >> 23) & 0xFF);
    const std::uint32_t fa = ua & 0x7FFFFF, fb = ub & 0x7FFFFF;
    if (ea == 0xFF || eb == 0xFF) {
        const bool nan  = (ea == 0xFF && fa != 0) || (eb == 0xFF && fb != 0);
        const bool zero = (ea == 0 && fa == 0) || (eb == 0 && fb == 0);
        if (nan || zero) { return -kFixedClamp; } // NaN, or infinity times zero
        return negative ? -kFixedClamp : kFixedClamp;
    }
    // value = ma * 2^xa, weight = mb * 2^xb with integer mantissas.
    const std::uint64_t ma = ea == 0 ? fa : (fa | 0x800000u);
    const std::uint64_t mb = eb == 0 ? fb : (fb | 0x800000u);
    const int xa = ea == 0 ? -149 : ea - 150, xb = eb == 0 ? -149 : eb - 150;
    const std::uint64_t m = ma * mb; // < 2^48
    if (m == 0) { return 0; }
    const int e = xa + xb + 32;
    std::uint64_t r;
    if (e >= 0) {
        // Both are normal here, so m >= 2^46: past a shift of 15 it exceeds the clamp.
        if (e > 15 || (m << e) > std::uint64_t(kFixedClamp)) {
            return negative ? -kFixedClamp : kFixedClamp;
        }
        r = m << e;
    } else {
        const int shift = -e;
        if (shift > 49) { return 0; } // below one half
        const std::uint64_t q    = m >> shift;
        const std::uint64_t rest = m & ((std::uint64_t(1) << shift) - 1);
        const std::uint64_t half = std::uint64_t(1) << (shift - 1);
        r = q + ((rest > half || (rest == half && (q & 1) != 0)) ? 1 : 0);
    }
    return negative ? -static_cast<long long>(r) : static_cast<long long>(r);
}

} // namespace ninfer::ops
