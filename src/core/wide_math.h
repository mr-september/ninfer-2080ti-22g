#pragma once

// Exact 64 x 64 -> 128-bit products for saturating cost and ranking arithmetic. MSVC has no
// unsigned __int128, so the product is exposed as a (hi, lo) pair on every platform.

#include <cstdint>
#include <limits>

#if defined(_MSC_VER) && !defined(__clang__)
#    include <intrin.h>
#endif

namespace ninfer {

struct WideU64 {
    std::uint64_t hi = 0;
    std::uint64_t lo = 0;

    friend constexpr auto operator<=>(const WideU64&, const WideU64&) = default;
};

[[nodiscard]] inline WideU64 mul_wide(std::uint64_t left, std::uint64_t right) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    WideU64 result;
    result.lo = _umul128(left, right, &result.hi);
    return result;
#else
    const unsigned __int128 product = static_cast<unsigned __int128>(left) * right;
    return {static_cast<std::uint64_t>(product >> 64U), static_cast<std::uint64_t>(product)};
#endif
}

// min(left * right, UINT64_MAX)
[[nodiscard]] inline std::uint64_t saturating_mul(std::uint64_t left, std::uint64_t right) noexcept {
    const WideU64 product = mul_wide(left, right);
    return product.hi != 0 ? std::numeric_limits<std::uint64_t>::max() : product.lo;
}

// min(left + right, UINT64_MAX)
[[nodiscard]] constexpr std::uint64_t saturating_add(std::uint64_t left,
                                                     std::uint64_t right) noexcept {
    return right > std::numeric_limits<std::uint64_t>::max() - left
               ? std::numeric_limits<std::uint64_t>::max()
               : left + right;
}

} // namespace ninfer
