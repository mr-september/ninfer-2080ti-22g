#pragma once

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

namespace ninfer::bench {

// Tensor references are dense rates with explicit input/accumulator precision.
// Nominal and measured reference provenance is recorded in bench/README.md.
// Unestablished references remain absent, including any metrics derived from them.
struct Roofline {
    const char* target;
    double memory_gbs;
    double sustained_read_gbs = std::numeric_limits<double>::quiet_NaN();
    double bf16_f32_tflops    = std::numeric_limits<double>::quiet_NaN();
    double fp8_f16_tflops     = std::numeric_limits<double>::quiet_NaN();
    double fp8_f32_tflops     = std::numeric_limits<double>::quiet_NaN();
    double mxfp8_f32_tflops   = std::numeric_limits<double>::quiet_NaN();
};

// RTX 2080 Ti (TU102, 22 GB): nominal GDDR6 bandwidth, and the sustained read rate measured
// with the dependent-free streaming kernel (docs/turing-port.md). Tensor-core rates are not
// established for this target.
inline constexpr Roofline kRoofline{
    .target             = "turing",
    .memory_gbs         = 616.0,
    .sustained_read_gbs = 584.0,
};

// Console uses "-"; CSV callers pass "" so absent references stay empty fields.
inline std::string format_metric(double value, int precision = 2, const char* absent = "-") {
    if (!std::isfinite(value)) return absent;
    char text[64];
    std::snprintf(text, sizeof(text), "%.*f", precision, value);
    return text;
}

} // namespace ninfer::bench
