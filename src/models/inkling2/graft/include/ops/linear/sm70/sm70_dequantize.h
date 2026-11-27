#pragma once

// FP16 rows of a resident weight for the sm_70 wide products (sm70_gemm.h): the represented value
// of every element, rounded once to FP16 (saturating at +-65504). Covers the row-split integer
// formats (Q2, Q4, Q5, Q6, Q8 and T2, row-major or panel-permuted), BF16, row-scaled FP8 and NVFP4;
// GGUF blocks dequantize through their own bridge (gguf::dequantize_rows).

#include "core/weight.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::sm70 {

// The per-row multiplier dequantize_rows leaves out: row-scaled FP8's BF16 row scale and NVFP4's
// weight divisor can lie far outside FP16's range, so their rows are written unscaled and the
// product's epilogue applies the factor. The other formats' factor is 1.
struct RowFactor {
    enum class Kind : std::uint8_t { One, Bf16Row, Divisors, Constant };
    Kind kind                   = Kind::One;
    const void* data            = nullptr;
    std::int32_t rows_per_entry = 1;
    float constant              = 1.0F;

#if defined(__CUDACC__)
    __device__ __forceinline__ float operator()(std::int32_t row) const {
        switch (kind) {
        case Kind::Bf16Row:
            return __bfloat162float(static_cast<const __nv_bfloat16*>(data)[row]);
        case Kind::Divisors:
            return 1.0F / static_cast<const float*>(data)[row / rows_per_entry];
        case Kind::Constant:
            return constant;
        case Kind::One:
            break;
        }
        return 1.0F;
    }
#endif
};

[[nodiscard]] RowFactor row_factor(const Weight& w);

// The formats with a dequantizer here (a weight of one may still lack its planes).
[[nodiscard]] bool dequantizable_format(QType format);
[[nodiscard]] bool dequantizable(const Weight& w);

// Rows [row0, row0 + rows) of `w` into out[(r - row0) * w.k + c] for every logical column c,
// without the row's RowFactor.
void dequantize_rows(const Weight& w, std::int32_t row0, std::int32_t rows, __half* out,
                     cudaStream_t stream);

} // namespace ninfer::ops::detail::sm70
