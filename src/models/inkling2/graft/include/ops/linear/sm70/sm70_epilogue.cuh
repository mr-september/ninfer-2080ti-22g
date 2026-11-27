#pragma once

// The destination epilogue shared by the sm_70 wide and small-T projections.

#include "ops/common/math.cuh"
#include "ops/linear/sm70/sm70_dequantize.h"
#include "ops/linear/sm70/sm70_project.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::sm70 {

// Every parent row lands in the destination that takes it, times its RowFactor: stored, or added
// onto the BF16 residual (or the FP32 partial) there.
struct DestinationEpilogue {
    Destination set[kMaxDestinations];
    int count = 0;
    RowFactor factor;

    __device__ void operator()(std::int32_t row, std::int32_t token, float value) const {
        value *= factor(row);
#pragma unroll
        for (int i = 0; i < kMaxDestinations; ++i) {
            if (i >= count) { return; }
            const Destination& d = set[i];
            if (row >= d.row_begin && row < d.row_begin + d.rows) {
                const std::size_t at = static_cast<std::size_t>(token) * d.leading +
                                       d.row_offset + (row - d.row_begin);
                if (d.fp32 != nullptr) {
                    d.fp32[at] = d.accumulate ? d.fp32[at] + value : value;
                    return;
                }
                __nv_bfloat16* p = d.data + at;
                *p = d.accumulate ? residual_add_bf16(value, *p) : __float2bfloat16_rn(value);
                return;
            }
        }
    }
};

} // namespace ninfer::ops::detail::sm70
