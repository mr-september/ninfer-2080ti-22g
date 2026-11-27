#pragma once

#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops {

struct DirectWeight {
    const void* data;
    bool fp32;

    __device__ __forceinline__ float operator[](std::int64_t i) const {
        return fp32 ? static_cast<const float*>(data)[i]
                    : __bfloat162float(static_cast<const __nv_bfloat16*>(data)[i]);
    }
};

inline DirectWeight direct_weight(const Tensor& tensor) {
    return {tensor.data, tensor.dtype == DType::FP32};
}

} // namespace ninfer::ops
