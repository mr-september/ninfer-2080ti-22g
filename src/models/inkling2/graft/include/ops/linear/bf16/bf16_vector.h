#pragma once

// The GEMV form of the runtime-shape BF16 route (bf16_general.cu) for several products of one
// activation in one launch.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <span>

namespace ninfer::ops::detail {

// Whether the runtime-shape route runs x [K,T] against `weight` in its GEMV form: T up to 16, K a
// multiple of 8, x and the weight 16-byte aligned.
[[nodiscard]] bool bf16_general_vector(const Tensor& x, const Weight& weight);

// A BF16 weight and its contiguous BF16 [N,T] output.
struct Bf16VectorProduct {
    const Weight* weight = nullptr;
    Tensor* out          = nullptr;
};

// Each product's out <- its weight's product of x, in one launch per four products. Every product
// must take the GEMV form (bf16_general_vector) into a BF16 out; each output row gets the bits that
// the form's own launch gives it. Throws for any other product.
void bf16_general_vector_together(const Tensor& x, std::span<const Bf16VectorProduct> products,
                                  cudaStream_t stream);

} // namespace ninfer::ops::detail
