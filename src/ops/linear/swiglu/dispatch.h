#pragma once
#include "core/weight.h"
#include "core/arena.h"
#include "core/device.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void validate_swiglu_geometry(QType type, int rows, int k);

using LinearSwiGluLaunch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);
LinearSwiGluLaunch select_q8_linear_swiglu(int rows, int k, int tokens);
LinearSwiGluLaunch select_q8_swiglu_n34816_k5120(int tokens);
std::size_t q4_linear_swiglu_capacity_workspace_bytes(int rows, int k, int first, int last);
void q4_linear_swiglu_dispatch(const Tensor&, const Weight&, Tensor&, MemoryArena&,
                               CudaExecutionView);
} // namespace ninfer::ops::detail
