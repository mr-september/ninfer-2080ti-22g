#pragma once

#include "ops/linear/q6/q6_mma_launch.cuh"
#include "ops/linear/q6/q6_simt_launch.cuh"
#include "ops/linear/q6/q6_sliced_k_launch.cuh"

namespace ninfer::ops::detail {

// A Q6 schedule writing FP32: a tensor-parallel rank's partial sum or the vocabulary head's
// unrounded logits.

template <class Schedule>
void launch_q6_fp32_simt(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q6_a16_simt<Schedule>(q6_linear_operands(x, w),
                                 LinearFp32Output{static_cast<float*>(out.data), w.n},
                                 LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_q6_fp32_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q6_a16_gemv<Schedule>(q6_linear_operands(x, w),
                                 LinearFp32Output{static_cast<float*>(out.data), w.n},
                                 LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_q6_fp32_mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_q6_a16_mma<Schedule>(q6_linear_operands(x, w),
                                LinearFp32Output{static_cast<float*>(out.data), w.n},
                                LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_q6_fp32_sliced_k_mma(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    launch_q6_a16_sliced_k_mma<Schedule>(q6_linear_operands(x, w),
                                         LinearFp32Output{static_cast<float*>(out.data), w.n},
                                         LinearIdentityEpilogue{}, stream);
}

} // namespace ninfer::ops::detail
