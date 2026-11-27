#pragma once

#include "core/device_launch.h"
#include "core/pdl.cuh"
#include "ops/linear/q8/q8_ksplit_mma.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {

template <class Geometry, int ColumnCapacity, class Schedule, class Epilogue>
void launch_q8_ksplit_schedule(const Tensor& x, const Weight& weight, Tensor& out,
                               cudaStream_t stream) {
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    CUDA_CHECK(pdl::launch_consumer(
        {dim3(Geometry::kOutputRows / Schedule::kRowsPerCta), dim3(Schedule::kThreads), 0, stream},
        q8_ksplit_mma_kernel<Geometry, ColumnCapacity, Schedule, Q8ContiguousOutput, Epilogue,
                             Q8KSplitIdentityRows, false, true>,
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, Epilogue{}, Q8KSplitIdentityRows{},
        x.ne[1]));
    CUDA_CHECK(cudaGetLastError());
}

// FP32 out: each pair of sums is stored unrounded (the vocabulary head's logits).
struct Q8KSplitFp32Epilogue {
    float* out;
    std::int32_t rows;

    __device__ __forceinline__ void store_pair(int row, int col0, float4 sum, int columns) const {
        if (col0 < columns) {
            out[static_cast<std::int64_t>(col0) * rows + row]     = sum.x;
            out[static_cast<std::int64_t>(col0) * rows + row + 8] = sum.z;
        }
        if (col0 + 1 < columns) {
            out[static_cast<std::int64_t>(col0 + 1) * rows + row]     = sum.y;
            out[static_cast<std::int64_t>(col0 + 1) * rows + row + 8] = sum.w;
        }
    }
};

template <class Geometry, int ColumnCapacity, class Schedule>
void launch_q8_ksplit_fp32_schedule(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream) {
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);
    const Q8ContiguousOutput output{nullptr, Geometry::kOutputRows};
    const Q8KSplitFp32Epilogue epilogue{static_cast<float*>(out.data), Geometry::kOutputRows};
    CUDA_CHECK(pdl::launch_consumer(
        {dim3(Geometry::kOutputRows / Schedule::kRowsPerCta), dim3(Schedule::kThreads), 0, stream},
        q8_ksplit_mma_kernel<Geometry, ColumnCapacity, Schedule, Q8ContiguousOutput,
                             Q8KSplitFp32Epilogue, Q8KSplitIdentityRows, true, true>,
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, epilogue, Q8KSplitIdentityRows{},
        x.ne[1]));
    CUDA_CHECK(cudaGetLastError());
}

// Shape files select the capacity and physical schedule; the device profile may run an 8-column
// tile on its four-K-warp form (q8_ksplit_config.h). The launcher supplies the live T, so every T
// in the selected interval reuses this compiled instance.
template <class Geometry, int ColumnCapacity, class RouteSchedule,
          class Epilogue = Q8KSplitStoreEpilogue>
void launch_q8_ksplit(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (weight.padded_shape[1] != Geometry::kInputRows) {
        throw std::invalid_argument("q8 K-split: padded K differs from the registered geometry");
    }
    if constexpr (kQ8KSplitHasFourWarpForm<RouteSchedule>) {
        if (q8_ksplit_four_warps<Geometry>(x.ne[1])) {
            launch_q8_ksplit_schedule<Geometry, ColumnCapacity,
                                      Q8KSplitFourWarpSchedule<RouteSchedule>, Epilogue>(
                x, weight, out, stream);
            return;
        }
    }
    launch_q8_ksplit_schedule<Geometry, ColumnCapacity, RouteSchedule, Epilogue>(x, weight, out,
                                                                                 stream);
}

// launch_q8_ksplit's route with FP32 out.
template <class Geometry, int ColumnCapacity, class RouteSchedule>
void launch_q8_ksplit_fp32(const Tensor& x, const Weight& weight, Tensor& out,
                           cudaStream_t stream) {
    if (weight.padded_shape[1] != Geometry::kInputRows) {
        throw std::invalid_argument("q8 K-split: padded K differs from the registered geometry");
    }
    if constexpr (kQ8KSplitHasFourWarpForm<RouteSchedule>) {
        if (q8_ksplit_four_warps<Geometry>(x.ne[1])) {
            launch_q8_ksplit_fp32_schedule<Geometry, ColumnCapacity,
                                           Q8KSplitFourWarpSchedule<RouteSchedule>>(x, weight, out,
                                                                                    stream);
            return;
        }
    }
    launch_q8_ksplit_fp32_schedule<Geometry, ColumnCapacity, RouteSchedule>(x, weight, out, stream);
}

} // namespace ninfer::ops::detail
