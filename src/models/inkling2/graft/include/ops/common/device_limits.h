#pragma once

#include "ops/common/device_tier.h"

namespace ninfer::ops {

// The largest dynamic shared memory one block may opt into on every card the build runs a
// single-shape kernel on: 64 KiB with Turing (sm_75) among its architectures, 99 KiB otherwise
// (ops/common/device_tier.h). A kernel of one shape for every card asserts against it, so a
// schedule that cannot launch on a target fails to compile instead of at its first launch; a kernel
// whose shape follows the shape set asserts against shape_max_shared_bytes_per_block of its set.
inline constexpr int kMaxSharedBytesPerBlock = shape_max_shared_bytes_per_block(kBuildTuringShapes);

// The most threads one SM holds resident in the image being compiled: 1,024 on Turing, 1,536 on
// sm_86, sm_89 and sm_120 (sm_70 and sm_80 parts hold 2,048, and their kernels ask for at most the
// sm_86 figure). It bounds only __launch_bounds__, which each image's device pass reads for itself;
// the host pass sees the sm_86 figure.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 800
inline constexpr int kMaxThreadsPerSm = 1024;
#else
inline constexpr int kMaxThreadsPerSm = 1536;
#endif

// The minimum resident CTAs a __launch_bounds__ of `threads` a CTA asks for: `requested`, or as
// many as an SM of the image's architecture holds where that is fewer. ptxas ignores a minimum it
// cannot meet, which leaves the kernel's registers unbounded instead of sized for full occupancy.
[[nodiscard]] constexpr int resident_ctas_bound(int threads, int requested) {
    return requested * threads <= kMaxThreadsPerSm ? requested : kMaxThreadsPerSm / threads;
}

// The sm_8x bound, for the unified-template Linear schedules (namespace detail::unified) and the
// Q8 schedule templates they share with the legacy table. A Turing card never selects the unified
// table (unified_linear_routes in ops/linear/common/route_table.h), so those schedules keep the
// 99 KiB they were sized for; the legacy Q8 routes that do run on Turing are qualified by their
// tests, where a schedule past 64 KiB fails its first launch.
inline constexpr int kMaxSharedBytesPerBlockSm8x = 99 * 1024;

} // namespace ninfer::ops
