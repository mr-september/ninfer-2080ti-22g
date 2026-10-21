#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace ninfer::pdl {

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t dynamic_smem_bytes = 0;
    cudaStream_t stream            = nullptr;
};

// Launches a consumer kernel as a programmatic dependent of the immediately preceding producer
// kernel in the same stream. Every consumer control path that reads producer output must first call
// wait_for_dependencies().
template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline cudaError_t
launch_dependent(const LaunchConfig& launch, void (*kernel)(KernelArgs...), CallArgs&&... args) {
    cudaLaunchConfig_t config{};
    config.gridDim          = launch.grid;
    config.blockDim         = launch.block;
    config.dynamicSmemBytes = launch.dynamic_smem_bytes;
    config.stream           = launch.stream;
    config.attrs            = nullptr;
    config.numAttrs         = 0;

    return cudaLaunchKernelEx(&config, kernel, std::forward<CallArgs>(args)...);
}

// Turing has no programmatic dependent launch: the launch is an ordinary one and the device calls
// are no-ops.
__device__ __forceinline__ void trigger_dependents() {}
__device__ __forceinline__ void wait_for_dependencies() {}

} // namespace ninfer::pdl
