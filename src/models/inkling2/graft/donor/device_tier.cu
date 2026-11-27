#include "ops/common/device_tier.h"

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kMaxDevices = 64;

// Compiled for every architecture of the build, like every other kernel of ninfer_ops, so the image
// CUDA loads for it on a device is the one that device runs for all of them.
__global__ void device_tier_probe() {}

DeviceTier tier_of_ptx(int ptx_version) {
    if (ptx_version < 75) { return DeviceTier::Volta; }
    if (ptx_version < 80) { return DeviceTier::Turing; }
    if (ptx_version < 100) { return DeviceTier::Ampere; }
    return DeviceTier::Blackwell;
}

constexpr DeviceTier only_built_tier() {
    return kBuildVolta    ? DeviceTier::Volta
           : kBuildTuring ? DeviceTier::Turing
           : kBuildAmpere ? DeviceTier::Ampere
                          : DeviceTier::Blackwell;
}

// 0 is unknown; a tier is stored as its value plus one.
std::array<std::atomic<std::uint8_t>, kMaxDevices>& known_tiers() {
    static std::array<std::atomic<std::uint8_t>, kMaxDevices> tiers{};
    return tiers;
}

DeviceTier probe(int device) {
    int previous = 0;
    if (cudaGetDevice(&previous) != cudaSuccess) {
        cudaGetLastError();
        previous = device;
    }
    cudaError_t err = previous == device ? cudaSuccess : cudaSetDevice(device);
    cudaFuncAttributes attributes{};
    if (err == cudaSuccess) { err = cudaFuncGetAttributes(&attributes, device_tier_probe); }
    if (previous != device) { cudaSetDevice(previous); }
    if (err != cudaSuccess) {
        cudaGetLastError();
        throw std::runtime_error("CUDA device " + std::to_string(device) +
                                 " runs none of this build's code: " + cudaGetErrorName(err));
    }
    return tier_of_ptx(attributes.ptxVersion);
}

} // namespace

DeviceTier device_tier(int device) {
    if constexpr (kBuildTiers == 1) {
        (void)device;
        return only_built_tier();
    } else {
        if (device < 0 || device >= kMaxDevices) { return probe(device); }
        std::atomic<std::uint8_t>& slot = known_tiers()[static_cast<std::size_t>(device)];
        const std::uint8_t known        = slot.load(std::memory_order_relaxed);
        if (known != 0) { return static_cast<DeviceTier>(known - 1); }
        const DeviceTier tier = probe(device);
        slot.store(static_cast<std::uint8_t>(static_cast<std::uint8_t>(tier) + 1),
                   std::memory_order_relaxed);
        return tier;
    }
}

DeviceTier current_device_tier() {
    if constexpr (kBuildTiers == 1) {
        return only_built_tier();
    } else {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) {
            cudaGetLastError();
            device = 0;
        }
        return device_tier(device);
    }
}

} // namespace ninfer::ops
