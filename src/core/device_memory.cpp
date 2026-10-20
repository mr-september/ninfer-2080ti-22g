#include "core/device.h"

namespace ninfer {

DeviceMemorySnapshot device_memory_snapshot(Device device) {
    CudaDeviceScope scope(device);
    DeviceMemorySnapshot snapshot;
    CUDA_CHECK(cudaMemGetInfo(&snapshot.free_bytes, &snapshot.total_bytes));
    return snapshot;
}

std::int64_t device_memory_growth(DeviceMemorySnapshot before, DeviceMemorySnapshot after) {
    return static_cast<std::int64_t>(before.free_bytes) -
           static_cast<std::int64_t>(after.free_bytes);
}

} // namespace ninfer
