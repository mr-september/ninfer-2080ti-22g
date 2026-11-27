#pragma once

// What a kernel launch needs from the device layer: the CUDA error check, the dynamic shared memory
// limit and the launch facts Ops receive. core/device.h, which owns the device contexts, includes
// this header; a translation unit that only launches kernels includes this one, so an edit of the
// context types does not recompile it.

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::ninfer::cuda_check((expr), #expr, __FILE__, __LINE__)

// Non-owning execution facts passed to Ops whose launch policy depends on physical device
// capacity. DeviceContext remains the owner and authoritative source of both values.
struct DeviceExecutionView {
    cudaStream_t stream               = nullptr;
    std::int32_t multiprocessor_count = 0;
};

// Raises `kernel`'s dynamic shared memory limit on the current device to at least `bytes`, once per
// device and kernel. Function attributes belong to one device context, so each GPU needs its own
// call. The key is the host stub the calling translation unit launches: CUDA 13's nvcc gives a
// template kernel's stub internal linkage in every translation unit that instantiates it
// (-static-global-template-stub, on by default), and each stub launches its own copy of the device
// function, with attributes of its own. A cache keyed by a type or by a function-local static in a
// header is merged across translation units by the linker, so it would configure only the first
// copy and leave the next launch of another to fail with cudaErrorInvalidValue.
void raise_dynamic_shared_memory(const void* kernel, int bytes);

template <class... Args>
void raise_dynamic_shared_memory(void (*kernel)(Args...), int bytes) {
    raise_dynamic_shared_memory(reinterpret_cast<const void*>(kernel), bytes);
}

} // namespace ninfer
