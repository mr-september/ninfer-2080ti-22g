#pragma once

// Debug-only (NINFER_SM75_RANGE_PROBE): the largest magnitude seen by the BF16 -> FP16 operand
// conversion that feeds Turing's FP16 tensor cores, and how often values exceeded the thresholds
// that matter for FP16's +-65504 range. Not part of any production build.

#if defined(NINFER_SM75_RANGE_PROBE)

#include <cuda_runtime.h>

namespace ninfer::ops {

#if defined(__CUDACC__)
extern __device__ unsigned ninfer_range_probe_max_bits;
extern __device__ unsigned long long ninfer_range_probe_over[3]; // > 16384, > 32768, > 65504

__device__ __forceinline__ void range_probe_record(float lo, float hi) {
    const float magnitude = fmaxf(fabsf(lo), fabsf(hi));
    const unsigned bits   = __float_as_uint(magnitude); // non-negative floats order like integers
    if (bits > *reinterpret_cast<volatile unsigned*>(&ninfer_range_probe_max_bits)) {
        atomicMax(&ninfer_range_probe_max_bits, bits);
    }
    if (magnitude > 16384.0f) {
        atomicAdd(&ninfer_range_probe_over[0], 1ULL);
        if (magnitude > 32768.0f) { atomicAdd(&ninfer_range_probe_over[1], 1ULL); }
        if (magnitude > 65504.0f) { atomicAdd(&ninfer_range_probe_over[2], 1ULL); }
    }
}

#endif // __CUDACC__

// Prints the table to stderr after synchronizing the device.
void print_range_probe();

} // namespace ninfer::ops

#endif
