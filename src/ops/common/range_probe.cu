#include "ops/common/range_probe.h"

#if defined(NINFER_SM75_RANGE_PROBE)

#include <cstdio>
#include <cstring>

namespace ninfer::ops {

__device__ unsigned ninfer_range_probe_max_bits = 0;
__device__ unsigned long long ninfer_range_probe_over[3] = {0, 0, 0};

void print_range_probe() {
    (void)cudaDeviceSynchronize();
    unsigned max_bits = 0;
    unsigned long long over[3] = {0, 0, 0};
    (void)cudaMemcpyFromSymbol(&max_bits, ninfer_range_probe_max_bits, sizeof(max_bits));
    (void)cudaMemcpyFromSymbol(over, ninfer_range_probe_over, sizeof(over));
    float max_value = 0.0f;
    static_assert(sizeof(float) == sizeof(unsigned));
    std::memcpy(&max_value, &max_bits, sizeof(max_value));
    std::fprintf(stderr,
                 "range probe | max |x| %.1f | elements > 16384: %llu | > 32768: %llu | > 65504 "
                 "(clamped): %llu\n",
                 max_value, over[0], over[1], over[2]);
}

} // namespace ninfer::ops

#endif
