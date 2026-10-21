#include <cuda_runtime.h>
#include <cstdint>

namespace {

__device__ __constant__ unsigned nf4_words[16] = {
    0xbf800000, 0xbf3239b1, 0xbf066b30, 0xbeca32a0, 0xbe91a24d, 0xbe3d353f, 0xbdba7871, 0x00000000,
    0x3da2faff, 0x3e24cae3, 0x3e7c04dd, 0x3ead033a, 0x3ee1a4b8, 0x3f1007ab, 0x3f3913b3, 0x3f800000,
};

__device__ double scale_value(int word) {
    const int exponent = word >> 3;
    return exponent ? double(__int_as_float(((exponent + 120) << 23) | ((word & 7) << 20)))
                    : (word & 7) * (1. / 512.);
}

__device__ double code_value(int code) { return double(__uint_as_float(nf4_words[code])); }

__device__ int nearest(double x, double s) {
    int low = 0, high = 15;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int mid          = (low + high) / 2;
        const double threshold = (code_value(mid) + code_value(mid + 1)) * .5 * s;
        if (x > threshold)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

__device__ double error(double x, double s, unsigned mask) {
    const int code     = nearest(x, s);
    const double delta = x - code_value(code) * s;
    double sum         = delta * delta;
#pragma unroll
    for (int offset = 8; offset; offset /= 2) sum += __shfl_xor_sync(mask, sum, offset, 16);
    return __shfl_sync(mask, sum, 0, 16);
}

__global__ void encode(const float* values, unsigned char* codes, unsigned char* scales,
                       int64_t groups, double divisor) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t group = index / 16;
    if (group >= groups) return;
    const int lane = threadIdx.x & 15;
    // Every G16 group takes its own zero/pruning branches. Synchronize only
    // its sixteen lanes, even when the other half warp evaluates another scale.
    const unsigned mask = 0xffffu << (threadIdx.x & 16);
    const double x      = double(values[index]) * divisor;
    double maximum      = fabs(x);
#pragma unroll
    for (int offset = 8; offset; offset /= 2)
        maximum = fmax(maximum, __shfl_xor_sync(mask, maximum, offset, 16));
    int best_word = 0, code = 7;
    if (maximum != 0.) {
        const double limit = 1.;
        int low = 1, high = 126;
        while (low < high) {
            const int mid = (low + high) / 2;
            if (scale_value(mid) * limit < maximum)
                low = mid + 1;
            else
                high = mid;
        }
        best_word         = low;
        double best_error = error(x, scale_value(best_word), mask);
        for (int word = 1; word <= 126; ++word) {
            const double s = scale_value(word);
            // One maximum-magnitude element already exceeds the incumbent error:
            // even endpoint clipping cannot improve this candidate.
            const double clipped = fmax(maximum - limit * s, 0.);
            if (clipped * clipped > best_error) continue;
            const double candidate = error(x, s, mask);
            if (candidate < best_error || (candidate == best_error && word < best_word)) {
                best_error = candidate;
                best_word  = word;
            }
        }
        code = nearest(x, scale_value(best_word));
    }
    const int next = __shfl_xor_sync(mask, code, 1, 16);
    if (!(lane & 1)) codes[index / 2] = static_cast<unsigned char>(code | (next << 4));
    if (!lane) scales[group] = static_cast<unsigned char>(best_word);
}

} // namespace

void launch_stored_scale(const float* values, unsigned char* codes, unsigned char* scales,
                         int64_t elements, double divisor, cudaStream_t stream) {
    const auto blocks = (elements + 255) / 256;
    encode<<<blocks, 256, 0, stream>>>(values, codes, scales, elements / 16, divisor);
}
