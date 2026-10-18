// Oracle test of the tensor-core wrappers in ops/common/mma.cuh. Each wrapper is compared with
// naive math using the PTX fragment layouts of the Ampere-style m16n8k16 (f16/bf16), m16n8k32
// (s8) and m16n8k8 (tf32) instructions, so that the Turing decompositions and SIMT emulations
// stay equivalent to the hardware forms every kernel is written against.
#include "ops/common/cuda/mma.cuh"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

using namespace ninfer::ops::common;

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        cudaError_t e = (x);                                                                       \
        if (e != cudaSuccess) {                                                                    \
            printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__);                      \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

// A: [16][K] row-major floats, B: [K][8], result C[16][8]. Fragment index helpers.
struct Frag { unsigned r[4]; unsigned b[2]; };

__device__ unsigned pack16(float lo, float hi, int kind) {
    if (kind == 0) {
        __half2 h = __floats2half2_rn(lo, hi);
        return *reinterpret_cast<unsigned*>(&h);
    }
    __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<unsigned*>(&h);
}

// kind 0: f16 m16n8k16, kind 1: bf16 m16n8k16, kind 2: bf16 exact SIMT reference (sm_75 only)
__global__ void kernel16(const float* A, const float* B, float* C, int kind) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    auto a = [&](int row, int k) { return A[row * 16 + k]; };
    auto b = [&](int k, int n) { return B[k * 8 + n]; };
    unsigned a0 = pack16(a(g, 2 * t), a(g, 2 * t + 1), kind);
    unsigned a1 = pack16(a(g + 8, 2 * t), a(g + 8, 2 * t + 1), kind);
    unsigned a2 = pack16(a(g, 2 * t + 8), a(g, 2 * t + 9), kind);
    unsigned a3 = pack16(a(g + 8, 2 * t + 8), a(g + 8, 2 * t + 9), kind);
    unsigned b0 = pack16(b(2 * t, g), b(2 * t + 1, g), kind);
    unsigned b1 = pack16(b(2 * t + 8, g), b(2 * t + 9, g), kind);
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    if (kind == 0) mma_f16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    else if (kind == 1) mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#if defined(NINFER_SM75)
    else mma_bf16_simt(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
#endif
    C[g * 8 + 2 * t] = c0; C[g * 8 + 2 * t + 1] = c1;
    C[(g + 8) * 8 + 2 * t] = c2; C[(g + 8) * 8 + 2 * t + 1] = c3;
}

// s8 m16n8k32: A [16][32] int8, B [32][8] int8
__global__ void kernel_s8(const signed char* A, const signed char* B, int* C) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    auto a = [&](int row, int k) { return (unsigned)(unsigned char)A[row * 32 + k]; };
    auto b = [&](int k, int n) { return (unsigned)(unsigned char)B[k * 8 + n]; };
    auto pk = [&](unsigned x0, unsigned x1, unsigned x2, unsigned x3) {
        return x0 | (x1 << 8) | (x2 << 16) | (x3 << 24);
    };
    // a0: row g, k 4t..4t+3 ; a1: row g+8, same k ; a2: row g, k 16+4t.. ; a3: row g+8, k 16+4t..
    unsigned a0 = pk(a(g, 4 * t), a(g, 4 * t + 1), a(g, 4 * t + 2), a(g, 4 * t + 3));
    unsigned a1 =
        pk(a(g + 8, 4 * t), a(g + 8, 4 * t + 1), a(g + 8, 4 * t + 2), a(g + 8, 4 * t + 3));
    unsigned a2 = pk(a(g, 4 * t + 16), a(g, 4 * t + 17), a(g, 4 * t + 18), a(g, 4 * t + 19));
    unsigned a3 =
        pk(a(g + 8, 4 * t + 16), a(g + 8, 4 * t + 17), a(g + 8, 4 * t + 18), a(g + 8, 4 * t + 19));
    unsigned b0 = pk(b(4 * t, g), b(4 * t + 1, g), b(4 * t + 2, g), b(4 * t + 3, g));
    unsigned b1 = pk(b(4 * t + 16, g), b(4 * t + 17, g), b(4 * t + 18, g), b(4 * t + 19, g));
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    mma_s8(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    C[g * 8 + 2 * t] = c0; C[g * 8 + 2 * t + 1] = c1;
    C[(g + 8) * 8 + 2 * t] = c2; C[(g + 8) * 8 + 2 * t + 1] = c3;
}

// tf32 m16n8k8: A [16][8], B [8][8]
__global__ void kernel_tf32(const float* A, const float* B, float* C) {
    const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
    auto a = [&](int row, int k) { return A[row * 8 + k]; };
    auto b = [&](int k, int n) { return B[k * 8 + n]; };
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    mma_tf32(c0, c1, c2, c3, a(g, t), a(g + 8, t), a(g, t + 4), a(g + 8, t + 4), b(t, g),
             b(t + 4, g));
    C[g * 8 + 2 * t] = c0; C[g * 8 + 2 * t + 1] = c1;
    C[(g + 8) * 8 + 2 * t] = c2; C[(g + 8) * 8 + 2 * t + 1] = c3;
}

static float rnd() { return (float)rand() / RAND_MAX * 2.0f - 1.0f; }
static float round_to(float x, int kind) {
    if (kind == 0) return __half2float(__float2half(x));
    return __bfloat162float(__float2bfloat16(x));
}

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;
    srand(1);
    int failures = 0;
#if defined(NINFER_SM75)
    constexpr int kKinds = 3;
#else
    constexpr int kKinds = 2;
#endif
    // Operand magnitudes: unit, large (still inside FP16's range) and small-but-normal.
    const float scales[] = {1.0f, 2048.0f, 1.0f / 1024.0f};
    for (int kind = 0; kind < kKinds; ++kind) {
        for (const float scale : scales) {
            std::vector<float> A(16 * 16), B(16 * 8), C(16 * 8), R(16 * 8), Mag(16 * 8);
            for (auto& x : A) x = round_to(rnd() * scale, kind == 0 ? 0 : 1);
            for (auto& x : B) x = round_to(rnd() * scale, kind == 0 ? 0 : 1);
            for (int m = 0; m < 16; ++m)
                for (int n = 0; n < 8; ++n) {
                    double s = 0, mag = 0;
                    for (int k = 0; k < 16; ++k) {
                        s += (double)A[m * 16 + k] * B[k * 8 + n];
                        mag += fabs((double)A[m * 16 + k] * B[k * 8 + n]);
                    }
                    R[m * 8 + n] = (float)s;
                    Mag[m * 8 + n] = (float)mag;
                }
            float *dA, *dB, *dC;
            CHECK(cudaMalloc(&dA, 16 * 16 * 4));
            CHECK(cudaMalloc(&dB, 16 * 8 * 4));
            CHECK(cudaMalloc(&dC, 16 * 8 * 4));
            CHECK(cudaMemcpy(dA, A.data(), 16 * 16 * 4, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(dB, B.data(), 16 * 8 * 4, cudaMemcpyHostToDevice));
            kernel16<<<1, 32>>>(dA, dB, dC, kind);
            CHECK(cudaGetLastError());
            CHECK(cudaMemcpy(C.data(), dC, 16 * 8 * 4, cudaMemcpyDeviceToHost));
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
            // FP32 accumulation: error relative to the sum of absolute products.
            double worst = 0;
            for (int i = 0; i < 128; ++i) worst = fmax(worst, fabs(C[i] - R[i]) / (Mag[i] + 1e-30));
            // The Turing BF16 route feeds FP16 tensor cores, whose subnormals (|x| < 2^-14) carry
            // fewer than BF16's 8 mantissa bits; everything else is exact with FP32 accumulation.
            double limit = 1e-6;
#if defined(NINFER_SM75)
            if (kind == 1 && scale < 1.0f) limit = 2e-5;
#endif
            const char* name = kind == 0 ? "f16 " : kind == 1 ? "bf16" : "bf16-simt";
            printf("%-9s m16n8k16 scale %-10g: worst relative error %.3g %s\n", name, scale, worst,
                   worst < limit ? "OK" : "FAIL");
            failures += worst >= limit;
        }
    }
    {
        std::vector<signed char> A(16 * 32), B(32 * 8); std::vector<int> C(128), R(128);
        for (auto& x : A) x = (signed char)(rand() % 255 - 127);
        for (auto& x : B) x = (signed char)(rand() % 255 - 127);
        for (int m = 0; m < 16; ++m)
            for (int n = 0; n < 8; ++n) {
                int s = 0;
                for (int k = 0; k < 32; ++k) s += (int)A[m * 32 + k] * (int)B[k * 8 + n];
                R[m * 8 + n] = s;
            }
        signed char *dA, *dB;
        int* dC;
        CHECK(cudaMalloc(&dA, 512));
        CHECK(cudaMalloc(&dB, 256));
        CHECK(cudaMalloc(&dC, 512));
        CHECK(cudaMemcpy(dA, A.data(), 512, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB, B.data(), 256, cudaMemcpyHostToDevice));
        kernel_s8<<<1, 32>>>(dA, dB, dC);
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(C.data(), dC, 512, cudaMemcpyDeviceToHost));
        int bad = 0; for (int i = 0; i < 128; ++i) bad += C[i] != R[i];
        printf("s8   m16n8k32: mismatches %d %s\n", bad, bad == 0 ? "OK" : "FAIL");
        failures += bad != 0;
    }
    {
        std::vector<float> A(16 * 8), B(8 * 8), C(128), R(128);
        for (auto& x : A) x = rnd(); for (auto& x : B) x = rnd();
        for (int m = 0; m < 16; ++m)
            for (int n = 0; n < 8; ++n) {
                double s = 0;
                for (int k = 0; k < 8; ++k) s += (double)A[m * 8 + k] * B[k * 8 + n];
                R[m * 8 + n] = (float)s;
            }
        float *dA, *dB, *dC;
        CHECK(cudaMalloc(&dA, 512));
        CHECK(cudaMalloc(&dB, 256));
        CHECK(cudaMalloc(&dC, 512));
        CHECK(cudaMemcpy(dA, A.data(), 512, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB, B.data(), 256, cudaMemcpyHostToDevice));
        kernel_tf32<<<1, 32>>>(dA, dB, dC);
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(C.data(), dC, 512, cudaMemcpyDeviceToHost));
        double maxerr = 0; for (int i = 0; i < 128; ++i) maxerr = fmax(maxerr, fabs(C[i] - R[i]));
        printf("tf32 m16n8k8 : max abs err %.3g %s\n", maxerr, maxerr < 1e-4 ? "OK" : "FAIL");
        failures += maxerr >= 1e-4;
    }
    return failures;
}
