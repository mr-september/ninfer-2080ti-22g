#pragma once

// Shared-memory MMA operands for the dequantizing matrix kernels.
//
// Native-BF16 targets keep activations and dequantized weights as BF16 pairs and feed the BF16
// tensor core. Turing has no BF16 tensor core, so these kernels keep both operand tiles in shared
// memory as FP16 pairs and feed the FP16 tensor core directly:
//   * weights: the integer code (exact in FP16) times the FP16 group scale, rounded once to FP16
//     (a finer rounding than the BF16 materialization it replaces);
//   * activations: BF16 -> FP16 while the tile is staged (saturating beyond +-65504), once per
//     tile element instead of once per MMA fragment.
// No fragment conversion remains inside the MMA loop, which was the limiter of the previous
// route (about eleven ALU instructions per tensor-core instruction).

#include "ops/common/integer_operands.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

namespace ninfer::ops {

#if defined(NINFER_SM75)

__device__ __forceinline__ int4 bf16x8_to_f16x8_operand(int4 v) {
    return make_int4(static_cast<int>(bf16x2_to_f16x2_operand(static_cast<unsigned>(v.x))),
                     static_cast<int>(bf16x2_to_f16x2_operand(static_cast<unsigned>(v.y))),
                     static_cast<int>(bf16x2_to_f16x2_operand(static_cast<unsigned>(v.z))),
                     static_cast<int>(bf16x2_to_f16x2_operand(static_cast<unsigned>(v.w))));
}

// Stage one 16-byte vector of BF16 activations as FP16 operands.
template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void stage_activation_operand(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 16, "activation operand staging moves 8 BF16 values");
    *reinterpret_cast<int4*>(smem_dst) =
        bf16x8_to_f16x8_operand(*reinterpret_cast<const int4*>(gmem_src));
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void stage_activation_operand_zfill(void* smem_dst, const void* gmem_src,
                                                               int src_bytes) {
    static_assert(Bytes == 16, "activation operand staging moves 8 BF16 values");
    if (src_bytes >= Bytes) {
        stage_activation_operand<Bytes, Policy>(smem_dst, gmem_src);
    } else {
        // Ragged K tails convert the valid prefix element-wise and zero the rest.
        const auto* source = reinterpret_cast<const unsigned short*>(gmem_src);
        auto* dest         = reinterpret_cast<unsigned short*>(smem_dst);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            unsigned short bits = 0;
            if (2 * i < src_bytes) {
                const unsigned packed = static_cast<unsigned>(source[i]) << 16;
                bits = static_cast<unsigned short>(bf16x2_to_f16x2_operand(packed) >> 16);
            }
            dest[i] = bits;
        }
    }
}

// Group-scaled integer weight pair: (q0 * scale, q1 * scale) as an FP16 pair.
__device__ __forceinline__ unsigned scaled_weight_pair_operand(int q0, int q1, __half scale) {
    const unsigned integers = integer_pair_operand(q0, q1);
    const __half2 product   = __hmul2(*reinterpret_cast<const __half2*>(&integers),
                                      __half2half2(scale));
    return *reinterpret_cast<const unsigned*>(&product);
}

// FP16 x FP16 -> FP32 m16n8k16 over operands that are already resident as FP16.
__device__ __forceinline__ void mma_operands(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    mma_f16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
}

#else

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void stage_activation_operand(void* smem_dst, const void* gmem_src) {
    cp_async<Bytes, Policy>(smem_dst, gmem_src);
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void stage_activation_operand_zfill(void* smem_dst, const void* gmem_src,
                                                               int src_bytes) {
    cp_async_zfill<Bytes, Policy>(smem_dst, gmem_src, src_bytes);
}

__device__ __forceinline__ unsigned scaled_weight_pair_operand(int q0, int q1, __half scale) {
    const float s = __half2float(scale);
    union {
        __nv_bfloat162 value;
        unsigned bits;
    } pair;
    pair.value = __floats2bfloat162_rn(static_cast<float>(q0) * s, static_cast<float>(q1) * s);
    return pair.bits;
}

__device__ __forceinline__ void mma_operands(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
}

#endif

} // namespace ninfer::ops
