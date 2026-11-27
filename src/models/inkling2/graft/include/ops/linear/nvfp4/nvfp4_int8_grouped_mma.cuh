#pragma once

// NVFP4 weights on the integer tensor cores (Tier-2, sm_86): the stored E2M1 codes are widened to
// s8 (v = 2*e2m1, an exact int8 grid) and fed to mma.m16n8k16.s8 -- one NVFP4 G16 group per MMA --
// with the per-16 E4M3 group scales rescaling the int32 partials in FP32. Activations are
// quantised to s8 with one FP32 scale per (token, 32 columns): a per-token scalar lets one outlier
// channel starve the whole token, and 32 columns is the width the own chain's MMQ and MMVQ rescale
// against, so both chains walk one sequence -- acc = fmaf((float)mma, w_scale * x_scale, acc) once
// per k16, a single FP32 accumulator per output element, no per-tile partial in between.
//
// The E2M1 1/2 factor lives in the E4M3 scale decode (nvfp4_e4m3_scale.h) -- the codec widens E2M1
// to 2*e2m1, and llama.cpp folds the same 1/2 into its ue4m3 decode -- so the global weight divisor
// reaches the mainloop as alpha = 1/divisor and is applied once to the FP32 accumulators after the
// K loop, matching the A16 route's contract. Two half factors would halve the result silently, one
// in each place is exactly right; the enumeration test pins the decode and PPL pins the product.
//
// The mainloop stages 128 output rows and serves every projection route on this representation.
// The plain Linear / LinearAdd routes and the two-projection gate_up fallback (the gate projection
// to an FP32 plane, the up projection with the GateProduct store that reads it back) differ only in
// their output adapter; the grouped gate_up route instead uses Nvfp4Int8SwiGluRowPolicy, which
// stages 64 (gate, up) row pairs in the same 128 rows and folds silu(gate) * up in the epilogue's
// registers -- no FP32 plane, no second launch. Performance model: the rescale core costs 13 issue
// slots per m16n8k16 MMA (1 MMA
// + 4 I2F + 4 FMUL for w_scale * x_scale + 4 FFMA into the accumulator) -- the same shape MMQ pays
// -- and that floor is what the mainloop is built around. Throughput follows from
// TOPS = SMs * schedulers * clock * achieved_issue * 4096 / (instructions per MMA), so the lever is
// everything the kernel emits ON TOP of those 13 slots. The mainloop therefore carries the contract
// core plus one staging pack per thread and one barrier per tile: 128 TOPS at 13.0 slots needs an
// issue rate of 64.3%, at 14.5 slots 71.8%.

#include "ops/common/mma.cuh"  // sm70:: and sm75:: mma emulations, and the NINFER_SM70_DEVICE guard
#include "ops/linear/common/output.cuh"
#include "ops/linear/nvfp4/nvfp4_codec_unified.cuh"
#include "ops/linear/nvfp4/nvfp4_e4m3_scale.h"
#include "ops/linear/nvfp4/nvfp4_int8_grouped_host.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops::detail::unified {

// ---------------------------------------------------------------------------
// E2M1 code -> s8, four codes per word (PRMT/LOP3 form).
//
// The nibble v = 2*e2m1 takes the values {0,1,2,3,4,6,8,12} over u = n&7 and negates with bit 3,
// so the sign select is a byte-wise mux between two PRMT tables and nvcc folds it into one LOP3
// (immLut 0xD8). Consuming the raw uint32 words directly (packed nibble selectors, low half then
// high half) makes the split/interleave round-trip of a byte-shaped selector unnecessary: the
// words leave the codec already in k order.
//
// The sign masks are extracted with inline prmt.b32, NOT the __byte_perm intrinsic: the intrinsic
// lowers to PRMT with the selector masked by 0x7777 (the sign-replicate mode bit is stripped --
// measured on CUDA 13.4 / sm_86, the intrinsic performs a plain nibble&7 byte select), while
// inline asm reaches the documented PTX sign-replicate mode. The table lookups below use
// 0x7777-masked selectors, so the intrinsic is safe for them.
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned prmt_sign_replicate(unsigned a, unsigned b, unsigned s) {
    unsigned result;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(s));
    return result;
}

__device__ __forceinline__ void e2m1_word(unsigned x, unsigned& q0, unsigned& q1) {
    const unsigned sel0 = x & 0x7777u;                               // low 16 bits: 4 nibbles
    const unsigned sel1 = (x >> 16) & 0x7777u;
    const unsigned p0 = __byte_perm(0x03020100u, 0x0C080604u, sel0); // +V(u) = {0,1,2,3,4,6,8,12}
    const unsigned p1 = __byte_perm(0x03020100u, 0x0C080604u, sel1);
    const unsigned n0 = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, sel0); // -V(u) = {0,-1,-2,-3,-4,-6,-8,-12}
    const unsigned n1 = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, sel1);
    const unsigned me = prmt_sign_replicate(x << 4, 0u, 0xBA98u);    // even-code sign masks
    const unsigned mo = prmt_sign_replicate(x, 0u, 0xBA98u);         // odd-code sign masks
    const unsigned m0 = __byte_perm(me, mo, 0x5140u);                // codes 0..3 signs
    const unsigned m1 = __byte_perm(me, mo, 0x7362u);                // codes 4..7 signs
    q0 = (p0 & ~m0) | (n0 & m0);   // nvcc synthesises one LOP3 (immLut 0xD8) per mux
    q1 = (p1 & ~m1) | (n1 & m1);
}

// ---------------------------------------------------------------------------
// E4M3FN scale byte -> FP32, llama.cpp's ue4m3 semantics (see nvfp4_e4m3_scale.h for the whole
// argument: the trailing /2, NaN -> 0, and the unreachable >= 0x80 domain). The full 256-entry
// table costs 1KB of constant-storage rom and one cached load per scale byte, which beats both the
// CUDA fp8 header conversion (compiled to an outlined call with BSSY/BSYNC bookkeeping on sm_86,
// ~53 instructions per k-tile per thread) and an inline bit assembly of similar size. The host and
// the device table expand one list, so the enumeration test on the host covers what this reads.
// ---------------------------------------------------------------------------
__device__ constexpr float kNvfp4E4M3ScaleLut[256] = {
    NINFER_E4M3_SCALE_LIST(NINFER_E4M3_SCALE_VALUE)};

// ---------------------------------------------------------------------------
// Activation quantiser: one FP32 scale per (token, 32 columns), at every width. A 32-column group
// confining a channel outlier to half a BK=64 tile is why the group is not per token, and it is the
// group the own chain's MMQ and MMVQ both rescale against, so the two chains read one plane and
// fold one sequence: acc = fmaf((float)mma, w_scale * x_scale, acc) once per k16. The gate is byte
// equality with MMQ, not a tolerance, because both are the same FP32 sequence.
//
// The scale PLANE is [token][k/32] and every 32-column block carries its own value, which is what the
// narrow band, the plan/dispatch translation units and the MMQ envelope assertion already read.
//
// One thread owns one group. The 32 BF16 values arrive as four uint4 loads, and the abs-max runs
// on the raw bit patterns: masking the sign bit orders non-negative BF16 exactly as the floats
// order, so the integer max is the float max without a single conversion. The per-element
// arithmetic is the scalar form's, untouched (convert, multiply by 127/amax, round-to-nearest,
// clamp), so codes and scales are bit-identical to it; only the loads are vectorised and the max
// runs on integers. Per-element scalar loads leave this pass far below streaming rate, which is
// what makes the four per-projection quantise passes a visible share of a prefill chunk.
// ---------------------------------------------------------------------------

__device__ __forceinline__ unsigned nvfp4_int8_bf16_absmax_word(unsigned word, unsigned m) {
    const unsigned both = word & 0x7FFF7FFFu; // abs of the two packed BF16 lanes
    m = max(m, both & 0xFFFFu);
    return max(m, both >> 16);
}

// One 32-column group: the integer abs-max of its four uint4 loads and the two uint4 of s8 codes
// they quantise to. Both activation layouts call this and only differ in where the results land,
// so the codes and the scale of a group cannot drift between the two-plane and the MMQ-row form.
// The values are re-read through L1 rather than aliased from the staged registers: taking the
// address of a local uint4 would push it to local memory.
__device__ __forceinline__ void nvfp4_int8_quantize_group(const uint4* source, float& scale,
                                                          uint4 (&packed)[2]) {
    const uint4 w0 = source[0];
    const uint4 w1 = source[1];
    const uint4 w2 = source[2];
    const uint4 w3 = source[3];
    unsigned m = 0;
    m = nvfp4_int8_bf16_absmax_word(w0.x, m);
    m = nvfp4_int8_bf16_absmax_word(w0.y, m);
    m = nvfp4_int8_bf16_absmax_word(w0.z, m);
    m = nvfp4_int8_bf16_absmax_word(w0.w, m);
    m = nvfp4_int8_bf16_absmax_word(w1.x, m);
    m = nvfp4_int8_bf16_absmax_word(w1.y, m);
    m = nvfp4_int8_bf16_absmax_word(w1.z, m);
    m = nvfp4_int8_bf16_absmax_word(w1.w, m);
    m = nvfp4_int8_bf16_absmax_word(w2.x, m);
    m = nvfp4_int8_bf16_absmax_word(w2.y, m);
    m = nvfp4_int8_bf16_absmax_word(w2.z, m);
    m = nvfp4_int8_bf16_absmax_word(w2.w, m);
    m = nvfp4_int8_bf16_absmax_word(w3.x, m);
    m = nvfp4_int8_bf16_absmax_word(w3.y, m);
    m = nvfp4_int8_bf16_absmax_word(w3.z, m);
    m = nvfp4_int8_bf16_absmax_word(w3.w, m);
    const float amax = fmaxf(__uint_as_float(m << 16), 1.0e-20F);
    scale            = amax / 127.0F;
    const float inv  = 127.0F / amax;
    const __nv_bfloat16* values = reinterpret_cast<const __nv_bfloat16*>(source);
    auto* bytes                 = reinterpret_cast<std::int8_t*>(packed);
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        const float value = __bfloat162float(values[j]) * inv;
        bytes[j] = static_cast<std::int8_t>(max(-127, min(127, __float2int_rn(value))));
    }
}

// A padding row of a ragged token tile: zero the activation planes for one token and do not touch
// x, whose rows stop at the caller's real token count. Zero is the only safe filler -- e2m1 code 0
// is the value 0, so a padded row contributes nothing to any accumulator, and the row is never
// stored, so it merely has to be defined. The vector path needs 16-byte alignment, which the
// route's own k%64==0 gate and the arena's 256-byte alignment already give; the scalar loop is the
// same defensive fallback the quantiser below uses for exotic callers.
static __device__ void zero_nvfp4_int8_activation_row(std::int8_t* __restrict__ codes,
                                                     float* __restrict__ scales,
                                                     std::int32_t token, std::int32_t k,
                                                     std::int32_t groups) {
    std::int8_t* const row = codes + static_cast<std::size_t>(token) * k;
    if ((k & 15) == 0 && (reinterpret_cast<std::uintptr_t>(row) & 15) == 0) {
        auto* words = reinterpret_cast<uint4*>(row);
        for (std::int32_t v = static_cast<std::int32_t>(threadIdx.x); v < (k >> 4);
             v += static_cast<std::int32_t>(blockDim.x)) {
            words[v] = make_uint4(0u, 0u, 0u, 0u);
        }
    } else {
        for (std::int32_t b = static_cast<std::int32_t>(threadIdx.x); b < k;
             b += static_cast<std::int32_t>(blockDim.x)) {
            row[b] = 0;
        }
    }
    for (std::int32_t g = static_cast<std::int32_t>(threadIdx.x); g < groups;
         g += static_cast<std::int32_t>(blockDim.x)) {
        scales[static_cast<std::size_t>(token) * groups + g] = 0.0F;
    }
}

// static: the header is included by several .cu files (linear, linear_swiglu, linear_add,
// attn/gdn input projections); without internal linkage each TU emits its own host stub and
// kernel symbol and the bundle exe link fails with LNK2005. Repo precedent: rope.cuh:230.
//
// One FP32 scale per (token, 32 columns), at every width: the plane both chains consume, so their
// results above the crossover leave the same bytes.
static __global__ void quantize_nvfp4_int8_grouped(const __nv_bfloat16* __restrict__ x,
                                            std::int32_t tokens, std::int32_t k,
                                            std::int8_t* __restrict__ codes,
                                            float* __restrict__ scales) {
    const std::int32_t token = blockIdx.x;
    const std::int32_t groups = k >> 5;
    // Launched over the padded width, so every row the mainloop will read is written: real tokens
    // quantise, the surplus rows zero.
    if (token >= tokens) {
        zero_nvfp4_int8_activation_row(codes, scales, token, k, groups);
        return;
    }
    // The vector path needs 32-byte alignment on both planes; k%32==0 keeps every group's
    // load and store 32-byte aligned for any token. The layer buffers are arena-aligned, so
    // the guard is uniform per launch and the scalar fallback only covers exotic callers.
    if ((k & 31) == 0 && (reinterpret_cast<std::uintptr_t>(x) & 31) == 0 &&
        (reinterpret_cast<std::uintptr_t>(codes) & 31) == 0) {
        for (std::int32_t g = threadIdx.x; g < groups; g += blockDim.x) {
            const uint4* source = reinterpret_cast<const uint4*>(
                x + static_cast<std::size_t>(token) * k + static_cast<std::size_t>(g) * 32);
            float scale  = 0.0F;
            uint4 packed[2];
            nvfp4_int8_quantize_group(source, scale, packed);
            scales[static_cast<std::size_t>(token) * groups + g] = scale;
            std::int8_t* dst = codes + static_cast<std::size_t>(token) * k +
                               static_cast<std::size_t>(g) * 32;
            reinterpret_cast<uint4*>(dst)[0] = packed[0];
            reinterpret_cast<uint4*>(dst)[1] = packed[1];
        }
        return;
    }
    for (std::int32_t g = threadIdx.x; g < groups; g += blockDim.x) {
        const __nv_bfloat16* source = x + static_cast<std::size_t>(token) * k + g * 32;
        float amax                  = 0.0F;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            amax = fmaxf(amax, fabsf(__bfloat162float(source[j])));
        }
        amax = fmaxf(amax, 1.0e-20F);
        scales[static_cast<std::size_t>(token) * groups + g] = amax / 127.0F;
        const float inv  = 127.0F / amax;
        std::int8_t* dst = codes + static_cast<std::size_t>(token) * k +
                           static_cast<std::size_t>(g) * 32;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const float value = __bfloat162float(source[j]) * inv;
            dst[j] = static_cast<std::int8_t>(max(-127, min(127, __float2int_rn(value))));
        }
    }
}

// ---------------------------------------------------------------------------
// Row layout. A block stages 128 consecutive output rows and every thread's fragment tiles land
// on rows the warp owns. The block's J tokens are shared by NT token warps, each taking the NIT
// eight-token tile_C columns its participation buys; the remaining 8/NT warps take consecutive
// output-row bands of kWarpRows rows, held as MT fragment tiles 16 rows apart.
//
// J = 128 reproduces the layout this kernel shipped with: NT = 2 token warps of 64 tokens against
// four row bands of 32 rows. Narrower tiles coalesce the token axis into one warp (NT = 1) and spend
// the seven warps that frees on eight row bands of 16 rows -- same 128 output rows, same eight
// warps, one MT fragment per warp instead of two. That is the whole point of the ladder: the row
// tile, the staging and the weight traffic do not move, and the tensor-core work drops in
// proportion to J because the activation side of every MMA is J tokens wide instead of 128.
// ---------------------------------------------------------------------------
template <int J, int Threads = detail::kInt8Threads>
struct Nvfp4Int8RowPolicy {
    static_assert(nvfp4_int8_j_admitted(J), "J must be a stage of the token-tile ladder");

    static constexpr bool kPaired  = false;
    static constexpr int kThreads  = Threads;
    static constexpr int kWarps    = Threads / 32;
    static constexpr int kNTiles   = J / 8;         // eight-token MMA columns in the tile
    static constexpr int NT        = (kNTiles + 7) / 8; // token warps the tile keeps busy
    static constexpr int NIT       = kNTiles / NT;      // tile_C columns per token warp
    static constexpr int kRowWarps = kWarps / NT;
    static constexpr int kWarpRows = detail::kInt8TileRows / kRowWarps;
    static constexpr int MT        = kWarpRows / 16;

    static_assert(NT > 0 && NT <= kWarps, "warp budget must cover the warps");
    static_assert(MT * 16 * kRowWarps == detail::kInt8TileRows, "row bands must tile the tile");
    static_assert(NIT * NT * 8 == J, "token warps must tile the token tile");

    static constexpr int kStagedRows    = detail::kInt8TileRows;
    static constexpr int kOutputRows    = detail::kInt8TileRows;
    static constexpr int kFragmentStep  = 16;
    // Two resident blocks per SM up to J=64: the half-tile temporaries past that raise the register
    // count above what two blocks can hold, and losing the residency costs more than the issue slots
    // the wider temporaries save. Naming the residency asks for the cap. J>=80 is already one block
    // per SM on its own, so asking for two there would only buy spills.
    static constexpr int kMinBlocks     = (J <= 64) ? 2 : 1;
    static constexpr int kWarpRowStride = kWarpRows;

    static __device__ int weight_row(int begin, int local, int rows) {
        (void)rows;
        return begin + local;
    }

    // Staged row of fragment tile (wm, m) for lane row `gid`.
    static __device__ int fragment_row(int wm, int m, int gid) {
        return wm * kWarpRowStride + gid + m * kFragmentStep;
    }

    // First token column of tile_C column index `n` for token warp `wt`.
    static __device__ int token_column(int wt, int n, int tig) {
        return wt * (NIT * 8) + n * 8 + tig * 2;
    }
};

// ---------------------------------------------------------------------------
// Paired row layout for gate_up: 128 staged rows are 64 (gate row, up row) pairs. Each 16-row MMA
// fragment stages eight gate rows followed by their eight up partners, so the lane that holds gate
// row g (fragment row g, g < 8) holds up row g in the same fragment's second half (row g + 8). The
// pair therefore meets inside one thread's accumulators -- acc[0]/acc[1] carry the gate row's two
// tokens, acc[2]/acc[3] the up row's -- and silu(gate) * up needs no shared-memory exchange, no FP32
// gate plane and no second launch.
//
// Everything else is the identity policy: same 128 staged rows, same warp mapping, same activation
// staging, same per-thread accumulator count. Only weight_row (which global row a staged row names)
// and the epilogue (which rows pair up) change.
// ---------------------------------------------------------------------------
template <int J, int Threads = detail::kInt8Threads>
struct Nvfp4Int8SwiGluRowPolicy : Nvfp4Int8RowPolicy<J, Threads> {
    using Base = Nvfp4Int8RowPolicy<J, Threads>;

    static constexpr bool kPaired    = true;
    static constexpr int kOutputRows = Base::kStagedRows / 2;

    // Staged row -> absolute weight row. Rows 0..7 of every 16-row fragment are gate rows, rows
    // 8..15 the matching up rows of the same pair, so `local & 8` selects the half and the pair index
    // is the same for both: `begin + pair` in the gate half, `rows/2 + begin + pair` in the up half.
    static __device__ int weight_row(int begin, int local, int rows) {
        const int pair = pair_index(local);
        return (local & 8) != 0 ? rows / 2 + begin + pair : begin + pair;
    }

    // Staged row -> index of the output (gate/up pair) row it belongs to.
    static __device__ int pair_index(int staged) {
        return (staged >> 4) * 8 + (staged & 7);
    }
};

namespace detail {

// Shared launch geometry lives in nvfp4_int8_grouped_host.h so host-built plan/dispatch
// translation units can query admission and workspace sizes.

__device__ __forceinline__ unsigned int8_smem_addr(const void* pointer) {
    return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

// A uint4 read through a char* is emitted as four 32-bit loads unless the compiler can prove the
// alignment, which it cannot through this pointer arithmetic.
__device__ __forceinline__ uint4 int8_lds128(const void* pointer) {
    uint4 value;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "r"(int8_smem_addr(pointer)));
    return value;
}

__device__ __forceinline__ uint4 int8_lds128(unsigned address) {
    uint4 value;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "r"(address));
    return value;
}

__device__ __forceinline__ uint2 int8_lds64(const void* pointer) {
    uint2 value;
    const unsigned address = int8_smem_addr(pointer);
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];"
                 : "=r"(value.x), "=r"(value.y)
                 : "r"(address));
    return value;
}

// ldmatrix.x4 over four 8x8 b16 matrices = four 8-row x 16-byte windows of s8 codes. The four
// results are the four MMA fragments of that window in ascending matrix order, and adjacent results
// are an aligned register pair -- exactly the {lo row, hi row} pair an s8 m16n8k16 A operand wants,
// or the four k-step B words when the matrices are the four 16-wide k-steps.
__device__ __forceinline__ void int8_ldmatrix_x4(unsigned (&out)[4], const void* pointer) {
#if defined(NINFER_SM70_DEVICE)
    sm70::ldmatrix<4>(out, int8_smem_addr(pointer));
#else
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "l"(pointer));
#endif
}

__device__ __forceinline__ void int8_cpasync16(void* dst, const void* src) {
    cp_async<16, Cache::cg>(dst, src);
}

__device__ __forceinline__ void int8_cpasync8(void* dst, const void* src) {
    cp_async<8>(dst, src);
}

__device__ __forceinline__ void int8_cpasync_commit() { cp_commit(); }

__device__ __forceinline__ void int8_cpasync_wait() { cp_wait<0>(); }

__device__ __forceinline__ void mma_s8_k16(int& c0, int& c1, int& c2, int& c3, unsigned a0,
                                           unsigned a1, unsigned b0) {
#if defined(NINFER_SM70_DEVICE)
    sm70::mma_m16n8k16_s8(c0, c1, c2, c3, a0, a1, b0);
#elif defined(NINFER_SM75_DEVICE)
    // m16n8k16 splits into two m8n8k16 over the row halves: a0 is row g, a1 is row g+8 (the same
    // contract mma_s8 uses for its k32 form), so the four accumulators pair up without reordering.
    sm75::mma_s8_k16(c0, c1, a0, b0);
    sm75::mma_s8_k16(c2, c3, a1, b0);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
#endif
}

// D = A*B + 0, with D and C in separate registers. `zero` is a live register the caller keeps at
// 0, which ptxas folds to RZ -- so a k-step costs no accumulator clear, where the in-place form
// ("+r") needs the four result registers initialised before every MMA.
__device__ __forceinline__ void mma_s8_k16_zero(int& d0, int& d1, int& d2, int& d3, unsigned a0,
                                                unsigned a1, unsigned b0, int zero) {
#if defined(NINFER_SM70_DEVICE) || defined(NINFER_SM75_DEVICE)
    d0 = d1 = d2 = d3 = zero;
    mma_s8_k16(d0, d1, d2, d3, a0, a1, b0);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(zero));
#endif
}

} // namespace detail

// ---------------------------------------------------------------------------
// Main kernel. One block covers kInt8TileRows output rows x J tokens x one k-tile of
// kInt8TileK columns per iteration, where J is the stage of the token-tile ladder the caller's
// width selects (nvfp4_int8_select_j) -- see Nvfp4Int8RowPolicy for how the block's warps are
// spread over those two axes.
//
// Dataflow:
//
//   * Both operand planes sit in shared memory in natural k order, one 80-byte row per output row
//     (weights) or per token (activations), and both reach the MMA through ldmatrix.x4. For the
//     weights that instruction yields an aligned {row g, row g+8} pair directly; for the
//     activations one instruction yields all four k-step words of an eight-token column, so no
//     per-thread word permutation is needed before the MMA.
//   * Two shared-memory buffer sets: tile t+1 is staged into the set tile t is not reading, so a
//     whole tile costs ONE barrier. Activation codes and their per-32 scales move with cp.async and
//     name no register; weight codes are prefetched into registers before the tile's MMAs and
//     decoded after them, so their global latency hides under the tensor work.
//   * Every MMA uses mma_s8_k16_zero: the int32 result lands in its own registers against a zero C,
//     so no accumulator clear is emitted per k-step.
//   * The 256-entry E4M3 table lives in shared memory: a scale byte is a per-thread divergent
//     index, and a shared lookup is a 32-bit add plus one LDS where a global table builds a 64-bit
//     address and issues a load.
//
// The arithmetic contract is untouched: one FP32 accumulator per output element, fed one 16-column
// weight group at a time in ascending k with the scale product rounded first,
//     acc = fmaf((float)mma_sum, w_scale * x_scale, acc)
// and one BF16 rounding at the store through the projection's output adapter.
// ---------------------------------------------------------------------------
template <int J, int Threads, class Output, class Policy = Nvfp4Int8RowPolicy<J, Threads>>
__global__ __launch_bounds__(Threads, Policy::kMinBlocks)
    void nvfp4_int8_grouped_kernel(
    const std::uint8_t* __restrict__ weight_codes, const std::uint8_t* __restrict__ weight_scales,
    const std::int8_t* __restrict__ x_codes, const float* __restrict__ x_scales, float alpha_half,
    Output output, std::int32_t k, std::int32_t tokens, std::int32_t rows) {
    namespace d  = detail;
    constexpr int BM       = d::kInt8TileRows;
    constexpr int BN       = J;
    constexpr int MT       = Policy::MT;
    constexpr int N_IT     = Policy::NIT;
    constexpr int NT       = Policy::NT;
    constexpr int kSRow    = d::kInt8SRow;
    constexpr int kThreads = Threads;
    constexpr int kQuarterBytes = 16;                 // one MMA k-step of s8 codes
    constexpr int kWeightsPerTile = BM * 4;           // (row, k-quarter) staging slots
    constexpr int kWeightPasses = kWeightsPerTile / kThreads;
    static_assert(kWeightsPerTile % kThreads == 0, "weight sweep must cover the whole tile");

    extern __shared__ __align__(16) unsigned char int8_smem[];
    auto* sa   = reinterpret_cast<std::int8_t*>(int8_smem);      // [2][BM][kSRow] weight codes
    auto* sb   = sa + 2 * BM * kSRow;                            // [2][BN][kSRow] activation codes
    auto* sws  = reinterpret_cast<float*>(sb + 2 * BN * kSRow);  // [2][BM][4] weight group scales
    auto* sxs  = sws + 2 * BM * d::kInt8ScaleGroups;             // [2][BN][2] per-32 activation scales
    auto* slut = sxs + 2 * BM * d::kInt8XScales;                 // [256] E4M3 scale byte -> FP32

    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm   = warp / NT; // row band the warp owns
    const int wt   = warp % NT; // token slice of the band the warp owns
    const int gid  = lane >> 2; // 0..7, row inside a 16-row tile
    const int tig  = lane & 3;  // 0..3, k quarter

    // Token tiles on the fast axis: the wave of blocks resident together reads one 128-row weight
    // tile J-tiles-many times instead of scattering across the whole weight plane.
    const int row_block = static_cast<int>(blockIdx.y) * Policy::kOutputRows;
    const int col_block = static_cast<int>(blockIdx.x) * BN;
    const int groups    = k >> 6; // 64-column k-tiles == weight scale tiles

    const auto epilogue = linear_output_tile<Policy::kOutputRows>(output, row_block);

    // The 256-entry E4M3 table moves into shared memory: a scale byte is a per-thread divergent
    // index, which a global table answers with a 64-bit address build (IMAD.WIDE) plus an LDG per
    // lookup, while a shared table is a 32-bit add plus one LDS.
    if (tid < 256) { slut[tid] = kNvfp4E4M3ScaleLut[tid]; }
    __syncthreads();

    // ---- per-thread staging bases, built once ------------------------------------------------
    const int ld_row0 = tid >> 2; // 0..63 for pass 0; pass p adds p * (kThreads / 4) rows
    const int ld_q    = tid & 3;  // 16-code quarter inside the 64-column tile
    const std::size_t w_row_bytes = static_cast<std::size_t>(k) / 2;
    const std::uint8_t* w_ptr[kWeightPasses];
    const std::uint8_t* ws_ptr[kWeightPasses];
    std::int8_t* sa_ptr[kWeightPasses];
    float* sws_ptr[kWeightPasses];
#pragma unroll
    for (int p = 0; p < kWeightPasses; ++p) {
        const int ld_row = ld_row0 + p * (kThreads / 4);
        const int grow   = Policy::weight_row(row_block, ld_row, rows);
        w_ptr[p]         = weight_codes + static_cast<std::size_t>(grow) * w_row_bytes + ld_q * 8;
        ws_ptr[p]        = weight_scales + nvfp4_scale_byte_offset(grow, 0, k);
        sa_ptr[p]        = sa + ld_row * kSRow + ld_q * kQuarterBytes;
        sws_ptr[p]       = sws + ld_row * d::kInt8ScaleGroups;
    }
    // Activation staging: one 16-byte chunk per (row, k-quarter), cp.async'd straight to shared
    // memory. A chunk past the tile's real token count is never copied.
    constexpr int kActPasses = kWeightsPerTile / kThreads;
    const std::int8_t* x_ptr[kActPasses];
    std::int8_t* sb_ptr[kActPasses];
#pragma unroll
    for (int p = 0; p < kActPasses; ++p) {
        const int row  = ld_row0 + p * (kThreads / 4);
        const int srow = row < BN ? row : 0; // rows past BN are never read, only skipped
        x_ptr[p]  = x_codes + static_cast<std::size_t>(col_block + srow) * k + ld_q * kQuarterBytes;
        sb_ptr[p] = sb + srow * kSRow + ld_q * kQuarterBytes;
    }
    const float* xs_ptr = x_scales; // only threads with tid < BN read through it
    if (tid < BN) {
        xs_ptr = x_scales + static_cast<std::size_t>(col_block + tid) * (k >> 5);
    }

    // ---- staged pack -------------------------------------------------------------------------
    struct WPack {
        uint2 w;     // 16 codes (one k-quarter of one row)
        unsigned ws; // the row's four E4M3 group scales for this tile
    };
    WPack wp[kWeightPasses];

    const auto prefetch = [&](int tile) {
        const std::size_t w_off = static_cast<std::size_t>(tile) * 32;
        const std::size_t s_off = static_cast<std::size_t>(tile) * 512;
#pragma unroll
        for (int p = 0; p < kWeightPasses; ++p) {
            wp[p].w = *reinterpret_cast<const uint2*>(w_ptr[p] + w_off);
            // The four quarter threads of a row each own one of its four G16 scale bytes. The old
            // code made every quarter load the same uint32 and then execute four LUT lookups and a
            // float4 store; this keeps the warp full while turning those four serial operations into
            // one scalar lookup/store per lane.
            wp[p].ws = static_cast<unsigned>(ws_ptr[p][s_off + ld_q]);
        }
    };

    const auto stage_acts = [&](int tile, int buf) {
        const std::size_t x_off = static_cast<std::size_t>(tile) * 64;
#pragma unroll
        for (int p = 0; p < kActPasses; ++p) {
            const int row = ld_row0 + p * (kThreads / 4);
            if (row < BN) {
                d::int8_cpasync16(sb_ptr[p] + buf * (BN * kSRow), x_ptr[p] + x_off);
            }
        }
        if (tid < BN) {
            d::int8_cpasync8(sxs + buf * (BN * 2) + tid * 2,
                          xs_ptr + static_cast<std::size_t>(tile) * 2);
        }
    };

    const auto publish = [&](int buf) {
        const int sa_off  = buf * (BM * kSRow);
        const int sws_off = buf * (BM * d::kInt8ScaleGroups);
#pragma unroll
        for (int p = 0; p < kWeightPasses; ++p) {
            unsigned q0, q1, q2, q3;
            e2m1_word(wp[p].w.x, q0, q1);
            e2m1_word(wp[p].w.y, q2, q3);
            *reinterpret_cast<uint4*>(sa_ptr[p] + sa_off) = make_uint4(q0, q1, q2, q3);
            sws_ptr[p][sws_off + ld_q] = slut[wp[p].ws];
        }
    };

    float acc[MT][N_IT][4]; // m: rows of the warp band, n: eight-token columns of the band
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int n = 0; n < N_IT; ++n)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[m][n][j] = 0.0F;

    const int zero = 0;

    const auto compute_tile = [&](int buf) {
        const std::int8_t* sa_b = sa + buf * (BM * kSRow);
        const float* sws_b      = sws + buf * (BM * d::kInt8ScaleGroups);
        const std::int8_t* sb_b = sb + buf * (BN * kSRow);
        const float* sxs_b      = sxs + buf * (BN * 2);

        // A operands and weight scales are hoisted for the whole BK=64 tile: two ldmatrix.x4 per m
        // row group (four k-steps) and two lds128 of scales, reused across every n tile.
        unsigned af[MT][4][2];      // [m][k-step][lo/hi row]
        float    ws_pair[MT][2][4]; // [m][row 0/1][k-step]
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int r0 = Policy::fragment_row(wm, m, gid);
            const int rb = wm * Policy::kWarpRowStride + m * Policy::kFragmentStep;
            // lane -> (matrix, row inside 8, k-step pair): matrices 0/1 are the rows of k-step
            // 2*j, matrices 2/3 the rows of k-step 2*j+1.
            const std::int8_t* am = sa_b + (rb + (lane & 7) + ((lane >> 3) & 1) * 8) * kSRow +
                                    ((lane >> 4) & 1) * 16;
            unsigned a0[4], a1[4];
            d::int8_ldmatrix_x4(a0, am);
            d::int8_ldmatrix_x4(a1, am + 32);
            af[m][0][0] = a0[0];
            af[m][0][1] = a0[1];
            af[m][1][0] = a0[2];
            af[m][1][1] = a0[3];
            af[m][2][0] = a1[0];
            af[m][2][1] = a1[1];
            af[m][3][0] = a1[2];
            af[m][3][1] = a1[3];
            const uint4 w0 = d::int8_lds128(sws_b + r0 * 4);
            const uint4 w1 = d::int8_lds128(sws_b + (r0 + 8) * 4);
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                ws_pair[m][0][ks] = __uint_as_float((&w0.x)[ks]);
                ws_pair[m][1][ks] = __uint_as_float((&w1.x)[ks]);
            }
        }
        // Walk the token axis with pointer increments: the bases carry the lane term once and each
        // step is one add, where rebuilding the address from (wt, n, lane) cost 3-4 integer ops per
        // n on top of the two loads.
        const std::int8_t* brow_base =
            sb_b + (wt * N_IT * 8 + (lane & 7)) * kSRow + (lane >> 3) * 16;
        const float* xs_base = sxs_b + (wt * N_IT * 8 + tig * 2) * 2;
#pragma unroll
        for (int n = 0; n < N_IT; ++n) {
            const std::int8_t* brow = brow_base + n * (8 * kSRow);
            unsigned b4[4];
            d::int8_ldmatrix_x4(b4, brow);
            const uint4 x4 = d::int8_lds128(xs_base + n * 16);
            // BK=64 spans two 32-column activation groups: steps 0/1 rescale against the low half,
            // steps 2/3 against the high half -- the own chain's db[l%2] indexed by k.
            const float xl0 = __uint_as_float(x4.x); // token c0,   columns 0..31
            const float xl1 = __uint_as_float(x4.z); // token c0+1, columns 0..31
            const float xh0 = __uint_as_float(x4.y); // token c0,   columns 32..63
            const float xh1 = __uint_as_float(x4.w); // token c0+1, columns 32..63
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                int s0, s1, s2, s3;
                int u0, u1, u2, u3;
                int v0, v1, v2, v3;
                int y0, y1, y2, y3;
                d::mma_s8_k16_zero(s0, s1, s2, s3, af[m][0][0], af[m][0][1], b4[0], zero);
                d::mma_s8_k16_zero(u0, u1, u2, u3, af[m][1][0], af[m][1][1], b4[1], zero);
                d::mma_s8_k16_zero(v0, v1, v2, v3, af[m][2][0], af[m][2][1], b4[2], zero);
                d::mma_s8_k16_zero(y0, y1, y2, y3, af[m][3][0], af[m][3][1], b4[3], zero);
                acc[m][n][0] = fmaf(static_cast<float>(s0), ws_pair[m][0][0] * xl0, acc[m][n][0]);
                acc[m][n][1] = fmaf(static_cast<float>(s1), ws_pair[m][0][0] * xl1, acc[m][n][1]);
                acc[m][n][2] = fmaf(static_cast<float>(s2), ws_pair[m][1][0] * xl0, acc[m][n][2]);
                acc[m][n][3] = fmaf(static_cast<float>(s3), ws_pair[m][1][0] * xl1, acc[m][n][3]);
                acc[m][n][0] = fmaf(static_cast<float>(u0), ws_pair[m][0][1] * xl0, acc[m][n][0]);
                acc[m][n][1] = fmaf(static_cast<float>(u1), ws_pair[m][0][1] * xl1, acc[m][n][1]);
                acc[m][n][2] = fmaf(static_cast<float>(u2), ws_pair[m][1][1] * xl0, acc[m][n][2]);
                acc[m][n][3] = fmaf(static_cast<float>(u3), ws_pair[m][1][1] * xl1, acc[m][n][3]);
                acc[m][n][0] = fmaf(static_cast<float>(v0), ws_pair[m][0][2] * xh0, acc[m][n][0]);
                acc[m][n][1] = fmaf(static_cast<float>(v1), ws_pair[m][0][2] * xh1, acc[m][n][1]);
                acc[m][n][2] = fmaf(static_cast<float>(v2), ws_pair[m][1][2] * xh0, acc[m][n][2]);
                acc[m][n][3] = fmaf(static_cast<float>(v3), ws_pair[m][1][2] * xh1, acc[m][n][3]);
                acc[m][n][0] = fmaf(static_cast<float>(y0), ws_pair[m][0][3] * xh0, acc[m][n][0]);
                acc[m][n][1] = fmaf(static_cast<float>(y1), ws_pair[m][0][3] * xh1, acc[m][n][1]);
                acc[m][n][2] = fmaf(static_cast<float>(y2), ws_pair[m][1][3] * xh0, acc[m][n][2]);
                acc[m][n][3] = fmaf(static_cast<float>(y3), ws_pair[m][1][3] * xh1, acc[m][n][3]);
            }
        }
    };

    prefetch(0);
    stage_acts(0, 0);
    publish(0);
    d::int8_cpasync_commit();
    d::int8_cpasync_wait();
    __syncthreads();

    for (int tile = 0; tile < groups; ++tile) {
        const int buf = tile & 1;
        if (tile + 1 < groups) {
            prefetch(tile + 1);
            stage_acts(tile + 1, buf ^ 1);
            d::int8_cpasync_commit();
        }
        compute_tile(buf);
        if (tile + 1 < groups) {
            publish(buf ^ 1);
            d::int8_cpasync_wait();
        }
        __syncthreads();
    }

    // alpha carries 0.5/weight_scale_divisor; the activation scales already left through the
    // per-group rescale. The adapter owns everything the projection adds on top.
    //
    // `tokens` is the caller's real width, not the padded width the grid was built from: surplus
    // columns of a padded tile are computed and dropped (their stores would read back a residual
    // or leak into the gate plane), and a width that lands on a whole tile guards nothing away.
#pragma unroll
    for (int m = 0; m < MT; ++m) {
        if constexpr (Policy::kPaired) {
            // Paired gate_up: the fragment's row g is a gate row and row g+8 its up partner, so the
            // product is formed from this thread's own accumulators with the same FP32 expression
            // the two-launch path evaluates through the gate plane:
            // up * (gate / (1 + exp(-gate))), one BF16 rounding at the store.
            const int row = row_block + Policy::pair_index(Policy::fragment_row(wm, m, gid));
#pragma unroll
            for (int n = 0; n < N_IT; ++n) {
                const int c0 = col_block + Policy::token_column(wt, n, tig);
                if (c0 >= tokens) { break; }
                const float g0 = acc[m][n][0] * alpha_half;
                const float u0 = acc[m][n][2] * alpha_half;
                epilogue.store(row, c0, u0 * (g0 / (1.0F + __expf(-g0))));
                if (c0 + 1 < tokens) {
                    const float g1 = acc[m][n][1] * alpha_half;
                    const float u1 = acc[m][n][3] * alpha_half;
                    epilogue.store(row, c0 + 1, u1 * (g1 / (1.0F + __expf(-g1))));
                }
            }
        } else {
            const int row0 = row_block + Policy::fragment_row(wm, m, gid);
            const int row1 = row0 + 8;
#pragma unroll
            for (int n = 0; n < N_IT; ++n) {
                const int c0 = col_block + Policy::token_column(wt, n, tig);
                if (c0 >= tokens) { break; }
                epilogue.store(row0, c0, acc[m][n][0] * alpha_half);
                epilogue.store(row1, c0, acc[m][n][2] * alpha_half);
                if (c0 + 1 < tokens) {
                    epilogue.store(row0, c0 + 1, acc[m][n][1] * alpha_half);
                    epilogue.store(row1, c0 + 1, acc[m][n][3] * alpha_half);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Launch front end for the token-tile ladder: the width picks the stage (nvfp4_int8_select_j), and
// the grid and padding follow it, so adding a stage is one line here rather than a change in every
// route. A J that no width can select is a compile-time error in every instantiation instead of a
// runtime surprise, so the switch below covers exactly the admitted stages.
// ---------------------------------------------------------------------------
template <int J, class Output, class Policy>
inline void nvfp4_int8_grouped_launch_policy(const std::uint8_t* weight_codes,
                                            const std::uint8_t* weight_scales,
                                            const std::int8_t* x_codes, const float* x_scales,
                                            float alpha_half, Output output, std::int32_t k,
                                            std::int32_t tokens, std::int32_t rows,
                                            cudaStream_t stream) {
    static_assert(nvfp4_int8_j_admitted(J), "J must be a stage of the token-tile ladder");
    static_assert(Policy::kStagedRows == detail::kInt8TileRows, "every row policy stages one tile");
    const std::int32_t padded = nvfp4_int8_padded_tokens(tokens);
    // Token tiles on the fast axis, row tiles on the slow one: see the kernel's block-index note.
    // Every block stages one 128-row weight tile, so the grid height is rows/128 whatever the policy;
    // the paired layout only changes which global rows those staged rows name (and thus how many
    // output rows one block answers).
    const dim3 grid(padded / J, rows / detail::kInt8TileRows);
    // Wide stages run 16 warps: the per-thread accumulator set halves (MT=1), which is what keeps
    // the block inside the 128-register budget that 512 threads impose. Narrow stages keep eight
    // warps, because 16 warps cannot cover 128 rows at MT>=1 once J is at or below 64.
    constexpr int kThreads = 256;
    nvfp4_int8_grouped_kernel<J, kThreads, Output, Policy>
        <<<grid, kThreads, nvfp4_int8_grouped_smem_bytes(), stream>>>(
            weight_codes, weight_scales, x_codes, x_scales, alpha_half, output, k, tokens, rows);
}

template <template <int, int> class PolicyFor, class Output>
inline void nvfp4_int8_grouped_dispatch(const std::uint8_t* weight_codes,
                                        const std::uint8_t* weight_scales,
                                        const std::int8_t* x_codes, const float* x_scales,
                                        float alpha_half, Output output, std::int32_t k,
                                        std::int32_t tokens, std::int32_t rows,
                                        cudaStream_t stream) {
    switch (nvfp4_int8_select_j(tokens)) {
        case 8:
            nvfp4_int8_grouped_launch_policy<8, Output, PolicyFor<8, detail::kInt8Threads>>(
                weight_codes, weight_scales, x_codes, x_scales, alpha_half, output, k, tokens, rows,
                stream);
            return;
        case 16:
            nvfp4_int8_grouped_launch_policy<16, Output, PolicyFor<16, detail::kInt8Threads>>(
                weight_codes, weight_scales, x_codes, x_scales, alpha_half, output, k, tokens, rows,
                stream);
            return;
        case 24:
            nvfp4_int8_grouped_launch_policy<24, Output, PolicyFor<24, detail::kInt8Threads>>(
                weight_codes, weight_scales, x_codes, x_scales, alpha_half, output, k, tokens, rows,
                stream);
            return;
        case 32:
            nvfp4_int8_grouped_launch_policy<32, Output, PolicyFor<32, detail::kInt8Threads>>(
                weight_codes, weight_scales, x_codes, x_scales, alpha_half, output, k, tokens, rows,
                stream);
            return;
        case 40:
            nvfp4_int8_grouped_launch_policy<40, Output, PolicyFor<40, detail::kInt8Threads>>(
                weight_codes, weight_scales, x_codes, x_scales, alpha_half, output, k, tokens, rows,
                stream);
            return;
        case 48:
            nvfp4_int8_grouped_launch_policy<48, Output, PolicyFor<48, detail::kInt8Threads>>(
                weight_codes, weight_scales, x_codes, x_scales, alpha_half, output, k, tokens, rows,
                stream);
            return;
        case 64:
            nvfp4_int8_grouped_launch_policy<64, Output, PolicyFor<64, detail::kInt8Threads>>(
                weight_codes, weight_scales, x_codes, x_scales, alpha_half, output, k, tokens, rows,
                stream);
            return;
        case 80:
            nvfp4_int8_grouped_launch_policy<80, Output, PolicyFor<80, detail::kInt8Threads>>(
                weight_codes, weight_scales, x_codes, x_scales, alpha_half, output, k, tokens, rows,
                stream);
            return;
        case 96:
            nvfp4_int8_grouped_launch_policy<96, Output, PolicyFor<96, detail::kInt8Threads>>(
                weight_codes, weight_scales, x_codes, x_scales, alpha_half, output, k, tokens, rows,
                stream);
            return;
        case 112:
            nvfp4_int8_grouped_launch_policy<112, Output, PolicyFor<112, detail::kInt8Threads>>(
                weight_codes, weight_scales, x_codes, x_scales, alpha_half, output, k, tokens, rows,
                stream);
            return;
        case 128:
            nvfp4_int8_grouped_launch_policy<128, Output, PolicyFor<128, detail::kInt8Threads>>(
                weight_codes, weight_scales, x_codes, x_scales, alpha_half, output, k, tokens, rows,
                stream);
            return;
        default:
            break;
    }
    throw std::invalid_argument("nvfp4 int8 grouped: no admissible token tile for T=" +
                                std::to_string(tokens));
}

template <class Output>
inline void nvfp4_int8_grouped_launch(const std::uint8_t* weight_codes,
                                      const std::uint8_t* weight_scales,
                                      const std::int8_t* x_codes, const float* x_scales,
                                      float alpha_half, Output output, std::int32_t k,
                                      std::int32_t tokens, std::int32_t rows, cudaStream_t stream) {
    nvfp4_int8_grouped_dispatch<Nvfp4Int8RowPolicy, Output>(
        weight_codes, weight_scales, x_codes, x_scales, alpha_half, output, k, tokens, rows, stream);
}

// The same mainloop with the paired gate_up row layout: `rows` is the full [gate; up] plane and the
// block's output row range is half of it. The caller stores the product through an ordinary output
// adapter, so `out` is the [rows/2, tokens] tensor.
template <class Output>
inline void nvfp4_int8_grouped_swiglu_launch(const std::uint8_t* weight_codes,
                                             const std::uint8_t* weight_scales,
                                             const std::int8_t* x_codes, const float* x_scales,
                                             float alpha_half, Output output, std::int32_t k,
                                             std::int32_t tokens, std::int32_t rows,
                                             cudaStream_t stream) {
    nvfp4_int8_grouped_dispatch<Nvfp4Int8SwiGluRowPolicy, Output>(
        weight_codes, weight_scales, x_codes, x_scales, alpha_half, output, k, tokens, rows, stream);
}

} // namespace ninfer::ops::detail::unified
