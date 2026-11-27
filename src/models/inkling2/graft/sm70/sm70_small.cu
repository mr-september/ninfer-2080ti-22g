#include "ops/linear/sm70/sm70_small.h"

#include "core/device_launch.h"
#include "core/qtype.h"
#include "ops/common/math.cuh"
#include "ops/linear/sm70/sm70_dequantize.h"
#include "ops/linear/sm70/sm70_epilogue.cuh"
#include "ops/linear/sm70/sm70_bf16_f16.cuh"
#include "ops/linear/sm70/sm70_gemm.h"
#include "ops/linear/sm70/sm70_small.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail::sm70 {
namespace {

template <class Decoder>
__host__ __device__ constexpr float value_scale() {
    if constexpr (requires { Decoder::kValueScale; }) {
        return Decoder::kValueScale;
    } else {
        return 1.0f;
    }
}

template <class Decoder, int Tiles, bool Direct, int Warps>
__launch_bounds__(Warps * 32) __global__
    void small_project_kernel(Decoder decoder, std::int32_t rows,
                              const std::uint16_t* __restrict__ x,
                              const std::uint16_t* __restrict__ xb,
                              const std::uint8_t* __restrict__ beyond, std::int32_t k,
                              std::int32_t col0, std::int32_t cols, DestinationEpilogue epilogue) {
    __shared__ SmallShared<Tiles, Decoder::kChunk, Warps> shared;
    const std::int32_t row0 = static_cast<std::int32_t>(blockIdx.x) * kSmallTileRows;
    small_tile<Decoder, Tiles, Direct, Warps>(decoder, decoder, row0, row0, kSmallTileRows, rows,
                                              rows, x, xb, beyond, k, col0, cols, shared);
    if (threadIdx.x >= 32) { return; }
    constexpr float kScale = value_scale<Decoder>();
    for (int e = static_cast<int>(threadIdx.x); e < kSmallTileRows * 8 * Tiles; e += 32) {
        const int r            = e / (8 * Tiles);
        const int c            = e % (8 * Tiles);
        const std::int32_t row = row0 + r;
        if (row < rows && c < cols) { epilogue(row, col0 + c, shared.partial[0][r][c] * kScale); }
    }
}

// Tile rows 0..15 are gate rows [i0, i0 + 16), rows 16..31 the matching up rows.
template <class Decoder, int Tiles, bool Direct, int Warps>
__launch_bounds__(Warps * 32) __global__
    void small_swiglu_kernel(Decoder gate, Decoder up, std::int32_t rows, std::int32_t gate_first,
                             std::int32_t up_first, std::int32_t gate_limit, std::int32_t up_limit,
                             const std::uint16_t* __restrict__ x,
                             const std::uint16_t* __restrict__ xb,
                             const std::uint8_t* __restrict__ beyond, std::int32_t k,
                             std::int32_t col0, std::int32_t cols, RowFactor gate_factor,
                             RowFactor up_factor, __nv_bfloat16* __restrict__ out) {
    __shared__ SmallShared<Tiles, Decoder::kChunk, Warps> shared;
    const std::int32_t i0 = static_cast<std::int32_t>(blockIdx.x) * (kSmallTileRows / 2);
    small_tile<Decoder, Tiles, Direct, Warps>(gate, up, gate_first + i0, up_first + i0,
                                              kSmallTileRows / 2, gate_limit, up_limit, x, xb,
                                              beyond, k, col0, cols, shared);
    if (threadIdx.x >= 32) { return; }
    constexpr float kScale = value_scale<Decoder>();
    for (int e = static_cast<int>(threadIdx.x); e < (kSmallTileRows / 2) * 8 * Tiles; e += 32) {
        const int r          = e / (8 * Tiles);
        const int c          = e % (8 * Tiles);
        const std::int32_t i = i0 + r;
        if (i < rows && c < cols) {
            const float g = shared.partial[0][r][c] * kScale * gate_factor(gate_first + i);
            const float u =
                shared.partial[0][r + kSmallTileRows / 2][c] * kScale * up_factor(up_first + i);
            out[static_cast<std::size_t>(col0 + c) * rows + i] = __float2bfloat16_rn(silu(g) * u);
        }
    }
}

[[maybe_unused]] RowSplitView row_split_view(const Weight& w, int group) {
    RowSplitView view;
    view.base           = static_cast<const std::uint8_t*>(w.qdata);
    view.high           = static_cast<const std::uint8_t*>(w.qhigh);
    view.scales         = static_cast<const __half*>(w.scales);
    view.groups_per_row = w.padded_shape[1] / group;
    view.panel_shift    = row_split_panel_shift(w.layout);
    return view;
}

// f.template operator()<Decoder>(decoder) with the weight's decoder.
template <class F>
void with_decoder(const Weight& w, F&& f) {
    switch (w.qtype) {
    case QType::Q2_G64_FP16:
        return f(Quartet2Decoder<false>{row_split_view(w, 64)});
    case QType::Q4_G64_FP16:
        return f(Q4Decoder{row_split_view(w, 64)});
    case QType::Q5_G64_FP16:
        return f(Q5Decoder{row_split_view(w, 64)});
    case QType::Q6_G64_FP16:
        return f(Q6Decoder{row_split_view(w, 64)});
    case QType::Q8_G32_FP16:
        return f(Q8Decoder{row_split_view(w, 32)});
    case QType::T2_G128_FP16:
        return f(Quartet2Decoder<true>{row_split_view(w, 128)});
    case QType::BF16:
        return f(Bf16Decoder{static_cast<const __nv_bfloat16*>(w.qdata), w.k});
    case QType::FP8_E4M3FN_ROW_BF16:
        return f(Fp8Decoder{static_cast<const std::uint8_t*>(w.qdata), w.k});
    case QType::NVFP4:
        return f(Nvfp4Decoder{static_cast<const std::uint8_t*>(w.qdata),
                              static_cast<const std::uint8_t*>(w.scales), w.k});
    default:
        break;
    }
    throw std::invalid_argument("sm70 small-T: unsupported weight format");
}

// Column launches of 8, 16 or 32 columns over [0, tokens): launch<Tiles, Direct>(x, xb, beyond,
// col0, cols), on the caller's BF16 plane at the direct widths and on the prepared P8 FP16 one with
// its chunk flags otherwise; xb is the BF16 plane.
template <class Launch>
void by_columns(const Activation& x, Launch&& launch) {
    const auto* bf16 = reinterpret_cast<const std::uint16_t*>(x.bf16);
    if (x.data == nullptr) {
        launch.template operator()<1, true>(bf16, bf16, nullptr, 0, x.tokens);
        return;
    }
    const auto* f16 = reinterpret_cast<const std::uint16_t*>(x.data);
    for (std::int32_t col0 = 0; col0 < x.tokens; col0 += 32) {
        const std::int32_t cols = std::min<std::int32_t>(32, x.tokens - col0);
        if (cols <= 8) {
            launch.template operator()<1, false>(f16, bf16, x.beyond_chunks, col0, cols);
        } else if (cols <= 16) {
            launch.template operator()<2, false>(f16, bf16, x.beyond_chunks, col0, cols);
        } else {
            launch.template operator()<4, false>(f16, bf16, x.beyond_chunks, col0, cols);
        }
    }
}

// Sixteen warps per CTA where eight would leave fewer than about 32 warps per SM of a V100 (80
// SMs) and K still gives each of sixteen warps six or more turns: narrow parents such as a 5120-row
// down projection, whose 160 tiles of 32 rows would otherwise run 16 warps per SM. Up to two column
// tiles (shared memory).
constexpr std::int32_t kWideWarpsBelowTiles = 80 * 32 / kSmallWarps;
constexpr std::int32_t kWideWarpsMinTurns   = 6;

template <class Decoder, int Tiles, class Launch>
void with_warps(std::int32_t tiles, std::int32_t k, Launch&& launch) {
    if constexpr (Tiles <= 2) {
        const std::int32_t turns = k / (Decoder::kChunk * Decoder::kUnroll);
        if (tiles < kWideWarpsBelowTiles && turns >= kSmallWideWarps * kWideWarpsMinTurns) {
            launch.template operator()<kSmallWideWarps>();
            return;
        }
    }
    launch.template operator()<kSmallWarps>();
}

// Eight values of a BF16 column per thread into the P8 order of sm70_small.cuh, saturated FP16,
// and per 64-value chunk (eight consecutive threads; K is a whole number of chunks) one flag byte:
// nonzero where the chunk holds a value beyond FP16.
__global__ void convert_activation_p8_kernel(const uint4* __restrict__ x, uint4* __restrict__ out,
                                             std::uint8_t* __restrict__ beyond,
                                             std::int64_t groups) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool valid     = i < groups;
    bool saturated       = false;
    if (valid) {
        const uint4 v      = x[i];
        const unsigned f01 = bf16x2_to_f16x2(v.x);
        const unsigned f23 = bf16x2_to_f16x2(v.y);
        const unsigned f45 = bf16x2_to_f16x2(v.z);
        const unsigned f67 = bf16x2_to_f16x2(v.w);
        out[i]    = make_uint4(__byte_perm(f01, f45, 0x5410u), __byte_perm(f01, f45, 0x7632u),
                               __byte_perm(f23, f67, 0x5410u), __byte_perm(f23, f67, 0x7632u));
        saturated = bf16x2_beyond_f16(v.x) || bf16x2_beyond_f16(v.y) ||
                    bf16x2_beyond_f16(v.z) || bf16x2_beyond_f16(v.w);
    }
    const unsigned ballot = __ballot_sync(0xffffffffu, saturated);
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    if (valid && (lane & 7) == 0) { beyond[i / 8] = ((ballot >> lane) & 0xffu) != 0u; }
}

// The bytes of a prepared plane's chunk flags.
std::size_t beyond_chunk_bytes(std::int32_t k, std::int32_t tokens) {
    return (static_cast<std::size_t>(k / kSmallChunk) * static_cast<std::size_t>(tokens) + 255) /
           256 * 256;
}

std::size_t plane_bytes(std::int32_t k, std::int32_t tokens) {
    return (static_cast<std::size_t>(k) * static_cast<std::size_t>(tokens) * sizeof(__half) + 255) /
           256 * 256;
}

} // namespace

bool small_takes(std::int32_t tokens) { return enabled() && tokens >= 1 && !takes(tokens); }

bool small_shape(const Weight& w) {
    // A T2 chunk is its whole G128 group (Quartet2Decoder<true>).
    const std::int32_t chunk = w.qtype == QType::T2_G128_FP16 ? 128 : kSmallChunk;
    return w.k % chunk == 0 && dequantizable(w);
}

std::size_t small_workspace_bytes(std::int32_t k, std::int32_t tokens) {
    return tokens <= kSmallDirectTokens ? 0
                                        : plane_bytes(k, tokens) + beyond_chunk_bytes(k, tokens);
}

namespace {

void require_activation(const Tensor& x) {
    if (x.dtype != DType::BF16 || x.data == nullptr || !x.is_contiguous() || x.ne[2] != 1 ||
        x.ne[3] != 1 || reinterpret_cast<std::uintptr_t>(x.data) % 16 != 0) {
        throw std::invalid_argument("sm70 small-T: x must be a contiguous, 16-byte aligned BF16 "
                                    "[K, T]");
    }
}

} // namespace

Activation small_direct(const Tensor& x) {
    require_activation(x);
    if (x.ne[1] > kSmallDirectTokens) {
        throw std::invalid_argument("sm70 small-T: a direct activation is at most eight columns");
    }
    Activation out;
    out.k      = x.ne[0];
    out.tokens = x.ne[1];
    out.bf16   = static_cast<const __nv_bfloat16*>(x.data);
    return out;
}

Activation small_prepare(const Tensor& x, WorkspaceArena& workspace, cudaStream_t stream) {
    require_activation(x);
    if (x.ne[1] <= kSmallDirectTokens) { return small_direct(x); }
    Activation out;
    out.k      = x.ne[0];
    out.tokens = x.ne[1];
    auto* f16 = static_cast<__half*>(workspace.alloc_bytes(plane_bytes(out.k, out.tokens)).data);
    auto* beyond = static_cast<std::uint8_t*>(
        workspace.alloc_bytes(beyond_chunk_bytes(out.k, out.tokens)).data);
    // K is a whole number of 64-value chunks (small_shape), so every eight values are one group.
    const std::int64_t groups = static_cast<std::int64_t>(out.k) * out.tokens / 8;
    constexpr int kThreads    = 256;
    convert_activation_p8_kernel<<<static_cast<unsigned>((groups + kThreads - 1) / kThreads),
                                   kThreads, 0, stream>>>(static_cast<const uint4*>(x.data),
                                                          reinterpret_cast<uint4*>(f16), beyond,
                                                          groups);
    CUDA_CHECK(cudaGetLastError());
    out.data          = f16;
    out.bf16          = static_cast<const __nv_bfloat16*>(x.data);
    out.beyond_chunks = beyond;
    return out;
}

void small_project(const Activation& x, const Weight& w, std::span<const Destination> destinations,
                   cudaStream_t stream) {
    if (w.k != x.k || !small_shape(w) || destinations.empty() ||
        destinations.size() > kMaxDestinations) {
        throw std::invalid_argument("sm70 small-T: unsupported weight or destinations");
    }
#if defined(NINFER_TIER_VOLTA)
    DestinationEpilogue epilogue;
    for (const Destination& d : destinations) { epilogue.set[epilogue.count++] = d; }
    epilogue.factor = row_factor(w);
    const dim3 grid(static_cast<unsigned>((w.n + kSmallTileRows - 1) / kSmallTileRows));
    with_decoder(w, [&](auto decoder) {
        using Decoder = decltype(decoder);
        by_columns(x, [&]<int Tiles, bool Direct>(
                          const std::uint16_t* plane, const std::uint16_t* xb,
                          const std::uint8_t* beyond, std::int32_t col0, std::int32_t cols) {
            with_warps<Decoder, Tiles>(static_cast<std::int32_t>(grid.x), x.k, [&]<int Warps>() {
                small_project_kernel<Decoder, Tiles, Direct, Warps>
                    <<<grid, Warps * 32, 0, stream>>>(decoder, w.n, plane, xb, beyond, x.k, col0,
                                                      cols, epilogue);
                CUDA_CHECK(cudaGetLastError());
            });
        });
    });
#else
    (void)stream;
    throw std::logic_error("sm70 small-T: this build does not target sm_70");
#endif
}

void small_swiglu(const Activation& x, const Weight& gate, const Weight* up, Tensor& out,
                  cudaStream_t stream) {
    const std::int32_t rows = out.ne[0];
    const bool joined       = up == nullptr;
    const Weight& up_weight = joined ? gate : *up;
    if (out.dtype != DType::BF16 || out.ne[1] != x.tokens || gate.k != x.k || !small_shape(gate) ||
        (joined ? gate.n != 2 * rows
                : (gate.n != rows || up->n != rows || up->k != x.k || up->qtype != gate.qtype ||
                   !small_shape(*up)))) {
        throw std::invalid_argument("sm70 small-T swiglu: unsupported weights or output");
    }
#if defined(NINFER_TIER_VOLTA)
    const std::int32_t up_first   = joined ? rows : 0;
    const std::int32_t gate_limit = joined ? rows : gate.n;
    const std::int32_t up_limit   = joined ? 2 * rows : up_weight.n;
    const dim3 grid(static_cast<unsigned>((rows + kSmallTileRows / 2 - 1) / (kSmallTileRows / 2)));
    with_decoder(gate, [&](auto gate_decoder) {
        using Decoder = decltype(gate_decoder);
        with_decoder(up_weight, [&](auto up_decoder) {
            if constexpr (std::is_same_v<decltype(up_decoder), Decoder>) {
                by_columns(x, [&]<int Tiles, bool Direct>(
                                  const std::uint16_t* plane, const std::uint16_t* xb,
                                  const std::uint8_t* beyond, std::int32_t col0,
                                  std::int32_t cols) {
                    with_warps<Decoder, Tiles>(
                        static_cast<std::int32_t>(grid.x), x.k, [&]<int Warps>() {
                            small_swiglu_kernel<Decoder, Tiles, Direct, Warps>
                                <<<grid, Warps * 32, 0, stream>>>(
                                    gate_decoder, up_decoder, rows, 0, up_first, gate_limit,
                                    up_limit, plane, xb, beyond, x.k, col0, cols,
                                    row_factor(gate), row_factor(up_weight),
                                    static_cast<__nv_bfloat16*>(out.data));
                            CUDA_CHECK(cudaGetLastError());
                        });
                });
            }
        });
    });
#else
    (void)stream;
    throw std::logic_error("sm70 small-T: this build does not target sm_70");
#endif
}

} // namespace ninfer::ops::detail::sm70
