#include "ops/linear/sm70/sm70_integer.h"

#include "core/device_launch.h"
#include "core/layout.h"
#include "core/qtype.h"
#include "core/weight_view.h"
#include "ops/common/device_multiprocessors.h"
#include "ops/common/math.cuh"
#include "ops/linear/sm70/sm70_dequantize.h"
#include "ops/linear/sm70/sm70_epilogue.cuh"
#include "ops/linear/sm70/sm70_gemm.h"
#include "ops/linear/sm70/sm70_small.cuh"
#include "ops/linear/t2/t2_a8.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail::sm70 {
namespace {

constexpr int kGroup   = 64;  // values per activation scale
constexpr int kWarps   = 8;   // two warps per four-row panel, so a CTA owns 16 rows
constexpr int kT2Group = 128; // values per T2 weight scale (a 32-byte record)

// Element e of a 64-value group in the stored order: within each sixteen-value block, word s holds
// elements 4i + s (i = 0..3) in byte i, the order in which a T2 code word's fields come out of
// (word >> 2s) & 0x03030303.
__device__ __forceinline__ int place(int e) { return (e & ~15) | ((e & 3) << 2) | ((e & 15) >> 2); }

// One warp per (column, 64-value group): codes at [T][K] in the stored order, scale FP16(absmax /
// 127) as a float and twice the codes' sum (the offset term of integer_dot) at [T][K / 64].
__global__ void integer_quantize_kernel(const __nv_bfloat16* __restrict__ x, std::int32_t k,
                                        std::int8_t* __restrict__ codes, float* __restrict__ scales,
                                        std::int32_t* __restrict__ sums) {
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int group  = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) >> 5) +
                       (static_cast<int>(threadIdx.x) >> 5);
    const int token  = static_cast<int>(blockIdx.y);
    const int groups = k / kGroup;
    if (group >= groups) { return; }
    const std::int64_t base = static_cast<std::int64_t>(token) * k + group * kGroup;
    const float2 value =
        __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + base + 2 * lane));
    float amax = fmaxf(fabsf(value.x), fabsf(value.y));
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
    }
    amax                              = fmaxf(amax, 1.0e-20F);
    const float inv                   = 127.0F / amax;
    const int low                     = max(-127, min(127, __float2int_rn(value.x * inv)));
    const int high                    = max(-127, min(127, __float2int_rn(value.y * inv)));
    codes[base + place(2 * lane)]     = static_cast<std::int8_t>(low);
    codes[base + place(2 * lane + 1)] = static_cast<std::int8_t>(high);
    int sum                           = low + high;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, mask); }
    if (lane == 0) {
        const std::int64_t at = static_cast<std::int64_t>(token) * groups + group;
        scales[at]            = __half2float(__float2half(amax / 127.0F));
        sums[at]              = 2 * sum;
    }
}

// The CTA's copy of the activations: codes [Cols][K], scales and offset terms [Cols][K / 64].
template <int Cols>
struct StagedActivations {
    const std::int8_t* codes;
    const float* scales;
    const std::int32_t* sums;
    std::int32_t groups;
};

template <int Cols>
__device__ __forceinline__ StagedActivations<Cols>
stage_activations(const std::int8_t* __restrict__ codes, const float* __restrict__ scales,
                  const std::int32_t* __restrict__ sums, std::int32_t k, std::int32_t cols) {
    extern __shared__ int4 shared_words[];
    auto* code_s      = reinterpret_cast<std::int8_t*>(shared_words);
    const int groups  = k / kGroup;
    auto* scale_s     = reinterpret_cast<float*>(code_s + Cols * k);
    auto* sum_s       = reinterpret_cast<std::int32_t*>(scale_s + Cols * groups);
    const int threads = static_cast<int>(blockDim.x);
    for (int i = static_cast<int>(threadIdx.x); i < Cols * k / 16; i += threads) {
        const int c = i / (k / 16);
        shared_words[i] =
            c < cols ? __ldg(reinterpret_cast<const int4*>(codes) + i) : make_int4(0, 0, 0, 0);
    }
    for (int i = static_cast<int>(threadIdx.x); i < Cols * groups; i += threads) {
        const bool live = i / groups < cols;
        scale_s[i]      = live ? __ldg(scales + i) : 0.0F;
        sum_s[i]        = live ? __ldg(sums + i) : 0;
    }
    __syncthreads();
    return {code_s, scale_s, sum_s, groups};
}

// The 64 codes of a T2 record half as sixteen offset words: byte i of word 4j + s is code 16j +
// 4i + s plus 2, a value in {1, 2, 3}.
__device__ __forceinline__ void offset_codes(uint4 record, unsigned (&b)[16]) {
    const unsigned words[4] = {record.x, record.y, record.z, record.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            b[4 * j + s] = ((words[j] >> (2 * s)) & 0x03030303u) ^ 0x02020202u;
        }
    }
}

// sum (code + 2) * x over a 64-value group, less 2 sum x: the exact integer dot product.
template <int Cols>
__device__ __forceinline__ int integer_dot(const unsigned (&b)[16],
                                           const StagedActivations<Cols>& x, int c, int group) {
    const int4* words = reinterpret_cast<const int4*>(
        x.codes + static_cast<std::int64_t>(c) * x.groups * kGroup + group * kGroup);
    int dot = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int4 v = words[j];
        dot          = __dp4a(static_cast<int>(b[4 * j + 0]), v.x, dot);
        dot          = __dp4a(static_cast<int>(b[4 * j + 1]), v.y, dot);
        dot          = __dp4a(static_cast<int>(b[4 * j + 2]), v.z, dot);
        dot          = __dp4a(static_cast<int>(b[4 * j + 3]), v.w, dot);
    }
    return dot - x.sums[c * x.groups + group];
}

// The lanes of a warp on the four-row panel at `row0`: lane l reads, at step s, half (l % 2) of
// row row0 + (l % 8) / 2's record for group 4s + l / 8. With the panel layout the warp's 32 reads
// are four whole 128-byte lines.
struct PanelLane {
    int row_in_panel;
    int half;
    int group_offset;
};

__device__ __forceinline__ PanelLane panel_lane(int lane) {
    return {(lane & 7) >> 1, lane & 1, lane >> 3};
}

// Adds row partials over the lanes that share a row (halves and group offsets); every lane ends
// with its row's sum.
__device__ __forceinline__ float row_sum(float value) {
    value += __shfl_xor_sync(0xffffffffu, value, 1);
    value += __shfl_xor_sync(0xffffffffu, value, 8);
    value += __shfl_xor_sync(0xffffffffu, value, 16);
    return value;
}

// A lane's record of one step: half p.half of row `row`'s record for group 4 step + p.group_offset
// and that group's weight scale (nothing past the row's groups).
struct Record {
    uint4 codes;
    float scale;
    int group;
};

__device__ __forceinline__ Record load_record(const RowSplitView& view, std::int32_t row,
                                              const PanelLane& p, int step, int t2_groups) {
    Record r{make_uint4(0u, 0u, 0u, 0u), 0.0F, 4 * step + p.group_offset};
    if (r.group < t2_groups) {
        r.codes = __ldg(reinterpret_cast<const uint4*>(
                            view.record(view.base, kT2Group / 4, row, r.group)) +
                        p.half);
        r.scale = view.scale(row, r.group);
    }
    return r;
}

template <int Cols>
__device__ __forceinline__ void accumulate(const Record& r, const PanelLane& p, int t2_groups,
                                           const StagedActivations<Cols>& x, int cols,
                                           float (&acc)[Cols]) {
    if (r.group >= t2_groups) { return; }
    unsigned b[16];
    offset_codes(r.codes, b);
    const int x_group = 2 * r.group + p.half;
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
        if (c >= cols) { break; }
        acc[c] = fmaf(static_cast<float>(integer_dot(b, x, c, x_group)),
                      r.scale * x.scales[c * x.groups + x_group], acc[c]);
    }
}

// The warps of a panel split its steps (kSplit warps, kUnroll steps of records in flight per
// warp): a narrow parent such as a 5120-row down projection otherwise runs too few warps per SM to
// keep the weight stream busy.
constexpr int kSplit  = 2;
constexpr int kUnroll = 2;
constexpr int kPanels = kWarps / kSplit; // four-row panels per CTA

// The partial sums of a warp's steps, over the lanes of each row.
template <int Cols, int Parts>
__device__ __forceinline__ void panel_sums(const RowSplitView (&views)[Parts],
                                           const std::int32_t (&rows)[Parts], const PanelLane& p,
                                           int split, int t2_groups,
                                           const StagedActivations<Cols>& x, int cols,
                                           float (&acc)[Parts][Cols]) {
#pragma unroll
    for (int part = 0; part < Parts; ++part) {
#pragma unroll
        for (int c = 0; c < Cols; ++c) { acc[part][c] = 0.0F; }
    }
    const int steps = (t2_groups + 3) / 4;
    for (int step0 = split * kUnroll; step0 < steps; step0 += kSplit * kUnroll) {
        Record r[Parts][kUnroll];
#pragma unroll
        for (int part = 0; part < Parts; ++part) {
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
                r[part][u] = load_record(views[part], rows[part], p, step0 + u, t2_groups);
            }
        }
#pragma unroll
        for (int part = 0; part < Parts; ++part) {
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
                accumulate<Cols>(r[part][u], p, t2_groups, x, cols, acc[part]);
            }
        }
    }
#pragma unroll
    for (int part = 0; part < Parts; ++part) {
#pragma unroll
        for (int c = 0; c < Cols; ++c) { acc[part][c] = row_sum(acc[part][c]); }
    }
}

template <int Cols>
__launch_bounds__(kWarps * 32) __global__
    void integer_project_kernel(RowSplitView view, std::int32_t rows, std::int32_t k,
                                const std::int8_t* __restrict__ codes,
                                const float* __restrict__ scales,
                                const std::int32_t* __restrict__ sums, std::int32_t cols,
                                DestinationEpilogue epilogue) {
    __shared__ float split_sums[kWarps][Cols][4];
    const StagedActivations<Cols> x = stage_activations<Cols>(codes, scales, sums, k, cols);
    const int lane                  = static_cast<int>(threadIdx.x) & 31;
    const int warp                  = static_cast<int>(threadIdx.x) >> 5;
    const int split                 = warp % kSplit;
    const std::int32_t row0 =
        (static_cast<std::int32_t>(blockIdx.x) * kPanels + warp / kSplit) * 4;
    const PanelLane p      = panel_lane(lane);
    const std::int32_t row = row0 + p.row_in_panel;
    const bool live        = row0 < rows;
    float acc[1][Cols];
    if (live) {
        const RowSplitView views[1]   = {view};
        const std::int32_t panel_rows[1] = {row};
        panel_sums<Cols, 1>(views, panel_rows, p, split, k / kT2Group, x, cols, acc);
        if ((lane & 0x19) == 0) {
#pragma unroll
            for (int c = 0; c < Cols; ++c) { split_sums[warp][c][p.row_in_panel] = acc[0][c]; }
        }
    }
    __syncthreads();
    if (!live || split != 0 || (lane & 0x19) != 0) { return; }
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
        if (c >= cols) { break; }
        float value = 0.0F;
#pragma unroll
        for (int s = 0; s < kSplit; ++s) { value += split_sums[warp + s][c][p.row_in_panel]; }
        epilogue(row, c, value);
    }
}

// Panel rows [i0, i0 + 4) of the gate and the matching up rows.
template <int Cols>
__launch_bounds__(kWarps * 32) __global__
    void integer_swiglu_kernel(RowSplitView gate, RowSplitView up, std::int32_t rows,
                               std::int32_t up_first, std::int32_t k,
                               const std::int8_t* __restrict__ codes,
                               const float* __restrict__ scales,
                               const std::int32_t* __restrict__ sums, std::int32_t cols,
                               RowFactor gate_factor, RowFactor up_factor,
                               __nv_bfloat16* __restrict__ out) {
    __shared__ float split_sums[kWarps][2][Cols][4];
    const StagedActivations<Cols> x = stage_activations<Cols>(codes, scales, sums, k, cols);
    const int lane                  = static_cast<int>(threadIdx.x) & 31;
    const int warp                  = static_cast<int>(threadIdx.x) >> 5;
    const int split                 = warp % kSplit;
    const std::int32_t i0 = (static_cast<std::int32_t>(blockIdx.x) * kPanels + warp / kSplit) * 4;
    const PanelLane p     = panel_lane(lane);
    const std::int32_t i  = i0 + p.row_in_panel;
    const bool live       = i0 < rows;
    float acc[2][Cols];
    if (live) {
        const RowSplitView views[2]   = {gate, up};
        const std::int32_t panel_rows[2] = {i, up_first + i};
        panel_sums<Cols, 2>(views, panel_rows, p, split, k / kT2Group, x, cols, acc);
        if ((lane & 0x19) == 0) {
#pragma unroll
            for (int c = 0; c < Cols; ++c) {
                split_sums[warp][0][c][p.row_in_panel] = acc[0][c];
                split_sums[warp][1][c][p.row_in_panel] = acc[1][c];
            }
        }
    }
    __syncthreads();
    if (!live || split != 0 || (lane & 0x19) != 0) { return; }
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
        if (c >= cols) { break; }
        float g = 0.0F;
        float u = 0.0F;
#pragma unroll
        for (int s = 0; s < kSplit; ++s) {
            g += split_sums[warp + s][0][c][p.row_in_panel];
            u += split_sums[warp + s][1][c][p.row_in_panel];
        }
        out[static_cast<std::size_t>(c) * rows + i] =
            __float2bfloat16_rn(silu(g * gate_factor(i)) * (u * up_factor(up_first + i)));
    }
}

// From two columns up (two requests, a verify round), where a 16-row CTA restaging every column of
// the activations would read them from L2 about as often as the weights, a CTA instead owns a range
// of four-row panels for the whole call (as many CTAs as the SMs hold at once), stages the activations one K tile
// at a time and keeps its panels' partial sums in shared memory. A lane owns two rows of a panel for
// one (group, half) of a step of eight groups -- lane = goff << 2 | q << 1 | half -- so every 16
// bytes of activations it reads serve two rows; the eight lanes of a quarter-warp read four
// adjacent chunks of the tile, which is stored chunk-major: chunk j of 64-value group g of column c
// at (c * 4 + j) * tile_groups + g. Past four columns the dp4a issue rate, not the weight stream,
// bounds the call (34,816 x 5,120 at eight columns: 132 us against 149 us on the tensor-core route).
constexpr int kWideRows    = 2;            // rows per lane
constexpr int kWideStep    = 8;            // T2 groups per warp step
constexpr int kWideValues  = kWideStep * kT2Group;
constexpr int kWideTileMax = 32 * 1024;    // activation bytes of one K tile
constexpr int kWideShared  = 48 * 1024;    // a CTA's shared memory
constexpr int kWideUnroll  = 2;            // steps of records in flight per warp

// Values per K tile: whole steps, Cols columns of them within kWideTileMax, spread evenly.
std::int32_t wide_tile(std::int32_t k, int cols) {
    const std::int32_t steps     = (k + kWideValues - 1) / kWideValues;
    const std::int32_t max_steps = std::max(1, kWideTileMax / (cols * kWideValues));
    const std::int32_t tiles     = (steps + max_steps - 1) / max_steps;
    return (steps + tiles - 1) / tiles * kWideValues;
}

std::size_t wide_activation_bytes(std::int32_t tile, int cols) {
    return static_cast<std::size_t>(cols) * tile +
           static_cast<std::size_t>(cols) * (tile / kGroup) * (sizeof(float) + sizeof(std::int32_t));
}

// Out of the four rows of a unit (a panel) and every column: one value per part.
struct WideProjectEpilogue {
    DestinationEpilogue destinations;
    __device__ void operator()(std::int32_t row, int c, const float (&v)[1]) const {
        destinations(row, c, v[0]);
    }
};

struct WideSwigluEpilogue {
    RowFactor gate_factor;
    RowFactor up_factor;
    std::int32_t up_first;
    std::int32_t rows;
    __nv_bfloat16* out;
    __device__ void operator()(std::int32_t i, int c, const float (&v)[2]) const {
        out[static_cast<std::size_t>(c) * rows + i] = __float2bfloat16_rn(
            silu(v[0] * gate_factor(i)) * (v[1] * up_factor(up_first + i)));
    }
};

// Part p's rows are rows [0, rows) of views[p] from row offsets[p]; the CTA owns panels [u0, u1)
// of every part.
template <int Cols, int Parts, class Epilogue>
__launch_bounds__(kWarps * 32) __global__
    void integer_wide_kernel(RowSplitView first, RowSplitView second, std::int32_t second_offset,
                             std::int32_t rows, std::int32_t k,
                             const std::int8_t* __restrict__ codes,
                             const float* __restrict__ scales,
                             const std::int32_t* __restrict__ sums, std::int32_t cols,
                             std::int32_t units_per_cta, std::int32_t tile, Epilogue epilogue) {
    extern __shared__ int4 shared_words[];
    const int tile_groups = tile / kGroup;
    int4* chunk_s         = shared_words;
    auto* scale_s         = reinterpret_cast<float*>(shared_words) + Cols * tile / 4;
    auto* sum_s           = reinterpret_cast<std::int32_t*>(scale_s + Cols * tile_groups);
    auto* acc_s           = reinterpret_cast<float*>(sum_s + Cols * tile_groups);
    const int x_groups    = k / kGroup;
    const std::int32_t units = rows / 4;
    const std::int32_t u0    = static_cast<std::int32_t>(blockIdx.x) * units_per_cta;
    const std::int32_t u1    = min(units, u0 + units_per_cta);
    const int threads        = kWarps * 32;
    for (int i = static_cast<int>(threadIdx.x); i < (u1 - u0) * Parts * 4 * Cols; i += threads) {
        acc_s[i] = 0.0F;
    }
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int half = lane & 1;
    const int q    = (lane >> 1) & 1;
    const int goff = lane >> 2;
    for (std::int32_t t0 = 0; t0 < k; t0 += tile) {
        const std::int32_t values = min(tile, k - t0);
        const int groups          = values / kGroup;
        __syncthreads();
        for (int i = static_cast<int>(threadIdx.x); i < Cols * values / 16; i += threads) {
            const int c = i / (values / 16);
            const int r = i % (values / 16);
            chunk_s[(c * 4 + (r & 3)) * tile_groups + (r >> 2)] =
                c < cols ? __ldg(reinterpret_cast<const int4*>(codes + static_cast<std::int64_t>(c) * k +
                                                               t0) + r)
                         : make_int4(0, 0, 0, 0);
        }
        for (int i = static_cast<int>(threadIdx.x); i < Cols * groups; i += threads) {
            const int c                  = i / groups;
            const int g                  = i % groups;
            const bool live              = c < cols;
            scale_s[c * tile_groups + g] = live ? __ldg(scales + c * x_groups + t0 / kGroup + g) : 0.0F;
            sum_s[c * tile_groups + g]   = live ? __ldg(sums + c * x_groups + t0 / kGroup + g) : 0;
        }
        __syncthreads();
        const int g0         = static_cast<int>(t0 / kT2Group);
        const int tile_t2    = values / kT2Group;
        const int steps      = (tile_t2 + kWideStep - 1) / kWideStep;
        for (std::int32_t slot = warp; slot < (u1 - u0) * Parts; slot += kWarps) {
            const int part          = Parts == 1 ? 0 : slot / (u1 - u0);
            const std::int32_t unit = u0 + slot % (u1 - u0);
            // Field by field: selecting either parameter struct whole puts it in local memory.
            RowSplitView view;
            view.base               = part == 0 ? first.base : second.base;
            view.scales             = part == 0 ? first.scales : second.scales;
            view.groups_per_row     = part == 0 ? first.groups_per_row : second.groups_per_row;
            view.panel_shift        = part == 0 ? first.panel_shift : second.panel_shift;
            const std::int32_t row0 = (part == 0 ? 0 : second_offset) + unit * 4 + q * kWideRows;
            float acc[kWideRows][Cols];
#pragma unroll
            for (int r = 0; r < kWideRows; ++r) {
#pragma unroll
                for (int c = 0; c < Cols; ++c) { acc[r][c] = 0.0F; }
            }
            // The lane's first record and scale of each row in the tile; a step advances them by
            // eight groups.
            const std::int64_t step_bytes = static_cast<std::int64_t>(kWideStep)
                                            << (5 + view.panel_shift);
            const std::uint8_t* record[kWideRows];
            const __half* weight_scale[kWideRows];
#pragma unroll
            for (int r = 0; r < kWideRows; ++r) {
                record[r] = view.record(view.base, kT2Group / 4, row0 + r, g0 + goff) + 16 * half;
                weight_scale[r] =
                    view.scales + static_cast<std::int64_t>(row0 + r) * view.groups_per_row + g0 +
                    goff;
            }
            for (int s0 = 0; s0 < steps; s0 += kWideUnroll) {
                uint4 w[kWideUnroll][kWideRows];
                float ws[kWideUnroll][kWideRows];
#pragma unroll
                for (int u = 0; u < kWideUnroll; ++u) {
                    const bool ok = kWideStep * (s0 + u) + goff < tile_t2;
#pragma unroll
                    for (int r = 0; r < kWideRows; ++r) {
                        w[u][r]  = ok ? __ldg(reinterpret_cast<const uint4*>(
                                           record[r] + (s0 + u) * step_bytes))
                                      : make_uint4(0u, 0u, 0u, 0u);
                        ws[u][r] =
                            ok ? __half2float(weight_scale[r][kWideStep * (s0 + u)]) : 0.0F;
                    }
                }
#pragma unroll
                for (int u = 0; u < kWideUnroll; ++u) {
                    const int g = kWideStep * (s0 + u) + goff;
                    if (g >= tile_t2) { break; }
                    const int xg = 2 * g + half;
                    int dot[kWideRows][Cols];
#pragma unroll
                    for (int r = 0; r < kWideRows; ++r) {
#pragma unroll
                        for (int c = 0; c < Cols; ++c) { dot[r][c] = 0; }
                    }
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        unsigned b[kWideRows][4];
#pragma unroll
                        for (int r = 0; r < kWideRows; ++r) {
                            const unsigned word = j == 0   ? w[u][r].x
                                                  : j == 1 ? w[u][r].y
                                                  : j == 2 ? w[u][r].z
                                                           : w[u][r].w;
#pragma unroll
                            for (int s = 0; s < 4; ++s) {
                                b[r][s] = ((word >> (2 * s)) & 0x03030303u) ^ 0x02020202u;
                            }
                        }
#pragma unroll
                        for (int c = 0; c < Cols; ++c) {
                            const int4 v = chunk_s[(c * 4 + j) * tile_groups + xg];
#pragma unroll
                            for (int r = 0; r < kWideRows; ++r) {
                                dot[r][c] = __dp4a(static_cast<int>(b[r][0]), v.x, dot[r][c]);
                                dot[r][c] = __dp4a(static_cast<int>(b[r][1]), v.y, dot[r][c]);
                                dot[r][c] = __dp4a(static_cast<int>(b[r][2]), v.z, dot[r][c]);
                                dot[r][c] = __dp4a(static_cast<int>(b[r][3]), v.w, dot[r][c]);
                            }
                        }
                    }
#pragma unroll
                    for (int c = 0; c < Cols; ++c) {
                        const float x_scale = scale_s[c * tile_groups + xg];
                        const int x_sum     = sum_s[c * tile_groups + xg];
#pragma unroll
                        for (int r = 0; r < kWideRows; ++r) {
                            acc[r][c] = fmaf(static_cast<float>(dot[r][c] - x_sum),
                                             ws[u][r] * x_scale, acc[r][c]);
                        }
                    }
                }
            }
            // The lanes of a row: both halves and the eight group offsets.
#pragma unroll
            for (int r = 0; r < kWideRows; ++r) {
#pragma unroll
                for (int c = 0; c < Cols; ++c) {
                    float v = acc[r][c];
                    v += __shfl_xor_sync(0xffffffffu, v, 1);
                    v += __shfl_xor_sync(0xffffffffu, v, 4);
                    v += __shfl_xor_sync(0xffffffffu, v, 8);
                    v += __shfl_xor_sync(0xffffffffu, v, 16);
                    acc[r][c] = v;
                }
            }
            if (lane < 4 && half == 0) {
                float* a = acc_s + slot * 4 * Cols;
#pragma unroll
                for (int r = 0; r < kWideRows; ++r) {
#pragma unroll
                    for (int c = 0; c < Cols; ++c) { a[(q * kWideRows + r) * Cols + c] += acc[r][c]; }
                }
            }
        }
    }
    __syncthreads();
    for (int i = static_cast<int>(threadIdx.x); i < (u1 - u0) * 4 * Cols; i += threads) {
        const int c = i % Cols;
        if (c >= cols) { continue; }
        const int local = i / Cols; // (unit - u0) * 4 + row in the panel
        float v[Parts];
#pragma unroll
        for (int p = 0; p < Parts; ++p) { v[p] = acc_s[(p * (u1 - u0) * 4 + local) * Cols + c]; }
        epilogue(u0 * 4 + local, c, v);
    }
}

RowSplitView t2_view(const Weight& w) {
    RowSplitView view;
    view.base           = static_cast<const std::uint8_t*>(w.qdata);
    view.scales         = static_cast<const __half*>(w.scales);
    view.groups_per_row = w.padded_shape[1] / kT2Group;
    view.panel_shift    = row_split_panel_shift(w.layout);
    return view;
}

std::size_t shared_bytes(std::int32_t k, int cols) {
    return static_cast<std::size_t>(cols) * k +
           static_cast<std::size_t>(cols) * (k / kGroup) * (sizeof(float) + sizeof(std::int32_t));
}

// f.template operator()<Cols>() for the call's width, Cols = tokens.
template <class F>
void with_cols(std::int32_t tokens, F&& f) {
    switch (tokens) {
    case 1: f.template operator()<1>(); break;
    case 2: f.template operator()<2>(); break;
    case 3: f.template operator()<3>(); break;
    case 4: f.template operator()<4>(); break;
    case 5: f.template operator()<5>(); break;
    case 6: f.template operator()<6>(); break;
    case 7: f.template operator()<7>(); break;
    default: f.template operator()<8>(); break;
    }
}

constexpr std::int32_t kPanelMaxTokens = 1;

// The kernels' shapes: whole T2 groups; at one column, staged activations within the static
// 48 KiB a CTA takes without opting in (17 KiB at the widest registered T2 input, 17,408); the
// wide kernel tiles K.
bool integer_shape(QType format, std::int32_t k, std::int32_t tokens) {
    return enabled() && format == QType::T2_G128_FP16 && tokens >= 1 &&
           tokens <= kIntegerMaxTokens && k % kT2Group == 0 &&
           (tokens > kPanelMaxTokens || shared_bytes(k, tokens) <= 48u * 1024u);
}

// The wide kernel over `rows` output rows of `Parts` parents: CTAs of whole panels, as many as the
// SMs hold at once (registers and shared memory, 80 to 156 registers a thread over the widths)
// while their partial sums fit beside the activation tile.
template <int Cols, int Parts, class Epilogue>
void launch_wide(const Activation& x, const RowSplitView& first, const RowSplitView& second,
                 std::int32_t second_offset, std::int32_t rows, const Epilogue& epilogue,
                 cudaStream_t stream) {
    auto kernel                  = integer_wide_kernel<Cols, Parts, Epilogue>;
    const std::int32_t units     = rows / 4;
    const std::int32_t tile      = wide_tile(x.k, Cols);
    const std::size_t staged     = wide_activation_bytes(tile, Cols);
    const std::size_t per_unit   = static_cast<std::size_t>(Parts) * 4 * Cols * sizeof(float);
    const std::int32_t sms       = current_device_multiprocessors(80);
    const auto most              = static_cast<std::int32_t>((kWideShared - staged) / per_unit);
    const auto units_for         = [&](std::int32_t ctas) {
        return std::max(1, std::min((units + ctas - 1) / ctas, most));
    };
    const auto shared_for        = [&](std::int32_t units_per_cta) {
        return staged + static_cast<std::size_t>(units_per_cta) * per_unit;
    };
    // Residency at the largest partial-sum block (one CTA an SM), then at the block it allows.
    int resident = 1;
    for (int pass = 0; pass < 2; ++pass) {
        int blocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks, kernel, kWarps * 32, shared_for(units_for(sms * resident))));
        resident = std::max(resident, blocks);
    }
    const std::int32_t units_per_cta = units_for(sms * resident);
    const std::int32_t grid          = (units + units_per_cta - 1) / units_per_cta;
    const std::size_t shared         = shared_for(units_per_cta);
    kernel<<<grid, kWarps * 32, shared, stream>>>(
        first, second, second_offset, rows, x.k, x.codes, x.code_scales, x.code_sums, x.tokens,
        units_per_cta, tile, epilogue);
    CUDA_CHECK(cudaGetLastError());
}

void require(const Activation& x, const Weight& w) {
    if (x.codes == nullptr || w.k != x.k || !integer_shape(w.qtype, w.k, x.tokens) ||
        !dequantizable(w) || w.n % 4 != 0) {
        throw std::invalid_argument("sm70 integer route: unsupported weight or activations");
    }
}

} // namespace

bool integer_format(QType format, std::int32_t k, std::int32_t tokens, LinearPolicy policy) {
    return t2_a8_admits(policy) && integer_shape(format, k, tokens);
}

bool integer_takes(const Weight& w, std::int32_t tokens, LinearPolicy policy) {
    return integer_format(w.qtype, w.k, tokens, policy) && dequantizable(w) && w.n % 4 == 0;
}

std::size_t integer_bytes(std::int32_t k, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    const auto groups = static_cast<std::size_t>(k / kGroup) * tokens;
    (void)layout.alloc_bytes(static_cast<std::size_t>(k) * tokens);
    (void)layout.alloc_bytes(groups * sizeof(float));
    (void)layout.alloc_bytes(groups * sizeof(std::int32_t));
    return layout.peak_bytes(1);
}

void integer_prepare(const Tensor& x, WorkspaceArena& workspace, cudaStream_t stream,
                     Activation& x_out) {
    const std::int32_t k      = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    if (x.dtype != DType::BF16 || x.data == nullptr || !x.is_contiguous() || x.ne[2] != 1 ||
        x.ne[3] != 1 || tokens < 1 || tokens > kIntegerMaxTokens || k % kT2Group != 0) {
        throw std::invalid_argument("sm70 integer route: x must be a contiguous BF16 [K, T<=8]");
    }
#if defined(NINFER_TIER_VOLTA)
    const auto groups = static_cast<std::size_t>(k / kGroup) * tokens;
    auto* codes =
        static_cast<std::int8_t*>(workspace.alloc_bytes(static_cast<std::size_t>(k) * tokens).data);
    auto* scales = static_cast<float*>(workspace.alloc_bytes(groups * sizeof(float)).data);
    auto* sums =
        static_cast<std::int32_t*>(workspace.alloc_bytes(groups * sizeof(std::int32_t)).data);
    constexpr int kQuantizeWarps = 8;
    const dim3 grid(static_cast<unsigned>((k / kGroup + kQuantizeWarps - 1) / kQuantizeWarps),
                    static_cast<unsigned>(tokens));
    integer_quantize_kernel<<<grid, kQuantizeWarps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), k, codes, scales, sums);
    CUDA_CHECK(cudaGetLastError());
    x_out.k           = k;
    x_out.tokens      = tokens;
    x_out.codes       = codes;
    x_out.code_scales = scales;
    x_out.code_sums   = sums;
#else
    (void)workspace;
    (void)stream;
    (void)x_out;
    throw std::logic_error("sm70 integer route: this build does not target sm_70");
#endif
}

void integer_project(const Activation& x, const Weight& w,
                     std::span<const Destination> destinations, cudaStream_t stream) {
    require(x, w);
    if (destinations.empty() || destinations.size() > kMaxDestinations) {
        throw std::invalid_argument("sm70 integer route: unsupported destinations");
    }
#if defined(NINFER_TIER_VOLTA)
    DestinationEpilogue epilogue;
    for (const Destination& d : destinations) { epilogue.set[epilogue.count++] = d; }
    epilogue.factor = row_factor(w);
    const dim3 grid(static_cast<unsigned>((w.n / 4 + kPanels - 1) / kPanels));
    with_cols(x.tokens, [&]<int Cols>() {
        if constexpr (Cols > kPanelMaxTokens) {
            const RowSplitView view = t2_view(w);
            launch_wide<Cols, 1>(x, view, view, 0, w.n, WideProjectEpilogue{epilogue}, stream);
        } else {
            const std::size_t shared = shared_bytes(x.k, Cols);
            integer_project_kernel<Cols><<<grid, kWarps * 32, shared, stream>>>(
                t2_view(w), w.n, x.k, x.codes, x.code_scales, x.code_sums, x.tokens, epilogue);
            CUDA_CHECK(cudaGetLastError());
        }
    });
#else
    (void)stream;
    throw std::logic_error("sm70 integer route: this build does not target sm_70");
#endif
}

void integer_swiglu(const Activation& x, const Weight& gate, const Weight* up, Tensor& out,
                    cudaStream_t stream) {
    const std::int32_t rows = out.ne[0];
    const bool joined       = up == nullptr;
    const Weight& up_weight = joined ? gate : *up;
    require(x, gate);
    require(x, up_weight);
    if (out.dtype != DType::BF16 || out.ne[1] != x.tokens || rows % 4 != 0 ||
        (joined ? gate.n != 2 * rows : (gate.n != rows || up->n != rows))) {
        throw std::invalid_argument("sm70 integer swiglu: unsupported weights or output");
    }
#if defined(NINFER_TIER_VOLTA)
    const dim3 grid(static_cast<unsigned>((rows / 4 + kPanels - 1) / kPanels));
    with_cols(x.tokens, [&]<int Cols>() {
        if constexpr (Cols > kPanelMaxTokens) {
            const WideSwigluEpilogue swiglu{row_factor(gate), row_factor(up_weight),
                                            joined ? rows : 0, rows,
                                            static_cast<__nv_bfloat16*>(out.data)};
            launch_wide<Cols, 2>(x, t2_view(gate), t2_view(up_weight), joined ? rows : 0, rows,
                                 swiglu, stream);
        } else {
            const std::size_t shared = shared_bytes(x.k, Cols);
            integer_swiglu_kernel<Cols><<<grid, kWarps * 32, shared, stream>>>(
                t2_view(gate), t2_view(up_weight), rows, joined ? rows : 0, x.k, x.codes,
                x.code_scales, x.code_sums, x.tokens, row_factor(gate), row_factor(up_weight),
                static_cast<__nv_bfloat16*>(out.data));
            CUDA_CHECK(cudaGetLastError());
        }
    });
#else
    (void)stream;
    throw std::logic_error("sm70 integer swiglu: this build does not target sm_70");
#endif
}

} // namespace ninfer::ops::detail::sm70
