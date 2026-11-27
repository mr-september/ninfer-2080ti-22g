#include "ops/linear/sm70/sm70_dequantize.h"

#include "core/device_launch.h"
#include "core/qtype.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <stdexcept>

namespace ninfer::ops::detail::sm70 {
namespace {

constexpr int kThreads = 128;

__device__ __forceinline__ __half to_half(float value) {
    return __float2half_rn(fminf(fmaxf(value, -65504.0f), 65504.0f));
}

// The row-split integer formats: code width, group size, base and high bytes per group, and the
// signed code of lane i from its base bytes and high bytes.
enum class RowSplitCode { Q2, Q4, Q5, Q6, Q8, T2 };

template <RowSplitCode Code>
struct RowSplitTraits;

template <>
struct RowSplitTraits<RowSplitCode::Q2> {
    static constexpr int kGroup = 64, kBase = 16, kHigh = 0;

    __device__ static int code(const std::uint8_t* base, const std::uint8_t*, int i) {
        return int((base[i >> 2] >> ((i & 3) * 2)) & 3u) - 1;
    }
};

template <>
struct RowSplitTraits<RowSplitCode::Q4> {
    static constexpr int kGroup = 64, kBase = 32, kHigh = 0;

    __device__ static int code(const std::uint8_t* base, const std::uint8_t*, int i) {
        const int u = (base[i >> 1] >> ((i & 1) * 4)) & 0xf;
        return (u ^ 8) - 8;
    }
};

template <>
struct RowSplitTraits<RowSplitCode::Q5> {
    static constexpr int kGroup = 64, kBase = 32, kHigh = 8;

    __device__ static int code(const std::uint8_t* base, const std::uint8_t* high, int i) {
        const int u =
            ((base[i >> 1] >> ((i & 1) * 4)) & 0xf) | (((high[i >> 3] >> (i & 7)) & 1) << 4);
        return (u ^ 16) - 16;
    }
};

template <>
struct RowSplitTraits<RowSplitCode::Q6> {
    static constexpr int kGroup = 64, kBase = 32, kHigh = 16;

    __device__ static int code(const std::uint8_t* base, const std::uint8_t* high, int i) {
        const int u =
            ((base[i >> 1] >> ((i & 1) * 4)) & 0xf) | (((high[i >> 2] >> ((i & 3) * 2)) & 3) << 4);
        return (u ^ 32) - 32;
    }
};

template <>
struct RowSplitTraits<RowSplitCode::Q8> {
    static constexpr int kGroup = 32, kBase = 32, kHigh = 0;

    __device__ static int code(const std::uint8_t* base, const std::uint8_t*, int i) {
        return static_cast<std::int8_t>(base[i]);
    }
};

template <>
struct RowSplitTraits<RowSplitCode::T2> {
    static constexpr int kGroup = 128, kBase = 32, kHigh = 0;

    __device__ static int code(const std::uint8_t* base, const std::uint8_t*, int i) {
        const int u = (base[i >> 2] >> ((i & 3) * 2)) & 3;
        return (u ^ 2) - 2;
    }
};

// Eight consecutive codes of a row-split format from the little-endian words holding them: the
// base bits (8 codes of kBits) and the high bits (Q5, Q6), as RowSplitTraits reads them bytewise.
template <RowSplitCode Code>
struct Codes8;

template <>
struct Codes8<RowSplitCode::Q2> {
    static constexpr int kBaseBytes = 2, kHighBytes = 0;

    __device__ static int code(std::uint64_t base, unsigned, int i) {
        return int((base >> (2 * i)) & 3u) - 1;
    }
};

template <>
struct Codes8<RowSplitCode::T2> {
    static constexpr int kBaseBytes = 2, kHighBytes = 0;

    __device__ static int code(std::uint64_t base, unsigned, int i) {
        return (int((base >> (2 * i)) & 3u) ^ 2) - 2;
    }
};

template <>
struct Codes8<RowSplitCode::Q4> {
    static constexpr int kBaseBytes = 4, kHighBytes = 0;

    __device__ static int code(std::uint64_t base, unsigned, int i) {
        return (int((base >> (4 * i)) & 0xfu) ^ 8) - 8;
    }
};

template <>
struct Codes8<RowSplitCode::Q5> {
    static constexpr int kBaseBytes = 4, kHighBytes = 1;

    __device__ static int code(std::uint64_t base, unsigned high, int i) {
        const int u = int((base >> (4 * i)) & 0xfu) | int(((high >> i) & 1u) << 4);
        return (u ^ 16) - 16;
    }
};

template <>
struct Codes8<RowSplitCode::Q6> {
    static constexpr int kBaseBytes = 4, kHighBytes = 2;

    __device__ static int code(std::uint64_t base, unsigned high, int i) {
        const int u = int((base >> (4 * i)) & 0xfu) | int(((high >> (2 * i)) & 3u) << 4);
        return (u ^ 32) - 32;
    }
};

template <>
struct Codes8<RowSplitCode::Q8> {
    static constexpr int kBaseBytes = 8, kHighBytes = 0;

    __device__ static int code(std::uint64_t base, unsigned, int i) {
        return static_cast<std::int8_t>((base >> (8 * i)) & 0xffu);
    }
};

template <int Bytes>
__device__ __forceinline__ std::uint64_t load_bytes(const std::uint8_t* p) {
    if constexpr (Bytes == 8) {
        return *reinterpret_cast<const std::uint64_t*>(p);
    } else if constexpr (Bytes == 4) {
        return *reinterpret_cast<const std::uint32_t*>(p);
    } else if constexpr (Bytes == 2) {
        return *reinterpret_cast<const std::uint16_t*>(p);
    } else if constexpr (Bytes == 1) {
        return *p;
    } else {
        return 0;
    }
}

// One thread per eight consecutive values of a row: their codes times the group's binary16 scale,
// stored as 16 bytes, so a warp writes 512 contiguous bytes. Under the panel layout the records of
// (1 << panel_shift) consecutive rows are contiguous within a group; a shift of zero is the
// row-major address. K is a whole number of eight values (dequantizable widths are of 64).
template <RowSplitCode Code>
__global__ void
row_split_kernel(const std::uint8_t* __restrict__ base, const std::uint8_t* __restrict__ high,
                 const __half* __restrict__ scales, std::int32_t groups_per_row, int panel_shift,
                 std::int32_t row0, std::int32_t k, __half* __restrict__ out) {
    using T              = RowSplitTraits<Code>;
    using C              = Codes8<Code>;
    const std::int32_t v = 8 * static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (v >= k) { return; }
    const std::int32_t g         = v / T::kGroup;
    const std::int32_t j         = v - g * T::kGroup;
    const std::int64_t row       = row0 + static_cast<std::int64_t>(blockIdx.y);
    const std::int64_t mask      = (std::int64_t{1} << panel_shift) - 1;
    const std::int64_t panel_row = (row & ~mask) * groups_per_row;
    const std::uint8_t* b        = base + (panel_row * T::kBase) + (row & mask) * T::kBase +
                                   static_cast<std::int64_t>(g) * (T::kBase << panel_shift) +
                                   (j / 8) * C::kBaseBytes;
    const std::uint64_t codes    = load_bytes<C::kBaseBytes>(b);
    unsigned high_bits           = 0;
    if constexpr (C::kHighBytes > 0) {
        const std::uint8_t* h = high + (panel_row * T::kHigh) + (row & mask) * T::kHigh +
                                static_cast<std::int64_t>(g) * (T::kHigh << panel_shift) +
                                (j / 8) * C::kHighBytes;
        high_bits             = static_cast<unsigned>(load_bytes<C::kHighBytes>(h));
    }
    const float scale = __half2float(scales[row * groups_per_row + g]);
    __align__(16) __half values[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        values[i] = to_half(float(C::code(codes, high_bits, i)) * scale);
    }
    *reinterpret_cast<uint4*>(out + static_cast<std::int64_t>(blockIdx.y) * k + v) =
        *reinterpret_cast<const uint4*>(values);
}

__global__ void bf16_kernel(const __nv_bfloat16* __restrict__ w, std::int32_t row0, std::int32_t k,
                            std::int64_t count, __half* __restrict__ out) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    out[i] = to_half(__bfloat162float(w[static_cast<std::int64_t>(row0) * k + i]));
}

// The E4M3 codes alone (at most 448 in magnitude); the BF16 row scale is the row's RowFactor.
__global__ void fp8_row_kernel(const std::uint8_t* __restrict__ codes, std::int32_t row0,
                               std::int32_t k, __half* __restrict__ out) {
    const std::int32_t c = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= k) { return; }
    const std::int64_t row = row0 + static_cast<std::int64_t>(blockIdx.y);
    __nv_fp8_e4m3 code;
    code.__x                                           = codes[row * k + c];
    out[static_cast<std::int64_t>(blockIdx.y) * k + c] = to_half(float(code));
}

// NVFP4 (block_scale_k16_m128x4_v1): E2M1 codes row-major, E4M3FN group scales in 128-row by
// four-group tiles of 512 bytes. Each value is code times group scale (at most 6 * 448); the
// source's FP32 divisor is the row's RowFactor.
__global__ void nvfp4_kernel(const std::uint8_t* __restrict__ codes,
                             const std::uint8_t* __restrict__ scales, std::int32_t row0,
                             std::int32_t k, __half* __restrict__ out) {
    const std::int32_t pair = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (pair * 2 >= k) { return; }
    const std::int32_t row = row0 + static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t c   = pair * 2;
    const int group        = c / 16;
    const int row_inner    = row & 127;
    const std::int64_t scale_at =
        (static_cast<std::int64_t>(row >> 7) * (k / 64) + group / 4) * 512 + (row_inner & 31) * 16 +
        (row_inner >> 5) * 4 + (group & 3);
    __nv_fp8_e4m3 scale_word;
    scale_word.__x = scales[scale_at];
    const float s  = float(scale_word);
    __nv_fp4x2_e2m1 value;
    value.__x            = codes[static_cast<std::int64_t>(row) * (k / 2) + pair];
    const float2 decoded = static_cast<float2>(value);
    __half* dst          = out + static_cast<std::int64_t>(blockIdx.y) * k + c;
    dst[0]               = to_half(decoded.x * s);
    dst[1]               = to_half(decoded.y * s);
}

template <RowSplitCode Code>
void launch_row_split(const Weight& w, std::int32_t row0, std::int32_t rows, __half* out,
                      cudaStream_t stream) {
    using T                           = RowSplitTraits<Code>;
    const std::int32_t groups_per_row = w.padded_shape[1] / T::kGroup;
    const std::int32_t eights         = w.k / 8;
    const dim3 grid(static_cast<unsigned>((eights + kThreads - 1) / kThreads),
                    static_cast<unsigned>(rows));
    row_split_kernel<Code><<<grid, kThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.qhigh),
        static_cast<const __half*>(w.scales), groups_per_row, row_split_panel_shift(w.layout), row0,
        w.k, out);
    CUDA_CHECK(cudaGetLastError());
}

bool row_split(const Weight& w) {
    return (w.layout == QuantLayout::RowSplit || w.layout == QuantLayout::RowSplitPanel) &&
           w.qdata != nullptr && w.scales != nullptr && w.scale_dtype == DType::FP16;
}

} // namespace

RowFactor row_factor(const Weight& w) {
    RowFactor factor;
    if (w.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        factor.kind = RowFactor::Kind::Bf16Row;
        factor.data = w.scales;
    } else if (w.qtype == QType::NVFP4 && w.weight_divisors != nullptr) {
        factor.kind           = RowFactor::Kind::Divisors;
        factor.data           = w.weight_divisors;
        factor.rows_per_entry = w.weight_divisor_rows > 0 ? w.weight_divisor_rows : w.n;
    } else if (w.qtype == QType::NVFP4) {
        factor.kind     = RowFactor::Kind::Constant;
        factor.constant = 1.0F / w.weight_scale_divisor;
    }
    return factor;
}

bool dequantizable_format(QType format) {
    switch (format) {
    case QType::Q2_G64_FP16:
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    case QType::Q8_G32_FP16:
    case QType::T2_G128_FP16:
    case QType::BF16:
    case QType::FP8_E4M3FN_ROW_BF16:
    case QType::NVFP4:
        return true;
    default:
        return false;
    }
}

bool dequantizable(const Weight& w) {
    if (w.n <= 0 || w.k <= 0) { return false; }
    switch (w.qtype) {
    case QType::Q2_G64_FP16:
    case QType::Q4_G64_FP16:
    case QType::Q8_G32_FP16:
    case QType::T2_G128_FP16:
        return row_split(w);
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
        return row_split(w) && w.qhigh != nullptr;
    case QType::BF16:
        return w.layout == QuantLayout::Contiguous && w.qdata != nullptr;
    case QType::FP8_E4M3FN_ROW_BF16:
        return w.layout == QuantLayout::RowScale && w.qdata != nullptr && w.scales != nullptr;
    case QType::NVFP4:
        return w.layout == QuantLayout::BlockScaleK16M128x4 && w.qdata != nullptr &&
               w.scales != nullptr && w.k % 64 == 0 &&
               (w.weight_divisors != nullptr || w.weight_scale_divisor > 0.0F);
    default:
        return false;
    }
}

void dequantize_rows(const Weight& w, std::int32_t row0, std::int32_t rows, __half* out,
                     cudaStream_t stream) {
    if (!dequantizable(w) || row0 < 0 || rows <= 0 || row0 + rows > w.n) {
        throw std::invalid_argument("sm70 dequantize: unsupported weight or row range");
    }
    switch (w.qtype) {
    case QType::Q2_G64_FP16:
        return launch_row_split<RowSplitCode::Q2>(w, row0, rows, out, stream);
    case QType::Q4_G64_FP16:
        return launch_row_split<RowSplitCode::Q4>(w, row0, rows, out, stream);
    case QType::Q5_G64_FP16:
        return launch_row_split<RowSplitCode::Q5>(w, row0, rows, out, stream);
    case QType::Q6_G64_FP16:
        return launch_row_split<RowSplitCode::Q6>(w, row0, rows, out, stream);
    case QType::Q8_G32_FP16:
        return launch_row_split<RowSplitCode::Q8>(w, row0, rows, out, stream);
    case QType::T2_G128_FP16:
        return launch_row_split<RowSplitCode::T2>(w, row0, rows, out, stream);
    case QType::BF16: {
        const std::int64_t count = static_cast<std::int64_t>(rows) * w.k;
        bf16_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(w.qdata), row0, w.k, count, out);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    case QType::FP8_E4M3FN_ROW_BF16: {
        const dim3 grid(static_cast<unsigned>((w.k + 255) / 256), static_cast<unsigned>(rows));
        fp8_row_kernel<<<grid, 256, 0, stream>>>(static_cast<const std::uint8_t*>(w.qdata), row0,
                                                 w.k, out);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    case QType::NVFP4: {
        const dim3 grid(static_cast<unsigned>((w.k / 2 + 255) / 256), static_cast<unsigned>(rows));
        nvfp4_kernel<<<grid, 256, 0, stream>>>(static_cast<const std::uint8_t*>(w.qdata),
                                               static_cast<const std::uint8_t*>(w.scales), row0,
                                               w.k, out);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    default:
        break;
    }
    throw std::invalid_argument("sm70 dequantize: unsupported weight format");
}

} // namespace ninfer::ops::detail::sm70
