#pragma once

// Host-resident GGUF routed experts computed on the CPU, from the same stored blocks the GPU reads:
// Q2_0, Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ1_S, IQ1_M, IQ2_XXS, IQ2_S, IQ3_XXS,
// IQ3_S, IQ4_XS and NVFP4. The Program
// splits a layer's missing experts between copies to the GPU and these products; it owns the
// threads, the activations and the weighted merge.
//
// Arithmetic: the GPU's decode products (ggml_bridge_vec.cuh), bit for bit. The activation is
// quantized as ggml's q8_1, the GPU's way: per 32 values d = amax / 127, codes round(v / d) (halves
// away from zero) and the scale used is d rounded to binary16. A row's 32-value slice s gives the
// exact integer dot of its stored codes with the activation's, and its product is
// (factor * scale) * dot, each step rounded, where the factor is the block's binary16 multiplier
// times the slice's integer scale and an i-quant's constant (exact); IQ2_S's and IQ1_M's are
// scale * fma(factor0, dot0, factor1 * dot1) over their two halves (IQ1_M's weights eight times its
// values, whose eighth its factors carry), Q3_K's, Q6_K's and NVFP4's
// scale * (factor0 * dot0 + factor1 * dot1) (NVFP4's factors its UE4M3 scales, halved),
// Q4_K's and Q5_K's (factor * scale) * dot less minimum * sum, and IQ1_S's
// factor0 * fma(scale, dot, factor1 * sum) over its codes 0..2 (factor1 its delta less one), the
// sum being the slice's activation values added as a warp adds them and rounded to binary16
// (q8_1's s). The products of slices s, s + 32,
// ... are added in turn to the partial sum of lane s mod 32, and the 32 partial sums meet as in a
// warp's butterfly (lane l with l + 16, then + 8, + 4, + 2, + 1). The middle
// silu(gate . x) * (up . x) (or gelu_tanh) uses ops/common/silu.h and is rounded to BF16 before it
// is quantized for the down product, as the GPU stores it. These translation units build without floating-point
// contraction, so no step fuses unless written as a fused multiply-add.

#include "core/qtype.h"
#include "core/weight.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::ops {

// Automatic takes AVX2 (with FMA and F16C) where the CPU has it; both give the same bits.
enum class GgufCpuBackend { Automatic, Scalar, Avx2 };
[[nodiscard]] bool gguf_cpu_backend_available(GgufCpuBackend backend) noexcept;
[[nodiscard]] const char* gguf_cpu_backend_name(GgufCpuBackend backend) noexcept;
[[nodiscard]] GgufCpuBackend gguf_cpu_backend_selected(GgufCpuBackend requested) noexcept;
// Whether `format` has a CPU product here.
[[nodiscard]] bool gguf_cpu_supported(QType format) noexcept;

// One stored projection: `rows` rows of `k` values, each row `row_bytes` of GGUF blocks.
struct GgufCpuMatrix {
    QType format             = QType::GGUF_Q2_0;
    const std::uint8_t* data = nullptr;
    std::int64_t row_bytes   = 0;
    int rows                 = 0;
    int k                    = 0;
};

// Activations of up to `tokens` columns of `k` values (a multiple of 32), quantized as q8_1 and
// laid out for one block format.
class GgufCpuActivation {
public:
    GgufCpuActivation() = default;
    GgufCpuActivation(int token_capacity, int k);
    // Quantizes `tokens` BF16 columns of k values (column-major, `stride` apart) for `format`.
    void prepare_bf16(QType format, const std::uint16_t* values, std::size_t stride, int tokens);
    // The same in pieces, which threads may fill at once: reset() sets the format and the token
    // count, then quantize_bf16() fills the 32-value groups [group_begin, group_end) of one
    // token from its BF16 column.
    void reset(QType format, int tokens);
    void quantize_bf16(int token, const std::uint16_t* column, int group_begin, int group_end);
    // Token `token` (after reset()) as token `from_token` of `from`, an activation of the same
    // format and k.
    void copy_token(int token, const GgufCpuActivation& from, int from_token);
    // The GPU's quantized columns `columns` of a planar vector activation (gguf vector_activation:
    // int8 codes [column][k], then a binary16 (d, s) pair [column][k / 32] at `ds`) for `format`.
    void load_q8_1(QType format, const std::int8_t* codes, const std::uint16_t* ds,
                   std::span<const int> columns);
    [[nodiscard]] int tokens() const noexcept { return tokens_; }
    [[nodiscard]] int k() const noexcept { return k_; }
    [[nodiscard]] QType format() const noexcept { return format_; }
    [[nodiscard]] const std::int8_t* codes(int token) const noexcept {
        return codes_.data() + std::size_t(token) * k_;
    }
    // One scale per 32 values: q8_1's d, rounded to binary16.
    [[nodiscard]] const float* scales(int token) const noexcept {
        return scales_.data() + std::size_t(token) * groups_;
    }
    // Per 32 values, the sum of their codes (Q2_0's offset term).
    [[nodiscard]] const std::int32_t* sums(int token) const noexcept {
        return sums_.data() + std::size_t(token) * groups_;
    }
    // Per 32 values, q8_1's s: their sum as a warp adds them, rounded to binary16 (the K-quants'
    // minimum term).
    [[nodiscard]] const float* value_sums(int token) const noexcept {
        return value_sums_.data() + std::size_t(token) * groups_;
    }

    [[nodiscard]] int groups() const noexcept { return groups_; }

private:
    void quantize_group(int token, int group, const float* values);
    int capacity_ = 0, k_ = 0, groups_ = 0, tokens_ = 0;
    QType format_ = QType::GGUF_Q2_0;
    std::vector<std::int8_t> codes_;
    std::vector<float> scales_, value_sums_;
    std::vector<std::int32_t> sums_;
};

// y[t * y_stride + r] = row r . x[t] for rows [row_begin, row_end) and every prepared token.
void gguf_cpu_rows(const GgufCpuMatrix& w, const GgufCpuActivation& x, int row_begin, int row_end,
                   float* y, std::size_t y_stride, GgufCpuBackend backend);
// out[i] = BF16(silu(gate[i]) * up[i]) for i < count, as the GPU's up kernels store the middle.
void gguf_cpu_swiglu_bf16(const float* gate, const float* up, std::size_t count, std::uint16_t* out,
                          GgufCpuBackend backend);
// out[i] = BF16(gelu_tanh(gate[i]) * up[i]), Gemma 4's middle.
void gguf_cpu_geglu_bf16(const float* gate, const float* up, std::size_t count, std::uint16_t* out,
                         GgufCpuBackend backend);
// out[i] = BF16(silu(min(g, limit)) * min(max(u, -limit), limit)) with g = BF16(gate[i] * scale)
// and u = BF16(up[i] * scale): DeepSeek-V4.1's middle, its input's guard factor `scale` taken back.
void gguf_cpu_clamped_swiglu_bf16(const float* gate, const float* up, std::size_t count, float scale,
                                  float limit, std::uint16_t* out, GgufCpuBackend backend);

namespace gguf_cpu_detail {
using Rows = void (*)(const GgufCpuMatrix&, const GgufCpuActivation&, int, int, float*,
                      std::size_t);
void rows_scalar(const GgufCpuMatrix&, const GgufCpuActivation&, int, int, float*, std::size_t);
void swiglu_scalar(const float*, const float*, std::size_t, std::uint16_t*);
void geglu_scalar(const float*, const float*, std::size_t, std::uint16_t*);
void clamped_swiglu_scalar(const float*, const float*, std::size_t, float, float, std::uint16_t*);
#if defined(NINFER_CPU_EXPERT_X86)
void rows_avx2(const GgufCpuMatrix&, const GgufCpuActivation&, int, int, float*, std::size_t);
void swiglu_avx2(const float*, const float*, std::size_t, std::uint16_t*);
void geglu_avx2(const float*, const float*, std::size_t, std::uint16_t*);
void clamped_swiglu_avx2(const float*, const float*, std::size_t, float, float, std::uint16_t*);
#endif
// Q2_0 activations are stored in the lane order of the vector decoders: within each 64-value
// block, position L * 16 + i holds value 4 * i + L, so a lane's first 8 positions belong to the
// block's first 32-value slice and its last 8 to the second.
inline int q2_lane_position(int j) noexcept { return (j % 4) * 16 + j / 4; }
float half_to_float(std::uint16_t bits) noexcept;
// Half the value of an NVFP4 scale byte (UE4M3, bit 7 ignored, 0x7F zero), as ggml-quants.c
// decodes it.
float ue4m3_half(std::uint8_t x) noexcept;
// The binary16 value nearest `value` (ties to even), as a float.
float round_to_half(float value) noexcept;
// The BF16 value nearest `value` (ties to even), as its word.
std::uint16_t float_to_bf16(float value) noexcept;

// One 32-value slice of a stored row as the GPU's decoder forms it: signed weights and the slice
// factors, each exact. factor[1] is IQ2_S's, IQ1_M's, Q3_K's, Q6_K's and NVFP4's for the second
// 16 values (the others use factor[0] for all 32), Q4_K's and Q5_K's minimum, or IQ1_S's delta
// less one.
struct GgufCpuSlice {
    std::int8_t weights[32];
    float factor[2];
};
// Slice `slice` of the row at `row` (Q2_0 weights are its codes less one).
void decode_slice(QType format, const std::uint8_t* row, int slice, GgufCpuSlice& out) noexcept;
// The 32 lanes' partial sums of a row's slice products, as a warp's butterfly adds them.
float butterfly(float* lanes) noexcept;
} // namespace gguf_cpu_detail

} // namespace ninfer::ops
