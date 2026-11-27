#pragma once

// Inkling 2's routed experts of a decode or verification call computed on CPU threads from the
// stored GGUF blocks of host or disk experts: ExpertDelivery's CPU share of a call's misses. For
// each pair, the expert's [gate; up] rows times its gate_up scale, SiTU (situ.h) into the BF16
// middle quantized as q8_1, its down rows, and each row weighted by the pair's route (which carries
// the expert's down scale) into its token's 2^-32 fixed-point sums, as the GPU's vector kernels and
// kernels::situ_rows compute them (gguf_expert_cpu, fixed_point.cuh). SiTU's middles stay below
// gate_cap * up_cap, so their F16 sums never overflow and need no guard. The products are the
// GPU's bit for bit and integer sums do not depend on order, so no split of a call's misses
// changes a bit of the layer's output.

#include "core/expert_delivery.h"
#include "core/qtype.h"
#include "core/weight.h"
#include "ops/moe_experts/gguf_expert_cpu.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::models::inkling2 {

// One expert layer's stored formats: [gate; up] rows of `hidden` values, then down rows of
// `width` values.
struct CpuExpertLayer {
    QType gate_up = QType::GGUF_Q4_0, down = QType::GGUF_Q4_0;
    std::int64_t gate_up_row_bytes = 0, down_row_bytes = 0;
};

class CpuExperts final : public ExpertCpuProducts {
public:
    // A call's tokens: the vector kernels' columns.
    static constexpr int kMaxTokens = 8;

    // Each layer's gate_up scales by expert (empty: 1), and SiTU's caps.
    CpuExperts(std::vector<CpuExpertLayer> layers, std::vector<std::vector<float>> gate_up_scales,
               int hidden, int width, int top_k, float gate_cap, float up_cap);

    // Whether the CPU computes this layer's products.
    [[nodiscard]] static bool computes(const CpuExpertLayer& layer, int hidden, int width) noexcept;
    // The inputs a call of `tokens` passes to ExpertDelivery::begin, in order: its quantized vector
    // activation (gguf vector_activation_bytes(hidden, tokens)) and its pairs' FP32 route weights.
    [[nodiscard]] std::size_t activation_bytes(int tokens) const;
    [[nodiscard]] std::size_t weight_bytes(int tokens) const;
    // ExpertCpuOptions' sizes for the largest call.
    [[nodiscard]] std::size_t input_bytes() const;
    // The products: int64 2^-32 fixed-point sums [tokens][hidden] of a call's CPU pairs.
    [[nodiscard]] std::size_t product_bytes() const;

    [[nodiscard]] bool supports(std::size_t layer) const override;
    Done compute(const Work& work) override;

private:
    struct Job {
        std::int32_t expert = -1;
        std::vector<int> pairs, tokens; // per column: the pair and its token
        const std::uint8_t* gate_up = nullptr;
        const std::uint8_t* down    = nullptr;
        ops::GgufCpuActivation input, middle;
        std::vector<float> gate, up, rows;
        std::vector<std::uint16_t> middle_bf16;
    };

    std::vector<CpuExpertLayer> layers_;
    std::vector<std::vector<float>> gate_up_scales_;
    int hidden_ = 0, width_ = 0, top_k_ = 0;
    float gate_cap_ = 0, up_cap_ = 0;
    ops::GgufCpuBackend backend_;
    std::vector<Job> jobs_;
};

} // namespace ninfer::models::inkling2
