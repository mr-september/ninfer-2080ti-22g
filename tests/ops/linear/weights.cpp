#include <cstdint>
#include "ops/linear/weights.h"
#include "ops/direct_bf16_weight.h"

namespace ninfer::test::linear_weights {

quantized_weight::PackedWeight make_q4_g64_fp16_weight(std::int32_t n, std::int32_t k,
                                                       std::uint32_t seed) {
    return quantized_weight::make_patterned_weight(QType::Q4_G64_FP16, n, k, seed,
                                                   {quantized_weight::RowSplitScalePattern::Small,
                                                    quantized_weight::RowSplitCodePattern::Hashed});
}

quantized_weight::PackedWeight make_q5_g64_fp16_weight(std::int32_t n, std::int32_t k,
                                                       std::uint32_t seed) {
    return quantized_weight::make_patterned_weight(QType::Q5_G64_FP16, n, k, seed,
                                                   {quantized_weight::RowSplitScalePattern::Small,
                                                    quantized_weight::RowSplitCodePattern::Hashed});
}

quantized_weight::PackedWeight make_q6_g64_fp16_weight(std::int32_t n, std::int32_t k,
                                                       std::uint32_t seed) {
    return quantized_weight::make_patterned_weight(QType::Q6_G64_FP16, n, k, seed,
                                                   {quantized_weight::RowSplitScalePattern::Small,
                                                    quantized_weight::RowSplitCodePattern::Hashed});
}

quantized_weight::PackedWeight make_q8_g32_fp16_weight(std::int32_t n, std::int32_t k,
                                                       std::uint32_t seed) {
    return quantized_weight::make_patterned_weight(QType::Q8_G32_FP16, n, k, seed,
                                                   {quantized_weight::RowSplitScalePattern::Small,
                                                    quantized_weight::RowSplitCodePattern::Hashed});
}

quantized_weight::PackedWeight make_fp8_weight(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    return quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16, n, k, seed);
}

quantized_weight::PackedWeight make_bf16_weight(std::int32_t n, std::int32_t k,
                                                std::uint32_t seed) {
    const auto host = direct_bf16_weight::make_patterned(n, k, seed);
    quantized_weight::PackedWeight result;
    result.payload.resize(host.bits.size() * sizeof(std::uint16_t));
    std::memcpy(result.payload.data(), host.bits.data(), result.payload.size());
    result.weight =
        host.device_weight({result.payload.data(), result.payload.size(), Device::cpu()});
    result.code_plane_bytes = result.payload.size();
    return result;
}


} // namespace ninfer::test::linear_weights
