#pragma once

// graft shim: ninfer owns the stored-format enum (QType) in core/weight.h, whose QType now
// carries the donor's formats past the port's own numbering (the ggml block types run 35-52,
// with GGUF_MXFP4 55 and GGUF_NVFP4 58 outside that run). This header keeps the donor-only
// helpers the grafted code reads.

#include "core/weight.h"

namespace ninfer {

[[nodiscard]] constexpr bool is_gguf(QType format) {
    return (format >= QType::GGUF_Q8_0 && format <= QType::GGUF_Q2_0) ||
           format == QType::GGUF_MXFP4 || format == QType::GGUF_NVFP4;
}

struct GgufBlockShape {
    int elements = 0;
    int bytes    = 0;
};

// ggml-common.h's block sizes; {0, 0} for every non-GGUF format.
[[nodiscard]] constexpr GgufBlockShape gguf_block_shape(QType format) {
    switch (format) {
    case QType::GGUF_Q8_0: return {32, 34};
    case QType::GGUF_Q2_K: return {256, 84};
    case QType::GGUF_Q3_K: return {256, 110};
    case QType::GGUF_Q4_K: return {256, 144};
    case QType::GGUF_Q5_K: return {256, 176};
    case QType::GGUF_Q6_K: return {256, 210};
    case QType::GGUF_IQ2_XXS: return {256, 66};
    case QType::GGUF_IQ2_XS: return {256, 74};
    case QType::GGUF_IQ2_S: return {256, 82};
    case QType::GGUF_IQ3_XXS: return {256, 98};
    case QType::GGUF_IQ3_S: return {256, 110};
    case QType::GGUF_IQ1_S: return {256, 50};
    case QType::GGUF_IQ1_M: return {256, 56};
    case QType::GGUF_IQ4_NL: return {32, 18};
    case QType::GGUF_IQ4_XS: return {256, 136};
    case QType::GGUF_Q4_0:
        return {32, 18};
    case QType::GGUF_Q5_0:
        return {32, 22};
    case QType::GGUF_Q2_0:
        return {64, 18};
    case QType::GGUF_MXFP4: return {32, 17};
    case QType::GGUF_NVFP4: return {64, 36};
    default: return {};
    }
}

} // namespace ninfer
