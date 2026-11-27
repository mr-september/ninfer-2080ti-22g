#pragma once

#include "core/dtype.h"
#include "core/memory.h"

#include <cstdint>

namespace ninfer {

enum class QType : std::uint16_t {
    Q4_G64_FP16                  = 0,
    Q5_G64_FP16                  = 1,
    Q6_G64_FP16                  = 2,
    Q8_G32_FP16                  = 3,
    BF16                         = 4,
    FP32                         = 5,
    INT32                        = 6,
    NVFP4                        = 7,
    FP8_E4M3FN_ROW_BF16          = 8,
    NF4_G16_E4M3FN               = 9,
    FP8_E4M3FN_BLOCK128X128_BF16 = 10,

    // The donor formats grafted for the Inkling-2 donor tree (graft/include/core/qtype.h
    // forwards here). The donor numbers T2 and the ggml block types 9-33; this port owns 9
    // and 10, so the donor-only formats are appended past them. Nothing stores these numbers
    // on disk: the port's binder names formats from artifact metadata, and this port binds
    // no ggml blocks.
    T2_G128_FP16 = 34,
    GGUF_Q8_0    = 35,
    GGUF_Q2_K    = 36,
    GGUF_Q3_K    = 37,
    GGUF_Q4_K    = 38,
    GGUF_Q5_K    = 39,
    GGUF_Q6_K    = 40,
    GGUF_IQ2_XXS = 41,
    GGUF_IQ2_XS  = 42,
    GGUF_IQ2_S   = 43,
    GGUF_IQ3_XXS = 44,
    GGUF_IQ3_S   = 45,
    GGUF_IQ1_S   = 46,
    GGUF_IQ1_M   = 47,
    GGUF_IQ4_NL  = 48,
    GGUF_IQ4_XS  = 49,
    GGUF_Q4_0    = 50,
    GGUF_Q5_0    = 51,
    GGUF_Q2_0    = 52,
    Q2_G64_FP16  = 53,
    FP8_E4M3FN_ROW_FP16 = 54,
    GGUF_MXFP4   = 55,
    FP16         = 56,
    UINT16       = 57,
    GGUF_NVFP4   = 58,
};

enum class QuantLayout : std::uint16_t {
    RowSplit                   = 0,
    Contiguous                 = 1,
    RowScale                   = 3,
    RowPagedG16_4K             = 4,
    BlockScaleM128K128         = 6,
};

struct Weight {
    Device device;
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4_G64_FP16;
    std::uint32_t group_size       = 0;
    std::int32_t shape[4]          = {1, 1, 1, 1};
    std::int32_t padded_shape[4]   = {1, 1, 1, 1};
    std::uint32_t ndim             = 0;

    const void* qdata          = nullptr;
    const void* qhigh          = nullptr;
    const void* scales         = nullptr;
    std::int32_t n             = 0;
    std::int32_t k             = 0;
    std::int32_t group         = 0;
    QuantLayout layout         = QuantLayout::RowSplit;
    DType scale_dtype          = DType::FP32;
    std::int32_t scale_ne[4]   = {1, 1, 1, 1};
    std::int64_t scale_nb[4]   = {0, 0, 0, 0};
    float weight_scale_divisor = 0.0F;
    float input_scale_divisor  = 0.0F;
};

} // namespace ninfer
