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
    // Donor names carried in by the inkling2 graft (sm70/gguf code reads these tags; the numbers
    // are in-process only — `.ninfer` layout ids keep the port values below). RowSplitPanel is
    // RowSplit's bytes permuted into kRowSplitPanelRows-row panels; GgufBlocks is rows of whole
    // ggml blocks as a GGUF stores them (the graft's gguf bridge tags its weights with this).
    BlockScaleK16M128x4        = 2,
    RowScale                   = 3,
    RowPagedG16_4K             = 4,
    GgufBlocks                 = 5,
    BlockScaleM128K128         = 6,
    RowSplitPanel              = 7,
    RowInterleaved             = 8,
};

// Rows per stored panel; kernels address panel and row-major forms with one shifted expression
// (donor core/weight.h — the graft's sm70 views call this on w.layout).
inline constexpr int kRowSplitPanelRows  = 4;
inline constexpr int kRowSplitPanelShift = 2;
static_assert((1 << kRowSplitPanelShift) == kRowSplitPanelRows,
              "the panel row count is addressed by shifting, so it must be a power of two");

[[nodiscard]] constexpr int row_split_panel_shift(QuantLayout layout) {
    return layout == QuantLayout::RowSplitPanel ? kRowSplitPanelShift : 0;
}

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

    // Donor fields (inkling2 graft): an NVFP4 plane assembled from several source matrices carries
    // one divisor per source after the scales; the divisor of row r is element r /
    // weight_divisor_rows (donor core/weight.h). input_columns is the INT32 [K] gather of a GGUF
    // matrix whose stored columns permute the input's; null otherwise.
    const void* weight_divisors      = nullptr;
    std::int32_t weight_divisor_rows = 0;
    const std::int32_t* input_columns = nullptr;

    // Device-side plane indirection for graph-captured expert dispatch (inkling2 MoE). When set,
    // it points to a device slot {qdata, qhigh, scales}: kernels admitted with a slot fetch the
    // live planes from it, so one captured launch serves every expert. The host pointers above
    // must remain the representative weight's (admission and validation read them); kernels must
    // not dereference them when the slot is set.
    const void* const* plane_slot = nullptr;
};

} // namespace ninfer
