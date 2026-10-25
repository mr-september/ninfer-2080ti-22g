#pragma once

#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail::gated_delta_net {
enum class FoldGeometryId { Layers48Heads48 };

inline FoldGeometryId select_fold_geometry(const GdnReplayRecordSpec& spec) {
    if (spec.layers == 48 && spec.qk_heads == 16 && spec.value_heads == 48 &&
        spec.conv_channels == 10240)
        return FoldGeometryId::Layers48Heads48;
    throw std::invalid_argument("gdn_replay_fold: unsupported record geometry");
}

struct alignas(16) GdnReplayFoldKernelRow {
    std::int32_t source_state_slot;
    std::int32_t destination_state_slot;
    std::int32_t commit_columns;
    std::int32_t reserved = 0;
};

struct alignas(16) GdnReplayFoldKernelRows {
    GdnReplayFoldKernelRow row[8];
};

void launch_recurrent_record(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                             const Tensor& beta, float scale, const Tensor& ssm_states,
                             const Tensor& valid_columns, const Tensor& initial_state_slots,
                             Tensor& key_record, Tensor& value_record, Tensor& gate_record,
                             Tensor& out, cudaStream_t stream);

void launch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                        const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                        cudaStream_t stream);

} // namespace ninfer::ops::detail::gated_delta_net
