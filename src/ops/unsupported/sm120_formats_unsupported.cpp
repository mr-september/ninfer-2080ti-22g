// Throwing definitions for the sm_120a-only routes on Turing builds.
//
// The FP8/NVFP4 weight kernels, the FP8/NVFP4/K8V4 KV-cache kernels and the TMA/block-scaled MMA
// launchers are excluded from the build on these targets (see src/ops/CMakeLists.txt). Shared
// wrappers still name them, so the declarations they use are defined here. The product artifacts
// for these targets contain only BF16 and grouped-integer tensors and use BF16/INT8 KV, so none of
// these is reachable from a supported route; reaching one reports a clear error.
#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_a4_tma_launch.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_a4_tma_launch.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma_launch.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma_launch.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma_launch.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/launch.h"

#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"
#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_topk/linear_topk_launch.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/plan.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void unsupported(const char* route) {
    throw std::runtime_error(std::string(route) +
                             " requires an sm_120a (Blackwell) build; this build targets "
                             "Turing and supports BF16 and grouped-integer artifacts "
                             "with BF16/INT8 KV");
}

} // namespace

// --- FP8 weights --------------------------------------------------------------------------------
const Fp8LinearShape kFp8N14336K5120{};
const Fp8LinearShape kFp8N16384K5120{};
const Fp8LinearShape kFp8N34816K5120{};
const Fp8LinearShape kFp8N5120K6144{};
const Fp8LinearShape kFp8N5120K17408{};

std::size_t fp8_attn_input_partial_capacity_bytes(std::int32_t) { unsupported("FP8 attn_input"); }
void fp8_attn_input_a8_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                              Fp8A8Workspace, cudaStream_t) {
    unsupported("FP8 attn_input");
}
std::size_t fp8_gdn_input_partial_capacity_bytes(std::int32_t) { unsupported("FP8 gdn_input"); }
void fp8_gdn_input_a8_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Fp8A8Workspace,
                             cudaStream_t) {
    unsupported("FP8 gdn_input");
}
std::size_t fp8_linear_add_partial_capacity_bytes(std::int32_t, std::int32_t) {
    unsupported("FP8 linear_add");
}
void fp8_linear_add_a8_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                              cudaStream_t) {
    unsupported("FP8 linear_add");
}
std::size_t fp8_linear_swiglu_partial_capacity_bytes(std::int32_t) {
    unsupported("FP8 linear_swiglu");
}
void fp8_linear_swiglu_a8_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                 cudaStream_t) {
    unsupported("FP8 linear_swiglu");
}

// --- NVFP4 weights ------------------------------------------------------------------------------
void launch_nvfp4_a4_quantize(const Tensor&, const Weight&, Nvfp4A4Workspace, Nvfp4ScaleLayout,
                              cudaStream_t) {
    unsupported("NVFP4 linear");
}
void launch_nvfp4_a4_tma_linear(Nvfp4GeometryId, const Nvfp4A4Operands&, __nv_bfloat16*,
                                cudaStream_t) {
    unsupported("NVFP4 linear");
}
void launch_nvfp4_a4_tma_attention(const Nvfp4A4Operands&, __nv_bfloat16*, __nv_bfloat16*,
                                   __nv_bfloat16*, __nv_bfloat16*, cudaStream_t) {
    unsupported("NVFP4 attn_input");
}
void launch_nvfp4_a4_tma_gdn(const Nvfp4A4Operands&, __nv_bfloat16*, __nv_bfloat16*,
                             cudaStream_t) {
    unsupported("NVFP4 gdn_input");
}
void launch_nvfp4_a4_tma_linear_add(const Nvfp4A4Operands&, __nv_bfloat16*, cudaStream_t) {
    unsupported("NVFP4 linear_add");
}
void launch_nvfp4_linear_swiglu_a4_tma(const Nvfp4A4Operands&, __nv_bfloat16*, cudaStream_t) {
    unsupported("NVFP4 linear_swiglu");
}

// --- NVFP4 / K8V4 KV caches ---------------------------------------------------------------------
void kv_cache_append_nvfp4_launch(const Tensor&, const Tensor&, const Tensor&, PagedKVLayerView,
                                  cudaStream_t) {
    unsupported("NVFP4 KV append");
}
void kv_cache_append_nvfp4_batch_launch(const Tensor&, const Tensor&, const Tensor&,
                                        const Tensor&, const Tensor&, PagedKVBatchLayerView,
                                        cudaStream_t) {
    unsupported("NVFP4 KV append");
}
void kv_cache_append_k8v4_launch(const Tensor&, const Tensor&, const Tensor&, PagedKVLayerView,
                                 cudaStream_t) {
    unsupported("K8V4 KV append");
}
void kv_cache_append_k8v4_batch_launch(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                                       const Tensor&, PagedKVBatchLayerView, cudaStream_t) {
    unsupported("K8V4 KV append");
}
void k8v4_kv_append_attention(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                              const Tensor&, const Tensor&, float, PagedKVBatchLayerView,
                              CausalAttentionExecutionEnvelope, WorkspaceArena&, Tensor&,
                              DeviceExecutionView) {
    unsupported("K8V4 KV attention");
}
void k8v4_kv_cached_attention(const Tensor&, const Tensor&, float, const PagedKVLayerView&,
                              CausalAttentionExecutionEnvelope, WorkspaceArena&, Tensor&,
                              DeviceExecutionView) {
    unsupported("K8V4 KV attention");
}
void nvfp4_kv_append_attention(const Tensor&, const Tensor&, const Tensor&, const Tensor&,
                               const Tensor&, const Tensor&, float, PagedKVBatchLayerView,
                               CausalAttentionExecutionEnvelope, WorkspaceArena&, Tensor&,
                               DeviceExecutionView) {
    unsupported("NVFP4 KV attention");
}
void nvfp4_kv_cached_attention(const Tensor&, const Tensor&, float, const PagedKVLayerView&,
                               CausalAttentionExecutionEnvelope, WorkspaceArena&, Tensor&,
                               DeviceExecutionView) {
    unsupported("NVFP4 KV attention");
}

// --- Dispatch and planning entry points of the same routes -------------------------------
void fp8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) { unsupported("fp8_attn_input_dispatch"); }
std::size_t fp8_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens) { unsupported("fp8_attn_input_workspace_capacity_bytes"); }
void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) { unsupported("fp8_dispatch"); }
void fp8_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z, LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) { unsupported("fp8_gdn_input_dispatch"); }
std::size_t fp8_gdn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens) { unsupported("fp8_gdn_input_workspace_capacity_bytes"); }
void fp8_gdn_record_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight, const Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot, Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& z, LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) { unsupported("fp8_gdn_record_dispatch"); }
std::size_t fp8_gdn_record_workspace_capacity_bytes(LinearPolicy policy, std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) { unsupported("fp8_gdn_record_workspace_capacity_bytes"); }
void fp8_gdn_snapshot_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight, Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot, const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value, Tensor& z, LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) { unsupported("fp8_gdn_snapshot_dispatch"); }
std::size_t fp8_gdn_snapshot_workspace_capacity_bytes(LinearPolicy policy, std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) { unsupported("fp8_gdn_snapshot_workspace_capacity_bytes"); }
void fp8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions, const Tensor& valid, const Tensor& rows, float scale, PagedKVBatchLayerView cache, CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace, Tensor& out, DeviceExecutionView execution) { unsupported("fp8_kv_append_attention"); }
void fp8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale, const PagedKVLayerView& cache, CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace, Tensor& out, DeviceExecutionView execution) { unsupported("fp8_kv_cached_attention"); }
std::size_t fp8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width, CausalAttentionExecutionEnvelope envelope, int multiprocessor_count) { unsupported("fp8_kv_workspace_bytes"); }
void fp8_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual, LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) { unsupported("fp8_linear_add_dispatch"); }
std::size_t fp8_linear_add_workspace_capacity_bytes(std::int32_t output_rows, std::int32_t input_rows, LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens) { unsupported("fp8_linear_add_workspace_capacity_bytes"); }
void fp8_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) { unsupported("fp8_linear_swiglu_dispatch"); }
std::size_t fp8_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens) { unsupported("fp8_linear_swiglu_workspace_capacity_bytes"); }
std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t output_rows, std::int32_t input_rows, LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens) { unsupported("fp8_linear_workspace_capacity_bytes"); }
std::size_t k8v4_kv_workspace_bytes(int heads, int batch, int min_width, int max_width, CausalAttentionExecutionEnvelope envelope, int multiprocessor_count) { unsupported("k8v4_kv_workspace_bytes"); }
void linear_topk_fp8_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows, const LinearTopKWorkspace& workspace, cudaStream_t stream) { unsupported("linear_topk_fp8_launch"); }
void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) { unsupported("nvfp4_attn_input_dispatch"); }
std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens) { unsupported("nvfp4_attn_input_workspace_capacity_bytes"); }
void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) { unsupported("nvfp4_dispatch"); }
Nvfp4GdnConvPlan nvfp4_gdn_conv_resolve_plan(LinearPolicy policy, std::int32_t tokens, std::int32_t batch_size) { unsupported("nvfp4_gdn_conv_resolve_plan"); }
void nvfp4_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z, LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) { unsupported("nvfp4_gdn_input_dispatch"); }
std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens) { unsupported("nvfp4_gdn_input_workspace_capacity_bytes"); }
void nvfp4_gdn_record_post_launch(const Tensor& conv_record, const Tensor& conv_weight, const Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot, Tensor& query, Tensor& key, Tensor& value, cudaStream_t stream) { unsupported("nvfp4_gdn_record_post_launch"); }
void nvfp4_gdn_record_small_t_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight, const Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot, Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) { unsupported("nvfp4_gdn_record_small_t_launch"); }
void nvfp4_gdn_snapshot_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight, Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot, const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value, Tensor& z, LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) { unsupported("nvfp4_gdn_snapshot_dispatch"); }
std::size_t nvfp4_gdn_snapshot_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens) { unsupported("nvfp4_gdn_snapshot_workspace_capacity_bytes"); }
std::size_t nvfp4_kv_workspace_bytes(int heads, int batch, int min_width, int max_width, CausalAttentionExecutionEnvelope envelope, int multiprocessor_count) { unsupported("nvfp4_kv_workspace_bytes"); }
void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual, LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) { unsupported("nvfp4_linear_add_dispatch"); }
std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows, std::int32_t input_rows, LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens) { unsupported("nvfp4_linear_add_workspace_capacity_bytes"); }
void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) { unsupported("nvfp4_linear_swiglu_dispatch"); }
std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens) { unsupported("nvfp4_linear_swiglu_workspace_capacity_bytes"); }
std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t output_rows, std::int32_t input_rows, LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens) { unsupported("nvfp4_linear_workspace_capacity_bytes"); }
Fp8WeightGeometry validate_fp8_weight(const Weight& weight, const char* operation) { unsupported("validate_fp8_weight"); }
Nvfp4WeightGeometry validate_nvfp4_weight(const Weight& weight, const char* operation) { unsupported("validate_nvfp4_weight"); }

} // namespace ninfer::ops::detail
