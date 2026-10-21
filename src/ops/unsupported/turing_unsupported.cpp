// Throwing definitions for the routes a Turing build does not compile.
//
// The FP8 weight kernels, the FP8 KV-cache kernels, the block-scaled MoE,
// and the TMA/sparse-attention families need sm_120a instructions or serve models that are
// not supported on this target. Their sources are excluded in src/ops/CMakeLists.txt; the
// shared dispatchers still name them, so these definitions keep the link complete and report
// a clear error if one is ever reached. Generated from the declarations they implement.
#include <stdexcept>
#include <string>

#include "ninfer/ops/gated_residual.h"
#include "ninfer/ops/kimi_delta_attention.h"
#include "ninfer/ops/qsa.h"
#include "ops/linear/add/fp8/dispatch.h"
#include "ops/linear/projection/fp8/fp8_dispatch.h"
#include "ops/linear/swiglu/fp8/dispatch.h"
#include "ops/linear/topk/dispatch.h"
#include "ops/linear_attention/gated_delta_net/input_conv/fp8/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/fp8/launch.h"
#include "ops/moe/routed_swiglu/cuda/dispatch.h"
#include "ops/moe/topk_softmax_swiglu_moe/dispatch.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/plan.h"

namespace {
[[noreturn]] void unsupported(const char* what) {
    throw std::runtime_error(std::string(what) + ": not available on the Turing (sm_75) target");
}
} // namespace

namespace ninfer::ops {
std::size_t
qsa_indexer_workspace_capacity_bytes(std::int32_t min_tokens, std::int32_t max_tokens,
                                     std::int32_t batch_size, QsaIndexerExecutionEnvelope envelope) { unsupported("qsa_indexer_workspace_capacity_bytes"); }
std::size_t
qsa_sparse_gqa_workspace_capacity_bytes(KvCacheStorage storage, std::int32_t min_tokens,
                                        std::int32_t max_tokens, std::int32_t batch_size,
                                        QsaSparseGqaExecutionEnvelope envelope,
                                        std::int32_t multiprocessor_count) { unsupported("qsa_sparse_gqa_workspace_capacity_bytes"); }
std::size_t hc_mix_workspace_capacity_bytes(const HcFinalMixProfile& profile,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens) { unsupported("hc_mix_workspace_capacity_bytes"); }
std::size_t hc_mix_workspace_capacity_bytes(const HcFinalMixWeights& weights,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens) { unsupported("hc_mix_workspace_capacity_bytes"); }
std::size_t hc_mix_workspace_capacity_bytes(const HcMixProfile& profile,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens) { unsupported("hc_mix_workspace_capacity_bytes"); }
std::size_t hc_mix_workspace_capacity_bytes(const HcMixWeights& weights,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens) { unsupported("hc_mix_workspace_capacity_bytes"); }
std::size_t kimi_delta_attention_workspace_capacity_bytes(std::int32_t qk_heads,
                                                                        std::int32_t value_heads,
                                                                        std::int32_t min_tokens,
                                                                        std::int32_t max_tokens) { unsupported("kimi_delta_attention_workspace_capacity_bytes"); }
void hc_mix(const Tensor& normalized, const HcFinalMixWeights& weights, MemoryArena& workspace,
            Tensor& sample_hidden, CudaExecutionView execution) { unsupported("hc_mix"); }
void hc_mix(const Tensor& normalized, const HcMixWeights& weights, MemoryArena& workspace,
            Tensor& block_input, Tensor& injection, CudaExecutionView execution) { unsupported("hc_mix"); }
void kimi_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, const Tensor& A_log, const Tensor& dt_bias,
                          float lower_bound, float scale, MemoryArena& workspace,
                          const Tensor& state_in, Tensor& state_out, Tensor& out,
                          CudaExecutionView execution) { unsupported("kimi_delta_attention"); }
void kimi_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, const Tensor& A_log, const Tensor& dt_bias,
                          float lower_bound, float scale, MemoryArena& workspace, Tensor& state,
                          Tensor& out, CudaExecutionView execution) { unsupported("kimi_delta_attention"); }
void kimi_delta_attention_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& g, const Tensor& beta, const Tensor& A_log,
                                       const Tensor& dt_bias, float lower_bound, float scale,
                                       Tensor& states, const Tensor& slots, Tensor& out,
                                       CudaExecutionView execution) { unsupported("kimi_delta_attention_batch_update"); }
void qsa_indexer(const Tensor& index_query, const QsaIndexCacheView& cache,
                 const QsaBatchView& batch, QsaIndexerExecutionEnvelope envelope,
                 MemoryArena& workspace, QsaBlockSelection& out, CudaExecutionView execution) { unsupported("qsa_indexer"); }
void qsa_prepare(const QsaProjectionView& projected, const QsaPrepareParameters& parameters,
                 const QsaBatchView& batch, const Tensor& rope_positions,
                 const QsaTailStateView& before, QsaCacheView cache, QsaRawRecordWriteView records,
                 Tensor& query, Tensor& index_query, CudaExecutionView execution) { unsupported("qsa_prepare"); }
void qsa_sparse_gqa(const Tensor& query, const Tensor& output_gate,
                    const PagedKVBatchLayerView& cache, const QsaBatchView& batch,
                    const QsaBlockSelection& selected, QsaSparseGqaExecutionEnvelope envelope,
                    MemoryArena& workspace, Tensor& out, CudaExecutionView execution) { unsupported("qsa_sparse_gqa"); }
void qsa_tail_extract(const QsaTailStateView& before, const QsaRawRecordReadView& records,
                      const Tensor& accepted_columns, QsaTailStateView after,
                      CudaExecutionView execution) { unsupported("qsa_tail_extract"); }
} // namespace ninfer::ops

namespace ninfer::ops::detail {
LinearTopKPlan select_topk_fp8_n248320_k5120_mma(int columns) { unsupported("select_topk_fp8_n248320_k5120_mma"); }
LinearTopKProducer select_topk_fp8_n248320_k5120_sliced(int columns) { unsupported("select_topk_fp8_n248320_k5120_sliced"); }
std::size_t fp8_gdn_input_partial_capacity_bytes(std::int32_t max_tokens) { unsupported("fp8_gdn_input_partial_capacity_bytes"); }
std::size_t fp8_kv_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                   CausalAttentionExecutionEnvelope envelope,
                                   int multiprocessor_count) { unsupported("fp8_kv_workspace_bytes"); }
std::size_t fp8_linear_add_workspace_capacity_bytes(std::int32_t rows, std::int32_t k,
                                                    LinearPolicy policy, std::int32_t first,
                                                    std::int32_t last) { unsupported("fp8_linear_add_workspace_capacity_bytes"); }
std::size_t fp8_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                                     std::int32_t min_tokens,
                                                                     std::int32_t max_tokens) { unsupported("fp8_linear_swiglu_workspace_capacity_bytes"); }
std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              LinearPolicy policy,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens) { unsupported("fp8_linear_workspace_capacity_bytes"); }
void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  MemoryArena* workspace, cudaStream_t stream) { unsupported("fp8_dispatch"); }
void fp8_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& positions, const Tensor& valid, const Tensor& rows,
                             float scale, PagedKVBatchLayerView cache,
                             CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                             Tensor& out, CudaExecutionView execution) { unsupported("fp8_kv_append_attention"); }
void fp8_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                             const PagedKVLayerView& cache,
                             CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                             Tensor& out, CudaExecutionView execution) { unsupported("fp8_kv_cached_attention"); }
void fp8_linear_add_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy, MemoryArena&,
                             cudaStream_t) { unsupported("fp8_linear_add_dispatch"); }
void fp8_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                LinearPolicy policy, MemoryArena& workspace, cudaStream_t stream) { unsupported("fp8_linear_swiglu_dispatch"); }
void launch_fp8_gdn_input_a8(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             Fp8A8Workspace workspace, cudaStream_t stream) { unsupported("launch_fp8_gdn_input_a8"); }
void launch_fp8_gdn_input_decode(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream) { unsupported("launch_fp8_gdn_input_decode"); }
void launch_fp8_gdn_input_matrix(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream) { unsupported("launch_fp8_gdn_input_matrix"); }
void launch_fp8_gdn_record_fused(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 const Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                                 Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) { unsupported("launch_fp8_gdn_record_fused"); }
void launch_fp8_gdn_snapshot_fused(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                   Tensor& conv_states, const Tensor& valid_columns,
                                   const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                   Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                   cudaStream_t stream) { unsupported("launch_fp8_gdn_snapshot_fused"); }
} // namespace ninfer::ops::detail

namespace ninfer::ops::detail::moe_complete {
const ShapePolicy& fp8_h2560_i640_e512_k10() { unsupported("fp8_h2560_i640_e512_k10"); }
} // namespace ninfer::ops::detail::moe_complete

namespace ninfer::ops::detail::moe_cuda::block_scaled::fp8 {
void launch_experts(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                    const Plan& plan, const Storage& storage, CudaExecutionView execution) { unsupported("launch_experts"); }
void launch_input(const Tensor& x, const CudaSwiGLUExperts& experts, const Storage& storage,
                  CudaExecutionView execution, bool clear_counts) { unsupported("launch_input"); }
void launch_vector(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, bool shared, MoeEpilogue mode, Tensor& destination,
                   const Storage& storage, CudaExecutionView execution) { unsupported("launch_vector"); }
} // namespace ninfer::ops::detail::moe_cuda::block_scaled::fp8

namespace ninfer::ops::detail::moe_routed {
const ShapePolicy& fp8_h2560_i640_e512_k10() { unsupported("fp8_h2560_i640_e512_k10"); }
} // namespace ninfer::ops::detail::moe_routed
