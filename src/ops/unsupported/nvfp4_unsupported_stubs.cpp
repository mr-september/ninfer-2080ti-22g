#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_dispatch.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"

#include <stdexcept>

namespace ninfer::ops::detail {

// TMA launches
void launch_nvfp4_w4a4_tma_linear(Nvfp4Problem, const std::uint8_t*,
                                  const std::uint8_t*, const std::uint8_t*,
                                  const std::uint8_t*, __nv_bfloat16*,
                                  std::int32_t, float, cudaStream_t) {
    throw std::runtime_error("NVFP4 TMA linear is supported only on sm_120a (Blackwell)");
}

void launch_nvfp4_w4a4_tma_attention(const std::uint8_t*, const std::uint8_t*,
                                     const std::uint8_t*, const std::uint8_t*,
                                     __nv_bfloat16*, __nv_bfloat16*,
                                     __nv_bfloat16*, __nv_bfloat16*,
                                     std::int32_t, float, cudaStream_t) {
    throw std::runtime_error("NVFP4 TMA attention is supported only on sm_120a (Blackwell)");
}

void launch_nvfp4_w4a4_tma_gdn(const std::uint8_t*, const std::uint8_t*,
                               const std::uint8_t*, const std::uint8_t*,
                               __nv_bfloat16*, __nv_bfloat16*,
                               std::int32_t, float, cudaStream_t) {
    throw std::runtime_error("NVFP4 TMA GDN is supported only on sm_120a (Blackwell)");
}

void launch_nvfp4_w4a4_tma_linear_add(Nvfp4Problem, const std::uint8_t*,
                                      const std::uint8_t*, const std::uint8_t*,
                                      const std::uint8_t*, __nv_bfloat16*,
                                      std::int32_t, float, cudaStream_t) {
    throw std::runtime_error("NVFP4 TMA linear_add is supported only on sm_120a (Blackwell)");
}

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t*, const std::uint8_t*,
                                         const std::uint8_t*, const std::uint8_t*,
                                         __nv_bfloat16*, std::int32_t, float,
                                         cudaStream_t) {
    throw std::runtime_error("NVFP4 TMA linear_swiglu is supported only on sm_120a (Blackwell)");
}

// Format
Nvfp4WeightGeometry validate_nvfp4_weight(const Weight&, const char*) {
    throw std::runtime_error("NVFP4 format validation is supported only on sm_120a (Blackwell)");
}

// Linear
std::size_t nvfp4_linear_workspace_capacity_bytes(std::int32_t, std::int32_t,
                                                  LinearPolicy, std::int32_t,
                                                  std::int32_t) {
    return 0;
}

void nvfp4_dispatch(const Tensor&, const Weight&, Tensor&, LinearPolicy,
                    WorkspaceArena*, cudaStream_t) {
    throw std::runtime_error("NVFP4 linear dispatch is supported only on sm_120a (Blackwell)");
}

// Linear Add
std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t, std::int32_t,
                                                      LinearPolicy, std::int32_t,
                                                      std::int32_t) {
    return 0;
}

void nvfp4_linear_add_decode_launch(const Tensor&, const Weight&, Tensor&,
                                    cudaStream_t) {
    throw std::runtime_error("NVFP4 linear_add decode is supported only on sm_120a (Blackwell)");
}

void nvfp4_linear_add_small_t_launch(const Tensor&, const Weight&, Tensor&,
                                     cudaStream_t) {
    throw std::runtime_error("NVFP4 linear_add small_t is supported only on sm_120a (Blackwell)");
}

void nvfp4_linear_add_w4a4_launch(const Tensor&, const Weight&, Tensor&,
                                  Nvfp4W4a4Workspace, cudaStream_t) {
    throw std::runtime_error("NVFP4 linear_add w4a4 is supported only on sm_120a (Blackwell)");
}

void nvfp4_linear_add_dispatch(const Tensor&, const Weight&, Tensor&,
                               LinearPolicy, WorkspaceArena&, cudaStream_t) {
    throw std::runtime_error("NVFP4 linear_add dispatch is supported only on sm_120a (Blackwell)");
}

// Linear SwiGLU
std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy,
                                                         std::int32_t,
                                                         std::int32_t) {
    return 0;
}

void nvfp4_linear_swiglu_decode_launch(const Tensor&, const Weight&, Tensor&,
                                       cudaStream_t) {
    throw std::runtime_error("NVFP4 linear_swiglu decode is supported only on sm_120a (Blackwell)");
}

void nvfp4_linear_swiglu_small_t_launch(const Tensor&, const Weight&, Tensor&,
                                        cudaStream_t) {
    throw std::runtime_error("NVFP4 linear_swiglu small_t is supported only on sm_120a (Blackwell)");
}

void nvfp4_linear_swiglu_w4a4_launch(const Tensor&, const Weight&, Tensor&,
                                     WorkspaceArena&, cudaStream_t) {
    throw std::runtime_error("NVFP4 linear_swiglu w4a4 is supported only on sm_120a (Blackwell)");
}

void nvfp4_linear_swiglu_dispatch(const Tensor&, const Weight&, Tensor&,
                                  LinearPolicy, WorkspaceArena&,
                                  cudaStream_t) {
    throw std::runtime_error("NVFP4 linear_swiglu dispatch is supported only on sm_120a (Blackwell)");
}

// Attention Input Proj
std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy,
                                                      std::int32_t,
                                                      std::int32_t) {
    return 0;
}

void nvfp4_attn_input_decode_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                    Tensor&, Tensor&, cudaStream_t) {
    throw std::runtime_error("NVFP4 attn_input decode is supported only on sm_120a (Blackwell)");
}

void nvfp4_attn_input_small_t_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                     Tensor&, Tensor&, cudaStream_t) {
    throw std::runtime_error("NVFP4 attn_input small_t is supported only on sm_120a (Blackwell)");
}

void nvfp4_attn_input_w4a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                  Tensor&, Tensor&, Nvfp4W4a4Workspace,
                                  cudaStream_t) {
    throw std::runtime_error("NVFP4 attn_input w4a4 is supported only on sm_120a (Blackwell)");
}

void nvfp4_attn_input_dispatch(const Tensor&, const Weight&, Tensor&, Tensor&,
                               Tensor&, Tensor&, LinearPolicy, WorkspaceArena*,
                               cudaStream_t) {
    throw std::runtime_error("NVFP4 attn_input dispatch is supported only on sm_120a (Blackwell)");
}

// GDN Input Proj
std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy,
                                                     std::int32_t,
                                                     std::int32_t) {
    return 0;
}

void nvfp4_gdn_input_decode_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                   cudaStream_t) {
    throw std::runtime_error("NVFP4 gdn_input decode is supported only on sm_120a (Blackwell)");
}

void nvfp4_gdn_input_small_t_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                    cudaStream_t) {
    throw std::runtime_error("NVFP4 gdn_input small_t is supported only on sm_120a (Blackwell)");
}

void nvfp4_gdn_input_w4a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                 Nvfp4W4a4Workspace, cudaStream_t) {
    throw std::runtime_error("NVFP4 gdn_input w4a4 is supported only on sm_120a (Blackwell)");
}

void nvfp4_gdn_input_dispatch(const Tensor&, const Weight&, Tensor&, Tensor&,
                              LinearPolicy, WorkspaceArena*, cudaStream_t) {
    throw std::runtime_error("NVFP4 gdn_input dispatch is supported only on sm_120a (Blackwell)");
}

// GDN Snapshot & Conv
Nvfp4GdnConvPlan nvfp4_gdn_conv_resolve_plan(LinearPolicy, std::int32_t,
                                             std::int32_t) {
    return Nvfp4GdnConvPlan{Nvfp4GdnConvScheduleId::Materialized};
}

std::size_t nvfp4_gdn_snapshot_workspace_capacity_bytes(LinearPolicy,
                                                        std::int32_t,
                                                        std::int32_t) {
    return 0;
}

void nvfp4_gdn_snapshot_decode_launch(const Tensor&, const Weight&,
                                      const Tensor&, Tensor&,
                                      const Tensor&, const Tensor&,
                                      const Tensor&, Tensor&, Tensor&,
                                      Tensor&, Tensor&, cudaStream_t) {
    throw std::runtime_error("NVFP4 gdn_snapshot decode is supported only on sm_120a (Blackwell)");
}

void nvfp4_gdn_snapshot_small_t_launch(const Tensor&, const Weight&,
                                       const Tensor&, Tensor&,
                                       const Tensor&, const Tensor&,
                                       const Tensor&, Tensor&, Tensor&,
                                       Tensor&, Tensor&, cudaStream_t) {
    throw std::runtime_error("NVFP4 gdn_snapshot small_t is supported only on sm_120a (Blackwell)");
}

void nvfp4_gdn_record_small_t_launch(const Tensor&, const Weight&,
                                     const Tensor&, const Tensor&,
                                     const Tensor&, const Tensor&,
                                     Tensor&, Tensor&, Tensor&, Tensor&,
                                     Tensor&, cudaStream_t) {
    throw std::runtime_error("NVFP4 gdn_record small_t is supported only on sm_120a (Blackwell)");
}

void nvfp4_gdn_snapshot_post_launch(const Tensor&, const Tensor&,
                                    Tensor&, const Tensor&,
                                    const Tensor&, const Tensor&,
                                    Tensor&, Tensor&, Tensor&, cudaStream_t) {
    throw std::runtime_error("NVFP4 gdn_snapshot post is supported only on sm_120a (Blackwell)");
}

void nvfp4_gdn_record_post_launch(const Tensor&, const Tensor&,
                                  const Tensor&, const Tensor&,
                                  const Tensor&, Tensor&, Tensor&,
                                  Tensor&, cudaStream_t) {
    throw std::runtime_error("NVFP4 gdn_record post is supported only on sm_120a (Blackwell)");
}

void nvfp4_gdn_snapshot_dispatch(const Tensor&, const Weight&, const Tensor&,
                                 Tensor&, const Tensor&,
                                 const Tensor&, const Tensor&,
                                 Tensor&, Tensor&, Tensor&, Tensor&,
                                 LinearPolicy, WorkspaceArena&,
                                 cudaStream_t) {
    throw std::runtime_error("NVFP4 gdn_snapshot dispatch is supported only on sm_120a (Blackwell)");
}

} // namespace ninfer::ops::detail
