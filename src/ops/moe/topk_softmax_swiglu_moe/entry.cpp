#include "core/nvtx.h"
#include "ninfer/ops/moe.h"
#include "ops/moe/operands.h"
#include "ops/moe/topk_softmax_swiglu_moe/dispatch.h"
#include "ops/moe/workspace.h"
#include "ops/moe/kernels/cuda/block_scaled/workspace.h"

namespace ninfer::ops {
namespace {
namespace kernel       = detail::moe_cuda;
namespace block_scaled = kernel::block_scaled;
namespace routing      = kernel::routing;
namespace impl         = detail::moe_complete;

struct CallStorage {
    ExpertRoutes routes;
    Tensor shared;
    routing::RouteStorage route;
    block_scaled::Storage block_scaled;
};

template <class Arena>
CallStorage allocate(Arena& arena, const impl::Plan& plan, int tokens, int experts, int top_k) {
    CallStorage out;
    out.routes = detail::allocate_routes(arena, top_k, tokens);
    out.shared = arena.alloc(DType::FP32, {tokens});
    // Router scratch is independent of expert activations, including during PDL overlap.
    out.route = routing::allocate_route(arena, plan.router, tokens, experts);
    out.block_scaled = block_scaled::allocate(arena, plan.block_scaled, tokens, true, true);
    return out;
}
} // namespace

std::size_t topk_softmax_swiglu_moe_workspace_capacity_bytes(const MoeProfile& profile,
                                                             MoeEpilogue mode, int first,
                                                             int last) {
    const auto& shape = impl::shape_dispatch(profile);
    detail::validate_output(DType::BF16, mode);
    detail::validate_interval(first, last);
    return detail::interval_capacity(
        first, last, shape.allocation_boundaries, impl::kSliceTokens, [&](int tokens) {
            WorkspaceLayoutBuilder layout;
            (void)allocate(layout, shape.select(tokens), tokens, profile.routed.experts,
                           profile.routed.top_k);
            return layout.peak_bytes(1);
        });
}

void topk_softmax_swiglu_moe(const Tensor& x, const MoeProfile& profile, const MoeWeights& weights,
                             MoeEpilogue mode, Tensor& destination, MemoryArena& workspace,
                             CudaExecutionView execution, const MoeExecutionHints&) {
    const auto& shape = impl::shape_dispatch(profile);
    if (destination.dtype != DType::BF16) {
        throw std::invalid_argument("complete MoE requires BF16 destination");
    }
    detail::MoeOperands operands("topk_softmax_swiglu_moe", execution);
    const auto& p    = profile.routed;
    const int tokens = operands.input(x, p.ffn.hidden);
    operands.output(destination, p.ffn.hidden, tokens, mode);
    operands.control(weights.control, {p.ffn.hidden, p.experts, p.top_k});
    operands.experts(weights.routed, p);
    operands.shared(weights.shared, p.ffn.hidden);
    operands.workspace(
        workspace, topk_softmax_swiglu_moe_workspace_capacity_bytes(profile, mode, tokens, tokens));
    operands.finish();
    nvtx::ScopedRange range(nvtx::Name::TopkSoftmaxSwiGLUMoe, nvtx::Category::Moe, tokens);
    for (std::int64_t start = 0; start < tokens; start += impl::kSliceTokens) {
        const int offset   = static_cast<int>(start);
        const int count    = std::min(tokens - offset, impl::kSliceTokens);
        const auto plan    = shape.select(count);
        const auto input   = x.slice(1, offset, count);
        auto output        = destination.slice(1, offset, count);
        auto scope         = workspace.scope();
        const auto storage = allocate(workspace, plan, count, p.experts, p.top_k);
        const ScaledSwiGLUTerm shared{weights.shared, storage.shared};
        block_scaled::launch_control_shared(input, weights, storage.routes, storage.shared,
                                            storage.route.scores, storage.block_scaled,
                                            plan.block_scaled, execution);
        block_scaled::launch(input, weights.routed, storage.routes, {}, &shared, mode, output,
                             plan.block_scaled, storage.block_scaled, execution, true);
    }
}
} // namespace ninfer::ops
