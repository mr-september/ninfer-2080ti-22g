#include "core/nvtx.h"
#include "ninfer/ops/moe.h"
#include "ops/moe/operands.h"
#include "ops/moe/topk_softmax_route/dispatch.h"
#include "ops/moe/workspace.h"

namespace ninfer::ops {
namespace {
std::size_t route_bytes(const RouteProfile& profile, int tokens) {
    WorkspaceLayoutBuilder layout;
    (void)detail::moe_cuda::routing::allocate_route(
        layout, detail::moe_route::select_h2560_e512_k10(tokens), tokens, profile.experts);
    return layout.peak_bytes(1);
}
} // namespace

std::size_t topk_softmax_route_workspace_capacity_bytes(const RouteProfile& profile, int first,
                                                        int last) {
    detail::validate_route_profile(profile);
    detail::validate_interval(first, last);
    return detail::interval_capacity(first, last, detail::moe_route::allocation_boundaries(),
                                     detail::moe_route::kSliceTokens,
                                     [&](int tokens) { return route_bytes(profile, tokens); });
}

void topk_softmax_route(const Tensor& x, const RouteProfile& profile, const Weight& control,
                        ExpertRoutes& routes, Tensor& shared_coefficients, MemoryArena& workspace,
                        CudaExecutionView execution) {
    detail::validate_route_profile(profile);
    detail::MoeOperands operands("topk_softmax_route", execution);
    const int tokens = operands.input(x, profile.hidden);
    operands.control(control, profile);
    operands.routes(routes, profile.top_k, tokens, true);
    operands.coefficient(shared_coefficients, tokens, true);
    operands.workspace(workspace,
                       topk_softmax_route_workspace_capacity_bytes(profile, tokens, tokens));
    operands.finish();
    nvtx::ScopedRange range(nvtx::Name::TopkSoftmaxRoute, nvtx::Category::Moe, tokens);
    for (std::int64_t start = 0; start < tokens; start += detail::moe_route::kSliceTokens) {
        const int offset = static_cast<int>(start);
        const int count  = std::min(tokens - offset, detail::moe_route::kSliceTokens);
        auto scope       = workspace.scope();
        const auto plan  = detail::moe_route::select_h2560_e512_k10(count);
        const auto storage =
            detail::moe_cuda::routing::allocate_route(workspace, plan, count, profile.experts);
        const ExpertRoutes result{routes.ids.slice(1, offset, count),
                                  routes.coefficients.slice(1, offset, count)};
        detail::moe_cuda::routing::launch_route(x.slice(1, offset, count), control, result,
                                                shared_coefficients.slice(0, offset, count), plan,
                                                storage, {}, execution);
    }
}
} // namespace ninfer::ops
