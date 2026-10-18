#include "ops/kv_cache/context_materialize/dispatch.h"
#include "ops/common/host/launch.h"
#include "ops/kv_cache/context_materialize/projection.cuh"

namespace ninfer::ops::detail::context_kv {
namespace {

template <int Rows, int Columns, int BlockK, int ColumnWarps>
void launch_mma(const Tensor& x, const Tensor& positions, const Tensor& counts, const Tensor& slots,
                DeviceLayers layers, ContextKVMaterializeExecutionEnvelope envelope,
                const Tensor& scratch, cudaStream_t stream) {
    constexpr int bytes   = sizeof(MaterializeStorage<Rows, Columns, BlockK>);
    constexpr auto kernel = context_kv_mma_kernel<Rows, Columns, BlockK, ColumnWarps>;
    (void)common::prepare_shared<bytes, kernel, true>();
    context_kv_mma_kernel<Rows, Columns, BlockK, ColumnWarps>
        <<<dim3(1024 / Rows, (envelope.max_count * x.ne[2] + Columns - 1) / Columns, 10),
           Rows / 16 * ColumnWarps * 32, bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const int*>(positions.data),
            static_cast<const int*>(counts.data), static_cast<const int*>(slots.data), layers,
            static_cast<float*>(scratch.data), x.ne[1], x.ne[2], envelope.min_count,
            envelope.max_count);
    CUDA_CHECK(cudaGetLastError());
}

template <int Columns, int KWarps = 8>
void launch_grouped(const Tensor& x, const Tensor& positions, const Tensor& counts,
                    const Tensor& slots, DeviceLayers layers,
                    ContextKVMaterializeExecutionEnvelope envelope, const Tensor& scratch,
                    cudaStream_t stream) {
    constexpr auto kernel = context_kv_grouped_kernel<Columns, KWarps>;
    const int shared =
        common::prepare_shared<GroupedSchedule<Columns, KWarps>::kSharedBytes, kernel>();
    kernel<<<dim3(64, (envelope.max_count * x.ne[2] + Columns - 1) / Columns, 10), KWarps * 32,
             shared, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const int*>(positions.data),
        static_cast<const int*>(counts.data), static_cast<const int*>(slots.data), layers,
        static_cast<float*>(scratch.data), x.ne[1], x.ne[2], envelope.min_count,
        envelope.max_count);
    CUDA_CHECK(cudaGetLastError());
}

DeviceLayers make_device_layers(
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers) {
    DeviceLayers result{};
    for (int index = 0; index < kLayers; ++index) {
        const ContextKVMaterializeLayerView& source = layers[static_cast<std::size_t>(index)];
        result.layer[index]                         = {
            static_cast<const std::uint8_t*>(source.key_weight.qdata),
            static_cast<const std::uint8_t*>(source.key_weight.scales),
            static_cast<const std::uint8_t*>(source.value_weight.qdata),
            static_cast<const std::uint8_t*>(source.value_weight.scales),
            static_cast<const __nv_bfloat16*>(source.key_norm_weight.data),
            static_cast<__nv_bfloat16*>(source.cache.k.data),
            static_cast<__half*>(source.cache.v.data),
            static_cast<std::int32_t>(source.cache.padded_capacity),
            source.cache.k.nb[3] / 2,
            source.cache.v.nb[3] / 2,
        };
    }
    return result;
}

template <auto Projection, bool KeyPost>
void launch_materialize(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    ContextKVMaterializeExecutionEnvelope envelope, const Tensor& key_scratch,
    cudaStream_t stream) {
    const auto device_layers = make_device_layers(layers);
    Projection(context, positions, counts, slots, device_layers, envelope, key_scratch, stream);
    if constexpr (KeyPost) {
        context_kv_key_post_kernel<<<dim3(envelope.max_count * context.ne[2], kLayers), 256, 0,
                                     stream>>>(
            static_cast<const float*>(key_scratch.data), static_cast<const int*>(positions.data),
            static_cast<const int*>(counts.data), static_cast<const int*>(slots.data),
            device_layers, context.ne[2], context.ne[1], envelope.min_count, envelope.max_count);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <auto Projection, bool KeyPost>
constexpr ContextKVMaterializeImplementation implementation() {
    return {launch_materialize<Projection, KeyPost>, KeyPost};
}

} // namespace
} // namespace ninfer::ops::detail::context_kv

namespace ninfer::ops::detail {

ContextKVMaterializeImplementation select_context_kv_materialize(std::int32_t columns) {
    using namespace context_kv;
    if (columns <= 16) return implementation<launch_grouped<16>, true>();
    if (columns <= 24) return implementation<launch_grouped<24>, true>();
    if (columns <= 32) return implementation<launch_grouped<16>, true>();
    if (columns <= 64) return implementation<launch_mma<64, 32, 128, 2>, true>();
    if (columns <= 80) return implementation<launch_mma<64, 80, 128, 5>, true>();
    if (columns <= 96) return implementation<launch_mma<64, 96, 128, 6>, true>();
#if !defined(NINFER_SM75)
    // The fused 128-row tile stages 66 KiB, over a Turing CTA's 64 KiB.
    if (columns <= 256) return implementation<launch_mma<128, 64, 128, 2>, false>();
#endif
    return implementation<launch_mma<64, 64, 64, 2>, true>();
}

} // namespace ninfer::ops::detail
