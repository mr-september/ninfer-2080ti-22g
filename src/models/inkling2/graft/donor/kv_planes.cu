// Inkling 2's quantized KV planes: kv_store (kernels.h, kv_planes.h). One warp encodes one row of
// one head; the codings are the engine's D128 variants of its KV codecs (kv_codec.cuh).

#include "models/inkling2/kv_planes.h"

#include "models/inkling2/kv_codec.cuh"

#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace ninfer::models::inkling2::kernels {
namespace {

constexpr int kWarps = 4; // warps per block, one (row, head) unit each

void check_launch(const char* what) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("inkling2 ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

template <KvCoding C>
__launch_bounds__(kWarps * 32) __global__ void kv_store_kernel(KvStoreArgs a) {
    constexpr int kCode = kv_code_bytes(C), kScale = kv_scale_bytes(C);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int unit = blockIdx.x * kWarps + warp;
    if (unit >= a.rows_count * a.heads) { return; }
    const int head = unit % a.heads;
    const int r    = unit / a.heads;
    const int slot = physical(a.rows, r);
    const int sequence = row_sequence(a.rows, r);
    const __half* src = a.src + head * a.src_head_stride + std::int64_t(r) * kKvRowDims;
    float x[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { x[i] = __half2float(src[lane + 32 * i]); }
    std::uint8_t* codes  = a.dst.codes + sequence * a.dst.code_seq + head * a.dst.code_head +
                          std::int64_t(slot) * a.dst.code_row;
    std::uint8_t* scales = a.dst.scales + sequence * a.dst.scale_seq + head * a.dst.scale_head +
                           std::int64_t(slot) * a.dst.scale_row;
    kv::encode_row<C>(x, codes, scales, lane);
}

} // namespace

void kv_store(const KvStoreArgs& args, cudaStream_t stream) {
    const int units = args.rows_count * args.heads;
    if (units <= 0) { return; }
    const dim3 grid((units + kWarps - 1) / kWarps), block(kWarps * 32);
    switch (args.dst.coding) {
    case KvCoding::Int8G64: kv_store_kernel<KvCoding::Int8G64><<<grid, block, 0, stream>>>(args); break;
    case KvCoding::Int4G32: kv_store_kernel<KvCoding::Int4G32><<<grid, block, 0, stream>>>(args); break;
    case KvCoding::Lloyd4: kv_store_kernel<KvCoding::Lloyd4><<<grid, block, 0, stream>>>(args); break;
    case KvCoding::Mxfp8: kv_store_kernel<KvCoding::Mxfp8><<<grid, block, 0, stream>>>(args); break;
    default: throw std::invalid_argument("inkling2 kv_store: not an Inkling 2 plane coding");
    }
    check_launch("kv_store");
}

} // namespace ninfer::models::inkling2::kernels
