#pragma once

// Inkling 2's quantized KV planes: storing kv_post's F16 rows into them (kv_codec.cuh encodes one
// row per warp). Reading them back happens in the attention's tile loader (kernels.h), which
// decodes eight-value chunks as it loads a tile — there is no F16 copy of a quantized cache.

#include "models/inkling2/kernels.h"
#include "models/inkling2/kv_format.h"

namespace ninfer::models::inkling2::kernels {

// Stores F16 rows into a quantized plane: row r's values, head h at src + h * src_head_stride +
// r * 128 (kv_post's `row_slots` layout), head h's plane row of position physical(rows, r) at slot
// physical(rows, r) (a full-attention visit: ring 0) of sequence row_sequence(rows, r).
struct KvStoreArgs {
    Rows rows;
    const __half* src = nullptr;
    std::int64_t src_head_stride = 0;
    int rows_count = 0, heads = 0;
    KvPlaneView dst;
};
void kv_store(const KvStoreArgs& args, cudaStream_t stream);

} // namespace ninfer::models::inkling2::kernels
