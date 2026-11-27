#pragma once

// llama.cpp's tensor-core flash attention (third_party/ggml-quants, fattn-mma-f16.cuh) behind plain
// pointers: one sequence's queries against a masked F16 key/value cache, for head widths 128 (query
// heads per key/value head a multiple of eight), 256 (two) and 512 (eight), optionally with a
// learned sink logit per query head. Gemma 4 and Flint run their prompt attention through it.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::gguf {

struct FlashAttentionArgs {
    int head_dim    = 0; // 128, 256 or 512; K and V share it
    int query_heads = 0;
    // query_heads / kv_heads: a multiple of 8 at widths 128 and 512, of 2 at width 256.
    int kv_heads    = 0;
    int tokens      = 0; // query rows
    // Q FP32: element (d, h, t) at q[t * q_token_stride + h * head_dim + d].
    const float* q              = nullptr;
    std::int64_t q_token_stride = 0;
    // K and V F16: element (d, slot, kv head) at base + head * head_bytes + slot * row_bytes + 2 d.
    const void* k            = nullptr;
    const void* v            = nullptr;
    std::int64_t row_bytes   = 0;
    std::int64_t head_bytes  = 0;
    int slots                = 0; // a whole number of 256
    // F16 [tokens][mask_row_halves]: added to the scores, -inf where a slot is not visible.
    const void* mask             = nullptr;
    std::int64_t mask_row_halves = 0;
    // FP32 [tokens][query_heads][head_dim].
    float* out  = nullptr;
    float scale = 1.0f;
    // Optional FP32 [query_heads]: head h's sink logit joins each of its softmax denominators as
    // exp(sinks[h] - max), a key without a value.
    const float* sinks = nullptr;
    // Optional, device [ceil(tokens / flash_attention_tile_rows(head_dim))]: per tile of query
    // rows, a multiple of 256 at or past which every slot is masked for every row of the tile. The
    // launch skips those slots.
    const int* kv_max = nullptr;
};

// Query rows per tile of the launch (the granularity of FlashAttentionArgs::kv_max).
[[nodiscard]] int flash_attention_tile_rows(int head_dim);

// Workspace the launch needs (the stream-k partial results), for any `tokens` up to the given count.
[[nodiscard]] std::size_t flash_attention_workspace_bytes(int head_dim, int query_heads,
                                                          int kv_heads, int max_tokens);

void flash_attention(const FlashAttentionArgs& args, void* workspace, std::size_t workspace_bytes,
                     cudaStream_t stream);

} // namespace ninfer::ops::gguf
