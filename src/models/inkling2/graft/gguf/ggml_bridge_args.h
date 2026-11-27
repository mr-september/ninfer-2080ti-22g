#pragma once

// The types the bridge's kernel units share with its entry points (ggml_bridge.h): the epilogues,
// the routing of a product over experts and the arguments of the decode-call kernels. The block
// type stays opaque here; the entry points name its values and map them to ggml's. An edit of the
// entry points then compiles no kernel unit.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::gguf {

// The block type's id in a GGUF tensor directory (ggml_type); its values are in ggml_bridge.h.
enum class GgmlType : int;

enum class Epilogue : int {
    Store,       // out = y
    Accumulate,  // out = out + y
    GateProduct, // out = act(gate) * y, gate read from a float plane
};

// The activation a gated product applies to its gate: SiLU (SwiGLU) or tanh-approximated GELU
// (GeGLU, Gemma 4).
enum class GateActivation : int {
    Silu     = 0,
    GeluTanh = 1,
};

// The pairs grouped by expert: `sorted` lists the pairs of expert e at [bounds[e], bounds[e + 1]),
// and `active` the `*active_count` experts that have any, ascending. Device arrays.
// moe_sort_routes also lists the matrix kernel's work for each of its column tile widths
// (16, 32, 64 and 128 pairs, list w for width 16 << w): every (expert, column tile) that holds a
// pair, expert | tile << 16, ascending, `tile_count[w]` of them at tiles + w * tile_capacity.
struct MoeRouting {
    const std::int32_t* bounds       = nullptr; // [experts + 1]
    const std::int32_t* sorted       = nullptr; // [pairs]
    const std::int32_t* active       = nullptr; // [experts]
    const std::int32_t* active_count = nullptr; // [1]
    const std::int32_t* tile_count   = nullptr; // [4]
    const std::int32_t* tiles        = nullptr; // [4][tile_capacity]
    int tile_capacity                = 0;
};

// The records and arguments of the decode-call kernels (moe_decode_prep, moe_decode_up and
// moe_decode_down in ggml_bridge.h).
struct MoeDecodeRecord {
    // The slot's expert and the id set it came from (0: ids_a, 1: ids_b), with that expert's
    // pairs in ascending order and their tokens; n == 0 when the slot is negative in its set or
    // repeats an earlier slot's expert.
    std::int32_t expert = -1, n = 0, source = 0, reserved = 0;
    std::int32_t pairs[8] = {}, columns[8] = {};
};
struct MoeDecodePrepArgs {
    const __nv_bfloat16* m = nullptr; // BF16 [tokens][2560]
    int tokens             = 0;
    const std::int32_t* ids = nullptr; // [slots], the first stage's set (ids_a)
    int slots = 0, per_token = 10;
    MoeDecodeRecord* records = nullptr; // [slots]
    // The inputs as `type`'s up kernel reads them: q8_1 values [tokens][2560], (d, sum) and the
    // integer sum of every 32 values [tokens][80].
    std::int8_t* qs          = nullptr;
    half2* ds                = nullptr;
    int* qsum                = nullptr;
    unsigned long long* zero = nullptr; // zeroed when given
    int zero_count           = 0;
};
struct MoeDecodeUpArgs {
    const __nv_bfloat16* m = nullptr; // BF16 [tokens][k]
    int k = 2560, rows = 640;
    const std::int32_t* ids = nullptr; // [slots]: negative entries skipped; null: all expert 0
    int slots = 0, per_token = 10;
    const std::uint8_t* const* gate = nullptr;
    const std::uint8_t* const* up   = nullptr;
    std::int64_t row_bytes          = 0;
    __nv_bfloat16* middle           = nullptr; // [slots][rows]
    unsigned long long* zero        = nullptr; // zeroed first when given
    int zero_count                  = 0;
    // Prepared: the slots' records (instead of scanning ids) and the quantized inputs (instead of m).
    const MoeDecodeRecord* records = nullptr;
    const std::int8_t* qs          = nullptr;
    const half2* ds                = nullptr;
    const int* qsum                = nullptr;
    // When given, every block quantizes its 32 middle values of every pair for the down kernel
    // (transposed when its decoder reads them so) instead of storing them: [slots][640], and
    // [slots][20] (d, sum) and integer sums. Requires rows to be whole groups of 32.
    std::int8_t* middle_qs         = nullptr;
    half2* middle_ds               = nullptr;
    int* middle_qsum               = nullptr;
    bool middle_transposed         = false;
    // Scanning `ids`, a block row also lists its slot in `write_records` as `record_source`: the
    // first stage (0) every slot, the second (1) the slots it serves.
    MoeDecodeRecord* write_records = nullptr;
    int record_source              = 1;
};
struct MoeDecodeDownArgs {
    int rows = 2560; // the down projection's outputs; its k is 640
    // A slot is served from table_a when ids_a[slot] >= 0, else from table_b when ids_b[slot] >= 0.
    const std::int32_t* ids_a = nullptr; // null: all expert 0
    const std::int32_t* ids_b = nullptr;
    int slots = 0, per_token = 10;
    const std::uint8_t* const* table_a = nullptr;
    const std::uint8_t* const* table_b = nullptr;
    std::int64_t row_bytes             = 0;
    const __nv_bfloat16* middle        = nullptr; // [slots][640]
    const float* weights               = nullptr; // [slots]
    unsigned long long* fixed          = nullptr; // [tokens][rows]
    int tokens                         = 0;
    // The call's last launch: y [tokens][rows], external products (as add_products), and a
    // completion counter that starts at zero and is left at zero.
    float* y                         = nullptr;
    unsigned int* counter            = nullptr;
    const float* products            = nullptr;
    const std::int32_t* product_pairs = nullptr;
    const std::int32_t* product_count = nullptr;
    // Prepared: the slots' records and the quantized middles (MoeDecodeUpArgs::middle_qs).
    const MoeDecodeRecord* records = nullptr;
    const std::int8_t* qs          = nullptr;
    const half2* ds                = nullptr;
    const int* qsum                = nullptr;
};

} // namespace ninfer::ops::gguf
