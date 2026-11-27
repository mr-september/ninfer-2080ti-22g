#pragma once

// Device side of Inkling 2's quantized KV planes (kv_format.h): the engine's 128-wide row codecs
// (ops/kv_cache/row128_codec.cuh), a warp encoding or decoding one row and the eight-value chunk
// reads the attention's tile loader makes (kernels.cu).

#include "models/inkling2/kv_format.h"
#include "ops/kv_cache/row128_codec.cuh"

namespace ninfer::models::inkling2 {

namespace kv = ::ninfer::ops::kv128;
using ::ninfer::ops::kv128::chunk_word;

} // namespace ninfer::models::inkling2
