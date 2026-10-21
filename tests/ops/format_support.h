#pragma once

namespace ninfer::test {

// FP8 weights, the FP8 KV cache and the TMA/block-scaled MMA routes need
// sm_120a instructions. Turing builds link throwing definitions for them (see
// src/ops/unsupported), so their qualification cases are skipped there.
inline constexpr bool kSm120Formats =
    false;

} // namespace ninfer::test
