#pragma once

namespace ninfer::test {

// FP8/NVFP4 weights, the FP8/NVFP4/K8V4 KV caches and the TMA/block-scaled MMA routes need
// sm_120a instructions. Turing builds link throwing definitions for them (see
// src/ops/unsupported), so their qualification cases are skipped there.
inline constexpr bool kSm120Formats =
#if defined(NINFER_SM75)
    false;
#else
    true;
#endif

} // namespace ninfer::test
