#pragma once

// The paged KV cache's page size: the one fact about pages that device code needs. The views and
// pools (core/paged_kv_view.h, core/paged_kv_cache.h) include it, and so do the kernels' address
// helpers (ops/kernel/paged_kv_address.cuh) without them.

#include <cstdint>

namespace ninfer {

inline constexpr std::int32_t kPagedKVPageSize = 64;

} // namespace ninfer
