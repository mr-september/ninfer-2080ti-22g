#pragma once

// Physical KV-cache storage formats (--kv-dtype). Renamed from
// KvCacheStorage for the lean port: the destination engine owns that name
// (ninfer/types.h); this superset stays vendored-kernel-local. Kernels, the paged KV store and the Engine options
// share it; it depends on nothing else in the public interface.

// graft: the engine's KvCacheStorage (ninfer/types.h) now carries this same superset, so the
// vendored name is an alias of it rather than a second enum. Kernels, the paged KV store and
// the Engine options share it.

#include "ninfer/types.h"

namespace ninfer {

using VendoredKvCacheStorage = ninfer::KvCacheStorage;

} // namespace ninfer
