#pragma once

// graft shim: the donor tree names this header its KV-cache storage enum. ninfer owns that
// name in ninfer/types.h, whose KvCacheStorage now carries the donor's full set; this shim
// also provides the vendored kernels' alias, which the graft's position on the include path
// otherwise shadows out of the vendor's own copy of this header.

#include "ninfer/types.h"

namespace ninfer {

using VendoredKvCacheStorage = ninfer::KvCacheStorage;

} // namespace ninfer
