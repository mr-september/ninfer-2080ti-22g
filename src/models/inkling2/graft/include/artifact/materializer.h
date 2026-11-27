#pragma once

// graft shim: the destination tree's materializer (artifact/materializer.h), plus the donor's
// MaterializeOptions, which the donor's dspine model signatures name and the port's header
// does not yet declare.

#include "../../../../../artifact/materializer.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace ninfer::artifact {

struct MaterializeOptions {
    // Device arenas at fixed addresses whose physical memory a model suspend can release; the
    // device objects are then uploaded again from `retained_reader` when the model resumes. The
    // reader keeps the artifact's files open, so the bytes it restores are the ones it loaded even
    // if a path is replaced meanwhile.
    bool suspendable = false;
    std::shared_ptr<const Reader> retained_reader;
    // At most this many bytes of Residency::PinnedOrFiles objects are pinned; the rest stay in
    // the artifact's files (--expert-host-mib). Unset: as many as the host can pin.
    std::optional<std::uint64_t> files_after_bytes;
};

} // namespace ninfer::artifact
