#pragma once

// Inkling-2-Mini single-sequence executor (M3 numerical core).
//
// Drives the vendored donor kernel API (execution/vendor/models/inkling2/kernels.h) with
// engine ops::linear for quantized projections and the native artifact store for weights.
// Visit flow mirrors the donor executor 1:1 (port/m3/visit-flow.md); the GGUF table and
// expert-delivery paths are intentionally replaced, not ported.
//
// Cross-token state: per-visit F16 K/V planes (full layers ring; sliding window) and the
// convolution histories per visit. Per-token state (residual bank, messages, controller
// ranks) is call scratch — nothing else persists across tokens (verified in donor
// control(): the rank state is recomputed per call from x + messages).

#include "core/device.h"
#include "models/inkling2/load/bindings.h"
#include "models/inkling2/port_config.h"
#include "models/inkling2/weights.h"
#include "models/common/bindings.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::inkling2::execution {

class Session;    // owns KV planes, conv histories; one per sequence

class Executor {
public:
    struct Impl;  // visible so Session can befriend it (Storage access)

    Executor(const Config&, std::span<const BoundWeight>, DeviceContext&, cudaStream_t,
             std::uint32_t max_context);
    ~Executor();

    [[nodiscard]] std::unique_ptr<Session> make_session() const;

    // One decode step for a single sequence: 48 visits, then the fused output mix + head.
    void decode_step(Session&, std::uint32_t token, std::int64_t position);

    // The last decode step's logits (bf16 [vocab], device-resident).
    [[nodiscard]] const void* logits() const noexcept;

private:
    std::unique_ptr<Impl> impl;
};

// Owns the cross-token state of one sequence.
class Session {
public:
    Session(const Config&, Device, std::size_t visits, std::uint32_t max_context);
    ~Session();
    [[nodiscard]] std::uint32_t length() const noexcept { return length_; }
    void advance(std::uint32_t steps) noexcept { length_ += steps; }

private:
    friend struct Executor::Impl;
    std::uint32_t length_ = 0;
    struct Storage;
    std::unique_ptr<Storage> store;   // KV planes + conv histories (site-addressed)
};

} // namespace ninfer::models::inkling2::execution
