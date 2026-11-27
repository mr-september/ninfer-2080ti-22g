#pragma once

// Which wide mainloop an NVFP4 int8 call runs, and where that answer comes from.
//
// The NVFP4 W4A8 family has three mainloops: the grouped s8-tensor-core ladder
// (nvfp4_int8_grouped_mma.cuh), the ggml-shaped MMQ mainloop (nvfp4_mmq_mma.cuh), and a row-form
// kernel that answers T <= 8 whichever wide mainloop the call would otherwise take -- llama.cpp's own
// crossover, so decode and verify widths have one implementation rather than three. The row form is
// on the table too, not on a name of its own: the legacy chain runs the MMVQ-shaped kernel
// (nvfp4_mmvq_mma.cuh) and the unified chain runs its own narrow kernel
// (nvfp4_int8_narrow_mma.cuh) at the call's exact width.
//
// The wide mainloop is a property of the ROUTE TABLE, not of a second switch: a call the own
// (legacy) table serves runs MMQ, a call the unified table serves runs the grouped mainloop. That
// is the same axis the device profile's `nvfp4_int8/*` bands calibrate, so a calibrated band chooses
// a mainloop and there is no separate name to keep in sync with it.
//
// The family itself has exactly one switch: --nvfp4-w4a8-int (LoadOptions::nvfp4_w4a8_int), which
// registers the per-shape AllowNvfp4Int8 policy at the model layer
// (models/qwen3_5/execution/parameters.cpp). While the flag is on no width falls back to A16 --
// prefill and decode are the same decision on purpose, because a per-phase switch would make a
// calibrated band mean different kernels at different phases.

#include <atomic>
#include <cstdint>
#include <optional>

namespace ninfer::ops::detail {

enum class Nvfp4Int8Mainloop : std::int8_t {
    Grouped, ///< The int8-grouped ladder above the crossover; T <= 8 goes to the row-form kernel.
    Mmq,     ///< The ggml-shaped MMQ mainloop above the crossover; same narrow-band exception.
};

inline const char* to_string(Nvfp4Int8Mainloop mainloop) {
    switch (mainloop) {
        case Nvfp4Int8Mainloop::Grouped: return "grouped";
        case Nvfp4Int8Mainloop::Mmq: return "mmq";
    }
    return "?";
}

namespace mainloop_detail {

// -1: no override; otherwise the forced mainloop, plus one. Function-local so the seam is a
// process-wide property, and atomic because op tests launch from more than one thread.
inline std::atomic<int>& forced() {
    static std::atomic<int> slot{-1};
    return slot;
}

} // namespace mainloop_detail

/// Test seam: override the mainloop for the whole process, returning whatever was in force so a
/// nested scope can put it back. Production passes the table's mainloop and never calls this.
///
/// The return value is load-bearing, not convenience: the linear harness pins a mainloop while a
/// case that owns it may pin another, and with an absolute setter the inner scope would clear the
/// outer one on exit and leave the rest of the block on the wrong kernel under test.
inline std::optional<Nvfp4Int8Mainloop> force_nvfp4_int8_mainloop(
    std::optional<Nvfp4Int8Mainloop> mainloop) {
    const int previous = mainloop_detail::forced().exchange(
        mainloop ? static_cast<int>(*mainloop) + 1 : -1, std::memory_order_relaxed);
    if (previous < 0) { return std::nullopt; }
    return static_cast<Nvfp4Int8Mainloop>(previous - 1);
}

/// The mainloop this call runs: the route table's, unless a test override is in force.
inline Nvfp4Int8Mainloop nvfp4_int8_mainloop(Nvfp4Int8Mainloop table) {
    const int value = mainloop_detail::forced().load(std::memory_order_relaxed);
    return value >= 0 ? static_cast<Nvfp4Int8Mainloop>(value - 1) : table;
}

} // namespace ninfer::ops::detail
