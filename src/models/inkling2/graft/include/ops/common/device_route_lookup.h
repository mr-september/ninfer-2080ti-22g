#pragma once

// The one question a launch asks of the current device's route profile (ops/common/device_route.h).
// Kernel units that ask it include this header alone, so an edit of the profiles, their
// installation or calibration's forcing compiles none of them.

#include <cstdint>
#include <string_view>

namespace ninfer::ops {

// The schedule the current device's profile names for `key` at `width`; empty when there is no
// profile, no entry for the key, or the width lies past the entry's last band.
[[nodiscard]] std::string_view device_route_schedule(std::string_view key, std::int32_t width);

} // namespace ninfer::ops
