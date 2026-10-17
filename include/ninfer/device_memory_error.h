#pragma once

#include <stdexcept>

namespace ninfer {

// Thrown when startup cannot place its device allocations: a failed cudaMalloc, or a runtime
// reservation larger than the memory left after the weights. Callers may retry with a smaller
// context; every other startup failure keeps its own type.
class DeviceMemoryError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

} // namespace ninfer
