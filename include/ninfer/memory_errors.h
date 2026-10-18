#pragma once

#include <cstddef>
#include <string>
#include <stdexcept>

namespace ninfer {

// Thrown when startup cannot place its device allocations: a failed cudaMalloc, or a runtime
// reservation larger than the memory left after the weights. Callers may retry with a smaller
// context; every other startup failure keeps its own type.
class DeviceMemoryError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Thrown when page-locked host memory cannot be allocated. requested_bytes() is the size that
// failed, so a caller can retry with a smaller host budget.
class HostMemoryError final : public std::runtime_error {
public:
    HostMemoryError(const std::string& message, std::size_t requested_bytes)
        : std::runtime_error(message), requested_bytes_(requested_bytes) {}

    [[nodiscard]] std::size_t requested_bytes() const noexcept { return requested_bytes_; }

private:
    std::size_t requested_bytes_;
};

} // namespace ninfer
