#include "core/io/file.h"

#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::io {
namespace {

void check_position(std::uint64_t value) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::invalid_argument("file offset exceeds the positional I/O range");
}

} // namespace

void validate_direct_read(FileView file, std::uint64_t offset, std::span<std::byte> destination) {
    if (file.descriptor < 0 || file.mode != ReadMode::Direct ||
        !std::has_single_bit(file.memory_alignment) ||
        !std::has_single_bit(file.offset_alignment) || offset % file.offset_alignment ||
        destination.size() % file.offset_alignment ||
        (!destination.empty() &&
         (!destination.data() ||
          reinterpret_cast<std::uintptr_t>(destination.data()) % file.memory_alignment)) ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::invalid_argument("invalid direct read range or alignment");
    check_position(offset);
}

DirectReadWindow direct_read_window(FileView file, std::uint64_t offset, std::size_t bytes) {
    if (file.descriptor < 0 || file.mode != ReadMode::Direct || !bytes ||
        !std::has_single_bit(file.offset_alignment) || offset > file.bytes ||
        bytes > file.bytes - offset)
        throw std::invalid_argument("invalid direct read window");
    const auto alignment = file.offset_alignment;
    const auto prefix    = offset % alignment;
    if (bytes > std::numeric_limits<std::size_t>::max() - prefix - (alignment - 1))
        throw std::overflow_error("direct read window size overflow");
    const auto required = bytes + prefix;
    const auto count    = (required + alignment - 1) / alignment * alignment;
    check_position(offset - prefix);
    return {offset - prefix, count, prefix, required};
}

} // namespace ninfer::io
