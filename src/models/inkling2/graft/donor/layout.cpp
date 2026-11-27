#include "core/layout.h"

// Graft layout implementations (donor src/core/layout.cpp), TRIMMED to the Device-span surface the
// port does not own. The port's ninfer_core layout.cpp already defines, with identical mangled
// names: LayoutBuilder::{Scope ctor/move/dtor, add, add_tensor, scope, finish} (its LayoutBuilder
// class body is character-identical to the donor's) and WorkspaceLayoutBuilder::{alloc,
// peak_bytes}. The port's WorkspaceLayoutBuilder bodies touch cursor_/peak_ at the same offsets as
// the donor's single `layout_` member (verified against both headers: LayoutBuilder is
// {cursor_, peak_}), so sharing the engine definitions is byte-compatible, not merely convenient.
// What stays here is the DeviceSpan-shaped surface — bind(DeviceSpan), alloc_bytes(DeviceSpan),
// and scope() returning LayoutBuilder::Scope — all distinct mangled names the port lacks.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace ninfer {
namespace {

bool is_power_of_two(std::size_t value) { return value != 0 && (value & (value - 1)) == 0; }

} // namespace

DeviceSpan LayoutRegion::bind(DeviceSpan backing) const {
    if (backing.data == nullptr) { throw std::invalid_argument("layout backing is null"); }
    if (!is_power_of_two(alignment)) {
        throw std::logic_error("layout region has an invalid alignment");
    }
    const auto base = reinterpret_cast<std::uintptr_t>(backing.data);
    if ((base + offset) % alignment != 0) {
        throw std::invalid_argument("layout backing does not satisfy region alignment");
    }
    if (offset > backing.bytes || bytes > backing.bytes - offset) {
        throw std::out_of_range("layout region exceeds its backing allocation");
    }
    return DeviceSpan{static_cast<unsigned char*>(backing.data) + offset, bytes};
}

Tensor TensorRegion::bind(DeviceSpan backing) const {
    Tensor expected(nullptr, dtype, {shape[0], shape[1], shape[2], shape[3]});
    if (expected.bytes() != region.bytes) {
        throw std::logic_error("tensor layout has an inconsistent byte size");
    }
    return Tensor(region.bind(backing).data, dtype, {shape[0], shape[1], shape[2], shape[3]});
}

DeviceSpan WorkspaceLayoutBuilder::alloc_bytes(std::size_t bytes, std::size_t alignment) {
    if (bytes == 0) { return {}; }
    (void)layout_.add(bytes, alignment, "workspace layout");
    return DeviceSpan{nullptr, bytes};  // layout_.add defined by ninfer_core (identical class)
}

WorkspaceLayoutBuilder::Scope WorkspaceLayoutBuilder::scope() noexcept { return layout_.scope(); }

} // namespace ninfer
