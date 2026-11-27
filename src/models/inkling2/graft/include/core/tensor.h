#pragma once

// graft: the destination engine OWNS Tensor. This mirrors the port's src/core/tensor.h — the
// authoritative LAYOUT (port-compiled Ops read these bytes) — and adds the donor's device-less
// construction (graft_bridge.cpp). The `Device` member is PRESENT: closing the layout seam
// between graft translation units and port-compiled ops (graft-checklist §4 risk 2).
#include "core/dtype.h"
#include "core/memory.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace ninfer {

struct Tensor {
    void* data  = nullptr;
    DType dtype = DType::BF16;
    Device device;
    std::int32_t ne[4] = {1, 1, 1, 1};
    std::int64_t nb[4] = {0, 0, 0, 0};

    Tensor() noexcept = default;
    Tensor(MemorySpan memory, DType dtype, std::initializer_list<std::int32_t> shape);
    Tensor(void* data, Device device, DType dtype, std::initializer_list<std::int32_t> shape);
    // graft: the donor's device-less form (the donor Tensor carried no device); defined in
    // graft_bridge.cpp against the caller's bound CUDA device.
    Tensor(void* data, DType dtype, std::initializer_list<std::int32_t> shape);
    [[nodiscard]] static Tensor metadata(DType dtype, std::initializer_list<std::int32_t> shape);

    std::int64_t numel() const;
    std::size_t bytes() const;
    bool is_contiguous() const;

    Tensor view(std::initializer_list<std::int32_t> shape) const;
    Tensor reshape(std::initializer_list<std::int32_t> shape) const;
    Tensor slice(int dim, std::int32_t start, std::int32_t len) const;
    Tensor permute(std::initializer_list<int> order) const;
};

} // namespace ninfer
