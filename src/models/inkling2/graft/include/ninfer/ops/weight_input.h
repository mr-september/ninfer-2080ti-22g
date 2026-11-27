#pragma once

// graft shim: the destination tree's WeightInput surface (ninfer/ops/weight_input.h), plus
// the donor's ops::joins, which the grafted executor reads and the port's header does not
// yet declare.

#include "../../../../../../../include/ninfer/ops/weight_input.h"

#include <span>

namespace ninfer::ops {

// Whether the rows of `inputs`, in order, are one contiguous region of one parent.
[[nodiscard]] bool joins(std::span<const WeightInput> inputs);

} // namespace ninfer::ops
