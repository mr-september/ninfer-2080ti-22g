#include "ops/linear/split/dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail::linear_split {
namespace {
const SplitShape& match(std::span<const SplitShape* const> shapes, ProjectionProfile weights) {
    for (const auto* shape : shapes)
        if (shape->first_rows == weights.first.n && shape->input_rows == weights.first.k &&
            shape->second_rows == (weights.second ? weights.second->n : 0) &&
            (!weights.second || weights.second->k == shape->input_rows))
            return *shape;
    throw std::invalid_argument("linear_split: unsupported parent geometry");
}
} // namespace

const SplitShape& select_split_shape(ProjectionProfile weights) {
    if (!valid_linear_policy(weights.first.policy))
        throw std::invalid_argument("linear_split: invalid activation permission");
    if (weights.second) {
        if (weights.first.qtype != QType::Q4_G64_FP16 ||
            weights.second->qtype != QType::Q5_G64_FP16 ||
            weights.first.policy != LinearPolicy::A16Only)
            throw std::invalid_argument("linear_split: paired weights require Q4/Q5 A16");
        const std::array shapes{&kQ4Q5SplitN7168N7168K5120, &kQ4Q5SplitN4096N12288K5120};
        return match(shapes, weights);
    }
    switch (weights.first.qtype) {
    case QType::BF16: {
        const std::array shapes{&kBf16SplitN14336K5120};
        return match(shapes, weights);
    }
    case QType::Q8_G32_FP16: {
        const std::array shapes{&kQ8SplitN6144K5120};
        return match(shapes, weights);
    }
    default:
        throw std::invalid_argument("linear_split: unsupported weight format");
    }
}
} // namespace ninfer::ops::detail::linear_split
