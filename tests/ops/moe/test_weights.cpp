#include "ninfer/ops/moe.h"

#include <limits>
#include <stdexcept>

namespace ninfer::test::moe {
namespace {

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template <class F>
void rejects(F&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid expert activation usage was accepted");
}

} // namespace

int test_weights() {
    const WeightView weight;
    ops::ExpertInputUsage usage;
    usage.include({weight, ops::LinearPolicy::AllowA8, 2.0F});
    usage.include({weight, ops::LinearPolicy::AllowA8, 2.0F});
    require(usage.policy() == ops::LinearPolicy::AllowA8,
            "matching expert calibration was not preserved");
    usage.include({weight, ops::LinearPolicy::AllowA8, {}});
    require(usage.policy() == ops::LinearPolicy::AllowA8,
            "A8 restriction was not preserved");
    usage.include({weight, ops::LinearPolicy::A16Only, 3.0F});
    usage.include({weight, ops::LinearPolicy::AllowA8, 2.0F});
    require(usage.policy() == ops::LinearPolicy::A16Only,
            "expert permissions were widened");

    for (const auto divisor : {0.0F, -1.0F, std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::quiet_NaN()})
        rejects([&] { usage.include({weight, ops::LinearPolicy::AllowA8, divisor}); });
    return 0;
}

} // namespace ninfer::test::moe
