#include "core/weight.h"
#include "ops/linear/swiglu/runner.h"
#include "ops/linear_common.h"

#include <array>
#include <vector>

namespace {

void add_wide_cases(ninfer::test::linear_suite::Suite& suite) {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;
    std::vector<std::int32_t> tokens;
    for (int t = 1; t <= 128; ++t) tokens.push_back(t);
    for (int t : {129, 256, 1024}) tokens.push_back(t);
    constexpr std::array graphs{1,  16, 32, 40, 41, 51, 52, 63,  64,
                                65, 80, 81, 88, 89, 96, 97, 128, 129};
    add_profile(suite, {QType::Q8_G32_FP16, 34816, 5120, 17408, 1603U, ops::LinearPolicy::A16Only},
                tokens, graphs);
}

} // namespace

void ninfer::test::linear_swiglu::add_q8_cases(Suite& suite) {
    add_wide_cases(suite);
}
