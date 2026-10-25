#include "ops/linear/projection/runner.h"

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace ninfer::test::linear_weights;
using namespace ninfer::test::linear;
namespace linear_suite = ninfer::test::linear_suite;

struct Geometry {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
};

const std::array kGeometries{
    Geometry{1024, 5120, 223U},  Geometry{4608, 4608, 277U},   Geometry{5120, 4608, 281U},
    Geometry{5120, 6144, 239U},  Geometry{5120, 10240, 211U},  Geometry{5120, 17408, 241U},
    Geometry{5120, 25600, 293U}, Geometry{6144, 5120, 227U},   Geometry{14336, 5120, 229U},
    Geometry{34816, 5120, 233U}, Geometry{248320, 5120, 197U}};

void q8_a16_conformance(Suite& suite) {
    for (const auto& shape : kGeometries) {
        std::vector<Invocation> calls;
        // Cover live-column tails and the transitions from K-split to tiled contractions.
        for (int t : {1,  2,  3,  4,  5,  7,  8,  9,  15,  16,  17,  23,  24,  25,
                      31, 32, 33, 39, 40, 41, 44, 47, 48,  49,  55,  56,  57,  63,
                      64, 65, 79, 80, 81, 95, 96, 97, 127, 128, 129, 256, 1024}) {
            calls.push_back({t});
        }
        if (shape.n == 6144 && shape.k == 5120) {
            for (int t : {191, 192, 193}) calls.push_back({t});
        }
        if (shape.k == 4608) {
            for (int t : {6, 11, 12, 13, 14, 19, 20, 21, 27, 28, 29}) calls.push_back({t});
            if (shape.n == 4608) {
                for (int t : {255, 257}) calls.push_back({t});
            }
        }
        if (shape.n == 248320) calls.push_back({34});
        for (int t : {1, 8, 16, 32, 48, 64, 65, 128}) {
            calls.push_back({t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true});
        }
        for (int t : {1, 16, 64, 128}) calls.push_back({t, CallForm::A16Convenience});
        calls.push_back({17, CallForm::Policy, ninfer::ops::LinearPolicy::AllowA8});
        add_shape(suite, ninfer::QType::Q8_G32_FP16, ActivationRecipe::TokenBlocks,
                  make_q8_g32_fp16_weight,
                  {shape.n, shape.k, shape.seed, Comparison::Sampled, true, calls});
    }
}
} // namespace

void ninfer::test::linear::add_q8_cases(Suite& suite) { q8_a16_conformance(suite); }
