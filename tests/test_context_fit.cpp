#include "product/context_fit.h"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using ninfer::DeviceMemoryError;
using ninfer::KvCapacityPolicy;
using ninfer::product::start_fitting_context;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

// Starts only when the context is at most `fits`; records every attempt.
struct Device {
    std::uint32_t fits;
    std::vector<std::uint32_t>* attempts;
    std::uint32_t* context;
    int operator()() const {
        attempts->push_back(*context);
        if (*context > fits) { throw DeviceMemoryError("out of memory"); }
        return 7;
    }
};

} // namespace

int main() {
    int failures = 0;
    {
        std::uint32_t context = 65536;
        auto kv               = KvCapacityPolicy::explicit_capacity(65536);
        std::vector<std::uint32_t> attempts;
        const int value = start_fitting_context(context, kv, false, false,
                                                Device{20000, &attempts, &context});
        failures += check(value == 7 && context == 16384 && kv.explicit_tokens == 16384,
                          "the default context did not shrink to the first one that fits");
        failures += check(attempts == std::vector<std::uint32_t>{65536, 32768, 16384},
                          "the fallback contexts were not tried in order");
    }
    {
        std::uint32_t context = 65536;
        auto kv               = KvCapacityPolicy::explicit_capacity(65536);
        std::vector<std::uint32_t> attempts;
        bool thrown = false;
        try {
            (void)start_fitting_context(context, kv, true, false,
                                        Device{20000, &attempts, &context});
        } catch (const DeviceMemoryError&) { thrown = true; }
        failures += check(thrown && context == 65536 && attempts.size() == 1,
                          "an explicit --max-context was lowered");
    }
    {
        std::uint32_t context = 65536;
        auto kv               = KvCapacityPolicy::explicit_capacity(131072);
        std::vector<std::uint32_t> attempts;
        (void)start_fitting_context(context, kv, false, true, Device{40000, &attempts, &context});
        failures += check(context == 32768 && kv.explicit_tokens == 131072,
                          "an explicit --kv-capacity was changed");
    }
    {
        std::uint32_t context = 65536;
        auto kv               = KvCapacityPolicy::explicit_capacity(65536);
        std::vector<std::uint32_t> attempts;
        bool thrown = false;
        try {
            (void)start_fitting_context(context, kv, false, false,
                                        Device{1000, &attempts, &context});
        } catch (const DeviceMemoryError&) { thrown = true; }
        failures += check(thrown && context == 65536 && kv.explicit_tokens == 65536 &&
                              attempts.size() == 3,
                          "nothing fitting did not restore the requested options and rethrow");
    }
    {
        std::uint32_t context = 65536;
        auto kv               = KvCapacityPolicy::explicit_capacity(65536);
        bool other            = false;
        try {
            (void)start_fitting_context(context, kv, false, false, [&]() -> int {
                throw std::invalid_argument("not a memory error");
            });
        } catch (const std::invalid_argument&) { other = true; }
        failures += check(other && context == 65536, "a non-memory failure was retried");
    }
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
