#pragma once

#include "ninfer/memory_errors.h"
#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <optional>

namespace ninfer::product {

// Contexts tried, in order, when the default one does not fit in free device memory.
inline constexpr std::array<std::uint32_t, 2> kFallbackMaxContexts = {32768, 16384};

// Runs start() with the options as given. When it fails for lack of device memory and the user did
// not pass --max-context, retries at each smaller fallback context and leaves the fitted values in
// max_context and kv_capacity (the KV pool follows the context unless --kv-capacity was explicit).
// An explicit --max-context is never lowered. The caller compares max_context with the value it
// requested to report a reduction.
template <class Start>
[[nodiscard]] auto start_fitting_context(std::uint32_t& max_context, KvCapacityPolicy& kv_capacity,
                                         bool max_context_explicit, bool kv_capacity_explicit,
                                         Start&& start) {
    const std::uint32_t requested        = max_context;
    const KvCapacityPolicy requested_kv  = kv_capacity;
    try {
        return start();
    } catch (const DeviceMemoryError&) {
        if (max_context_explicit) { throw; }
        const std::exception_ptr first = std::current_exception();
        for (const std::uint32_t fallback : kFallbackMaxContexts) {
            if (fallback >= requested) { continue; }
            max_context = fallback;
            if (!kv_capacity_explicit) { kv_capacity = KvCapacityPolicy::explicit_capacity(fallback); }
            try {
                return start();
            } catch (const DeviceMemoryError&) {}
        }
        max_context = requested;
        kv_capacity = requested_kv;
        std::rethrow_exception(first);
    }
}

// Smallest page-locked host-context budget the fit below will pick.
inline constexpr std::size_t kMinimumHostContextBytes = 256ULL << 20;
// What the default budget needs: kDefaultHostKvBytes plus eight Host StateImages (about 150 MiB each
// at 27B scale; the Engine uses the model's actual layout).
inline constexpr std::size_t kHostContextEstimateBytes = kDefaultHostKvBytes + (1280ULL << 20);
// Page-locked memory the Engine pins besides the Host context: the token embedding (about 1.3 GiB at
// 27B scale) and the weight staging buffers.
inline constexpr std::uint64_t kOtherPinnedBytes = 1700ULL << 20;

// The Host context budget to request when the default would not fit, found by halving it (never
// below kMinimumHostContextBytes); nullopt when the default fits or the available RAM is unknown.
// Measured on Windows with the model resident: pinning succeeded only while the available physical
// RAM was about twice everything pinned, so the budget plus kOtherPinnedBytes must stay within half
// of it. The rule is empirical; a budget that fits it can still fail to pin on a busier machine.
[[nodiscard]] inline std::optional<std::size_t>
fit_host_context_to_ram(std::uint64_t available_bytes) noexcept {
    if (available_bytes == 0) { return std::nullopt; }
    std::size_t budget = kHostContextEstimateBytes;
    while (budget > kMinimumHostContextBytes &&
           2 * (budget + kOtherPinnedBytes) > available_bytes) {
        budget /= 2;
    }
    budget = std::max(budget, kMinimumHostContextBytes);
    if (budget >= kHostContextEstimateBytes) { return std::nullopt; }
    return budget;
}

} // namespace ninfer::product
