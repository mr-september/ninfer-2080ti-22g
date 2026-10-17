#pragma once

#include "ninfer/device_memory_error.h"
#include "ninfer/types.h"

#include <array>
#include <cstdint>
#include <exception>

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

} // namespace ninfer::product
