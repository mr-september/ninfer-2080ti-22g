#pragma once

#include "runtime/contract/request.h"
#include "core/transfer_work.h"
#include "core/wide_math.h"
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ninfer::runtime {

enum class CheckpointRole : std::uint8_t { InputReplay, LongAnchor, SharedPrefix, Continuation };

// Physical quantities are sampled from stores, never a scheduler-owned occupancy ledger.
struct ContextResourceUsage {
    std::uint32_t state_slots                                                   = 0;
    std::uint32_t main_kv_pages                                                 = 0;
    std::uint32_t backend_kv_pages                                              = 0;
    std::size_t host_bytes                                                      = 0;
    friend bool operator==(ContextResourceUsage, ContextResourceUsage) noexcept = default;
};

struct ResourceReservation {
    bool reserved = false;
    ContextResourceUsage shortage;

    explicit operator bool() const noexcept { return reserved; }
};

// Exact features for the startup-selected static prefill cost model. They describe only the
// suffix rebuilt after a selected prefix and remain separate from Scheduler service work.
struct PrefillWork {
    std::uint64_t chunks          = 0;
    std::uint64_t tokens          = 0;
    std::uint64_t attention_pairs = 0;
    std::uint64_t vision_items    = 0;
    std::uint64_t vision_patches  = 0;

    [[nodiscard]] friend constexpr bool operator==(PrefillWork, PrefillWork) noexcept = default;
};

// Exact prefill feature definition for a suffix beginning after prefix_tokens. Attention work is
// prefix*suffix + suffix*(suffix+1)/2 and all arithmetic saturates.
[[nodiscard]] inline PrefillWork make_prefill_work(std::uint64_t prefix_tokens,
                                                   std::uint64_t suffix_tokens,
                                                   std::uint64_t vision_items,
                                                   std::uint64_t vision_patches,
                                                   std::uint32_t prefill_chunk) noexcept {
    PrefillWork result;
    result.chunks =
        suffix_tokens == 0 || prefill_chunk == 0 ? 0 : 1U + (suffix_tokens - 1U) / prefill_chunk;
    result.tokens                       = suffix_tokens;
    result.vision_items                 = vision_items;
    result.vision_patches               = vision_patches;
    // prefix * suffix + suffix * (suffix + 1) / 2, saturated to uint64. The triangular term halves
    // whichever factor is even so neither factor overflows.
    const bool odd_suffix      = (suffix_tokens & 1U) != 0;
    const WideU64 linear       = mul_wide(prefix_tokens, suffix_tokens);
    const WideU64 triangular   = mul_wide(odd_suffix ? suffix_tokens : suffix_tokens / 2U,
                                          odd_suffix ? suffix_tokens / 2U + 1U : suffix_tokens + 1U);
    result.attention_pairs = linear.hi != 0 || triangular.hi != 0
                                 ? std::numeric_limits<std::uint64_t>::max()
                                 : saturating_add(linear.lo, triangular.lo);
    return result;
}

enum class ContextResourceClass : std::uint8_t {
    State,
    MainKV,
    BackendKV,
};

enum class ContextTransferDirection : std::uint8_t {
    DeviceToHost,
    HostToDevice,
    DeviceToDevice,
};

struct ContextTransferObservation {
    ContextResourceClass resource      = ContextResourceClass::State;
    ContextTransferDirection direction = ContextTransferDirection::DeviceToHost;
    std::uint64_t units                = 0; // State images for State; bytes for typed KV.
    std::uint32_t page_count           = 0;
    TransferWork work;
    std::uint64_t elapsed_ns = 0;
};

struct ContextTransferRequirement {
    ContextResourceClass resource      = ContextResourceClass::State;
    ContextTransferDirection direction = ContextTransferDirection::DeviceToHost;
    std::uint64_t units                = 0;
    std::uint32_t page_count           = 0;
    TransferWork work;

    [[nodiscard]] friend constexpr bool operator==(ContextTransferRequirement,
                                                   ContextTransferRequirement) noexcept = default;
};

struct ContextOperationCounts {
    std::uint64_t state_moves            = 0;
    std::uint64_t state_forks            = 0;
    std::uint64_t state_restores         = 0;
    std::uint64_t pressure_spill_pages   = 0;
    std::uint64_t partial_tail_cow_pages = 0;
};

struct SequenceCapacityCurve {
    std::uint32_t main_page_tokens                   = 0;
    std::uint32_t minimum_main_page_groups           = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::size_t minimum_device_reservation_bytes     = 0;
    std::size_t bytes_per_additional_main_page_group = 0;

    [[nodiscard]] std::size_t reservation_bytes(std::uint32_t main_page_groups) const;
    [[nodiscard]] std::uint32_t resolved_tokens(std::uint32_t main_page_groups) const;
};

struct KvCapacityResolution {
    KvCapacityMode mode                              = KvCapacityMode::Explicit;
    std::uint32_t main_page_groups                   = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::uint32_t resolved_tokens                    = 0;
    std::size_t minimum_runtime_reservation_bytes    = 0;
    std::size_t bytes_per_additional_main_page_group = 0;
    std::size_t runtime_reservation_bytes            = 0;
    std::size_t available_after_weights_bytes        = 0;
    std::size_t available_after_startup_bytes        = 0;
    std::size_t automatic_headroom_bytes             = 0;
    std::size_t planned_slack_bytes                  = 0;
};

} // namespace ninfer::runtime
