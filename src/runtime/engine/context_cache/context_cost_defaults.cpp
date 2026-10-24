#include "runtime/engine/context_cache/context_cost.h"

#include <array>
#include <vector>

namespace ninfer::runtime {
namespace {

constexpr ContextPrefillCost kGroupwise27bPrefill{
    .chunk_ns              = 40'813'570,
    .token_ns_q32          = 1'012'273'154'411'951,
    .attention_pair_ns_q32 = 30'497'396'515,
    .vision_item_ns        = 5'986'585,
    .vision_patch_ns_q32   = 23'710'212'854'694,
};

} // namespace

const std::array<ContextTransferCost, 3>& generic_context_transfer_cost() {
    // Conservative but useful on an unmeasured machine. These values preserve numerical ranking;
    // they are not a switch that disables the cost model.
    static constexpr std::array<ContextTransferCost, 3> value{
        ContextTransferCost{
            .batch_ns = 100'000, .operation_ns = 12'000, .ns_per_byte_q32 = 107'374'182},
        ContextTransferCost{
            .batch_ns = 50'000, .operation_ns = 12'000, .ns_per_byte_q32 = 107'374'182},
        ContextTransferCost{
            .batch_ns = 25'000, .operation_ns = 12'000, .ns_per_byte_q32 = 42'949'673},
    };
    return value;
}

const ContextPrefillCost& generic_context_prefill_cost() {
    // The slower measured 27B configuration is the conservative generic estimate. An unknown model
    // still receives continuous recomputation costs instead of falling back to a discrete tuple.
    return kGroupwise27bPrefill;
}

// Accepted project defaults live only in this table and are compiled into the binary. Runtime JSON
// presets are independent local-machine overrides and never become a build dependency. No machine
// has accepted defaults yet, so every machine resolves to the generic numerical defaults above.
// Add a ContextCostMachinePreset here after calibrating transfer and prefill on the RTX 2080 Ti.
const std::vector<ContextCostMachinePreset>& compiled_context_cost_defaults() {
    static const std::vector<ContextCostMachinePreset> defaults{};
    return defaults;
}

} // namespace ninfer::runtime
