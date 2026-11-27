#pragma once

#include "core/expert_disk.h"
#include "core/expert_tier.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer {

struct RankContext;

// A blob is copied without decoding or repacking. File extents concatenate in order.
struct ExpertExtent {
    std::size_t file = 0;
    std::uint64_t offset = 0, bytes = 0;
};
struct ExpertBlob {
    const void* pinned = nullptr;
    // The base of the host allocation `pinned` lies in. Neighbouring pinned blobs of one nonzero
    // region are copied together; a copy never spans two allocations.
    std::uintptr_t pinned_region = 0;
    std::vector<ExpertExtent> extents;
    std::uint64_t bytes = 0;
    // Zero bytes a kernel may read past the blob's end (whole K steps past a row's end decode the
    // following bytes as blocks of the same format, so they must be zeros or that format's data).
    std::uint64_t zero_tail = 0;
};
struct ExpertAlias {
    std::size_t blob = 0;
    std::uint64_t offset = 0;
};
struct ExpertDeliveryLayer {
    // [expert][blob]. Aliases describe the projection-major device pointer table.
    std::vector<std::vector<ExpertBlob>> experts;
    std::vector<ExpertAlias> aliases;
};
class CpuSpinPool;

// A family's CPU products of the routed experts a narrow call leaves to the CPU: computed from the
// experts' stored bytes in host memory for the call's inputs that begin() copied to the host, into
// the products the family's second stage adds after await(). They reproduce the GPU's products bit
// for bit, so how a call splits its misses never changes a result.
class ExpertCpuProducts {
public:
    virtual ~ExpertCpuProducts() = default;
    // Whether the CPU computes `layer`'s experts.
    [[nodiscard]] virtual bool supports(std::size_t layer) const = 0;
    struct Work {
        std::size_t layer = 0;
        std::span<const std::int32_t> experts; // the CPU's experts, each listed once
        // blobs[i][b]: blob b of experts[i] in host memory (a disk expert's read for this call).
        std::span<const std::vector<const std::byte*>> blobs;
        std::span<const std::int32_t> ids; // every pair's expert, as begin() received them
        const std::byte* inputs = nullptr; // the device inputs begin() copied, in order
        std::byte* products     = nullptr; // ExpertCpuOptions::product_bytes
        CpuSpinPool* pool       = nullptr; // the service thread and its pinned workers
    };

    struct Done {
        // The bytes of the product buffer the pairs used, from its start (at most
        // ExpertCpuOptions::product_bytes): only those reach the device.
        std::size_t bytes   = 0;
        double microseconds = 0; // the products' time
    };

    // Writes the products of the pairs routed to work.experts. Runs on the delivery's service
    // thread, after the call's copies are issued.
    virtual Done compute(const Work& work) = 0;
};

struct ExpertCpuOptions {
    ExpertCpuProducts* products = nullptr; // null: the GPU copies every miss
    std::size_t input_bytes = 0, product_bytes = 0;
    double share = 0;          // the share of a narrow call's missing experts the CPU computes
    bool balance = false;      // instead, the split whose copies and products end soonest
    std::uint32_t threads = 0; // the CPU pool, the service thread included; 0: pool_cpu_count()
};

struct ExpertDeliveryOptions {
    std::uint32_t pair_capacity = 0;
    // Persistent cache only; the layer banks are separate. Both explicit and automatic cache keep
    // keep_free_bytes free after physical allocation rounding.
    std::optional<std::uint64_t> cache_bytes;
    std::uint64_t keep_free_bytes = 512ULL << 20;
    std::uint64_t staging_bytes = 128ULL << 20; // disk only, two pinned halves
    // Disk only: a drive mirror and the direct path (core/expert_disk.h).
    ExpertDiskReads disk{};
    // Whole-layer device windows that hold the experts the cache does not. Two let the next layer
    // of a wide call (a prompt chunk, a wide verification) arrive while the current one computes;
    // one halves their memory and gives that up. Two fall back to one when memory is short.
    std::uint32_t banks = 2;
    // Between wide calls the banks serve as cache slots: once two narrow passes have run without a
    // wide call, each bank holds whole slots for the narrow calls' misses the tier does not admit,
    // and the next wide call takes its bank back.
    bool lend_banks = true;
    bool suspendable = false; // existing VmmRegion allocations, fixed virtual addresses
    // The cache reserves addresses for every expert and backs only the slots it has, so its owner
    // can give slots back to memory that grows (shrink_cache) and take them again (grow_cache).
    // Needs CUDA virtual memory management.
    bool elastic = false;
    // Which experts the cache holds once it is full (core/expert_tier.h), a pool per slot size.
    ExpertTierOptions tier{};
    ExpertCpuOptions cpu{};
};
struct ExpertDeliveryStats {
    std::uint64_t requests = 0, routes = 0, hits = 0, misses = 0, evictions = 0;
    std::uint64_t pairs = 0, pair_hits = 0, admissions = 0;
    // Missing experts and their pairs the CPU computed.
    std::uint64_t cpu_experts = 0, cpu_pairs = 0;
    // Misses found already present in a layer bank (a prefetched layer, a repeated wide call).
    std::uint64_t bank_reuses = 0, prefetched_layers = 0;
    // Misses that entered a slot of a lent bank, and experts that left one (for another miss, or
    // because a wide call took the bank back).
    std::uint64_t lent_admissions = 0, lent_evictions = 0;
    // Host-to-device expert bytes (misses and prefetches); device copies from a bank into the cache.
    std::uint64_t read_bytes = 0, copied_bytes = 0, promoted_bytes = 0;
    double service_seconds = 0;
};
// Counters accumulated after `start` (startup warm-up work excluded from request statistics).
[[nodiscard]] inline ExpertDeliveryStats operator-(ExpertDeliveryStats now,
                                                   const ExpertDeliveryStats& start) noexcept {
    now.requests -= start.requests;
    now.routes -= start.routes;
    now.hits -= start.hits;
    now.misses -= start.misses;
    now.evictions -= start.evictions;
    now.pairs -= start.pairs;
    now.pair_hits -= start.pair_hits;
    now.admissions -= start.admissions;
    now.cpu_experts -= start.cpu_experts;
    now.cpu_pairs -= start.cpu_pairs;
    now.bank_reuses -= start.bank_reuses;
    now.prefetched_layers -= start.prefetched_layers;
    now.lent_admissions -= start.lent_admissions;
    now.lent_evictions -= start.lent_evictions;
    now.read_bytes -= start.read_bytes;
    now.copied_bytes -= start.copied_bytes;
    now.promoted_bytes -= start.promoted_bytes;
    now.service_seconds -= start.service_seconds;
    return now;
}
// Sum over ranks.
[[nodiscard]] inline ExpertDeliveryStats operator+(ExpertDeliveryStats a,
                                                   const ExpertDeliveryStats& b) noexcept {
    a.requests += b.requests;
    a.routes += b.routes;
    a.hits += b.hits;
    a.misses += b.misses;
    a.evictions += b.evictions;
    a.pairs += b.pairs;
    a.pair_hits += b.pair_hits;
    a.admissions += b.admissions;
    a.cpu_experts += b.cpu_experts;
    a.cpu_pairs += b.cpu_pairs;
    a.bank_reuses += b.bank_reuses;
    a.prefetched_layers += b.prefetched_layers;
    a.lent_admissions += b.lent_admissions;
    a.lent_evictions += b.lent_evictions;
    a.read_bytes += b.read_bytes;
    a.copied_bytes += b.copied_bytes;
    a.promoted_bytes += b.promoted_bytes;
    a.service_seconds += b.service_seconds;
    return a;
}
struct ExpertDeliveryMemory {
    std::uint64_t cache_bytes = 0, bank_bytes = 0, metadata_bytes = 0, pinned_bytes = 0;
    // lent_slots: the slots the lent banks hold now (ExpertDeliveryOptions::lend_banks).
    std::uint32_t cache_slots = 0, banks = 0, lent_slots = 0;
};

// One Program, rank and serial compute stream. Sources outlive this object. All device addresses
// stay fixed until destruction. The caller drains graph users before destroying their graphs.
//
// The cache's slots form pools, one for the layers whose experts need the same room (each blob's
// largest bytes and tail): every layer holds about as many slots as any other whatever its
// experts' size, and an expert enters only a slot of its own pool.
//
// Each call splits its routes into experts the persistent cache holds and the rest. A narrow call
// (decode, verification) leaves a share of its missing experts to the CPU when the family computes
// them there (ExpertCpuOptions): those cross no bus and take no slot. Its other misses enter a free
// cache slot, else the slot of the coldest expert the call does not route when the expert tier
// admits them, else a layer bank. The tier counts the narrow calls' routes; a call to a layer
// below the last narrow call's ends a pass, and every few passes a round swaps the most routed
// experts the cache lacks into the slots of the least routed ones, copied on a side stream a few
// at a time after the calls' own work, each candidate's entries moved once its copy has landed.
// A wide call (half the layer's experts or more: a prompt chunk) brings its misses into a layer
// bank, laid out like a resident bank (expert e of blob b at a fixed stride) with one copy per run
// of consecutive experts (repeated wide calls to one layer, a prompt span's chunks, fill only what
// its bank lacks), starts the bank of the other layer whose wide call followed its last wide call
// (narrow calls between prompts do not change that order), and promotes its
// most routed experts from the bank into free cache slots by device copies, up to the layer's
// share; it evicts nothing and counts nothing, so a prompt does not push out the decode working set.
// While only narrow calls run, the banks are lent to the cache (ExpertDeliveryOptions::lend_banks):
// a narrow call's miss that no cache slot takes enters a free slot of a lent bank, else the slot of
// its least routed expert the call does not route, and stays there for later calls. A round
// candidate that a narrow call routes from a lent slot moves to its cache slot only at a later call,
// so no miss of the call takes the slot its cached pairs read. A wide call treats the lent slots as
// misses and takes back the bank it fills.
class ExpertDelivery {
public:
    ExpertDelivery(const RankContext& rank, std::vector<std::filesystem::path> files,
                   std::vector<ExpertDeliveryLayer> layers, ExpertDeliveryOptions options);
    ~ExpertDelivery();
    ExpertDelivery(const ExpertDelivery&) = delete;
    ExpertDelivery& operator=(const ExpertDelivery&) = delete;

    // The call's cached and missing sets on the device: per pair, its expert or -1 when the pair
    // lies in the other set, and as lists of experts (their count, then the experts ascending).
    struct Call {
        const std::int32_t* cached_ids      = nullptr;
        const std::int32_t* missing_ids     = nullptr;
        const std::int32_t* cached_experts  = nullptr;
        const std::int32_t* missing_experts = nullptr;
        // With a CPU share, after await(): the CPU's products (ExpertCpuOptions::product_bytes),
        // valid when *cpu is nonzero.
        const void* products = nullptr;
        const std::int32_t* cpu = nullptr;
    };
    struct Input {
        const void* data = nullptr;
        std::size_t bytes = 0;
    };

    // Capturable. The next begin must follow every consumer of the preceding call on stream, and
    // every begin needs its await. Cached pairs can run immediately. Missing pairs run only after
    // await(); with a CPU share they exclude the CPU's pairs, whose products the family adds then.
    // Neither set drops or renormalizes routes; an expert's pairs all fall in one set. `inputs`
    // (device ranges, ExpertCpuOptions::input_bytes in all) reach the CPU's products; a call
    // without them leaves nothing to the CPU (call.products and call.cpu stay null).
    [[nodiscard]] Call begin(std::size_t layer, const std::int32_t* ids, std::uint32_t pairs,
                             cudaStream_t stream, std::span<const Input> inputs = {});
    void await(cudaStream_t stream);
    // Projection-major [aliases][experts], including aliases into the same blob.
    [[nodiscard]] const void* const* tables(std::size_t layer) const;
    // Whether every entry of alias `alias` in tables(layer) is a multiple of `bytes` (a power of
    // two up to 256). Cache slots, lent slots and layer banks start at 256-byte boundaries, so the
    // alias's offset and the bank stride of its blob decide.
    [[nodiscard]] bool alias_aligned(std::size_t layer, std::size_t alias,
                                     std::uint64_t bytes) const;
    void check() const;
    // Drain before suspend. Invalidate preserves allocations and graph addresses for restore.
    void drain();
    void invalidate();
    void release_device_backing(); // drain first, then release every transport device allocation
    void restore_device_backing(); // map at the same addresses and invalidate residency
    [[nodiscard]] std::uint64_t state_backing_bytes() const noexcept;
    [[nodiscard]] ExpertDeliveryStats stats() const;
    [[nodiscard]] ExpertDeliveryMemory memory() const noexcept;
    // Once warm-up is done, before the first request: empties the cache, forgets the routes the
    // tier and the totals learned, and fills the slots with the most routed (layer, expert) pairs
    // of `profile` ([layer][expert], a routing profile of these layers) over every layer.
    void prime(const std::vector<std::vector<std::uint64_t>>* profile);
    // Every route the narrow calls took through each layer's experts since prime().
    [[nodiscard]] std::vector<std::vector<std::uint64_t>> totals() const;
    // With a CPU share: its threads and vector backend, e.g. "15 pinned threads"; else empty.
    [[nodiscard]] std::string cpu_description() const;

    // An elastic cache (ExpertDeliveryOptions::elastic), after the compute stream's last call:
    // shrink_cache gives at least `bytes` of whole granules back to the device, or every slot when
    // the cache has fewer, from the pools whose layers hold the most slots each, keeping each
    // pool's most routed experts (moved into its remaining slots when they sat past them);
    // grow_cache takes free device memory, beyond keep_free_bytes, for empty slots, to the pools
    // whose layers hold the fewest. Both return the device bytes they freed or took. Graphs stay
    // valid.
    std::uint64_t shrink_cache(std::uint64_t bytes);
    std::uint64_t grow_cache();
    // The free device memory grow_cache leaves (ExpertDeliveryOptions::keep_free_bytes).
    void set_keep_free_bytes(std::uint64_t bytes);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
