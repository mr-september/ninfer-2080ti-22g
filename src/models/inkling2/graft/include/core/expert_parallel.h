#pragma once

#include "core/device.h"
#include "core/device_buffer.h"
#include "core/stage_link.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace ninfer {

// The data path of expert parallelism. Rank 0 runs a MoE layer up to its routing; every rank holds
// a range of the layer's routed experts. For one MoE call rank 0 sends its input (whatever the
// experts read: activations, routes, weights) to every other rank, each other rank computes the
// part of the experts it holds and replies with it, and rank 0 collects the replies to add them to
// its own part.
//
// A call is `send` (rank 0's stream), then per peer `receive`, the peer's work into `part`, and
// `reply` (the peer's stream), then `collect` (rank 0's stream). Every half is a StageLink half,
// so the protocol is legal eagerly and inside one capture that spans every rank's stream: the
// peers' streams join rank 0's capture at `receive` and rejoin it at `collect`. Calls alternate
// between two slots; a call reuses a slot only after rank 0 has collected the call before it,
// which is after every peer drained that call's input.
class ExpertParallelExchange {
public:
    // `input_bytes` and `part_bytes`: the largest input and part of one call.
    ExpertParallelExchange(const DeviceContext& context, std::size_t input_bytes,
                           std::size_t part_bytes);
    ~ExpertParallelExchange();

    ExpertParallelExchange(const ExpertParallelExchange&)            = delete;
    ExpertParallelExchange& operator=(const ExpertParallelExchange&) = delete;

    [[nodiscard]] std::size_t ranks() const noexcept { return peers_.size(); }

    // Rank 0: queues `bytes` of device memory at `source` to every other rank. Returns the call's
    // slot, which the other halves of the call take.
    std::size_t send(const void* source, std::size_t bytes, cudaStream_t stream);
    // Rank `rank` (> 0): queues the receipt of the call's input; returns where it lands, device
    // memory of that rank.
    void* receive(std::size_t rank, std::size_t bytes, std::size_t slot, cudaStream_t stream);
    // Rank `rank` (> 0): its part buffer (device memory of that rank, `part_bytes`), and the reply
    // of `bytes` of it.
    [[nodiscard]] void* part(std::size_t rank) const;
    void reply(std::size_t rank, std::size_t bytes, std::size_t slot, cudaStream_t stream);
    // Rank 0: queues the receipt of every reply; out[r - 1] is rank r's part, device memory of
    // rank 0.
    [[nodiscard]] std::vector<void*> collect(std::size_t bytes, std::size_t slot,
                                             cudaStream_t stream);
    // The same with a size per peer: bytes[r - 1] from rank r.
    [[nodiscard]] std::vector<void*> collect(std::span<const std::size_t> bytes, std::size_t slot,
                                             cudaStream_t stream);

    // Device bytes the exchange holds on `rank`.
    [[nodiscard]] std::size_t device_bytes(std::size_t rank) const;

private:
    struct Peer {
        std::unique_ptr<StageLink> down, up; // rank 0 to the peer, and back
        DeviceBuffer input, part;            // on the peer
        DeviceBuffer received;               // on rank 0
    };

    std::size_t input_bytes_ = 0, part_bytes_ = 0;
    std::vector<Peer> peers_; // by rank; rank 0's is empty
    std::size_t next_slot_ = 0;
};

} // namespace ninfer
