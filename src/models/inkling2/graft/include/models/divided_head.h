#pragma once

#include "core/device.h"
#include "core/device_buffer.h"
#include "core/expert_parallel.h"
#include "core/weight.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models {

// A GGUF vocabulary head divided over the ranks of an expert-parallel model, whose other ranks
// would otherwise wait while rank 0 projects it. Rank r projects rows [rows[r], rows[r + 1]) from a
// piece of the head it holds (rank 0 a view of its whole head, which prompt chunks keep using), and
// the other ranks return their logits to rank 0 over an exchange of their own. A call's input (the
// final-normed columns in whatever form the family's products read) reaches every rank, each rank
// projects its rows of every column at once, and rank 0 places the rows into the call's logits.
// Each row is the product one device computes for it, so the logits are one device's, byte for
// byte, for a family whose products do not depend on the head's row count.
class DividedHead {
public:
    // Rank `rank` projects `columns` columns of its piece's rows from `input` (the call's input on
    // that rank) into BF16 [columns][piece.n] at `out`, on `stream`; `plane` is FP32 scratch of
    // columns * piece.n values on that rank.
    using Project = std::function<void(std::size_t rank, const void* input, const Weight& piece,
                                       int columns, float* plane, __nv_bfloat16* out,
                                       cudaStream_t stream)>;

    // `head`: rank 0's head (`vocab` rows of `row_bytes` bytes); its rows are cut in proportion
    // to `speed` at multiples of 256. `input_bytes` and `columns`: a call's largest input and most
    // columns. Copies each other rank's piece from rank 0 and allocates every rank's buffers.
    DividedHead(const DeviceContext& device, const Weight& head, std::int64_t row_bytes,
                std::span<const double> speed, std::size_t input_bytes, int columns);
    ~DividedHead();

    DividedHead(const DividedHead&)            = delete;
    DividedHead& operator=(const DividedHead&) = delete;

    // The head is divided only where it holds more than one 256-row unit per rank and has no
    // input gather.
    [[nodiscard]] static bool divides(const Weight& head, std::size_t ranks);

    // One call: `input` (`input_bytes` of device memory on rank 0) to the logits of `columns`
    // columns, BF16 [columns][vocab] at `logits` on rank 0. `streams[r]` is rank r's stream
    // (`streams[0]` rank 0's, on which the call starts and ends); `project` runs with its rank's
    // device current, rank 0's being the caller's.
    void project(const void* input, std::size_t input_bytes, int columns, __nv_bfloat16* logits,
                 std::span<const cudaStream_t> streams, const Project& project);

    [[nodiscard]] std::span<const std::uint64_t> rows() const noexcept { return rows_; }
    // Device bytes the division adds on `rank`: its piece (none on rank 0) and buffers.
    [[nodiscard]] std::size_t device_bytes(std::size_t rank) const;

private:
    struct Rank {
        Weight piece;
        DeviceBuffer storage; // the piece's rows, on ranks after the first
        DeviceBuffer plane;   // FP32 [columns][rows]
        DeviceBuffer own;     // rank 0: its BF16 [columns][rows]
    };

    std::vector<int> devices_; // by rank
    std::vector<std::uint64_t> rows_;
    std::vector<Rank> ranks_;
    std::unique_ptr<ExpertParallelExchange> exchange_;
    int columns_ = 0;
    std::size_t input_bytes_ = 0;
};

} // namespace ninfer::models
