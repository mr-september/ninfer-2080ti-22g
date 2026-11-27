#pragma once

// The DSpine drafter over one target model (docs/maintainer/dspine.md): a masked block of K + 1
// columns (the anchor and K masks) at a sequence's next positions through the drafter's
// sliding-window layers, causal inside the block, against a per-sequence ring of context K/V
// projected from the target's tapped hidden states; a code-bridge write after every layer; then
// the target's head over the block's final hidden, the 16 best candidates of each draft row and
// the serial refinement of the last write that chains them into drafts (ops::dspine_path).
//
// The target supplies its token rows, its head and the storage this class writes: the context
// ring (one layer view per drafter layer), and the head-write table that prepare() derives from
// the head once. Every call enqueues on the given stream and allocates only from the given
// workspace, so a CUDA graph may capture it.

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/tensor.h"
#include "models/dspine/model.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>

namespace ninfer::models::dspine {

// The target model the drafter runs over.
struct Target {
    std::uint32_t hidden = 0, vocab = 0, layers = 0;
    // Candidates come from the ids [0, domain): the public vocabulary.
    std::int32_t domain = 0;
    // The head's logits: BF16 or FP32.
    DType logits = DType::BF16;
    // Token rows as the drafter's first layer receives them: ids I32 [N] into BF16 [H, N].
    std::function<void(const Tensor& ids, Tensor& rows, cudaStream_t)> embed;
    // The target's output head without a final soft cap (the drafter needs logits linear in the
    // hidden): BF16 [H, U] into logits [vocab, U] of the type above.
    std::function<void(const Tensor& hidden, Tensor& logits, WorkspaceArena&, cudaStream_t)> head;
    // Optional: the head and its 16 best candidates of each column in one pass, without dense
    // logits: BF16 [H, U] into ids I32 [16, U] and their logits FP32 [16, U], in descending
    // order (ops::linear_topk). A shortlist head ranks only its own rows. When set, proposals
    // take their candidates here instead of from `head` and ops::candidate_topk.
    std::function<void(const Tensor& hidden, Tensor& ids, Tensor& scores, WorkspaceArena&,
                       cudaStream_t)>
        head_topk;
};

// Committed target features into the ring: row b's first counts[b] columns (positions[i, b], a
// sequential run that continues the lane's live interval) are stored in lane lanes[b]; its other
// columns are inert.
struct ContextCall {
    Tensor features;  // BF16 [taps * H, W, B]: the tapped hidden states side by side
    Tensor positions; // I32 [W, B]
    Tensor counts;    // I32 [B]
    Tensor lanes;     // I32 [B]
    // The host's promise on counts (min <= counts[b] <= max <= W). A row commits at most the
    // ring's capacity, except a single row whose W columns all commit: its last capacity
    // columns are then stored.
    std::uint32_t min_count = 0, max_count = 0;
};

// One proposal per row: the block [anchor, mask x K] at positions frontier .. frontier + K
// against the ring's committed context [frontier - window, frontier).
struct ProposalCall {
    Tensor anchors;       // I32 [B]: each row's anchor token
    Tensor frontiers;     // I32 [B]: the anchor's position
    Tensor valid_columns; // I32 [B]: live block columns 1..W
    Tensor lanes;         // I32 [B]: ring lanes
    const ops::SamplingConfig* configs = nullptr; // device [B]
    std::int32_t batch = 0, width = 0;             // W = K + 1 in [2, block_size]
    Tensor drafts;                                 // I32 [K, B]
    Tensor candidates;                             // I32 [16, K, B]
    Tensor proposal_q;                             // FP32 [16, K, B]
    Tensor confidence;                             // FP32 [K, B], or empty
};

class Drafter {
public:
    // `head_write` is the table's storage, BF16 [R, vocab] (dspine::head_write_bytes()); `window`
    // the ring capacity every call's ring views have (ring_capacity()); max_context bounds every
    // frontier, max_batch every proposal's rows (a commit takes up to eight) and max_width
    // (2..block_size) every block.
    Drafter(const Model& model, Target target, Tensor head_write, std::uint32_t window,
            std::uint32_t max_context, std::int32_t max_batch, std::int32_t max_width);

    // Derives the head-write table from the target's head (column v: the head's row v through
    // the last write, scaled by the final norm's weight). Run once, before the first propose.
    void prepare(WorkspaceArena& work, cudaStream_t stream) const;
    void materialize(const ContextCall& call, std::span<const CyclicKVCacheLayerView> ring,
                     WorkspaceArena& work, cudaStream_t stream) const;
    void propose(const ProposalCall& call, std::span<const CyclicKVCacheLayerView> ring,
                 WorkspaceArena& work, cudaStream_t stream) const;

    // Workspace of prepare() beside `head_bytes`, the head's own workspace at up to
    // prepare_columns() columns; `logits` is the head's output type.
    [[nodiscard]] static std::size_t prepare_workspace_bytes(const Model& model, DType logits,
                                                             std::int32_t max_batch,
                                                             std::int32_t max_width,
                                                             std::size_t head_bytes);
    // The head's widest call in prepare(): min(rank, (max_width - 1) * max_batch) columns.
    [[nodiscard]] static std::int32_t prepare_columns(const Model& model, std::int32_t max_batch,
                                                      std::int32_t max_width) noexcept;
    // Workspace of materialize() over calls of up to `width` columns in `batch` rows.
    [[nodiscard]] static std::size_t context_workspace_bytes(const Model& model,
                                                             std::uint32_t window,
                                                             std::int32_t width,
                                                             std::int32_t batch);
    // Workspace of propose() at widths 2..`width` over `batch` rows beside `head_bytes`, the
    // head's own workspace at up to (width - 1) * batch columns.
    [[nodiscard]] static std::size_t proposal_workspace_bytes(const Model& model, DType logits,
                                                              std::uint32_t window,
                                                              std::uint32_t max_context,
                                                              std::int32_t width,
                                                              std::int32_t batch,
                                                              std::size_t head_bytes);

    [[nodiscard]] const Config& config() const noexcept { return model_.config(); }
    [[nodiscard]] std::uint32_t window() const noexcept { return window_; }

private:
    const Model& model_;
    Target target_;
    Tensor head_write_;
    std::uint32_t window_ = 0, max_context_ = 0;
    std::int32_t max_batch_ = 0, max_width_ = 0;
};

} // namespace ninfer::models::dspine
