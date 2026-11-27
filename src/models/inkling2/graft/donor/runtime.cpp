#include "models/dspine/runtime.h"

#include "core/device_launch.h"
#include "core/device_wait.h"
#include "core/layout.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scalar.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::dspine {
namespace {

constexpr std::size_t kAlign   = 256;
constexpr std::size_t kMaxRows = 8;

std::size_t aligned(std::size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::invalid_argument("DSpine runtime: " + message); }
}

// Operand regions, each 256-aligned: a proposal's inputs (anchors, frontiers, live columns,
// lanes; kMaxRows each), a commit's count and lane, kMaxRows ones (commit_rows' counts); drafts
// [K, B], candidates and their probabilities [16, K, B], confidences [K, B]; B greedy sampling
// configs.
struct OperandLayout {
    std::size_t inputs = 0, count = 0, lane = 0, ones = 0, drafts = 0, candidates = 0, q = 0,
                confidence = 0, configs = 0, bytes = 0;
};

OperandLayout operand_layout(std::uint32_t k, std::uint32_t rows) {
    const std::size_t cells = std::size_t(k) * rows;
    OperandLayout o;
    o.inputs     = 0;
    o.count      = kAlign;
    o.lane       = 2 * kAlign;
    o.ones       = 3 * kAlign;
    o.drafts     = 4 * kAlign;
    o.candidates = o.drafts + aligned(cells * 4);
    o.q          = o.candidates + aligned(16 * cells * 4);
    o.confidence = o.q + aligned(16 * cells * 4);
    o.configs    = o.confidence + aligned(cells * 4);
    o.bytes      = o.configs + aligned(rows * sizeof(ops::SamplingConfig));
    return o;
}

// Pinned staging: a proposal's inputs (4 x kMaxRows I32), then its drafts and confidences.
std::size_t staging_bytes(std::uint32_t k, std::uint32_t rows) {
    return 4 * kMaxRows * 4 + 2 * std::size_t(k) * rows * 4;
}

// The device allocations of a runtime with these options.
struct Sizes {
    std::uint32_t window = 0, lanes = 0;
    std::size_t ring = 0, table = 0, features = 0, operands = 0, workspace = 0;
};

Sizes sizes(const Model& model, DType logits, const RuntimeOptions& o) {
    const Config& c = model.config();
    require(o.sequences >= 1 && o.draft_tokens >= 1 && o.draft_tokens + 1 <= c.block_size &&
                o.max_context >= 1 && o.feature_columns >= 1 && o.draft_rows >= 1 &&
                o.draft_rows <= kMaxRows && o.commit_rows >= 1 && o.commit_rows <= kMaxRows &&
                o.commit_rows <= o.feature_columns && o.piece_columns >= 1,
            "one sequence, 1..block_size - 1 drafts, a context, 1..8 rows a proposal and a "
            "commit, and a feature column a committed row and a column a piece at least");
    Sizes s;
    s.window = ring_capacity(c, o.max_context);
    s.lanes  = o.sequences * (1 + o.kept_states);
    LayoutBuilder builder;
    (void)plan_cyclic_kv_cache(builder, c.num_hidden_layers, s.window,
                               static_cast<std::int32_t>(c.num_key_value_heads),
                               static_cast<std::int32_t>(c.head_dim),
                               static_cast<std::int32_t>(s.lanes));
    s.ring     = builder.finish(kAlign, "DSpine context ring");
    s.table    = aligned(head_write_bytes(c));
    s.features = aligned(std::size_t(c.taps()) * c.hidden_size * o.feature_columns * 2);
    s.operands = operand_layout(o.draft_tokens, o.draft_rows).bytes;
    // A store or a commit goes in pieces of at most piece_columns columns, a commit within the
    // window.
    const std::uint32_t piece = std::min({o.piece_columns, o.feature_columns, s.window});
    const auto width   = static_cast<std::int32_t>(o.draft_tokens + 1);
    const auto rows    = static_cast<std::int32_t>(o.draft_rows);
    const std::size_t store =
        aligned(std::size_t(c.hidden_size) * std::min(o.piece_columns, o.feature_columns) * 2);
    const std::size_t commit =
        std::max(aligned(std::size_t(piece) * 4) +
                     Drafter::context_workspace_bytes(model, s.window,
                                                      static_cast<std::int32_t>(piece), 1),
                 Drafter::context_workspace_bytes(model, s.window, 1,
                                                  static_cast<std::int32_t>(o.commit_rows)));
    s.workspace = std::max({store, commit,
                            Drafter::proposal_workspace_bytes(model, logits, s.window,
                                                              o.max_context, width, rows,
                                                              o.head_bytes),
                            Drafter::prepare_workspace_bytes(model, logits, rows, width,
                                                             o.head_bytes)}) +
                  (1U << 20);
    return s;
}

} // namespace

Runtime::Runtime(const Model& model, Target target, RuntimeOptions options)
    : model_(model), options_(options) {
    const Config& c = model_.config();
    const Sizes s   = sizes(model_, target.logits, options_);
    window_         = s.window;
    tap_of_.assign(c.num_target_layers, -1);
    for (std::size_t t = 0; t < c.target_layer_ids.size(); ++t) {
        tap_of_.at(c.target_layer_ids[t]) = static_cast<int>(t);
    }
    LayoutBuilder builder;
    const CyclicKVCacheLayout layout = plan_cyclic_kv_cache(
        builder, c.num_hidden_layers, window_, static_cast<std::int32_t>(c.num_key_value_heads),
        static_cast<std::int32_t>(c.head_dim), static_cast<std::int32_t>(s.lanes));
    ring_backing_ = DeviceBuffer(builder.finish(kAlign, "DSpine context ring"));
    // Finite values in every slot before a lane's positions reach it.
    ring_backing_.fill(0);
    ring_ = std::make_unique<CyclicKVCache>(DeviceSpan{ring_backing_.p, ring_backing_.bytes},
                                            layout);
    for (std::uint32_t l = 0; l < ring_->layer_count(); ++l) {
        views_.push_back(ring_->layer_view(l));
    }
    table_    = DeviceBuffer(s.table);
    features_ = DeviceBuffer(s.features);
    operands_ = DeviceBuffer(s.operands);
    operands_.fill(0);
    const OperandLayout o = operand_layout(options_.draft_tokens, options_.draft_rows);
    std::array<std::int32_t, kMaxRows> ones{};
    ones.fill(1);
    operands_.copy_from_host(ones.data(), sizeof(ones), o.ones);
    const std::vector<ops::SamplingConfig> greedy(options_.draft_rows);
    operands_.copy_from_host(greedy.data(), greedy.size() * sizeof(ops::SamplingConfig),
                             o.configs);
    workspace_backing_ = DeviceBuffer(s.workspace);
    work_    = std::make_unique<WorkspaceArena>(
        DeviceSpan{workspace_backing_.p, workspace_backing_.bytes});
    staging_ = std::make_unique<PinnedHostBuffer>(
        staging_bytes(options_.draft_tokens, options_.draft_rows));
    drafter_ = std::make_unique<Drafter>(
        model_, std::move(target),
        Tensor(table_.p, DType::BF16,
               {static_cast<std::int32_t>(c.rank), static_cast<std::int32_t>(c.vocab_size)}),
        window_, options_.max_context, static_cast<std::int32_t>(options_.draft_rows),
        static_cast<std::int32_t>(options_.draft_tokens + 1));
}

Runtime::~Runtime() = default;

int Runtime::tap_of(std::uint32_t layer) const noexcept {
    return layer < tap_of_.size() ? tap_of_[layer] : -1;
}

std::uint32_t Runtime::lane(std::uint32_t sequence, std::uint32_t state) const {
    if (sequence >= options_.sequences || state > options_.kept_states) {
        throw std::out_of_range("DSpine runtime: no such lane");
    }
    return state * options_.sequences + sequence;
}

std::uint64_t Runtime::device_bytes() const noexcept {
    return ring_backing_.bytes + table_.bytes + features_.bytes + operands_.bytes +
           workspace_backing_.bytes;
}

std::uint64_t Runtime::planned_bytes(const Model& model, DType logits,
                                     const RuntimeOptions& options) {
    const Sizes s = sizes(model, logits, options);
    return s.ring + s.table + s.features + s.operands + s.workspace;
}

void Runtime::prepare(cudaStream_t stream) {
    work_->reset();
    drafter_->prepare(*work_, stream);
    work_->reset();
}

void Runtime::store_tap(std::uint32_t tap, const Tensor& rows, std::uint32_t column,
                        cudaStream_t stream) {
    const Config& c       = config();
    const std::int32_t h  = static_cast<std::int32_t>(c.hidden_size);
    const std::int32_t n  = rows.ne[1];
    require(tap < c.taps() && rows.ne[0] == h && n >= 1 && rows.ne[2] == 1 && rows.ne[3] == 1 &&
                (rows.dtype == DType::FP32 || rows.dtype == DType::BF16) &&
                std::uint64_t(column) + std::uint64_t(n) <= options_.feature_columns,
            "a tap is FP32 or BF16 [hidden, n] within the feature columns");
    const std::size_t row          = std::size_t(h) * 2;
    const std::size_t column_bytes = row * c.taps();
    const auto piece = static_cast<std::int32_t>(options_.piece_columns);
    for (std::int32_t done = 0; done < n; done += piece) {
        const std::int32_t m = std::min(piece, n - done);
        work_->reset();
        Tensor source = rows.slice(1, done, m);
        if (rows.dtype == DType::FP32) {
            source = work_->alloc(DType::BF16, {h, m});
            ops::cast_fp32_to_bf16(rows.slice(1, done, m), source, stream);
        }
        auto* destination = static_cast<std::uint8_t*>(features_.p) +
                            (std::size_t(column) + std::size_t(done)) * column_bytes +
                            std::size_t(tap) * row;
        CUDA_CHECK(cudaMemcpy2DAsync(destination, column_bytes, source.data, row, row,
                                     std::size_t(m), cudaMemcpyDeviceToDevice, stream));
    }
    work_->reset();
}

void Runtime::commit(std::uint32_t lane, std::uint32_t position, std::uint32_t column,
                     std::uint32_t count, cudaStream_t stream) {
    if (count == 0) { return; }
    const Config& c = config();
    require(lane < std::uint32_t(ring_->lane_capacity()) &&
                std::uint64_t(column) + count <= options_.feature_columns &&
                std::uint64_t(position) + count <= options_.max_context,
            "a commit names a lane, feature columns and positions in range");
    const OperandLayout o = operand_layout(options_.draft_tokens, options_.draft_rows);
    const auto rows       = static_cast<std::int32_t>(std::size_t(c.taps()) * c.hidden_size);
    if (count > window_) {
        // No later block sees a position before the last window.
        column += count - window_;
        position += count - window_;
        count = window_;
    }
    auto* operands = static_cast<std::uint8_t*>(operands_.p);
    Tensor counts(operands + o.count, DType::I32, {1});
    Tensor lanes(operands + o.lane, DType::I32, {1});
    ops::set_i32_scalar(lanes, static_cast<std::int32_t>(lane), stream);
    const std::uint32_t piece = std::min(options_.piece_columns, window_);
    for (std::uint32_t done = 0; done < count; done += piece) {
        const std::uint32_t m = std::min(piece, count - done);
        const auto n          = static_cast<std::int32_t>(m);
        work_->reset();
        Tensor positions = work_->alloc(DType::I32, {n});
        ops::fill_i32_positions(positions, static_cast<std::int32_t>(position + done), stream);
        ops::set_i32_scalar(counts, n, stream);
        const Tensor features(static_cast<std::uint8_t*>(features_.p) +
                                  (std::size_t(column) + done) * std::size_t(rows) * 2,
                              DType::BF16, {rows, n, 1});
        drafter_->materialize({.features  = features,
                               .positions = positions.view({n, 1}),
                               .counts    = counts,
                               .lanes     = lanes,
                               .min_count = m,
                               .max_count = m},
                              views_, *work_, stream);
    }
    work_->reset();
}

void Runtime::commit_rows(const Tensor& lanes, const Tensor& positions, std::uint32_t column,
                          cudaStream_t stream) {
    const Config& c = config();
    const std::int32_t n = lanes.ne[0];
    require(lanes.dtype == DType::I32 && positions.dtype == DType::I32 && n >= 1 &&
                std::uint32_t(n) <= options_.commit_rows && positions.ne[0] == n &&
                std::uint64_t(column) + std::uint64_t(n) <= options_.feature_columns,
            "a row commit names up to commit_rows lanes and positions within the features");
    const OperandLayout o = operand_layout(options_.draft_tokens, options_.draft_rows);
    const auto rows       = static_cast<std::int32_t>(std::size_t(c.taps()) * c.hidden_size);
    const Tensor features(static_cast<std::uint8_t*>(features_.p) +
                              std::size_t(column) * std::size_t(rows) * 2,
                          DType::BF16, {rows, 1, n});
    const Tensor counts(static_cast<std::uint8_t*>(operands_.p) + o.ones, DType::I32, {n});
    work_->reset();
    drafter_->materialize({.features  = features,
                           .positions = Tensor(positions.data, DType::I32, {1, n}),
                           .counts    = counts,
                           .lanes     = Tensor(lanes.data, DType::I32, {n}),
                           .min_count = 1,
                           .max_count = 1},
                          views_, *work_, stream);
    work_->reset();
}

void Runtime::draft(std::span<const std::uint32_t> lanes, std::span<const std::int32_t> anchors,
                    std::span<const std::uint32_t> positions, std::uint32_t k,
                    std::span<std::int32_t> drafts, std::span<float> confidence,
                    cudaStream_t stream) {
    const std::size_t b = lanes.size();
    require(b >= 1 && b <= options_.draft_rows && anchors.size() == b && positions.size() == b &&
                k >= 1 && k <= options_.draft_tokens && drafts.size() == b * k &&
                confidence.size() == b * k,
            "a draft names 1..draft_rows rows of 1..draft_tokens drafts");
    for (std::size_t j = 0; j < b; ++j) {
        require(lanes[j] < std::uint32_t(ring_->lane_capacity()) && anchors[j] >= 0 &&
                    anchors[j] < std::int32_t(config().vocab_size) &&
                    std::uint64_t(positions[j]) + k <= options_.max_context,
                "a row names a lane, an anchor and a position in range");
    }
    const OperandLayout o = operand_layout(options_.draft_tokens, options_.draft_rows);
    auto* host            = static_cast<std::int32_t*>(staging_->data());
    for (std::size_t j = 0; j < b; ++j) {
        host[j]                = anchors[j];
        host[kMaxRows + j]     = static_cast<std::int32_t>(positions[j]);
        host[2 * kMaxRows + j] = static_cast<std::int32_t>(k + 1);
        host[3 * kMaxRows + j] = static_cast<std::int32_t>(lanes[j]);
    }
    auto* operands = static_cast<std::uint8_t*>(operands_.p);
    CUDA_CHECK(cudaMemcpyAsync(operands + o.inputs, host, 4 * kMaxRows * 4,
                               cudaMemcpyHostToDevice, stream));
    const auto rows  = static_cast<std::int32_t>(b);
    const auto steps = static_cast<std::int32_t>(k);
    auto* inputs     = reinterpret_cast<std::int32_t*>(operands + o.inputs);
    const Tensor anchor_ids(inputs, DType::I32, {rows});
    const Tensor frontiers(inputs + kMaxRows, DType::I32, {rows});
    const Tensor valid(inputs + 2 * kMaxRows, DType::I32, {rows});
    const Tensor lane_ids(inputs + 3 * kMaxRows, DType::I32, {rows});
    Tensor out_drafts(operands + o.drafts, DType::I32, {steps, rows});
    Tensor candidates(operands + o.candidates, DType::I32, {16, steps, rows});
    Tensor q(operands + o.q, DType::FP32, {16, steps, rows});
    Tensor out_confidence(operands + o.confidence, DType::FP32, {steps, rows});
    work_->reset();
    drafter_->propose({.anchors       = anchor_ids,
                       .frontiers     = frontiers,
                       .valid_columns = valid,
                       .lanes         = lane_ids,
                       .configs = reinterpret_cast<const ops::SamplingConfig*>(operands + o.configs),
                       .batch      = rows,
                       .width      = steps + 1,
                       .drafts     = out_drafts,
                       .candidates = candidates,
                       .proposal_q = q,
                       .confidence = out_confidence},
                      views_, *work_, stream);
    work_->reset();
    auto* back = host + 4 * kMaxRows;
    CUDA_CHECK(cudaMemcpyAsync(back, out_drafts.data, b * k * 4, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(back + b * k, out_confidence.data, b * k * 4,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(ninfer::wait_for_stream(stream));
    std::memcpy(drafts.data(), back, b * k * 4);
    std::memcpy(confidence.data(), back + b * k, b * k * 4);
}

void Runtime::copy_lane(std::uint32_t from, std::uint32_t to, cudaStream_t stream) {
    if (from == to) { return; }
    require(from < std::uint32_t(ring_->lane_capacity()) &&
                to < std::uint32_t(ring_->lane_capacity()),
            "a lane copy names two lanes");
    const CyclicKVCacheSlotView source = ring_->slot_view(static_cast<std::int32_t>(from));
    const CyclicKVCacheSlotView target = ring_->slot_view(static_cast<std::int32_t>(to));
    for (std::uint32_t layer = 0; layer < source.layers; ++layer) {
        const auto* k = static_cast<const std::uint8_t*>(source.k_layer0.data) +
                        std::ptrdiff_t(layer) * source.k_layer_pitch_bytes;
        const auto* v = static_cast<const std::uint8_t*>(source.v_layer0.data) +
                        std::ptrdiff_t(layer) * source.v_layer_pitch_bytes;
        auto* dk = static_cast<std::uint8_t*>(target.k_layer0.data) +
                   std::ptrdiff_t(layer) * target.k_layer_pitch_bytes;
        auto* dv = static_cast<std::uint8_t*>(target.v_layer0.data) +
                   std::ptrdiff_t(layer) * target.v_layer_pitch_bytes;
        CUDA_CHECK(cudaMemcpyAsync(dk, k, source.k_layer_bytes, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dv, v, source.v_layer_bytes, cudaMemcpyDeviceToDevice, stream));
    }
}

std::uint64_t Runtime::image_bytes(std::uint32_t position) const noexcept {
    const std::uint64_t used = std::min<std::uint64_t>(position, window_);
    return used * std::uint64_t(ring_->head_dim()) * 2 * std::uint64_t(ring_->num_kv_heads()) *
           2 * ring_->layer_count();
}

} // namespace ninfer::models::dspine
