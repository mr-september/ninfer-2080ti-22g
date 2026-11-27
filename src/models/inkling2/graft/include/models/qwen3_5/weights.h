#pragma once

#include "core/stage_plan.h"
#include "core/weight_view.h"
#include "models/qwen3_5/config.h"
#include "ninfer/ops/linear_policy.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5 {

struct WeightId {
    std::size_t index                          = std::numeric_limits<std::size_t>::max();
    friend bool operator==(WeightId, WeightId) = default;
};

struct WeightUseId {
    WeightId parameter;
    std::size_t use_index = std::numeric_limits<std::size_t>::max();
};

struct WeightUse {
    std::string input;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    std::optional<float> activation_input_divisor;
    // BF16 sign vector of a Hadamard-rotated matrix, bound as its own device weight. The Use reads
    // its K elements from `hadamard_begin`: a tensor-parallel piece of a matrix's columns turns its
    // own whole 1024-column blocks of the input.
    std::optional<WeightId> hadamard_signs;
    std::uint64_t hadamard_begin = 0;
    // INT32 [K] input gather of a GGUF matrix stored over permuted input columns.
    std::optional<WeightId> input_columns;
};

struct BoundWeight {
    std::string name;
    std::vector<std::string> source_objects;
    WeightView view;
    std::vector<WeightUse> uses;
    // The device rank the weight is materialized on: its layer's pipeline stage, 0 unless placed.
    std::size_t rank = 0;
};

// Byte extent inside the pinned Host weight block.
struct PinnedRange {
    std::size_t offset = 0;
    std::size_t bytes  = 0;
};

// Overlay Vision residency: the pinned groups a window stages through borrowed device memory. Each
// range covers exactly its group's parents; staging_bytes is the prelude, the merger and two layer
// slots of slot_bytes, each 256-aligned.
struct VisionOverlayLayout {
    PinnedRange prelude; // patch embedding, its bias, position embedding
    std::vector<PinnedRange> layers;
    PinnedRange merger; // merger norm, fc1, fc2 and biases
    std::size_t slot_bytes    = 0;
    std::size_t staging_bytes = 0;
};

struct AttentionWeights {
    WeightId query, key, gate, value;
    WeightId query_norm, key_norm, output;
};

struct GdnWeights {
    WeightId query, key, value, z;
    WeightId a_projection, b_projection, a_log, dt_bias;
    WeightId convolution, norm, output;
};

struct DenseWeights {
    WeightId gate, up, down;
};

struct MoeWeights {
    WeightId router, shared_score;
    std::vector<DenseWeights> experts;
    DenseWeights shared;
    // Absent for a resident bank, including MTP's explicit load-time Q4 storage trade.
    std::optional<std::size_t> delivery_layer;

    // Expert parallelism, a tensor-parallel rank's share of the layer: its experts are
    // [expert_begin, expert_begin + experts.size()) of the router's, and only rank 0 holds the
    // shared expert (`shared` is unbound elsewhere). Where the ranks divide every expert's
    // intermediate columns instead, a rank holds all experts and the shared one, each with the
    // `intermediate` columns of its share (0: whole experts). Absent for a whole layer.
    struct ExpertShare {
        std::uint32_t expert_begin = 0;
        bool shared_expert         = true;
        std::uint32_t intermediate = 0;
    };

    std::optional<ExpertShare> share;
};

struct BlockWeights {
    WeightId input_norm, post_attention_norm;
    std::variant<AttentionWeights, GdnWeights> mixer;
    std::variant<DenseWeights, MoeWeights> ffn;
};

// A tensor-parallel rank's share of one layer beyond rank 0: the norms its share reads, its GDN
// heads or its attention heads (neither in an attention layer when the rank holds no KV heads),
// and its dense FFN rows or its experts with a replica of the router.
struct RankLayerWeights {
    WeightId input_norm, post_attention_norm;
    std::optional<GdnWeights> gdn;
    std::optional<AttentionWeights> attention;
    std::variant<DenseWeights, MoeWeights> ffn;
};

// A vocabulary head divided over the tensor-parallel ranks for decode: rank r holds rows
// [rows[r], rows[r + 1]) of it as shares[r], on its own device, read through uses[r].
struct HeadShareWeights {
    std::vector<WeightId> shares;
    std::vector<WeightUseId> uses;
    std::vector<std::uint64_t> rows;
};

struct TextWeights {
    WeightId token_embedding, output_head, final_norm;
    WeightUseId output_head_use;
    // Tensor parallelism: every rank's rows of the output head, which decode and verification
    // project divided; rank 0 also keeps the whole head for every other projection. Empty
    // otherwise, and with --lm-head-q4/--lm-head-q6, whose narrower head only rank 0 holds.
    HeadShareWeights output_head_shares;
    // With tensor parallelism, rank 0's share of each layer's GDN or attention heads and dense FFN
    // rows or experts.
    std::vector<BlockWeights> layers;
    // Tensor parallelism: rank_layers[r - 1][layer] is rank r's share of a layer. Empty otherwise.
    std::vector<std::vector<RankLayerWeights>> rank_layers;
    // Tensor parallelism: each rank's GDN heads, KV heads and FFN rows or experts, the same in
    // every layer. Empty otherwise.
    std::vector<TensorShare> tensor_shares;
    // Which stage's device holds each whole layer. The identity mapping for one device, which is
    // what keeps every consumer a no-op there; `--devices` adds stages.
    StagePlan stages{1};
};

struct MtpWeights {
    WeightId input_projection, embedding_norm, hidden_norm, final_norm;
    BlockWeights layer;
    WeightId token_embedding, output_head;
    WeightUseId output_head_use;
    // Tensor parallelism without a proposal head: the output head's shares, read through MTP's
    // input for the drafts. Empty otherwise.
    HeadShareWeights output_head_shares;
    // Tensor parallelism: ranks 1 to N-1's rows of the layer's dense FFN, each on its own device,
    // with rank 0's in `layer.ffn`. Empty where rank 0 holds the whole FFN: one device, a MoE FFN,
    // or stored runs that do not cut at the shares' ends.
    std::vector<DenseWeights> rank_ffn;
    // Tensor parallelism: the attention of ranks 1 to A-1, the ranks after rank 0 that hold KV
    // heads, each on its own device, with rank 0's share in `layer.mixer`; and ranks 1 to N-1's
    // copies of the layer's input and post-attention norms. Empty where rank 0 holds the whole
    // attention: one device, position-split attention, or stored runs that do not cut at the
    // shares' heads.
    std::vector<AttentionWeights> rank_attention;
    std::vector<WeightId> rank_input_norm, rank_post_attention_norm;
    // With the attention divided, rank 0's whole K and V rows besides its share: it projects a
    // prompt chunk's keys and values and sends each rank its heads'. Invalid otherwise.
    WeightId whole_key, whole_value;
};

struct NormWeights {
    WeightId weight, bias;
};

struct VisionBlockWeights {
    NormWeights norm1, norm2;
    WeightId query, key, value, query_bias, key_bias, value_bias;
    WeightId output, output_bias;
    WeightId fc1, fc1_bias, fc2, fc2_bias;
};

struct VisionWeights {
    WeightId patch_embedding, patch_embedding_bias, position_embedding;
    std::vector<VisionBlockWeights> layers;
    NormWeights merger_norm;
    WeightId merger_fc1, merger_fc1_bias, merger_fc2, merger_fc2_bias;
};

struct DraftAttentionWeights {
    WeightId query, key, value, context_key, context_value;
    WeightId query_norm, key_norm, output;
};

struct DynamicConvWeights {
    WeightId base_kernel, kernel_projection;
};

struct DraftBlockWeights {
    WeightId input_norm, post_attention_norm;
    DraftAttentionWeights attention;
    DenseWeights mlp;
    std::optional<DynamicConvWeights> attention_conv, mlp_conv;
};

struct SelectorWeights {
    WeightId hidden_projection, predecessor_codebook, successor_codebook;
};

struct DraftWeights {
    WeightId feature_projection, context_norm, final_norm;
    std::vector<DraftBlockWeights> layers;
    std::optional<SelectorWeights> selector;
    WeightId token_embedding, output_head;
    WeightUseId output_head_use;
};

struct ProposalWeights {
    WeightId head;
    std::optional<WeightId> token_ids;
    std::uint32_t rows = 0;
    std::vector<std::int32_t> global_token_ids;
    // Tensor parallelism: every rank's rows of the head, each read through its one use. Empty
    // otherwise.
    HeadShareWeights head_shares;
};

// Handles refer to the frozen model's weight array. No artifact ID lookup is needed in execution.
struct ModelWeights {
    TextWeights text;
    std::optional<VisionWeights> vision;
    std::optional<MtpWeights> mtp;
    std::optional<DraftWeights> draft;
    std::optional<ProposalWeights> proposal;
};

} // namespace ninfer::models::qwen3_5
