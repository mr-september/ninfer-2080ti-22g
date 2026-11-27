#pragma once

// Inkling 2 (Inkling2LoopForConditionalGeneration): the immutable model a v3 artifact describes.
// Matrices stay in the GGUF block formats the converter imported; vectors and the small float
// projections (sinks, dynamic convolution taps, router, depth-mix queries, loop controllers) are
// FP32. The routed experts sit on the device (divided over several with expert parallelism), in
// pinned host memory or in the artifact's files (ExpertResidency). The frontend resources are
// Inkling's tokenizer and chat template, which the shared frontend serves in its Inkling dialect.
// With LoadOptions::vision the vision tower binds, with LoadOptions::dspark the DSpark drafter,
// with LoadOptions::dspine a standalone DSpine drafter artifact loads beside it.

#include "artifact/materializer.h"
#include "core/device.h"
#include "core/expert_delivery.h"
#include "core/startup.h"
#include "models/dspine/model.h"
#include "models/inkling2/config.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/model.h"
#include "models/qwen3_5/weights.h"
#include "ninfer/ops/weight_input.h"

#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::artifact {
class Reader;
}

namespace ninfer::models::inkling2 {

using qwen3_5::BoundWeight;
using qwen3_5::FrontendResources;
using qwen3_5::InstanceInfo;
using qwen3_5::WeightId;

// A layer's attention: the projections of its query input (Q, the gate and the relative feature),
// of its key and value inputs, and the output; the sink projection [heads][hidden], Q's
// per-channel clip scale [heads][head_dim], the relative feature's [heads][16], the distance
// table [distances][16], the convolutions' taps [channels][taps]; a full-attention layer's three
// input norms and dynamic-tap projections [kv heads * taps][hidden], a sliding layer's one norm.
struct AttentionWeights {
    WeightId query, gate, relative, key, value, output;
    WeightId sinks, query_clip, relative_clip, distances, key_conv, value_conv, output_conv;
    std::optional<WeightId> query_norm, key_norm, value_norm, key_taps, value_taps; // full
    std::optional<WeightId> input_norm;                                              // sliding
};

// The router [experts][hidden] and its selection bias, the routed experts' banks when they are
// resident on the device (each expert's [gate; up] rows, then its down rows), the per-expert
// scales of the two products (layers whose GGUF stores them), and the output's convolution.
struct MoeWeights {
    WeightId router, router_bias, output_conv;
    std::optional<WeightId> gate_up, down; // device experts without expert parallelism
    // Expert parallelism: each rank's rows of the two banks (its experts), by rank.
    std::vector<WeightId> gate_up_shares, down_shares;
    std::optional<WeightId> gate_up_scale, down_scale;
};

// A loop controller of rank r: down [r][2 hidden], gain and forget [hidden][r].
struct ControllerWeights {
    WeightId down, gain, forget;
};

// A core layer's message queries [4][hidden] and its controllers of the Q, K, V and MoE inputs.
struct LoopWeights {
    WeightId queries;
    std::array<ControllerWeights, 4> controllers;
};

struct LayerWeights {
    AttentionWeights attention;
    // The depth-mix queries of the attention's inputs (1 sliding, 3 full; none in layer 0) and
    // of the MoE input.
    std::optional<WeightId> attention_mix;
    WeightId ffn_mix, ffn_norm;
    MoeWeights moe;
    std::optional<LoopWeights> loop;
};

struct TextWeights {
    WeightId token_embedding, embedding_norm, output_head, final_norm, output_mix;
    std::vector<LayerWeights> layers;
};

struct ExpertBanks {
    QType gate_up_type{}, down_type{};
    std::uint64_t gate_up_row_bytes = 0, down_row_bytes = 0;
    ExpertDeliveryLayer delivery;
};

// The vision tower: BF16 matrices, FP32 vectors and the learned rows.
struct VisionLayerWeights {
    WeightId attention_norm, attention_norm_bias, query, query_bias, key, key_bias, value,
        value_bias, output, output_bias, attention_scale, mlp_norm, mlp_norm_bias, up, up_bias,
        down, down_bias, mlp_scale;
};

struct VisionWeights {
    WeightId patch_embedding, class_embedding, class_position, positions, pre_norm, pre_norm_bias;
    WeightId projector_up, projector_out;
    std::vector<VisionLayerWeights> layers;
};

// Where the tower's matrices live: on the device, or in pinned host memory streamed through a
// window of two device blocks as the tower runs. Its vectors stay on the device either way.
enum class VisionPlacement : std::uint8_t {
    None,
    Device,
    Pinned,
};

// The DSpark drafter: the fusion weights of its feature mixes [layers + 1][taps], the norms, the
// mask rows' embedding, the Markov embedding [vocab][rank] (BF16), the recurrent head's joint
// projection [3 rank][2 rank + 2 hidden] and bias, its output [hidden][rank] and bias (FP32), and
// three blocks.
struct DSparkLayerWeights {
    WeightId context_norm, attention_norm, ffn_norm;
    WeightId query, key, value, output, context_key, context_value;
    WeightId query_norm, key_norm, context_key_norm;
    WeightId gate, up, down;
};

struct DSparkWeights {
    WeightId fuse, context_norm, final_norm, mask_embedding, markov_embed, joint, joint_bias,
        hidden_out, hidden_out_bias;
    std::vector<DSparkLayerWeights> layers;
};

struct LoadOptions {
    std::filesystem::path artifact;
    ExpertResidency experts = ExpertResidency::Device;
    // Host experts: at most this many bytes pinned, the rest read from the artifact's files
    // (--expert-host-mib); unset, as many as the host can pin.
    std::optional<std::uint64_t> expert_host_bytes;
    // Expert parallelism (`--tensor-parallel`): each device rank's relative share of every layer's
    // routed experts; empty for one device. Everything else stays on rank 0, and the experts must
    // be device resident.
    std::vector<double> expert_split;
    // Binds the artifact's `vision` component, which must then exist, in this placement.
    VisionPlacement vision = VisionPlacement::None;
    bool dspark            = false; // bind the artifact's DSpark drafter
    // Loads the standalone DSpine drafter artifact at this path (--spec dspine --draft-model) on
    // rank 0; empty for none. It excludes the DSpark drafter.
    std::filesystem::path dspine;
};

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;

    [[nodiscard]] const TextConfig& config() const noexcept { return config_; }
    [[nodiscard]] const TextWeights& weights() const noexcept { return weights_; }
    [[nodiscard]] ExpertResidency expert_residency() const noexcept { return expert_residency_; }
    // Expert parallelism: whether the routed experts are divided over the ranks, and the experts
    // [first, second) rank `rank` holds.
    [[nodiscard]] bool expert_parallel() const noexcept { return !expert_split_.empty(); }
    [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> expert_range(std::size_t rank) const;
    // Host or disk experts: one bank per stored layer, in layer order.
    [[nodiscard]] std::span<const ExpertBanks> expert_banks() const noexcept { return expert_banks_; }
    [[nodiscard]] std::uint64_t expert_pinned_bytes() const noexcept {
        std::uint64_t bytes = 0;
        for (const auto& bank : expert_banks_) {
            for (const auto& expert : bank.delivery.experts) {
                for (const auto& blob : expert) { if (blob.pinned) { bytes += blob.bytes; } }
            }
        }
        return bytes;
    }

    // Bytes of host experts the pinned block could not hold, read from the artifact's files.
    [[nodiscard]] std::uint64_t expert_file_bytes() const noexcept {
        std::uint64_t bytes = 0;
        for (const auto& bank : expert_banks_) {
            for (const auto& expert : bank.delivery.experts) {
                for (const auto& blob : expert) { if (!blob.pinned) { bytes += blob.bytes; } }
            }
        }
        return bytes;
    }
    [[nodiscard]] const std::vector<std::filesystem::path>& expert_files() const noexcept {
        return expert_files_;
    }
    [[nodiscard]] const std::optional<VisionConfig>& vision_config() const noexcept {
        return vision_config_;
    }
    [[nodiscard]] const std::optional<VisionWeights>& vision() const noexcept { return vision_; }
    [[nodiscard]] VisionPlacement vision_placement() const noexcept { return vision_placement_; }
    [[nodiscard]] const std::optional<DSparkConfig>& dspark_config() const noexcept {
        return dspark_config_;
    }
    [[nodiscard]] const std::optional<DSparkWeights>& dspark() const noexcept { return dspark_; }
    [[nodiscard]] const dspine::Model* dspine() const noexcept { return dspine_.get(); }
    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }
    [[nodiscard]] std::span<const BoundWeight> weight_data() const noexcept { return bound_; }
    [[nodiscard]] ops::WeightInput input(WeightId id) const;
    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }
    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }
    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

private:
    friend std::unique_ptr<Model> load_model(const artifact::Reader&, const LoadOptions&,
                                             DeviceContext&, const StartupObserver*);
    Model(TextConfig config, TextWeights weights, std::vector<BoundWeight> bound,
          FrontendResources resources, InstanceInfo info, artifact::MaterializedArtifact backing);

    artifact::MaterializedArtifact backing_;
    TextConfig config_;
    TextWeights weights_;
    std::vector<BoundWeight> bound_;
    FrontendResources resources_;
    InstanceInfo info_;
    ExpertResidency expert_residency_ = ExpertResidency::Device;
    std::vector<double> expert_split_;
    std::vector<ExpertBanks> expert_banks_;
    std::vector<std::filesystem::path> expert_files_;
    std::optional<VisionConfig> vision_config_;
    std::optional<VisionWeights> vision_;
    VisionPlacement vision_placement_ = VisionPlacement::None;
    std::optional<DSparkConfig> dspark_config_;
    std::optional<DSparkWeights> dspark_;
    std::unique_ptr<dspine::Model> dspine_;
};

// Whether the artifact's text component is Inkling 2, and whether it carries the vision tower and
// the DSpark drafter.
[[nodiscard]] bool is_inkling2(const artifact::Reader& reader);
[[nodiscard]] bool has_vision(const artifact::Reader& reader);
[[nodiscard]] bool has_dspark(const artifact::Reader& reader);

[[nodiscard]] std::unique_ptr<Model> load_model(const artifact::Reader& reader,
                                                const LoadOptions& options, DeviceContext& device,
                                                const StartupObserver* observer = nullptr);

// Expert parallelism's split of the routed experts when none is given: by `speed`, within each
// rank's `free_bytes` beside what the load puts on it (models::default_expert_split), a DSpine
// drafter at `dspine` (empty for none) on rank 0 included.
[[nodiscard]] std::vector<double> default_expert_split(const artifact::Reader& reader,
                                                       std::span<const std::uint64_t> free_bytes,
                                                       std::span<const double> speed,
                                                       const std::filesystem::path& dspine = {});

} // namespace ninfer::models::inkling2
