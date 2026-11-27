#pragma once

// A standalone DSpine drafter artifact materialized on its target's head device: the drafter's
// layers and code bridge as device operands. The head-dependent table and the context ring belong
// to the Drafter a target builds over this Model (models/dspine/drafter.h).

#include "artifact/materializer.h"
#include "core/device.h"
#include "core/startup.h"
#include "core/tensor.h"
#include "models/dspine/config.h"
#include "models/qwen3_5/weights.h"
#include "ninfer/ops/weight_input.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::artifact {
class Reader;
}

namespace ninfer::models::dspine {

struct LayerOperands {
    Tensor input_norm, post_attention_norm; // BF16 [H]
    Tensor query_norm, key_norm;            // BF16 [D]
    ops::SingleProjectionWeight query, key, value, output;
    // The context's K/V: the key and value matrices through the context's input.
    ops::SingleProjectionWeight context_key, context_value;
    ops::SingleProjectionWeight gate, up, down;
    // The query, key and value rows as one parent [Q + 2K, H], and the gate and up rows as one
    // [2I, H], where the artifact stores each group in one parent: the fused routes read them.
    std::optional<ops::SingleProjectionWeight> query_key_value, gate_up;
    // The bridge after this layer: its read [C, H] and its write [H, R].
    ops::SingleProjectionWeight read, write;
};

struct Operands {
    Tensor context_norm, final_norm; // BF16 [H]
    ops::SingleProjectionWeight feature_projection; // [H, taps * H]
    ops::SingleProjectionWeight bridge_projection;  // [3R, C]: gate, predecessor gate, value
    Tensor gate_bias;                                // BF16 [R]
    Tensor head_write_columns;                       // BF16 [H, R]: column r is the last write's
    Tensor sender_table;                             // BF16 [2R, V]
    std::vector<LayerOperands> layers;
};

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;

    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] const Operands& operands() const noexcept { return operands_; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
    // The artifact's identity, which a stored image of the drafter's ring depends on.
    [[nodiscard]] std::string artifact_hex() const;
    // Device bytes of the materialized weights.
    [[nodiscard]] std::uint64_t device_bytes() const noexcept { return device_bytes_; }

    // Model suspend, as the target's weights take it (artifact::MaterializedArtifact): the
    // drafter's arena gives its memory back and later receives the same bytes at the same
    // addresses. Only a load with MaterializeOptions::suspendable supports these.
    [[nodiscard]] artifact::MaterializedArtifact& backing() noexcept { return backing_; }
    [[nodiscard]] const artifact::MaterializedArtifact& backing() const noexcept {
        return backing_;
    }

private:
    friend std::unique_ptr<Model> load_model(const std::filesystem::path&, DeviceContext&,
                                             const StartupObserver*,
                                             const artifact::MaterializeOptions&);
    Model() = default;

    std::shared_ptr<const artifact::Reader> reader_;
    artifact::MaterializedArtifact backing_;
    std::vector<qwen3_5::BoundWeight> bound_;
    Config config_;
    Operands operands_;
    std::string name_;
    std::filesystem::path path_;
    std::uint64_t device_bytes_ = 0;
};

// Materializes the drafter at `path` on rank 0 of `device` (the target's head device). A
// suspendable load keeps the artifact's files open to upload the weights again on resume;
// `materialize.retained_reader` is ignored.
[[nodiscard]] std::unique_ptr<Model> load_model(const std::filesystem::path& path,
                                                DeviceContext& device,
                                                const StartupObserver* observer = nullptr,
                                                const artifact::MaterializeOptions& materialize =
                                                    {});

} // namespace ninfer::models::dspine
