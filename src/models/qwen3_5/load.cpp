#include "core/platform.h"
#include "models/qwen3_5/load.h"

#include "artifact/reader.h"
#include "models/qwen3_5/load/bindings.h"

#include <utility>

namespace ninfer::models::qwen3_5 {

struct LoadPlan::Impl {
    Config config;
    LoadOptions options;
    ModelWeights weights;
    std::vector<PendingWeight> pending;
    const artifact::Reader* reader = nullptr;
    artifact::PreparedResources prepared_resources;
    std::vector<moe::BankDescription> expert_banks;
    qwen::FrontendResources resources;
    InstanceInfo info;
    std::vector<WeightParent> geometry_parents;
    std::vector<WeightView> geometry_views;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const LoadOptions& LoadPlan::options() const { return impl_->options; }

ops::WeightInput LoadPlan::input(WeightUseId id) const {
    const auto& use = impl_->pending.at(id.parameter.index).uses.at(id.use_index);
    return {impl_->geometry_views.at(id.parameter.index), use.policy, use.activation_input_divisor};
}

ops::WeightInput LoadPlan::input(WeightId id) const {
    if (impl_->pending.at(id.index).uses.size() != 1) {
        throw artifact::ArtifactError("weight description requires an explicit Use");
    }
    return input(WeightUseId{id, 0});
}

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

const qwen::FrontendResources& LoadPlan::resources() const { return impl_->resources; }

std::span<const moe::BankDescription> LoadPlan::expert_banks() const { return impl_->expert_banks; }

const artifact::ParameterReference& LoadPlan::parameter(WeightId id) const {
    return impl_->pending.at(id.index).reference;
}

std::span<const WeightUse> LoadPlan::uses(WeightId id) const {
    return impl_->pending.at(id.index).uses;
}

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->reader  = &reader;
    out->options = options;
    out->config  = parse_config(reader.directory(), options);
    artifact::Binder binder(reader);
    out->resources = loading::bind_resources(binder, out->config);
    Bindings bindings(binder);
    const auto& text  = out->config.text;
    out->weights.text = loading::bind_text(bindings, text, options);
    if (out->config.vision) {
        out->weights.vision = qwen::bind_vision(bindings, *out->config.vision, text.hidden_size);
    }
    if (out->config.mtp) {
        out->weights.mtp = loading::bind_mtp(bindings, text, out->weights.text);
    }
    if (out->config.draft) {
        out->weights.draft =
            loading::bind_draft(bindings, *out->config.draft, text, out->weights.text,
                                std::string(options.speculative_component()));
    }
    if (options.proposal_enabled()) {
        const auto& proposal = reader.directory().component("text").proposal;
        if (!proposal) {
            throw artifact::ArtifactError("selected proposal head is absent from artifact");
        }
        out->weights.proposal = loading::bind_proposal(bindings, *proposal, text, options,
                                                       out->resources.public_token_count);
        if (out->config.draft && out->config.draft->dflash2) {
            const auto domain =
                proposal->indexed ? proposal->rows : out->resources.public_token_count;
            if (out->config.draft->dflash2->selector_top_k > domain) {
                throw artifact::ArtifactError(
                    "proposal domain is smaller than DFlash2 selector_top_k");
            }
        }
        if (out->weights.mtp) { out->weights.mtp->output_head = out->weights.proposal->head; }
        if (out->weights.draft) { out->weights.draft->output_head = out->weights.proposal->head; }
    }
    out->weights.text.output_head_use =
        bindings.use(out->weights.text.output_head, "text/final_hidden");
    if (out->weights.mtp) {
        out->weights.mtp->output_head_use =
            bindings.use(out->weights.mtp->output_head, "mtp/final_hidden");
    }
    if (out->weights.draft) {
        out->weights.draft->output_head_use =
            bindings.use(out->weights.draft->output_head,
                         std::string(options.speculative_component()) + "/final_hidden");
    }
    out->pending            = std::move(bindings.weights);
    out->expert_banks       = describe_expert_weights(reader, out->weights, out->pending);
    out->prepared_resources = std::move(binder).finish();
    out->info.name          = reader.directory().metadata.value(
        "name", std::string(architecture_name(text.architecture)));
    out->geometry_parents.resize(reader.directory().objects.size());
    for (const auto& pending : out->pending) {
        WeightView view{pending.reference.shape, {}};
        for (const auto& part : pending.reference.binding.parts) {
            auto& parent = out->geometry_parents.at(part.object.index);
            if (!parent.geometry.elements) parent.geometry = reader.geometry(part.object);
            view.parts.push_back({&parent, part.begin, part.end});
        }
        out->geometry_views.push_back(std::move(view));
    }
    return LoadPlan(std::move(out));
}

ResolvedLoadPlan::ResolvedLoadPlan(LoadPlan&& logical,
                                   artifact::MaterializationPlan&& materialization,
                                   std::vector<Device>&& devices,
                                   std::vector<Device>&& bank_devices)
    : materialization_(std::move(materialization)), logical_(std::move(logical)),
      devices_(std::move(devices)), bank_devices_(std::move(bank_devices)) {}

namespace {
struct Placement {
    artifact::MaterializationPlan storage;
    std::vector<Device> weights, banks;
};

template <class Data>
Placement describe_placement(const Data& data, Device cuda, Device expert_device) {
    if (cuda.type() != DeviceType::CUDA ||
        (expert_device != cuda && expert_device != Device::cpu())) {
        throw artifact::ArtifactError("weight placement requires the selected GPU or CPU");
    }
    Placement out;
    out.weights.assign(data.pending.size(), cuda);
    out.banks.assign(data.expert_banks.size(), cuda);
    for (const auto& layer : data.weights.text.layers)
        if (const auto* parameters = std::get_if<MoeWeights>(&layer.ffn))
            out.banks[parameters->bank.index] = expert_device;
    const auto place = [&](const BlockWeights& layer) {
        if (const auto* moe = std::get_if<MoeWeights>(&layer.ffn)) {
            for (const auto& expert : moe->experts) {
                for (const auto id : {expert.gate, expert.up, expert.down}) {
                    out.weights[id.index] = out.banks[moe->bank.index];
                }
            }
        }
    };
    for (const auto& layer : data.weights.text.layers) place(layer);
    if (data.weights.mtp) place(data.weights.mtp->layer);
    std::vector<artifact::PlacementRequest> requests;
    for (std::size_t i = 0; i < data.pending.size(); ++i) {
        const bool in_place =
            NINFER_HOST_MAPPED_EMBEDDING != 0 && i == data.weights.text.token_embedding.index;
        for (const auto& part : data.pending[i].reference.binding.parts) {
            requests.push_back({part.object, out.weights[i],
                                out.weights[i] == Device::cpu() ? artifact::HostAccess::CudaTransfer
                                : in_place                      ? artifact::HostAccess::CudaInPlace
                                                                : artifact::HostAccess::CpuOnly});
        }
    }
    out.storage = artifact::plan_materialization(*data.reader, requests);
    return out;
}
} // namespace

artifact::MaterializationPlan LoadPlan::placement(Device cuda, Device main_experts) const {
    return describe_placement(*impl_, cuda, main_experts).storage;
}

ResolvedLoadPlan resolve_load(LoadPlan&& plan, Device cuda, std::optional<Device> main_experts) {
    if (!plan.impl_) throw artifact::ArtifactError("load plan was already consumed");
    auto placement = describe_placement(*plan.impl_, cuda, main_experts.value_or(cuda));
    placement.storage.resources = std::move(plan.impl_->prepared_resources);
    return ResolvedLoadPlan(std::move(plan), std::move(placement.storage),
                            std::move(placement.weights), std::move(placement.banks));
}

std::unique_ptr<Model> materialize_model(ResolvedLoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer) {
    if (!plan.logical_.impl_) { throw artifact::ArtifactError("load plan was already consumed"); }
    auto data    = std::move(plan.logical_.impl_);
    auto backing = artifact::materialize(*data->reader, std::move(plan.materialization_),
                                         device.transfer_view(), observer);
    auto bound   = resolve_weights(std::move(data->pending), backing, plan.devices_);
    auto banks   = moe::bind_banks(data->expert_banks, backing, plan.bank_devices_);
    return std::unique_ptr<Model>(new Model(
        std::move(data->config), data->options, std::move(data->weights), std::move(bound),
        std::move(banks), std::move(data->resources), std::move(data->info), std::move(backing)));
}

std::unique_ptr<Model> load_model(const std::filesystem::path& path, LoadOptions options,
                                  DeviceContext& device, const StartupObserver* observer) {
    artifact::Reader reader(path);
    return materialize_model(resolve_load(plan_load(reader, options), device.device()), device,
                             observer);
}

} // namespace ninfer::models::qwen3_5
