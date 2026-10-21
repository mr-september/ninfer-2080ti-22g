#include "artifact/fixture.h"
#include "models/common/moe/weights.h"
#include "models/registry.h"

#include <bit>
#include <iostream>

namespace {
using namespace ninfer;
using namespace test::artifact_fixture;

void bank_binding() {
    Fixture f;
    constexpr std::size_t gu_bytes = 98316, dn_bytes = 49158, dn_offset = 98560;
    f.payload.assign(dn_offset + dn_bytes, std::byte{0});
    f.root["objects"]    = Json::array({{{"id", "gu"},
                                         {"kind", "tensor"},
                                         {"shape", {3, 256, 128}},
                                         {"format", "fp8_e4m3fn_block128x128_bf16"},
                                         {"layout", "block_scale_m128_k128_v1"},
                                         {"offset", 0},
                                         {"bytes", gu_bytes}},
                                        {{"id", "dn"},
                                         {"kind", "tensor"},
                                         {"shape", {3, 128, 128}},
                                         {"format", "fp8_e4m3fn_block128x128_bf16"},
                                         {"layout", "block_scale_m128_k128_v1"},
                                         {"offset", dn_offset},
                                         {"bytes", dn_bytes}}});
    f.root["components"] = {{"text", {{"config", Json::object()}}}};
    f.root["bindings"]   = Json::object();
    f.root["uses"]       = Json::array();
    f.root["files"]      = Json::array({{{"path", nullptr}, {"payload_bytes", f.payload.size()}}});
    for (const char* role : {"gate", "up", "down"}) {
        auto parts = Json::array();
        for (unsigned e = 0; e < 3; ++e) {
            const auto begin =
                (std::string_view(role) == "down" ? e : e * 2 + (std::string_view(role) == "up")) *
                16384;
            parts.push_back({{"object", std::string_view(role) == "down" ? "dn" : "gu"},
                             {"range", {begin, begin + 16384}}});
        }
        f.root["bindings"][role] = {{"parts", parts}};
    }
    f.write();
    artifact::Reader reader(f.entry);
    artifact::Binder binder(reader);
    auto g          = binder.parameter("gate", {3, 128, 128});
    auto u          = binder.parameter("up", {3, 128, 128});
    auto d          = binder.parameter("down", {3, 128, 128});
    const auto bank = models::moe::describe_bank(reader, models::moe::ExpertBinding{g, u, d});
    require(bank.gate_up.layout.codes.expert_stride == 32768 &&
                bank.gate_up.layout.scales.expert_stride == 4 &&
                !bank.gate_up.layout.has_divisors &&
                bank.down.layout.codes.expert_stride == 16384,
            "rank-3 bank plane strides");
    const std::array requests{artifact::PlacementRequest{bank.gate_up.object, Device::cpu()},
                              artifact::PlacementRequest{bank.down.object, Device::cpu()}};
    auto storage = artifact::materialize(reader, artifact::plan_materialization(reader, requests));
    const std::array devices{Device::cpu()};
    const auto sources = models::moe::bind_banks({&bank, 1}, storage, devices);
    for (unsigned e = 0; e < 3; ++e) {
        const auto gu = matrix_weight(sources[0].gate_up, e);
        const auto dn = matrix_weight(sources[0].down, e);
        require(static_cast<const std::byte*>(gu.qdata) - sources[0].gate_up.parent->data ==
                    e * 32768,
                "expert codes offset");
        require(static_cast<const std::byte*>(dn.qdata) - sources[0].down.parent->data ==
                    e * 16384,
                "expert down codes offset");
    }
    std::swap(u.binding.parts[0], u.binding.parts[1]);
    rejects([&] { (void)models::moe::describe_bank(reader, models::moe::ExpertBinding{g, u, d}); },
            "out-of-order experts admitted");
}
} // namespace

int main() {
    try {
        bank_binding();
        require(models::resolve_architecture("Qwen4ExpForCausalLM", "qwen4_exp_text") ==
                    models::Architecture::Qwen4Exp,
                "architecture identity");
        std::cout << "PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
