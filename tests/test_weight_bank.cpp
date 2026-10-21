#include "core/buffer.h"
#include "core/weight_bank.h"

#include <array>
#include <iostream>
#include <stdexcept>

namespace {
using namespace ninfer;

void require(bool value) {
    if (!value) { throw std::runtime_error("weight bank addressing differs"); }
}

template <class F>
void rejects(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid weight bank accepted");
}

void integers() {
    for (const auto format :
         {QType::Q4_G64_FP16, QType::Q5_G64_FP16, QType::Q6_G64_FP16, QType::Q8_G32_FP16}) {
        const auto g =
            weight_geometry(format, QuantLayout::RowSplit, std::array<std::uint64_t, 2>{1536, 130});
        HostBuffer bytes(g.bytes);
        WeightParent parent{g, Device::cpu(), static_cast<const std::byte*>(bytes.data())};
        const auto bank = bind_weight_bank(parent, describe_weight_bank(g, {128, 3, 128, 512}));
        rejects([&] { (void)device_weight_bank(bank); });
        for (unsigned e = 0; e < 3; ++e) {
            const auto w     = matrix_weight(bank, e);
            const auto row   = 128 + e * 512;
            const auto* base = parent.data;
            const bool q8    = format == QType::Q8_G32_FP16;
            require(w.qdata == base + row * (q8 ? 256 : 128));
            require(w.scales == base + g.scale_offset + row * (q8 ? 16 : 8));
            if (format == QType::Q5_G64_FP16) {
                require(w.qhigh == base + g.high_offset + row * 32);
            }
            if (format == QType::Q6_G64_FP16) {
                require(w.qhigh == base + g.high_offset + row * 64);
            }
            require(bank.layout.codes.expert_stride == bank.layout.codes.expert_bytes * 4);
        }
        rejects([&] { (void)describe_weight_bank(g, {128, 4, 128, 512}); });
    }
}

void block_scales() {
    const auto g = weight_geometry(QType::FP8_E4M3FN_BLOCK128X128_BF16,
                                   QuantLayout::BlockScaleM128K128,
                                   std::array<std::uint64_t, 3>{3, 256, 384});
    HostBuffer bytes(g.bytes);
    WeightParent parent{g, Device::cpu(), static_cast<const std::byte*>(bytes.data())};
    const auto bank = bind_weight_bank(parent, describe_weight_bank(g, {128, 3, 128, 256}));
    for (unsigned e = 0; e < 3; ++e) {
        const auto w = matrix_weight(bank, e, 7);
        require(w.qdata == parent.data + (e * 256 + 128) * 384);
        require(w.scales == parent.data + (294912 + (e * 2 + 1) * 6));
        require(bank.divisor(e) == 1);
    }
    rejects([&] { (void)describe_weight_bank(g, {64, 1, 128, 256}); });
}
} // namespace

int main() {
    try {
        integers();
        block_scales();
        std::cout << "weight bank views passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
