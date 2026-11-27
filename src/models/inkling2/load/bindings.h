#pragma once

#include "models/common/bindings.h"
#include "models/inkling2/port_config.h"
#include "models/inkling2/weights.h"

#include <string>

namespace ninfer::models::inkling2::loading {

// Names and conditional presence mirror the artifact `bindings` table exactly (port/m3/
// binding-shapes.md); the Binder resolves per-expert / qkv-gate row-ranges itself.
[[nodiscard]] AttentionLayerWeights bind_layer(Bindings& bindings, const Config& config,
                                               std::uint32_t layer);
[[nodiscard]] TextWeights bind_text(Bindings& bindings, const Config& config);

} // namespace ninfer::models::inkling2::loading
