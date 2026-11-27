#pragma once

// Graft shim for the donor's models/qwen3_5/frontend/resources.h. The grafted executor reads
// exactly one field (executor.cpp:714 — resources().public_token_count); the port's tokenizer
// lives in the port loader, outside this layer. The adapter sets public_token_count.

#include <cstdint>

namespace ninfer::models::qwen3_5 {

struct FrontendResources {
    std::uint32_t public_token_count = 0;
};

} // namespace ninfer::models::qwen3_5
