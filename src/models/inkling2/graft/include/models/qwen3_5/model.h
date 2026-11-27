#pragma once

// Graft shim for the donor's models/qwen3_5/model.h. The port's qwen3_5 tree has no model.h;
// the grafted inkling2 model/executor layer needs only InstanceInfo from it. The full donor
// header drags auxiliary_replicas/expert_sources/LoadPlan, none of which the grafted executor
// references (verified by grep: executor.cpp and inkling2/model.h use only InstanceInfo,
// FrontendResources, BoundWeight, WeightId from the donor qwen3_5 namespace).

#include <string>

namespace ninfer::models::qwen3_5 {

struct InstanceInfo {
    std::string name;
    std::string metadata_json;
    std::string provenance_json;
};

} // namespace ninfer::models::qwen3_5
