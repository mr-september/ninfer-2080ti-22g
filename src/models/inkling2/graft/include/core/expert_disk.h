#pragma once

#include "ninfer/model_enums.h"

#include <filesystem>

namespace ninfer {

// How disk experts reach the device: from the artifact's files, or striped over them and a drive
// mirror's copies (core/expert_mirror.h); through host memory, or with ExpertDirect straight into
// device memory (core/expert_direct.h).
struct ExpertDiskReads {
    std::filesystem::path mirror;
    ExpertDirect direct = ExpertDirect::Auto;
    // Tests: with ExpertDirect::On, the direct path's host window (DirectExpertWindow::open).
    bool host_window = false;
};

} // namespace ninfer
