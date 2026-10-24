#pragma once

#include <cstddef>

namespace ninfer::core {

enum class CudaArch {
    SM75, // Turing (e.g. RTX 2080 Ti, TU102)
};

template <CudaArch Arch>
struct ArchTraits;

template <>
struct ArchTraits<CudaArch::SM75> {
    static constexpr CudaArch arch = CudaArch::SM75;
    static constexpr const char* name = "sm_75";
    static constexpr int compute_capability = 75;

    static constexpr std::size_t max_smem_per_block = 64 * 1024; // opt-in dynamic limit
};

using CurrentArchTraits = ArchTraits<CudaArch::SM75>;

// Opt-in dynamic shared-memory limit of one CTA on the compiled architecture. Kernel schedules
// tuned for the 99 KiB Blackwell limit are validated against this at compile time.
inline constexpr int kMaxCtaSharedBytes = static_cast<int>(CurrentArchTraits::max_smem_per_block);

} // namespace ninfer::core
