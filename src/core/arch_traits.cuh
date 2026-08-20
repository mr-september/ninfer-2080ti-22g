#pragma once

#include <cstddef>

namespace ninfer::core {

enum class CudaArch {
    SM75,   // Turing (e.g. RTX 2080 Ti, TU102)
    SM86,   // Ampere (e.g. RTX 3090, GA102)
    SM120A, // Blackwell (e.g. RTX 5090, GB202)
};

template <CudaArch Arch>
struct ArchTraits;

template <>
struct ArchTraits<CudaArch::SM75> {
    static constexpr CudaArch arch = CudaArch::SM75;
    static constexpr const char* name = "sm_75";

    static constexpr bool has_cp_async  = false;
    static constexpr bool has_bf16_mma  = false;
    static constexpr bool has_tf32_mma  = false;
    static constexpr bool has_pdl       = false;
    static constexpr bool has_nvfp4     = false;
    static constexpr bool has_ldmatrix  = true;

    // MMA tile dimensions
    static constexpr int fp16_mma_m = 16;
    static constexpr int fp16_mma_n = 8;
    static constexpr int fp16_mma_k = 8;

    static constexpr int int8_mma_m = 8;
    static constexpr int int8_mma_n = 8;
    static constexpr int int8_mma_k = 16;

    // Hardware limits
    static constexpr std::size_t max_smem_per_sm = 64 * 1024;
    static constexpr int max_warps_per_sm        = 32;
    static constexpr int max_threads_per_sm      = 1024;
    static constexpr int max_blocks_per_sm       = 16;
};

template <>
struct ArchTraits<CudaArch::SM86> {
    static constexpr CudaArch arch = CudaArch::SM86;
    static constexpr const char* name = "sm_86";

    static constexpr bool has_cp_async  = true;
    static constexpr bool has_bf16_mma  = true;
    static constexpr bool has_tf32_mma  = true;
    static constexpr bool has_pdl       = false;
    static constexpr bool has_nvfp4     = false;
    static constexpr bool has_ldmatrix  = true;

    static constexpr int fp16_mma_m = 16;
    static constexpr int fp16_mma_n = 8;
    static constexpr int fp16_mma_k = 16;

    static constexpr int int8_mma_m = 16;
    static constexpr int int8_mma_n = 8;
    static constexpr int int8_mma_k = 32;

    static constexpr std::size_t max_smem_per_sm = 100 * 1024;
    static constexpr int max_warps_per_sm        = 48;
    static constexpr int max_threads_per_sm      = 1536;
    static constexpr int max_blocks_per_sm       = 16;
};

template <>
struct ArchTraits<CudaArch::SM120A> {
    static constexpr CudaArch arch = CudaArch::SM120A;
    static constexpr const char* name = "sm_120a";

    static constexpr bool has_cp_async  = true;
    static constexpr bool has_bf16_mma  = true;
    static constexpr bool has_tf32_mma  = true;
    static constexpr bool has_pdl       = true;
    static constexpr bool has_nvfp4     = true;
    static constexpr bool has_ldmatrix  = true;

    static constexpr int fp16_mma_m = 16;
    static constexpr int fp16_mma_n = 8;
    static constexpr int fp16_mma_k = 16;

    static constexpr int int8_mma_m = 16;
    static constexpr int int8_mma_n = 8;
    static constexpr int int8_mma_k = 32;

    static constexpr std::size_t max_smem_per_sm = 100 * 1024;
    static constexpr int max_warps_per_sm        = 48;
    static constexpr int max_threads_per_sm      = 1536;
    static constexpr int max_blocks_per_sm       = 32;
};

#if defined(NINFER_SM75)
using CurrentArchTraits = ArchTraits<CudaArch::SM75>;
#elif defined(NINFER_SM86)
using CurrentArchTraits = ArchTraits<CudaArch::SM86>;
#else
using CurrentArchTraits = ArchTraits<CudaArch::SM120A>;
#endif

} // namespace ninfer::core
