#pragma once

// Shared by the bridge's translation units only: the vendored ggml headers, the device facts the
// kernels' host side needs and the matrix kernel's column tiles. The vector kernel's launches are
// in ggml_bridge_vec_launch.cuh and the matrix kernel's in ggml_bridge_mmq_launch.cuh, so an edit
// of one family's launches compiles that family's units and the dispatch.

#include "ggml_bridge_args.h"

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/vecdotq.cuh"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::gguf::detail {

struct DeviceFacts {
    int cc         = 0; // ggml's compute capability: 100 * major + 10 * minor
    int sm_count   = 0;
    std::size_t shared_per_block_optin = 0;
};

[[nodiscard]] const DeviceFacts& device_facts();

inline void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("gguf ") + what + ": " + cudaGetErrorString(status));
    }
}

// Column tiles instantiated for the matrix kernel. A product is launched with the smallest number
// of column tiles these allow, which is where ggml's own selection lands for them as well.
inline constexpr int kMatrixColumnTiles[] = {16, 32, 64, 128};

} // namespace ninfer::ops::gguf::detail
