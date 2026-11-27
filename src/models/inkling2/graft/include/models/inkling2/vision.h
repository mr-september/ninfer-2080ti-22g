#pragma once

// Inkling 2's vision tower on the device (llama.cpp's inkling2v graph): an image's patches in, its
// prompt span's rows out. The patch projection (no bias) plus the learned position rows resized to
// the grid, after the class row (its embedding plus its position row); the pre-LayerNorm; then
// every block adds to the FP32 residual the attention of its LayerNormed input (Q, K and V with
// biases, the 2D RoPE of Q and K, softmax attention with scale 1/sqrt(D) over every row, the
// output projection with bias) times its learned scale, and its GELU MLP (biases) times its
// learned scale. The class row is dropped, each 2 x 2 cell of patches becomes one row, and the
// projector (two matrices without biases, the erf GELU between) takes it to the text model's
// width. Matrices multiply BF16 inputs into FP32 results (cuBLAS); a matrix of GGUF block rows (a
// llama.cpp Q8_0 mmproj) first decodes to BF16 rows in the workspace.
//
// The matrices are resident on the device, or (VisionPlacement::Pinned) the blocks' stay in pinned
// host memory and pass through a window of two device slots, a block's matrices uploading on a copy
// stream while the block before it computes; the patch projection and the projector stay resident.

#include "core/arena.h"
#include "core/device_buffer.h"
#include "core/device_launch.h"
#include "core/qtype.h"
#include "models/inkling2/model.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::models::inkling2 {

class VisionTower {
public:
    // `max_patches`: the most patches one image may have, which sizes the scratch.
    VisionTower(const Model& model, std::uint32_t max_patches);
    ~VisionTower();
    VisionTower(const VisionTower&)            = delete;
    VisionTower& operator=(const VisionTower&) = delete;

    [[nodiscard]] const VisionConfig& config() const noexcept { return config_; }
    [[nodiscard]] std::uint32_t max_patches() const noexcept { return max_patches_; }
    [[nodiscard]] bool streamed() const noexcept { return streamed_; }
    // Device bytes the tower owns: the window and, streamed, its resident copies.
    [[nodiscard]] std::size_t owned_bytes() const noexcept { return owned_bytes_; }
    // Scratch one encode() of up to max_patches patches takes from its workspace.
    [[nodiscard]] std::size_t workspace_bytes() const;
    // The span rows of an image of grid_w x grid_h patches (both even): one per 2 x 2 cell.
    [[nodiscard]] static std::uint32_t span_rows(int grid_w, int grid_h) noexcept {
        return std::uint32_t(grid_w / 2) * std::uint32_t(grid_h / 2);
    }

    // The patches (BF16 [grid_h * grid_w][3 patch^2], channel, row and column within the patch,
    // on the device) to the span's rows, BF16 [span_rows][text width] in `out` on the device.
    void encode(const __nv_bfloat16* patches, int grid_w, int grid_h, __nv_bfloat16* out,
                WorkspaceArena& workspace, cudaStream_t stream) const;

private:
    struct Matrix {
        // [n][k]: BF16 values or rows of GGUF blocks, on the device or in pinned host memory.
        const void* data = nullptr;
        int n = 0, k = 0;
        QType format = QType::BF16;
        [[nodiscard]] bool blocks() const noexcept { return is_gguf(format); }
        [[nodiscard]] std::size_t row_bytes() const noexcept {
            if (!blocks()) { return std::size_t(k) * 2; }
            const auto block = gguf_block_shape(format);
            return std::size_t(k) / block.elements * block.bytes;
        }
        [[nodiscard]] std::size_t bytes() const noexcept { return std::size_t(n) * row_bytes(); }
    };
    struct Layer {
        Matrix query, key, value, output, up, down;
        const float *attention_norm = nullptr, *attention_norm_bias = nullptr,
                    *query_bias = nullptr, *key_bias = nullptr, *value_bias = nullptr,
                    *output_bias = nullptr, *attention_scale = nullptr, *mlp_norm = nullptr,
                    *mlp_norm_bias = nullptr, *up_bias = nullptr, *down_bias = nullptr,
                    *mlp_scale = nullptr;
    };

    // out[t][j] = sum_k w[j][k] x[t][k] (FP32, row stride ldc) for the w rows.
    void gemm(const __nv_bfloat16* x, int tokens, const Matrix& w, float* out, int ldc,
              WorkspaceArena& workspace, cudaStream_t stream) const;
    [[nodiscard]] Layer staged(std::size_t i, int slot) const;
    void upload(std::size_t i, int slot) const;
    [[nodiscard]] const void* own(const Matrix& host);

    VisionConfig config_;
    std::uint32_t max_patches_ = 0;
    bool streamed_             = false;
    Matrix patch_embedding_, projector_up_, projector_out_;
    const float *class_embedding_ = nullptr, *class_position_ = nullptr, *positions_ = nullptr,
                *pre_norm_ = nullptr, *pre_norm_bias_ = nullptr;
    std::vector<Layer> layers_;
    DeviceBuffer inv_freq_; // [head_dim / 4]
    std::vector<DeviceBuffer> owned_;
    std::size_t owned_bytes_ = 0;
    cublasHandle_t blas_     = nullptr;
    DeviceBuffer slots_[2];
    std::size_t slot_bytes_   = 0;
    cudaStream_t copy_stream_ = nullptr;
    cudaEvent_t copied_[2]    = {nullptr, nullptr};
    cudaEvent_t used_[2]      = {nullptr, nullptr};
};

} // namespace ninfer::models::inkling2
