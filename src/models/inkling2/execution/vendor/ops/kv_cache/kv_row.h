#pragma once

// The engine's KV row codings: how a quantized cache stores one KV head of one position in blocks
// of D dimensions (128 for Inkling 2's and Flint's planes, 256 for Gemma 4's), its codes and its
// scales apart. The codings are the engine's codecs (int8_g64_codec.cuh and its neighbours):
//
//   Int8G64   signed int8, one FP16 scale per 64 dimensions, FP16-RNE(absmax / 127)
//   Int4G32   two signed 4-bit codes per byte, one FP16 scale per 32, FP16-RNE(absmax / 7)
//   Lloyd4    4-bit Lloyd-Max indices with a G64 FP16 least-squares scale (rk4v4 keys)
//   Mxfp8     E4M3 codes, one UE8M0 power-of-two scale per 32 (Inkling 2's trained format)
//   Fp8Row    E4M3FN codes, one FP16 scale per 256 (D = 256 only)
//   Nvfp4G16  two E2M1 codes per byte, one E4M3 scale per 16
//   Int4E8    4-bit codes of the nearest E8 point, G64 FP16 scale (rk4v4-e8 keys)
//   RootE8    two bytes per 8 dimensions: E8 root, radius and residual axis, G64 FP16 scale
//             (rk2v4-e8 keys)
//
// Device side: kv_row_codec.cuh.

#include <cstdint>

// The sizes below are also compile-time constants of the CUDA kernels.
#if defined(__CUDACC__)
#    define NINFER_KV_ROW_HOST_DEVICE __host__ __device__
#else
#    define NINFER_KV_ROW_HOST_DEVICE
#endif

namespace ninfer::ops::kv_row {

enum class KvCoding : std::uint8_t {
    F16,
    Int8G64,
    Int4G32,
    Lloyd4,
    Mxfp8,
    Fp8Row,
    Nvfp4G16,
    Int4E8,
    RootE8
};

// Bytes of one D-wide block's codes and of its scales.
[[nodiscard]] NINFER_KV_ROW_HOST_DEVICE constexpr int code_bytes(KvCoding c, int dims) noexcept {
    switch (c) {
    case KvCoding::F16: return 2 * dims;
    case KvCoding::Int8G64:
    case KvCoding::Mxfp8:
    case KvCoding::Fp8Row: return dims;
    case KvCoding::Int4G32:
    case KvCoding::Lloyd4:
    case KvCoding::Nvfp4G16:
    case KvCoding::Int4E8: return dims / 2;
    case KvCoding::RootE8: return dims / 4;
    }
    return 0;
}

[[nodiscard]] NINFER_KV_ROW_HOST_DEVICE constexpr int scale_bytes(KvCoding c, int dims) noexcept {
    switch (c) {
    case KvCoding::F16: return 0;
    case KvCoding::Int8G64:
    case KvCoding::Lloyd4:
    case KvCoding::Int4E8:
    case KvCoding::RootE8: return 2 * (dims / 64);
    case KvCoding::Int4G32: return 2 * (dims / 32);
    case KvCoding::Mxfp8: return dims / 32;   // a UE8M0 exponent byte per 32
    case KvCoding::Fp8Row: return 2 * (dims / 256);
    case KvCoding::Nvfp4G16: return dims / 16; // an E4M3 byte per 16
    }
    return 0;
}

} // namespace ninfer::ops::kv_row
