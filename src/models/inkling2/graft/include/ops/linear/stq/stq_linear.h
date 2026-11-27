#pragma once

// Products of STQ matrices (Sushi Trellis Quant, beamster/sushi, MIT), decoded as they are read.
// A matrix of k inputs and n outputs stores 16 x 16 tiles of weights, tile (i, j) covering inputs
// [16i, 16i + 16) and outputs [16j, 16j + 16), as u16 trellis words [k/16][n/16][words]: the tile's
// bit stream is its words in u32 pairs (word 2m low, 2m + 1 high), each read from its most
// significant bit; slot s of the tile reads the 16-bit window that ends at bit ((s + 1) * words) / 16,
// wrapping around the stream, and slot s holds the weight of lane s / 8 of an MMA fragment (rows
// r, r + 1, r + 8, r + 9 with r = 2 (lane % 4) for slots 0..3 at column lane / 4, slots 4..7 at
// column lane / 4 + 8; rows are inputs). A codebook turns a window into binary16 weights, every one
// an integer times 2^-12. Beside the tiles, the input signs suh (F16 [k]) and output signs svh (F16
// [n]) give the product
//
//   y = svh * H(W^T H(suh * x)),
//
// H the orthonormal Walsh-Hadamard transform of each 128 consecutive elements (Sylvester order).
// Activations arrive in BF16; every sum is FP32.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::stq {

enum class Codebook : std::uint8_t {
    // SPEC1, 15-bit codewords, 64 words a tile (4 bits a weight): p = (cw & 0x7FFF) * 0x3FED4CEF,
    // weight = binary16(lo(p & 0x8FF08FF0) + hi(p & 0x8FF08FF0)) * 4096.
    Spec1W15N64,
    // PAIR1, 16-bit codewords, 32 words a tile (2 bits a weight): slots 2m and 2m + 1 both read slot
    // 2m + 1's window cw; p = cw * 0x3B22BE15, weight 2m from p & 0x8ED78C13 and weight 2m + 1 from
    // rotl(p, 3) & 0x8ECE8C03, each the binary16 sum of its halves times 4096.
    Pair1W16N32,
};

[[nodiscard]] constexpr int tile_words(Codebook codebook) {
    return codebook == Codebook::Spec1W15N64 ? 64 : 32;
}

// Rows of one item a product reads at once.
inline constexpr int kMaxRows = 8;

// Matrices of one shape: matrix m's tiles at trellis + m * trellis_stride, its signs at suh + m *
// suh_stride and svh + m * svh_stride (strides in elements).
struct Matrices {
    const std::uint16_t* trellis = nullptr;
    const __half* suh            = nullptr;
    const __half* svh            = nullptr;
    std::int64_t trellis_stride = 0, suh_stride = 0, svh_stride = 0;
    int k = 0, n = 0;
    Codebook codebook = Codebook::Spec1W15N64;
};

// `items` products of `rows` (1..kMaxRows) BF16 rows each: item i multiplies matrix ids[i] (i
// without ids), its row r at x + (i / x_divisor) * x_item_stride + r * x_row_stride, and stores
// output o of that row at out + i * out_item_stride + r * out_row_stride + o (FP32), or at the same
// place of out_bf16 rounded to BF16.
struct Product {
    Matrices m;
    const std::int32_t* ids = nullptr;
    int items = 1, rows = 1;
    const __nv_bfloat16* x = nullptr;
    std::int64_t x_item_stride = 0, x_row_stride = 0;
    int x_divisor = 1;
    float* out                = nullptr;
    __nv_bfloat16* out_bf16   = nullptr;
    std::int64_t out_item_stride = 0, out_row_stride = 0;
};

// Throws unless the shapes are whole 128-element transform blocks and the rows and strides fit.
void validate(const Product& product);
// Transient bytes of product() on a device of `sms` multiprocessors.
[[nodiscard]] std::size_t workspace_bytes(const Product& product, int sms);
void product(const Product& product, void* workspace, int sms, cudaStream_t stream);

} // namespace ninfer::ops::stq
