#pragma once

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::gguf::detail {

inline int matrix_work_blocks(int row_tiles, int column_tiles, int k_blocks) {
    constexpr std::int64_t limit = std::int64_t(1) << 30;
    if (row_tiles <= 0 || column_tiles <= 0 || k_blocks <= 0) {
        throw std::invalid_argument("gguf matrix product: nonpositive geometry");
    }
    const std::int64_t tiles = std::int64_t(row_tiles) * column_tiles;
    if (tiles >= limit || k_blocks > (limit - 1) / tiles) {
        throw std::invalid_argument("gguf matrix product: stream-k index range exceeded");
    }
    return static_cast<int>(tiles * k_blocks);
}

}
