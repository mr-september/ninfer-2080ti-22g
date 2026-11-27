#pragma once

// How tensor-parallel ranks divide a layer's work when the cards differ.
//
// Every rank holds a contiguous share of each divided dimension (GDN key heads, dense FFN rows or
// MoE experts), and the exchange after each projection waits for every rank, so a layer takes as
// long as its slowest share. Shares in proportion to each card's speed keep the ranks finishing
// together; the stored blocks of the projections a share cuts decide where a share may end.

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer {

// The ends of `weights.size()` contiguous shares of [0, total): out[0] = 0, out[N] = total, and
// share r is [out[r], out[r + 1]). Every inner end is one of `allowed` (ascending positions inside
// (0, total)) and every share is non-empty. Each inner end is the allowed position nearest to where
// the weights put it (the cumulative weight's fraction of `total`), the lower one on a tie, among
// those that leave every later share a position. Equal weights over evenly spaced positions give
// equal shares. Throws std::invalid_argument when the positions cannot give every rank a share or a
// weight is not positive and finite.
[[nodiscard]] std::vector<std::uint64_t> split_shares(std::uint64_t total,
                                                      std::span<const std::uint64_t> allowed,
                                                      std::span<const double> weights);

// The multiples of `unit` inside (0, total).
[[nodiscard]] std::vector<std::uint64_t> unit_positions(std::uint64_t total, std::uint64_t unit);

// Weights for dividing work over ranks of `speed` when rank 0 also runs `first` (work counted in
// fractions of the divided work) while the other ranks run their shares: rank 0's share shrinks so
// that every rank's work over its speed comes out the same. A rank 0 whose `first` alone outlasts
// the others' shares keeps a share of almost nothing; `first` = 0 gives the speeds' proportions.
[[nodiscard]] std::vector<double> shares_beside_first(std::span<const double> speed, double first);

// The fractions of the divided work each rank takes: in proportion to `speed`, except that a rank
// takes at most `capacity[r]` (a fraction of the divided work it has room for); what a full rank
// cannot take goes to the others, still in proportion to their speed. The fractions add up to one.
// When the capacities add up to less than one the work does not fit anywhere. If part of it then
// shrinks to fit (`shrinks`: automatic KV capacity) and every rank has some room, the fractions
// follow the capacities, so each rank outgrows its room by the same factor and the shrinking part
// keeps the most it can; otherwise they follow speed alone, and the caller's later memory planning
// reports the device that runs out.
[[nodiscard]] std::vector<double>
fill_shares(std::span<const double> speed, std::span<const double> capacity, bool shrinks = false);

// The fractions of the routed experts (`bytes` of weights in all) each rank takes when it serves its
// share from host memory or the files through a device cache of its own: rank r's cache holds
// room[r] bytes of its share and its bus copies the rest at bus[r] bytes per second, so the
// fractions give every rank's copies the same time, and a rank with more room or a faster bus takes
// more experts. When the caches together hold every expert nothing is copied, and the fractions
// follow `speed` within the room (fill_shares). The fractions add up to one.
[[nodiscard]] std::vector<double> delivery_shares(std::span<const double> room,
                                                  std::span<const double> bus, double bytes,
                                                  std::span<const double> speed);

} // namespace ninfer
