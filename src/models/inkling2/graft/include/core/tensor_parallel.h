#pragma once

#include "core/device.h"
#include "ninfer/model_enums.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ninfer {

// The all-reduce of a tensor-parallel group. Every rank of a DeviceContext holds a replica of the
// BF16 residual and an FP32 partial of one row-split projection; each rank adds the sum of all
// partials, taken in rank order, to its replica and rounds once:
//
//   x_r <- bf16(float(x_r) + ((p_0 + p_1) + ... + p_{N-1}))
//
// Every rank evaluates the same expression on the same values, so the replicas stay identical, and
// the one rounding is the one a single device's residual-adding projection makes. A projection only
// rank 0 runs is an exchange with one contributor: every replica adds rank 0's output.
//
// A rank writes its partial into its slot (`partial`), publishes it (`publish`) and, once every
// rank has published, reduces (`reduce`) and completes (`complete`). Every rank publishes in every
// exchange, one outside the contributors with zero bytes. Two slots alternate between exchanges. A
// rank writes a slot (its own, or its row of another rank's) again two exchanges later, after its
// reduce of the exchange in between, which waits for every other rank's publish of that exchange,
// which each rank issues after its own reduce and complete of the earlier one: so no rank
// overwrites a slot another is still reading. Consecutive passes are separate in the same way as
// long as a pass ends in a synchronize, as every round does.
//
// Records and waits pair within one exchange, so the protocol is legal eagerly and inside one
// capture that spans every rank's stream.
//
// Transports (TensorParallelTransport; Auto only chooses among them):
//   Host       a publish copies the slot with the copy engine into a pinned, mapped host slot, and
//              a reduce reads the other ranks' host slots across the bus. Works between any cards.
//   PeerRead   every pair of ranks has peer access or shares a card: a reduce reads the other
//              ranks' device slots directly.
//   PeerWrite  the same access the other way: a publish stores the partial into a row of every
//              other rank's inbox, in that rank's memory, and a reduce reads only its own card.
//              Stores to another card are posted writes, which do not wait for an answer across
//              the bus as reads do: on four RTX A4000s they beat peer reads at every size.
//
// Forms (TensorParallelForm):
//   Words    an exchange of at most `words_bytes` per partial, with every rank on its own card,
//            skips the copies and the cross-device fences: the reduce is one kernel that stores
//            each FP32 value of the rank's partial with the exchange's 32-bit epoch as one 8-byte
//            word, then polls the other ranks' words until they carry the same epoch, and adds in
//            rank order as above. Under Host a rank stores its words into its own mapped host row
//            and polls the others' rows across the bus; under PeerWrite it stores them into its row
//            in every other rank's device memory and polls the rows in its own. A rank outside the
//            contributors stores one word. Each rank counts its own exchanges in device memory, so
//            every rank derives the same epoch and a captured graph replays without new arguments;
//            the row's slot is the epoch's parity. A rank writes a slot again two exchanges later,
//            after its kernel of the exchange in between, which waited for every other rank's word
//            of that exchange, which each rank stores only after its own earlier kernel ended: no
//            rank overwrites words another is still reading. One count serves both kinds of rows,
//            since that order holds whichever rows the exchange in between used. Rows start
//            zeroed, as if carrying epoch 0, so after 2^32 - 1 exchanges the count goes on at 2,
//            which no unwritten word matches and which keeps the parity alternating. A wait that
//            outlasts kWordsTimeoutNs traps the context. Ranks that share a card exchange no words,
//            since two spinning kernels on one device need not run together.
//   Whole    every rank reduces every value.
//   Scatter  each rank owns one contiguous range of the values, whole 64-value blocks as even as
//            they divide. A rank's reduce sums only its own range of the contributors' partials, in
//            rank order as above, into its output; its `complete` brings in every other rank's
//            finished range. Each value is still evaluated once, by the same expression on the same
//            operands, so every output is bit-identical to a whole reduce, while a rank moves 1/N
//            of each other rank's FP32 partial and (N-1)/N of an output instead of N-1 whole
//            partials. Two ranks storing an FP32 sum would move half a partial and half an output,
//            as many bytes as a whole partial, so they reduce whole. Under Host a publish leaves
//            the rank's own range of its partial on the device, since no other rank reads it, the
//            owner copies its finished range into that range of its host slot, and `complete` reads
//            the others' ranges there; under PeerRead `complete` reads them from the owners'
//            outputs; under PeerWrite a publish stores each owner's range of the partial into its
//            row of that owner's inbox, and the owner stores its finished range straight into every
//            other rank's output, which a publish names: `complete` only waits for it. A rank's
//            output is written by others only after its publish of the exchange, which follows
//            every read of it before, and read by its own kernels only after its `complete`. The
//            caller issues `complete` for every rank after `reduce` for every rank, since a wait
//            issued before its event's record would see an earlier record.
//   ScatterFlags  Scatter with every rank on its own card and at most kScatterFlagsBytes per
//            partial, whose reduce and complete are one kernel per rank that waits on the other
//            ranks through flags instead of cross-device fences. Under Host the flags lie in host
//            memory and a publish copies the partial as Scatter's does. Block b of every rank's
//            kernel handles part b of every rank's range (whole 64-value blocks), so a block waits
//            only on the
//            same block of the other ranks: it raises its first flag at once (its rank's copy ended
//            before the kernel began), waits for the contributors' first flags, sums its part of
//            the rank's own range in rank order as above into the output and into a finished row of
//            its rank's mapped host area, raises its second flag once those stores have reached
//            host memory, waits for every other rank's second flag and reads their finished parts
//            into its output. A flag carries the exchange's epoch from the count Words keeps, and
//            finished rows and flags alternate by its parity. Every block waits, in every exchange,
//            for a flag every other rank's same block raised in it, which that rank raises only
//            after its kernel of the exchange before ended: so no rank overwrites a host slot, row
//            or flag another is still reading, and Words and ScatterFlags share the count. Under a
//            peer transport a publish moves nothing, and the flags lie in each rank's device
//            memory, raised by the other ranks' stores, so a rank polls only its own card: block b
//            raises its first flag in every other rank's memory, under PeerWrite after storing part
//            b of every other owner's range of its partial into its row of that owner's inbox;
//            waits for its own flags from the contributors (under PeerWrite from every rank, which
//            has then begun its kernel and stopped reading its output); sums its part of its rank's
//            range in rank order, reading the others' partials across the link (PeerRead) or from
//            its inbox (PeerWrite), into the output and either every other rank's output
//            (PeerWrite) or a finished row in its own memory (PeerRead); raises its second flag
//            everywhere, waits for every other rank's, and under PeerRead reads their finished
//            parts across the link. A rank's kernel in the same slot two exchanges later begins
//            only after its kernel in between saw every other rank's second flag of that exchange,
//            which each rank raises only after its own kernel of this one ended: so no rank
//            overwrites a partial, finished row, inbox row or flag another is still reading, nor
//            writes an output another is still reading. Two ranks storing an FP32 sum reduce
//            whole, as for Scatter.
//
// Pieces. A prompt's exchange may run as up to kPieces pieces, each a contiguous range of the
// partial's values (`TensorParallelPiece`) with fences of its own, so a piece's exchange crosses on
// a rank's side stream (`side_stream`) while its compute stream runs the next piece. Each piece is
// an exchange as above over its own values: its route is its own size's, its reduce-scatter divides
// its own values, and under PeerWrite an inbox row holds every piece's range at a place of its own,
// in the bytes its partials move as (rounded ones take fewer); a later piece whose range would pass
// the row's end reduces whole by peer reads instead. A whole reduce's place is the piece's offset,
// past every earlier piece's range, while a reduce-scatter's lies lower, among an earlier whole
// piece's values: after a piece reduced whole over peer writes, every later piece of the exchange
// reduces whole too. A rank publishes a piece after its reduce of every earlier piece, whose routes
// it keeps. Neither Words nor ScatterFlags takes pieces: a later piece whose band names one reduces
// whole. The slot alternation holds piece by piece: a rank writes piece p of a slot again only
// after its own reduce and complete of piece p in the exchange in between.
//
// Rounded partials (TensorParallelPartials::Bf16 or Int8, a prompt chunk's exchanges by choice). A
// Whole or Scatter exchange can move its partials rounded, in fewer bytes: to BF16, half of them,
// or to 8-bit blocks, 9/32 of them: each 32 values from a multiple of 32 as an FP32 scale (their
// largest magnitude over 127) and 32 signed codes (each value over the scale, rounded to the
// nearest integer), a value being its code times the scale. A publish rounds the rank's whole
// partial into a device copy (`rounded_partials` allocates it) and moves that as the transport
// moves partials (Host: into its host slot, the values from the slot's start; PeerWrite: into the
// inbox rows, the rounded values of a range at the byte its place gives them; PeerRead: nothing,
// the others read the device copy), and a reduce adds the contributors' rounded partials, its own
// from the device copy, in FP32 in rank order, into the output the sum asks for. Every rank sums
// the same rounded values, so the replicas stay identical; the sum differs from the FP32
// partials' by one rounding of each. 8-bit partials need every piece offset and every range to
// start on a multiple of 32 values, which reduce-scatter ranges (multiples of 64) and pieces of
// whole columns of a hidden width that is a multiple of 32 do. Under Host a Scatter owner's
// finished range lies past the slot's first half, which holds every rounded partial; host slots
// are half a slot longer for an FP32 output.
// Words and ScatterFlags take the partials as computed whatever the call asks, and every rank of
// an exchange asks the same.
//
// Routes. Which transport and form is fastest depends on the partial's size, the number of ranks
// and the platform's links, and no fixed choice holds everywhere: on one host peer reads beat host
// memory at every size, on another the reverse. Unless `routes` names them, the exchange times
// every route `transport` and the topology allow, in captured graphs on every rank at once (each
// rank contributing, a BF16 residual), at partial sizes from 4 KiB doubling up to its slot and at
// the last size of each route that stops short of it (words, scatter flags, PeerWrite's whole
// reduce into its inbox rows). Between two of these sizes a route's time is taken on the line
// through its two times, at sixteenths of the step, and an exchange takes the fastest route at the
// smallest of all these sizes that holds it (`bands`). Rounded partials move fewer bytes, so their
// fastest route can differ (two ranks reducing 8-bit partials over peer writes: whole, where FP32
// partials take a reduce-scatter): an exchange that holds rounded copies also times its Whole and
// Scatter routes with partials rounded as `rounded_partials` names, and an exchange of rounded
// partials takes its route from those bands (`rounded_bands`; Words and ScatterFlags keep their
// FP32 times there, since they move the partials as computed). The measurement then frees the
// buffers of the routes it did not choose.
enum class TensorParallelForm : std::uint8_t { Words, Whole, Scatter, ScatterFlags };

struct TensorParallelRoute {
    TensorParallelTransport transport = TensorParallelTransport::Host;
    TensorParallelForm form           = TensorParallelForm::Whole;

    bool operator==(const TensorParallelRoute&) const = default;
};

// Partials of at most `bytes` (and more than the band before's) take `route`; `microseconds` is
// what one exchange of `bytes` took on it, zero where nothing was timed.
struct TensorParallelBand {
    std::size_t bytes = 0;
    TensorParallelRoute route;
    double microseconds = 0.0;
};

// What a reduce does with the sum of the partials: adds it to a BF16 residual and rounds once (the
// residual stream of a Qwen3.5-core model), or stores it in place of the output, rounded to BF16 or
// kept in FP32 (a block output that the caller then adds itself, as Qwen3.8-Flash-Next's
// hyper-connections do).
enum class TensorParallelSum : std::uint8_t { AddBf16, StoreBf16, StoreFp32 };

// A piece of an exchange: the values from `offset` of the partial and of the output, fenced by
// events of its own (`index`, below kPieces). Pieces of one exchange take increasing offsets and
// indices. The default is a whole exchange.
struct TensorParallelPiece {
    std::size_t index   = 0;
    std::int64_t offset = 0;
};

struct TensorParallelExchangeOptions {
    // Largest partial one slot holds.
    std::size_t slot_bytes = 0;
    // The transport every exchange takes, or Auto: whichever was measured fastest at each size. A
    // peer transport needs peer access between every pair of ranks.
    TensorParallelTransport transport = TensorParallelTransport::Auto;
    // Instead of measuring, the route of every size (tests, the bench): bands in increasing
    // `bytes`, the last covering the slot. A route a size cannot take falls back as `route` says.
    std::vector<TensorParallelBand> routes;
    // Largest partial exchanged as words.
    std::size_t words_bytes = 32U << 10U;
    // How many exchanges every rank has already counted (tests: an epoch close to the wrap).
    std::uint32_t words_count = 0;
    // The rounded partials prompt exchanges ask for, Bf16 or Int8 (Fp32: none). Each rank then
    // holds a device copy of its slots, rounded, which exchanges of either rounded format need, and
    // the measurement times the routes of partials rounded so (see Routes).
    TensorParallelPartials rounded_partials = TensorParallelPartials::Fp32;
};

class TensorParallelExchange {
public:
    static constexpr std::size_t kSlots                 = 2;
    static constexpr std::size_t kPieces                = 4;
    static constexpr unsigned long long kWordsTimeoutNs = 300'000'000'000ULL;
    // Largest partial exchanged as scatter flags.
    static constexpr std::size_t kScatterFlagsBytes = 1U << 20U;
    // A PeerWrite inbox row holds an owner's range of a full slot or this much, whichever is more
    // (never more than the slot): the largest partial PeerWrite reduces whole.
    static constexpr std::size_t kPeerWriteWholeBytes = 1U << 20U;

    TensorParallelExchange(const DeviceContext& context, TensorParallelExchangeOptions options);
    ~TensorParallelExchange();

    TensorParallelExchange(const TensorParallelExchange&)            = delete;
    TensorParallelExchange& operator=(const TensorParallelExchange&) = delete;
    TensorParallelExchange(TensorParallelExchange&&)                 = delete;
    TensorParallelExchange& operator=(TensorParallelExchange&&)      = delete;

    // Device memory one rank's exchange of `slot_bytes` slots takes at most while it is built, over
    // `ranks` ranks that do (`peer`) or do not all have peer access, with `transport` and, where
    // `rounded_partials`, the rounded copies of its slots.
    [[nodiscard]] static std::size_t device_bytes(std::size_t slot_bytes, std::size_t ranks,
                                                  bool peer, TensorParallelTransport transport,
                                                  bool rounded_partials = false);

    [[nodiscard]] std::size_t ranks() const noexcept { return ranks_.size(); }
    [[nodiscard]] std::size_t slot_bytes() const noexcept { return slot_bytes_; }
    // Device memory one rank's exchange holds once built.
    [[nodiscard]] std::size_t device_bytes() const noexcept;
    // The bands every exchange of a sum chooses from, and those of `publish_whole` and `gather`.
    [[nodiscard]] const std::vector<TensorParallelBand>& bands() const noexcept { return bands_; }
    [[nodiscard]] const std::vector<TensorParallelBand>& whole_bands() const noexcept {
        return whole_bands_;
    }

    // The bands an exchange of rounded partials chooses from: the bands themselves where nothing
    // was measured or the exchange holds no rounded copies.
    [[nodiscard]] const std::vector<TensorParallelBand>& rounded_bands() const noexcept {
        return rounded_bands_.empty() ? bands_ : rounded_bands_;
    }

    // Every route timed at every size, in the order measured, FP32 partials; empty where nothing
    // was timed.
    [[nodiscard]] const std::vector<TensorParallelBand>& timings() const noexcept {
        return timings_;
    }
    // One line naming the bands, for the startup log.
    [[nodiscard]] std::string describe() const;

    // The route an exchange of `bytes` per partial whose reduce does `sum` takes: its band's, with
    // the form a whole reduce where the band's cannot hold it (words past `words_bytes`, scatter
    // flags past kScatterFlagsBytes, either between ranks on one card, a reduce-scatter of two
    // ranks storing an FP32 sum), from the whole bands. Rounded `partials` take the rounded bands.
    [[nodiscard]] TensorParallelRoute
    route(std::size_t bytes, TensorParallelSum sum = TensorParallelSum::AddBf16,
          TensorParallelPartials partials = TensorParallelPartials::Fp32) const noexcept;
    // The route of a `publish_whole` and `gather` of `bytes`.
    [[nodiscard]] TensorParallelRoute gather_route(std::size_t bytes) const noexcept;

    // Where `rank` writes its FP32 partial for an exchange in `slot`: device memory of that rank.
    [[nodiscard]] float* partial(std::size_t rank, std::size_t slot) const;
    // Makes `rank`'s partial of `bytes` in `slot` available to the other ranks; `out` is the output
    // the rank's reduce will write and `sum` the exchange's reduce. With a `piece`, `bytes` and
    // `out` are the piece's, starting at its offset. `partials` asks for rounded partials (above).
    void publish(std::size_t rank, std::size_t slot, std::size_t bytes, void* out,
                 cudaStream_t stream, TensorParallelSum sum = TensorParallelSum::AddBf16,
                 TensorParallelPiece piece        = {},
                 TensorParallelPartials partials = TensorParallelPartials::Fp32);
    // Adds the partials ranks [0, contributors) published in `slot` to `rank`'s BF16 residual `out`
    // of `count` values, or stores their sum there (`sum`); for a reduce-scatter, only in the rank's
    // own range.
    void reduce(std::size_t rank, std::size_t slot, void* out, std::int64_t count,
                std::size_t contributors, cudaStream_t stream,
                TensorParallelSum sum = TensorParallelSum::AddBf16, TensorParallelPiece piece = {},
                TensorParallelPartials partials = TensorParallelPartials::Fp32);
    // Brings the other ranks' finished ranges of a reduce-scatter into `rank`'s output; nothing for
    // another form. Every rank's `complete` follows every rank's `reduce` of the exchange, with the
    // same `out`, `count`, `sum` and `partials`.
    void complete(std::size_t rank, std::size_t slot, void* out, std::int64_t count,
                  cudaStream_t stream, TensorParallelSum sum = TensorParallelSum::AddBf16,
                  TensorParallelPiece piece        = {},
                  TensorParallelPartials partials = TensorParallelPartials::Fp32);
    // Pieces: `rank`'s side stream, and the events with which its compute stream hands piece
    // `piece` to the side stream (`piece_ready`) and takes its exchange back (`piece_done`).
    [[nodiscard]] cudaStream_t side_stream(std::size_t rank) const;
    [[nodiscard]] cudaEvent_t piece_ready(std::size_t rank, std::size_t piece) const;
    [[nodiscard]] cudaEvent_t piece_done(std::size_t rank, std::size_t piece) const;
    // A gather in place of the sum (a position-split attention's partials, merged by
    // ops::causal_attention_merge_partials): `publish_whole` makes `rank`'s partial of `bytes` in
    // `slot` available to the others, never as words; `gather` points `parts[r]` at every rank's
    // partial of `bytes` in `slot` as `rank`'s kernels on `stream` read it (its own device slot,
    // another's device slot under PeerRead, its own inbox row under PeerWrite, another's mapped host
    // slot under Host), after `stream` waits for the other ranks' publishes. It takes its place in
    // the slot alternation as an exchange does: a rank's next exchange in the slot follows its own
    // reads of it.
    void publish_whole(std::size_t rank, std::size_t slot, std::size_t bytes, cudaStream_t stream);
    void gather(std::size_t rank, std::size_t slot, std::size_t bytes,
                std::span<const float*> parts, cudaStream_t stream);

private:
    struct Rank {
        int device = 0;
        std::array<float*, kSlots> device_slot{};
        std::array<float*, kSlots> host_slot{};
        // The rank's partial rounded, per slot (`rounded_partials`): half a slot of bytes, BF16
        // values or 8-bit blocks.
        std::array<unsigned char*, kSlots> rounded{};
        // PeerWrite: [ranks - 1][inbox_values_] partials, or owners' ranges of them, the other
        // ranks stored here, in rank order without the rank itself.
        std::array<float*, kSlots> inbox{};
        // Per slot and piece: the fence of the rank's publish, the output its publish named,
        // whether its reduce-scatter has finished it, and the fence of that reduce.
        std::array<std::array<cudaEvent_t, kPieces>, kSlots> published{};
        std::array<std::array<void*, kPieces>, kSlots> output{};
        std::array<std::array<bool, kPieces>, kSlots> reduced_ready{};
        std::array<std::array<cudaEvent_t, kPieces>, kSlots> reduced{};
        // Per slot and piece: the route the rank's last reduce of that piece took.
        std::array<std::array<TensorParallelRoute, kPieces>, kSlots> piece_route{};
        // Pieces: the side stream and the hand-over events (see `piece_ready`).
        cudaStream_t side = nullptr;
        std::array<cudaEvent_t, kPieces> piece_ready{};
        std::array<cudaEvent_t, kPieces> piece_done{};
        // Words: the rank's mapped host row, [kSlots][words_values_], the rows the other ranks
        // store into in its device memory, [ranks][kSlots][words_values_], and its {exchanges
        // counted, blocks finished} in device memory.
        unsigned long long* host_words = nullptr;
        unsigned long long* peer_words = nullptr;
        std::uint32_t* epoch           = nullptr;
        // ScatterFlags: the rank's mapped host area, per slot a row of its finished outputs, then
        // the flags of both phases per slot and block.
        unsigned char* scatter_flags = nullptr;
        // ScatterFlags under a peer transport: the rank's area in its device memory, per slot a
        // row of its finished outputs, then the flags the other ranks raise in it.
        unsigned char* peer_flags = nullptr;
    };

    // Which buffers the bands need.
    struct Needs {
        bool host_slots    = false;
        bool host_words    = false;
        bool inbox         = false;
        bool peer_words    = false;
        bool scatter_flags = false;
        bool peer_flags    = false;
    };

    void release() noexcept;
    void check_slot(std::size_t rank, std::size_t slot, std::size_t bytes) const;
    void check_piece(std::size_t rank, std::size_t slot, std::size_t bytes,
                     TensorParallelPiece piece) const;
    [[nodiscard]] bool legal(TensorParallelRoute route, std::size_t bytes) const noexcept;
    [[nodiscard]] Needs needs(const std::vector<TensorParallelBand>& bands) const noexcept;
    void provide(Needs needs);
    void measure(const std::vector<TensorParallelTransport>& transports);
    // PeerWrite's copies by the copy engines rather than one kernel from which size
    // (`engine_bytes_`), timed on rank 0's rows of the other ranks' inboxes.
    void measure_engine_bytes();
    [[nodiscard]] float* inbox_row(std::size_t rank, std::size_t sender, std::size_t slot) const;
    // Where in an inbox row a piece's values begin (see Pieces), and the route of `rank`'s
    // exchange of it in `slot`, which follows the routes its earlier pieces took.
    [[nodiscard]] std::int64_t inbox_offset(TensorParallelPiece piece,
                                            TensorParallelForm form) const noexcept;
    [[nodiscard]] TensorParallelRoute piece_route(std::size_t rank, std::size_t slot,
                                                  std::size_t bytes, TensorParallelSum sum,
                                                  TensorParallelPiece piece,
                                                  TensorParallelPartials partials) const;
    // The whole route of `bytes`, for `partials`: PeerRead past PeerWrite's inbox rows.
    [[nodiscard]] TensorParallelRoute whole_route(std::size_t bytes,
                                                  TensorParallelPartials partials) const noexcept;

    // How an exchange on `route` moves the partials, as `partials` asks (see Rounded partials):
    // Fp32 where the route keeps them as computed.
    [[nodiscard]] TensorParallelPartials moved(TensorParallelRoute route,
                                               TensorParallelPartials partials) const;
    // Under Host, where `owner`'s finished output from value `begin` of a reduce-scatter, of
    // `size` bytes a value, lies in its host slot.
    [[nodiscard]] unsigned char* finished_host(std::size_t owner, std::size_t slot,
                                               TensorParallelPiece piece, std::int64_t begin,
                                               bool rounded, std::size_t size) const noexcept;
    // Bytes of a host slot: half a slot more where partials may be rounded, for an FP32 output
    // past them.
    [[nodiscard]] std::size_t host_slot_bytes() const noexcept;

    void publish_on(TensorParallelRoute route, std::size_t rank, std::size_t slot,
                    std::size_t bytes, void* out, cudaStream_t stream, TensorParallelSum sum,
                    TensorParallelPiece piece       = {},
                    TensorParallelPartials partials = TensorParallelPartials::Fp32);
    void reduce_on(TensorParallelRoute route, std::size_t rank, std::size_t slot, void* out,
                   std::int64_t count, std::size_t contributors, cudaStream_t stream,
                   TensorParallelSum sum, TensorParallelPiece piece = {},
                   TensorParallelPartials partials = TensorParallelPartials::Fp32);
    void complete_on(TensorParallelRoute route, std::size_t rank, std::size_t slot, void* out,
                     std::int64_t count, cudaStream_t stream, TensorParallelSum sum,
                     TensorParallelPiece piece       = {},
                     TensorParallelPartials partials = TensorParallelPartials::Fp32);

    std::vector<Rank> ranks_;
    std::size_t slot_bytes_           = 0;
    bool peer_                        = false;
    std::size_t words_values_         = 0; // 0 where ranks share a card
    std::size_t scatter_flags_values_ = 0; // 0 where ranks share a card
    std::size_t inbox_values_         = 0; // floats per inbox row
    // PeerWrite copies of at least this many bytes per row go by the copy engines, smaller ones by
    // one kernel: measured with the routes, or 1 MiB, which four RTX A4000 favoured (2026-11-02).
    std::int64_t engine_bytes_        = 1 << 20;
    bool engine_measured_             = false;
    bool measured_                    = false;
    TensorParallelPartials rounded_   = TensorParallelPartials::Fp32;
    std::vector<TensorParallelBand> bands_;
    std::vector<TensorParallelBand> whole_bands_;
    // Measured with rounded partials; empty where they are the bands above.
    std::vector<TensorParallelBand> rounded_bands_;
    std::vector<TensorParallelBand> rounded_whole_bands_;
    std::vector<TensorParallelBand> timings_;
};

// Each rank's speed for dividing work by: the device-to-device copy bandwidth of its card in bytes
// per second, shared evenly by the ranks one card holds (`--devices 0,0`). A decode step reads every
// weight once, so a rank's share of the weights takes time in proportion to this. Measured with a
// short copy on each card; cards of one model (name, memory, multiprocessors, capability) take the
// mean of theirs, so the shares they divide stay the same from start to start.
[[nodiscard]] std::vector<double> rank_copy_bandwidth(const DeviceContext& context);

// Each rank's bus speed: the bandwidth of a copy from pinned host memory to its card in bytes per
// second, shared evenly by the ranks one card holds. Measured with a short copy on each card.
[[nodiscard]] std::vector<double> rank_host_copy_bandwidth(const DeviceContext& context);

// Each rank's free device memory now, shared evenly by the ranks one card holds.
[[nodiscard]] std::vector<std::uint64_t> rank_free_bytes(const DeviceContext& context);

// "host", "peer-read", "peer-write" (Auto: "auto"); "words", "whole", "scatter", "scatter-flags".
[[nodiscard]] const char* tensor_parallel_transport_name(TensorParallelTransport transport) noexcept;
[[nodiscard]] const char* tensor_parallel_form_name(TensorParallelForm form) noexcept;

} // namespace ninfer
