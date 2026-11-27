#pragma once

// The per-16 NVFP4 weight scale byte -- FP8-E4M3FN, ggml's "ue4m3" -- decoded to FP32, and the
// 256-entry table the NVFP4 integer routes read it through.
//
// The semantics are llama.cpp's `ggml_cuda_ue4m3_to_fp32` (ggml/src/ggml-cuda/common.cuh:847),
// carried over so the two codebases can be cross-checked byte for byte:
//
//   * The trailing /2 belongs to the decode. llama halves here because it feeds the 2x-widened
//     `kvalues_mxfp4` e2m1 grid; the ninfer codec widens E2M1 to s8 = 2*e2m1 for the same reason.
//     THE HALF FACTOR THEREFORE APPEARS EXACTLY ONCE ON THIS PATH, which is why every caller of
//     these routes passes alpha = 1/weight_scale_divisor and not 0.5/weight_scale_divisor.
//   * NaN is mapped to 0.0f (byte 0x7F), llama's explicit "to match CPU implementation".
//   * Bit 7 is not a sign bit: llama's software fallback masks the byte to its low 7 bits, so
//     0x80..0xFF decode as 0x00..0x7F would. That domain is unreachable -- a block scale is a
//     magnitude, so a stored scale byte never carries bit 7 -- and llama's own two paths disagree
//     about it: its CUDA intrinsic path reads bit 7 as the sign (0xFF -> 0), its software fallback
//     ignores it (0xFF -> 240). We follow the software fallback, which is also the path the 256-byte
//     enumeration oracle is built from, rather than introduce a third variant. The reachable
//     domain 0x00..0x7E is unaffected by the choice.
//   * Byte 0x00 and 0x7F both give +0.0f; the old table's -0.0f at 0x80 is gone with bit 7.
//
// Free of CUDA so a host-side test can enumerate the table against an independent reimplementation
// of llama's expression; the device table is initialised from the same list, so a host check
// covers what the kernels read.
//
// Because the /2 is a power of two, `sum * scale * alpha` with alpha = 1/divisor is one FP32
// expression across every route that reads the table, the SwiGLU gate plane and up projection
// included.

#include <cstdint>

namespace ninfer::ops::detail::nvfp4 {

// ggml's expression: exp = (byte >> 3) & 0xF, man = byte & 7, ldexp(man, -9) for exp == 0 and
// ldexp(1 + man/8, exp - 7) otherwise, then /2. Written with exact power-of-two factors so no
// rounding is introduced beyond the one product llama also rounds.
constexpr float ue4m3_to_fp32(unsigned byte) {
    if (byte == 0u || byte == 0x7Fu) { return 0.0F; }
    const unsigned exponent = (byte >> 3) & 0xFu;
    const unsigned mantissa = byte & 7u;
    if (exponent == 0u) {
        // ldexp(man, -9) / 2
        return static_cast<float>(mantissa) * (1.0F / 1024.0F);
    }
    float pow2 = 1.0F;
    for (unsigned i = 0u; i < exponent; ++i) { pow2 *= 2.0F; }
    // ldexp(1 + man/8, exp - 7) / 2
    return pow2 * (1.0F / 256.0F) * (1.0F + static_cast<float>(mantissa) / 8.0F);
}

#define NINFER_E4M3_SCALE_VALUE(n) ::ninfer::ops::detail::nvfp4::ue4m3_to_fp32(n)

// One entry per byte, in byte order. Both the host table below and the device table in
// nvfp4_int8_grouped_mma.cuh expand this list, so the two cannot drift apart.
#define NINFER_E4M3_SCALE_LIST(V)                                                               \
    V(0), V(1), V(2), V(3), V(4), V(5), V(6), V(7),                                           \
    V(8), V(9), V(10), V(11), V(12), V(13), V(14), V(15),                                     \
    V(16), V(17), V(18), V(19), V(20), V(21), V(22), V(23),                                   \
    V(24), V(25), V(26), V(27), V(28), V(29), V(30), V(31),                                   \
    V(32), V(33), V(34), V(35), V(36), V(37), V(38), V(39),                                   \
    V(40), V(41), V(42), V(43), V(44), V(45), V(46), V(47),                                   \
    V(48), V(49), V(50), V(51), V(52), V(53), V(54), V(55),                                   \
    V(56), V(57), V(58), V(59), V(60), V(61), V(62), V(63),                                   \
    V(64), V(65), V(66), V(67), V(68), V(69), V(70), V(71),                                   \
    V(72), V(73), V(74), V(75), V(76), V(77), V(78), V(79),                                   \
    V(80), V(81), V(82), V(83), V(84), V(85), V(86), V(87),                                   \
    V(88), V(89), V(90), V(91), V(92), V(93), V(94), V(95),                                   \
    V(96), V(97), V(98), V(99), V(100), V(101), V(102), V(103),                               \
    V(104), V(105), V(106), V(107), V(108), V(109), V(110), V(111),                           \
    V(112), V(113), V(114), V(115), V(116), V(117), V(118), V(119),                           \
    V(120), V(121), V(122), V(123), V(124), V(125), V(126), V(127),                           \
    V(128), V(129), V(130), V(131), V(132), V(133), V(134), V(135),                           \
    V(136), V(137), V(138), V(139), V(140), V(141), V(142), V(143),                           \
    V(144), V(145), V(146), V(147), V(148), V(149), V(150), V(151),                           \
    V(152), V(153), V(154), V(155), V(156), V(157), V(158), V(159),                           \
    V(160), V(161), V(162), V(163), V(164), V(165), V(166), V(167),                           \
    V(168), V(169), V(170), V(171), V(172), V(173), V(174), V(175),                           \
    V(176), V(177), V(178), V(179), V(180), V(181), V(182), V(183),                           \
    V(184), V(185), V(186), V(187), V(188), V(189), V(190), V(191),                           \
    V(192), V(193), V(194), V(195), V(196), V(197), V(198), V(199),                           \
    V(200), V(201), V(202), V(203), V(204), V(205), V(206), V(207),                           \
    V(208), V(209), V(210), V(211), V(212), V(213), V(214), V(215),                           \
    V(216), V(217), V(218), V(219), V(220), V(221), V(222), V(223),                           \
    V(224), V(225), V(226), V(227), V(228), V(229), V(230), V(231),                           \
    V(232), V(233), V(234), V(235), V(236), V(237), V(238), V(239),                           \
    V(240), V(241), V(242), V(243), V(244), V(245), V(246), V(247),                           \
    V(248), V(249), V(250), V(251), V(252), V(253), V(254), V(255)

constexpr float kNvfp4E4M3Scale[256] = {NINFER_E4M3_SCALE_LIST(NINFER_E4M3_SCALE_VALUE)};

} // namespace ninfer::ops::detail::nvfp4
