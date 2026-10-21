from __future__ import annotations

import struct

import pytest
import torch

from tools.artifact.codecs.direct import decode_direct, encode_direct
from tools.artifact.codecs.row_split import decode_row_split_codes, encode_row_split


def _signed_word(word: int, bits: int) -> int:
    return word if word < 1 << (bits - 1) else word - (1 << bits)


@pytest.mark.parametrize(
    ("format_name", "tensor", "words", "word_format", "word_view"),
    [
        (
            "bf16",
            torch.tensor(
                [_signed_word(word, 16) for word in (0x0000, 0x8000, 0x0001, 0x7FC1)],
                dtype=torch.int16,
            ).view(torch.bfloat16),
            (0x0000, 0x8000, 0x0001, 0x7FC1),
            "H",
            torch.int16,
        ),
        (
            "fp32",
            torch.tensor(
                [
                    _signed_word(word, 32)
                    for word in (0x00000000, 0x80000000, 0x00000001, 0x7FC01234)
                ],
                dtype=torch.int32,
            ).view(torch.float32),
            (0x00000000, 0x80000000, 0x00000001, 0x7FC01234),
            "I",
            torch.int32,
        ),
        (
            "int32",
            torch.tensor((0, -1, -(1 << 31), (1 << 31) - 1), dtype=torch.int32),
            (0, -1, -(1 << 31), (1 << 31) - 1),
            "i",
            torch.int32,
        ),
    ],
)
def test_direct_layout_preserves_exact_little_endian_words(
    format_name, tensor, words, word_format, word_view
):
    expected = struct.pack("<" + word_format * len(words), *words)
    payload = encode_direct(tensor, format_name)
    assert payload == expected
    decoded = decode_direct(payload, format_name, tensor.shape)
    assert torch.equal(decoded.view(word_view), tensor.view(word_view))

    if format_name == "bf16":
        with pytest.raises(TypeError):
            encode_direct(tensor.float(), format_name)


@pytest.mark.parametrize(
    (
        "format_name",
        "k",
        "group_size",
        "k_pad",
        "scale_offset",
        "prefix",
        "base_prefix",
        "high_prefix",
    ),
    [
        pytest.param(
            "q4_g64_fp16",
            65,
            64,
            128,
            256,
            (-8, -7, -1, 0, 1, 7),
            b"\x98\x0f\x71",
            b"",
            id="q4",
        ),
        pytest.param(
            "q5_g64_fp16",
            130,
            64,
            256,
            512,
            (-16, -15, -1, 0, 1, 15),
            b"\x10\x0f\xf1",
            b"\x07",
            id="q5",
        ),
        pytest.param(
            "q6_g64_fp16",
            65,
            64,
            128,
            512,
            (-32, -31, -17, -16, -1, 0, 15, 31),
            b"\x10\x0f\x0f\xff",
            b"\xea\x43",
            id="q6",
        ),
        pytest.param(
            "q8_g32_fp16",
            33,
            32,
            128,
            256,
            (-127, -1, 0, 1, 127),
            b"\x81\xff\x00\x01\x7f",
            b"",
            id="q8",
        ),
    ],
)
def test_row_split_matches_known_packed_bytes(
    format_name, k, group_size, k_pad, scale_offset, prefix, base_prefix, high_prefix
):
    groups = k_pad // group_size
    codes = torch.zeros((1, groups, group_size), dtype=torch.int8)
    codes[0, 0, : len(prefix)] = torch.tensor(prefix, dtype=torch.int8)
    scales = torch.zeros((1, groups), dtype=torch.float16)
    scales[0, :2] = torch.tensor([1.5, 0.25], dtype=torch.float16)

    # Literal wire positions and words come from the format, independently of layout helpers.
    expected = bytearray(scale_offset + groups * 2)
    expected[: len(base_prefix)] = base_prefix
    expected[256 : 256 + len(high_prefix)] = high_prefix
    expected[scale_offset:] = struct.pack(
        "<" + "H" * groups, 0x3E00, 0x3400, *([0] * (groups - 2))
    )
    assert encode_row_split(codes, scales, format_name, (1, k)) == expected
    decoded_scales, decoded_codes = decode_row_split_codes(
        expected, format_name, (1, k)
    )
    assert torch.equal(decoded_scales, scales)
    assert torch.equal(decoded_codes, codes)

