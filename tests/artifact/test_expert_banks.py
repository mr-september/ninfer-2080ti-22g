import struct

import pytest
import torch

from tools.artifact.codecs.fp8_block import (
    decode_fp8_block_scaled_words,
    dequantize_fp8_block_scaled,
    encode_fp8_block_scaled,
)


@pytest.mark.parametrize("bank", [False, True])
def test_fp8_blocks_have_exact_words_and_two_dimensional_scale_ownership(bank):
    shape = (2, 256, 384) if bank else (256, 384)
    codes = torch.full(shape, 0x38, dtype=torch.uint8)
    codes[..., 127, 128] = 0xB8
    codes[..., 128, 127] = 0x01
    codes[..., 255, 383] = 0x7E
    scales = torch.tensor([1, 2, 3, 4, 5, 6] * (2 if bank else 1), dtype=torch.bfloat16)
    if bank:
        scales[6:] *= 2
    scales = scales.reshape((*shape[:-2], 2, 3))
    scale_words = scales.view(torch.int16).flatten().tolist()
    expected_bytes = codes.numpy().tobytes() + struct.pack(
        "<" + "H" * len(scale_words), *scale_words
    )
    assert encode_fp8_block_scaled(codes, scales, shape) == expected_bytes
    q, s = decode_fp8_block_scaled_words(expected_bytes, shape)
    assert torch.equal(q, codes)
    assert torch.equal(s.view(torch.int16), scales.view(torch.int16))
    expected = torch.ones(shape)
    expected[..., 127, 128] = -1
    expected[..., 128, 127] = 2**-9
    expected[..., 255, 383] = 448
    for row in range(256):
        for column in range(384):
            expected[..., row, column] *= scales[..., row // 128, column // 128].float()
    assert torch.equal(dequantize_fp8_block_scaled(expected_bytes, shape), expected)


def test_fp8_block_word_validation():
    codes = torch.zeros((128, 128), dtype=torch.uint8)
    scales = torch.zeros((1, 1), dtype=torch.bfloat16)
    encode_fp8_block_scaled(codes, scales, (128, 128))
    codes[0, 0] = 0x38
    with pytest.raises(ValueError, match="zero block scale"):
        encode_fp8_block_scaled(codes, scales, (128, 128))
    scales.fill_(1)
    codes[0, 0] = 0x7F
    with pytest.raises(ValueError, match="finite E4M3FN"):
        encode_fp8_block_scaled(codes, scales, (128, 128))
    codes[0, 0] = 0x38
    scales.fill_(-1)
    with pytest.raises(ValueError, match="nonnegative finite"):
        encode_fp8_block_scaled(codes, scales, (128, 128))
