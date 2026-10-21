from __future__ import annotations

import struct

import torch
from safetensors.torch import save_file

from tools.convert.sources.safetensors import SafetensorsSource
from tools.convert.sources.compressed_tensors import (
    compressed_matrix_source,
    matrix_source,
)
from tools.convert.sources.logical import select_rows


def test_row_fp8_source_and_reordered_encoded_rows(tmp_path):
    codes = torch.tensor([[0x38, 0xB8, 0x40], [0x30, 0xB0, 0x80]], dtype=torch.uint8)
    scales = torch.tensor([[2.0], [0.5]], dtype=torch.bfloat16)
    save_file(
        {
            "proj.weight": codes.view(torch.float8_e4m3fn),
            "proj.weight_scale": scales,
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = compressed_matrix_source(store, "proj", (2, 3), "fp8_e4m3fn_row_bf16")
        assert torch.equal(
            source.values().reshape(2, 3),
            torch.tensor([[2.0, -2.0, 4.0], [0.25, -0.25, -0.0]]),
        )
        reordered = select_rows(source, ((1, 2), (0, 1)))
        words = reordered.read_encoded(0, 2)
        assert torch.equal(words.codes, codes.flip(0))
        assert torch.equal(words.scales, scales.flatten().flip(0))

