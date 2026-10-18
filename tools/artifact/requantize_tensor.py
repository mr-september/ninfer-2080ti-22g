"""Write a copy of a v3 artifact with one grouped-integer tensor re-encoded in another format.

The main use is shrinking the output head (``text/output_head``, Q8 -> Q6) of an existing artifact when
the BF16 source checkpoint is not at hand. Values are decoded from the stored representation and
re-quantized with the converter's own grouped quantizer, so the result carries one extra rounding stage
compared with a conversion from BF16; every other object is copied byte for byte.

    python -m tools.artifact.requantize_tensor SRC.ninfer OUT.ninfer \
        --object text/output_head --format q6_g64_fp16

Run it from the repository root. Pass ``CUDA_VISIBLE_DEVICES=`` to keep the quantizer on the CPU.
"""

from __future__ import annotations

import argparse
import time

import torch

from tools.artifact.codecs.row_split import (
    RowPlanes,
    assemble_row_planes,
    dequantize_row_split,
    encode_row_split,
    split_row_planes,
)
from tools.artifact.formats import get_format
from tools.artifact.layouts import row_split_geometry
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceObject, ResourceSpec, TensorObject, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.quantization.groupwise import quantize_matrix

BLOCK_ROWS = 8192


def _plane_bytes(plane) -> bytes:
    return plane.cpu().numpy().tobytes() if isinstance(plane, torch.Tensor) else bytes(plane)


def requantize(payload: torch.Tensor, shape, old_format: str, new_format: str, device) -> bytes:
    """Re-encode one logical [N,K] row-split tensor, a block of rows at a time."""

    old_spec, new_spec = get_format(old_format), get_format(new_format)
    n, k = shape
    old_geometry = row_split_geometry(old_spec, shape)
    base, high, scale = [], [], []
    for begin in range(0, n, BLOCK_ROWS):
        count = min(BLOCK_ROWS, n - begin)
        planes = split_row_planes(payload, old_geometry, begin, count)
        values = dequantize_row_split(planes, old_format, [count, k], device, dtype=torch.float32)
        quantized = quantize_matrix(values, new_format, device=device)
        block = encode_row_split(quantized.codes, quantized.scales, new_format, [count, k])
        block_planes = split_row_planes(
            torch.frombuffer(bytearray(block), dtype=torch.uint8),
            row_split_geometry(new_spec, [count, k]),
        )
        base.append(_plane_bytes(block_planes.base))
        high.append(_plane_bytes(block_planes.high))
        scale.append(_plane_bytes(block_planes.scale))
        print(f"  rows {begin + count}/{n}", end="\r", flush=True)
    print()
    joined = RowPlanes(
        torch.frombuffer(bytearray(b"".join(base)), dtype=torch.uint8),
        torch.frombuffer(bytearray(b"".join(high)), dtype=torch.uint8),
        torch.frombuffer(bytearray(b"".join(scale)), dtype=torch.uint8),
        n,
    )
    out = assemble_row_planes(joined, new_spec, k)
    return out.cpu().numpy().tobytes() if isinstance(out, torch.Tensor) else bytes(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--object", default="text/output_head")
    parser.add_argument("--format", default="q6_g64_fp16")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    with Artifact.open(args.source) as source:
        directory = source.directory
        target = source.object(args.object)
        if not isinstance(target, TensorObject):
            raise SystemExit("--object must name a tensor object")
        print(f"{target.id}: {target.format} {target.shape} -> {args.format} on {device}")
        payload = torch.frombuffer(bytearray(source.read_object(target.id)), dtype=torch.uint8)
        converted = requantize(payload, target.shape, target.format, args.format, device)
        del payload
        specs = []
        for obj in directory.objects:
            if obj.id == target.id:
                specs.append(TensorSpec(obj.id, obj.shape, args.format, obj.layout))
            elif isinstance(obj, TensorObject):
                specs.append(TensorSpec(obj.id, obj.shape, obj.format, obj.layout))
            else:
                assert isinstance(obj, ResourceObject)
                specs.append(ResourceSpec(obj.id, obj.bytes, obj.encoding))
        provenance = dict(directory.provenance)
        provenance["requantized"] = f"{args.object}: {target.format} -> {args.format}"
        with ArtifactWriter(
            args.output,
            specs,
            components=directory.components,
            bindings=directory.bindings,
            uses=directory.uses,
            metadata=directory.metadata,
            provenance=provenance,
        ) as writer:
            for obj in directory.objects:
                if obj.id == target.id:
                    writer.write_object(obj.id, converted)
                else:
                    writer.write_object(obj.id, source.iter_object(obj.id))
                print(f"wrote {obj.id}", end="\r", flush=True)
    print(f"\ndone in {time.time() - started:.0f}s -> {args.output}")


if __name__ == "__main__":
    main()
