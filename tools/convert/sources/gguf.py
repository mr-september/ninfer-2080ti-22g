"""GGUF container reader: metadata, tensor inventory, and logical value sources.

The Inkling Unbound checkpoints use a file-local quantization type id space that is
shifted +1 from the llama.cpp enum for the K-quant and Q5/Q8 types (12=Q4_K, 8=Q8_0,
6=Q5_0, 14=Q6_K); the block layouts themselves are the standard llama.cpp ones and
are pinned by their exact bytes-per-block in the table below, so a misidentified
type fails at open time rather than silently corrupting values. Block index
semantics mirror the gguf-py reference decoders.
"""

from __future__ import annotations

import struct
from math import prod
from pathlib import Path

import torch

from .logical import LogicalSource

_MAGIC = b"GGUF"

# id -> (name, elements per block, bytes per block)
_GGUF_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("BF16", 1, 2),
    6: ("Q5_0", 32, 22),
    8: ("Q8_0", 32, 34),
    12: ("Q4_K", 256, 144),
    14: ("Q6_K", 256, 210),
}


def _f16(raw: torch.Tensor) -> torch.Tensor:
    return raw.view(torch.float16).to(torch.float32)


def _decode_q8_0(raw: torch.Tensor) -> torch.Tensor:
    blocks = raw.view(-1, 34)
    scales = _f16(blocks[:, 0:2])  # _f16 yields (n, 1): pair viewed as float16
    codes = blocks[:, 2:34].view(torch.int8).to(torch.float32)
    return (codes * scales).reshape(-1)


def _decode_q5_0(raw: torch.Tensor) -> torch.Tensor:
    blocks = raw.view(-1, 22).contiguous()
    n = blocks.shape[0]
    d = _f16(blocks[:, 0:2])  # (n, 1)
    # qh: 4 little-endian bytes as one uint32; bit i is the high bit of element i.
    # clone() (not contiguous()) because a 1-row slice reports stride (1,1) and
    # would skip the copy, leaving storage_offset 2 invalid for an int32 view.
    qh = blocks[:, 2:6].clone().view(-1, 4).view(torch.int32)
    bits = (qh >> torch.arange(32, dtype=torch.int32, device=qh.device)) & 1
    qs = blocks[:, 6:22]
    # Element order: low nibbles of all 16 bytes first, then high nibbles.
    nibbles = torch.stack((qs & 0x0F, qs >> 4), dim=1).view(n, 32)
    codes = (nibbles | (bits.to(torch.uint8) << 4)).view(torch.int8).to(torch.float32) - 16.0
    return (codes * d).reshape(-1)


def _q4_k_scale_min(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # 12 bytes (gguf-py get_scale_min): fields a, b carry the low 6 bits of the
    # eight scales/mins; bytes 8..11 carry the remaining high 2 bits.
    fields = scales.view(-1, 3, 4)
    a, b, c = fields[:, 0], fields[:, 1], fields[:, 2]
    sc = torch.cat((a & 0x3F, (c & 0x0F) | ((a >> 2) & 0x30)), dim=1).to(torch.float32)
    minimum = torch.cat((b & 0x3F, (c >> 4) | ((b >> 2) & 0x30)), dim=1).to(torch.float32)
    return sc, minimum


def _decode_q4_k(raw: torch.Tensor) -> torch.Tensor:
    blocks = raw.view(-1, 144).contiguous()
    n = blocks.shape[0]
    d = _f16(blocks[:, 0:2])
    dmin = _f16(blocks[:, 2:4])
    sc, minimum = _q4_k_scale_min(blocks[:, 4:16])
    d = (d * sc).view(n, 8, 1)
    dm = (dmin * minimum).view(n, 8, 1)
    # qs[128]: byte chunks of 32; a chunk's low nibbles fill sub-block 2c, highs 2c+1.
    qs = blocks[:, 16:144].view(n, 4, 32)
    codes = torch.stack((qs & 0x0F, qs >> 4), dim=2).view(n, 8, 32).to(torch.float32)
    return (d * codes - dm).reshape(-1)


def _decode_q6_k(raw: torch.Tensor) -> torch.Tensor:
    blocks = raw.view(-1, 210).contiguous()
    n = blocks.shape[0]
    d = _f16(blocks[:, 208:210])
    # One signed scale byte per 16-element block.
    scales = blocks[:, 192:208].view(torch.int8).to(torch.float32)
    d = (d * scales).view(n, 16, 1)
    # ql[128]: byte chunks of 64; a chunk's low nibbles fill rows 2c, highs 2c+1 (32 each).
    ql = blocks[:, 0:128].view(n, 2, 64)
    ql = torch.stack((ql & 0x0F, ql >> 4), dim=2).view(n, 8, 32)
    # qh[64]: four 2-bit codes per byte, same chunk/row structure over 32-byte chunks.
    qh = blocks[:, 128:192].view(n, 2, 32)
    qh = torch.stack(
        tuple(((qh >> shift) & 0x03) for shift in (0, 2, 4, 6)), dim=2
    ).view(n, 8, 32)
    codes = (ql | (qh << 4)).view(torch.int8).to(torch.float32) - 32.0
    return (d * codes.view(n, 16, 16)).reshape(-1)


_DECODERS = {6: _decode_q5_0, 8: _decode_q8_0, 12: _decode_q4_k, 14: _decode_q6_k}


def decode_block(type_id: int, raw: torch.Tensor) -> torch.Tensor:
    if type_id not in _GGUF_TYPES:
        raise ValueError(f"unknown GGUF type id {type_id}")
    if type_id == 0:
        return raw.view(torch.float32).clone()
    if type_id == 1:
        return _f16(raw)
    if type_id == 2:
        return raw.view(torch.bfloat16).to(torch.float32)
    return _DECODERS[type_id](raw)


class GGUFSource:
    """Seek-based GGUF reader with exact byte accounting per tensor."""

    def __init__(self, path: str | Path, device: str = "cpu"):
        self.path = Path(path)
        self.device = torch.device(device)
        with self.path.open("rb") as stream:
            if stream.read(4) != _MAGIC:
                raise ValueError(f"{self.path}: not a GGUF file")
            version = struct.unpack("<I", stream.read(4))[0]
            if version < 2:
                raise ValueError(f"{self.path}: GGUF version {version} is not supported")
            n_tensors, n_kv = struct.unpack("<QQ", stream.read(16))
            self.metadata = self._read_kv(stream, n_kv)
            self.tensors: dict[str, tuple[tuple[int, ...], int, int]] = {}
            self._tensor_offsets: dict[str, int] = {}
            for _ in range(n_tensors):
                name = self._read_string(stream)
                n_dims = struct.unpack("<I", stream.read(4))[0]
                dims = struct.unpack(f"<{n_dims}Q", stream.read(8 * n_dims))
                # This writer orders tensor info as dims, then type, then offset
                # (verified byte-by-byte in the Inkling file header; M1 gate (a)).
                (type_id,) = struct.unpack("<i", stream.read(4))
                (data_offset,) = struct.unpack("<Q", stream.read(8))  # from data start
                shape = tuple(reversed(dims))  # GGUF stores the fastest axis first
                if type_id not in _GGUF_TYPES:
                    raise ValueError(f"{self.path}: tensor {name} has unknown type id {type_id}")
                elements = prod(shape) if shape else 1
                _, per_block, block_bytes = _GGUF_TYPES[type_id]
                if elements % per_block:
                    raise ValueError(f"{self.path}: tensor {name}: {elements} not block-aligned")
                n_bytes = elements // per_block * block_bytes
                self.tensors[name] = (shape, type_id, n_bytes)
                self._tensor_offsets[name] = data_offset
            header_end = stream.tell()
        align = self.metadata.get("general.alignment", 32)
        if not isinstance(align, int) or align <= 0:
            raise ValueError(f"{self.path}: invalid general.alignment")
        self._data_start = (header_end + align - 1) // align * align
        total = sum(n_bytes for _, _, n_bytes in self.tensors.values())
        remaining = self.path.stat().st_size - self._data_start
        if remaining < total or remaining > total + (len(self.tensors) + 1) * align:
            raise ValueError(
                f"{self.path}: data region {remaining} bytes is inconsistent "
                f"with {total} payload bytes"
            )
        for name, (_, _, n_bytes) in self.tensors.items():
            offset = self._tensor_offsets[name]
            if offset % align or offset + n_bytes > remaining:
                raise ValueError(
                    f"{self.path}: tensor {name} region [{offset},{offset + n_bytes}) "
                    "is outside the data section"
                )
        self._file = self.path.open("rb")

    @staticmethod
    def _read_string(stream) -> str:
        length = struct.unpack("<Q", stream.read(8))[0]
        return stream.read(length).decode("utf-8")

    @classmethod
    def _read_value(cls, stream):
        kind = struct.unpack("<I", stream.read(4))[0]
        scalar = {
            0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
            6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
        }
        if kind == 8:
            return cls._read_string(stream)
        if kind == 9:
            element_kind = struct.unpack("<I", stream.read(4))[0]
            length = struct.unpack("<Q", stream.read(8))[0]
            if element_kind == 8:
                return [cls._read_string(stream) for _ in range(length)]
            fmt = scalar[element_kind]
            width = struct.calcsize(fmt)
            data = stream.read(width * length)
            return list(struct.unpack("<" + fmt[1] * length, data))
        if kind in scalar:
            width = struct.calcsize(scalar[kind])
            return struct.unpack(scalar[kind], stream.read(width))[0]
        raise ValueError(f"unknown GGUF metadata value kind {kind}")

    def _read_kv(self, stream, n_kv: int) -> dict:
        result = {}
        for _ in range(n_kv):
            key = self._read_string(stream)
            result[key] = self._read_value(stream)
        return result

    def tensor(self, name: str) -> LogicalSource:
        """Values for the tensor in its on-disk C-order logical shape."""
        if name not in self.tensors:
            raise ValueError(f"{self.path}: no tensor {name}")
        shape, type_id, _n_bytes = self.tensors[name]
        offset = self._tensor_offsets[name]
        _, per_block, block_bytes = _GGUF_TYPES[type_id]
        label = f"{self.path.name}:{name}"

        def read(begin: int, end: int) -> torch.Tensor:
            if type_id in (0, 1, 2):
                self._file.seek(self._data_start + offset + block_bytes * begin)
                raw = torch.frombuffer(
                    bytearray(self._file.read(block_bytes * (end - begin))), dtype=torch.uint8
                )
                return decode_block(type_id, raw.to(self.device))
            low_block, high_block = begin // per_block, (end + per_block - 1) // per_block
            self._file.seek(self._data_start + offset + low_block * block_bytes)
            raw = torch.frombuffer(
                bytearray(self._file.read((high_block - low_block) * block_bytes)),
                dtype=torch.uint8,
            ).to(self.device)
            values = decode_block(type_id, raw)
            return values[begin - low_block * per_block : end - low_block * per_block]

        return LogicalSource(shape, label, read)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "GGUFSource":
        return self

    def __exit__(self, *args) -> None:
        self.close()
