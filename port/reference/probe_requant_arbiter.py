"""Requant arbiter: authoritative reference math on RAW GGUF weights vs TIER-ROUNDTRIPPED weights.

The port's PPL defect is flat (+5.7 nats/token at every window, target-1 nll ~19.6).
Two candidate causes: (a) the v3 artifact's requant (Q4-g64 gate/up, Q5-g64 down,
Q8-g32 attn_output/attn_rel/token_embd, Q6-g64 output_head) versus the donor blocks;
(b) port math. This script runs the M2-authoritative reference (whose math matched
the donor per-op) on the SAME GGUF source values twice:

  arm 1  raw dequant values                       -> what the donor engine scores on
  arm 2  values requantized through the v3 tiers  -> exactly what our artifact stores

using the exact converter encoder (quantize_matrix; value = codes x fp16 scale).
If arm 2's distribution flattens (nll of the oracle continuation token jumps toward
~15-20) the requant is the defect; if arm 1 ~ arm 2, the artifact is exonerated and
port math is implicated. Port step-1 on this prompt was 167815 @ 25.875.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from reference import Inkling2Reference, Experts  # noqa: E402
from tools.convert.quantization.groupwise import quantize_matrix  # noqa: E402
from tools.artifact.formats import get_format  # noqa: E402

GGUF = "E:/AI_Models/LLM/nokeepai/Inkling-2-Mini-Unbound-v2-Q4_K_M.gguf"
BASELINE = Path(__file__).resolve().parents[1] / "m0" / "parity" / "reference_baseline.json"

TIERS = [
    ("ffn_gate_up_exps.weight", "q4_g64_fp16"),
    ("ffn_down_exps.weight", "q5_g64_fp16"),
    ("attn_output.weight", "q8_g32_fp16"),
    ("attn_rel.weight", "q8_g32_fp16"),
    ("token_embd.weight", "q8_g32_fp16"),
    ("output.weight", "q6_g64_fp16"),
]


def tier_for(name):
    for suffix, fmt in TIERS:
        if name.endswith(suffix):
            return fmt
    return None


def roundtrip(matrix, fmt):
    spec = get_format(fmt)
    n, k = matrix.shape
    q = quantize_matrix(matrix, fmt, device="cpu")
    g = spec.group_size
    out = (q.codes.float().reshape(n, -1, g) * q.scales.float().unsqueeze(-1)).reshape(n, -1)
    return out[:, :k]


class RequantProxy:
    """Same interface the reference uses; .values() returns tier-roundtripped FP32."""

    def __init__(self, src, fmt):
        self.src, self.fmt = src, fmt
        self.shape = src.shape

    def values(self, begin=0, end=None):
        flat = self.src.values(begin, end)
        k = self.shape[-1]
        rows = flat.numel() // k
        if rows * k != flat.numel() or begin % k:
            raise ValueError(f"window {begin}:{end} is not row-aligned for {self.src}")
        return roundtrip(flat.reshape(rows, k).float(), self.fmt).reshape(-1)


def patch(store):
    orig = store.tensor

    def tensor(name):
        src = orig(name)
        fmt = tier_for(name)
        return src if fmt is None else RequantProxy(src, fmt)

    store.tensor = tensor


def report(tag, logits, oracle_tok):
    lf = logits.detach().float().cpu()
    logsum = torch.logsumexp(lf, dim=-1)
    vals, idx = torch.topk(lf, 5)
    nll_oracle = float(logsum - lf[oracle_tok])
    nll_top1 = float(logsum - lf[idx[0]])
    print(
        f"{tag}: top1={int(idx[0])} logit={vals[0]:.4f} margin={float(vals[0]-vals[1]):.4f} "
        f"nll(oracle {oracle_tok})={nll_oracle:.4f} nll(top1)={nll_top1:.4f} "
        f"logsumexp={float(logsum):.4f}",
        flush=True,
    )
    print(f"{tag}: top5={[(int(i), round(float(v), 3)) for i, v in zip(idx, vals)]}", flush=True)


entry = [e for e in json.loads(BASELINE.read_text(encoding="utf-8"))["entries"] if e["id"] == "factual-1"][0]
ids = entry["prompt_token_ids"]
oracle_tok = entry["continuation_token_ids"][0]
print(f"prompt len={len(ids)} oracle_tok={oracle_tok}", flush=True)

mode = sys.argv[1] if len(sys.argv) > 1 else "both"

if mode in ("both", "raw"):
    m = Inkling2Reference(GGUF)
    m.reset()
    report("RAW   ", m(list(ids)), oracle_tok)
    del m
    Experts.CACHE.clear()
    gc_collect = __import__("gc").collect
    gc_collect()

if mode in ("both", "requant"):
    m = Inkling2Reference(GGUF)
    patch(m.store)
    m.reset()
    report("REQUANT", m(list(ids)), oracle_tok)
