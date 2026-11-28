"""Per-position nll curve: reference on factual-1 ids, positions 1..5 (raw GGUF weights)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from reference import Inkling2Reference  # noqa: E402

GGUF = "E:/AI_Models/LLM/nokeepai/Inkling-2-Mini-Unbound-v2-Q4_K_M.gguf"
ids = [int(x) for x in Path("E:/ninfer/.agent-tmp/factual1.ids").read_text().split()]

m = Inkling2Reference(GGUF)
m.reset()
prev = None
for pos, tok in enumerate(ids):
    if pos == 0:
        prev = m([tok])
        continue
    lf = prev.detach().float().cpu()  # logits after ids[pos-1], predicting ids[pos]
    logsum = torch.logsumexp(lf, dim=-1)
    vals, idx = torch.topk(lf, 5)
    nll = float(logsum - lf[tok])
    rank = 1 + int((vals > lf[tok]).sum()) if tok in idx.tolist() else ">5"
    print(f"pos {pos} target={tok} nll={nll:.4f} top1={int(idx[0])} "
          f"top5={[(int(i), round(float(v), 2)) for i, v in zip(idx, vals)]} rank={rank}", flush=True)
    prev = m([tok])
