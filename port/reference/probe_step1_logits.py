"""Step-1 logits arbiter: reference's first-token logits vs the engine probe dump.

Engine (v3 battery, factual-1) emitted top-1 = 167815 @ logit 25.8750.
This prints the reference's top-5 (token id, logit) on the same prompt so we can
tell whether engine and reference agree end-to-end at step 1 and the production
server is the outlier, or the two implementations already disagree.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference import Inkling2Reference  # noqa: E402

BASELINE = Path(__file__).resolve().parents[1] / "m0" / "parity" / "reference_baseline.json"
GGUF = "E:/AI_Models/LLM/nokeepai/Inkling-2-Mini-Unbound-v2-Q4_K_M.gguf"

entry = [e for e in json.loads(BASELINE.read_text(encoding="utf-8"))["entries"] if e["id"] == "factual-1"][0]
ids = entry["prompt_token_ids"]
print("prompt ids:", ids, flush=True)

import torch  # noqa: E402

model = Inkling2Reference(GGUF)
model.reset()
logits = model(list(ids))  # [vocab] logits at last position, post-softcap
vals, idx = torch.topk(logits.detach().float().cpu(), 10)
for v, i in zip(vals.tolist(), idx.tolist()):
    print(f"top: {i} {v:.4f}", flush=True)
print("ORACLE first continuation token:", entry["continuation_token_ids"][0], flush=True)
