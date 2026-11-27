"""M2 qualification driver: FP32 reference greedy continuation vs the pinned
server baseline (port/m0/parity/reference_baseline.json, 20 prompts x 64 tokens).

Contract (parity README): compare decoded text first; report the first divergent
token index in the README's format. Early divergence = semantics bug; the README
records why late divergence is bounded noise, not proof.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference import Inkling2Reference  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

PROMPTS = Path(__file__).resolve().parents[1] / "m0" / "parity" / "prompts.json"
BASELINE = Path(__file__).resolve().parents[1] / "m0" / "parity" / "reference_baseline.json"
GGUF = "E:/AI_Models/LLM/nokeepai/Inkling-2-Mini-Unbound-v2-Q4_K_M.gguf"
TOKENIZER = "E:/ninfer-all/models/inkling2/hf/tokenizer.json"
OUT = Path(__file__).resolve().parent / "m2_results.json"


def main(limit=None, max_tokens=64):
    baseline_all = json.loads(BASELINE.read_text(encoding="utf-8"))["entries"]
    tok = Tokenizer.from_file(TOKENIZER)
    model = Inkling2Reference(GGUF)
    results = OUT.read_text(encoding="utf-8") if OUT.exists() else "[]"
    done = {r["prompt_id"] for r in json.loads(results)}
    results = json.loads(results)

    for e in baseline_all if limit is None else baseline_all[:limit]:
        pid = e["id"]
        if pid in done:
            print(f"{pid}: already recorded, skipping")
            continue
        ids = e["prompt_token_ids"]
        gen = model.generate(ids, max_tokens)[len(ids):]
        text = tok.decode(gen)
        # First-divergent token on the baseline's own continuation IDs.
        ref_ids = e["continuation_token_ids"]
        k = 0
        while k < min(len(gen), len(ref_ids)) and gen[k] == ref_ids[k]:
            k += 1
        entry = {"prompt_id": pid, "first_divergent_token_index": k if gen[:len(ref_ids)] != ref_ids else -1,
                 "reference_id": ref_ids[k] if k < len(ref_ids) else None,
                 "port_id": gen[k] if k < len(gen) else None,
                 "prefix_match_length": k, "text": text}
        results.append(entry)
        print(f"{pid}: prefix {k}/{len(ref_ids)}; got[:60]={text[:60]!r}")
        OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")

    n = len(results)
    ok = sum(1 for r in results if r["first_divergent_token_index"] == -1)
    print(f"\nn={n} exact={ok} first-divergences={[r['first_divergent_token_index'] for r in results]}")


if __name__ == "__main__":
    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
