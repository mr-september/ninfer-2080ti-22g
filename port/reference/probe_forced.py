"""Probe: prefill prompt + the baseline's own 64-token continuation, then decode 8.
Coherent output => the chunked-prefill path is healthy and the bug is single-row
decode state; incoherent => a general semantics bug."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference import Inkling2Reference
from tokenizers import Tokenizer

base = json.loads(Path(__file__).resolve().parents[1].joinpath("m0/parity/reference_baseline.json").read_text(encoding="utf-8"))["entries"]
e = [x for x in base if x["id"] == sys.argv[1]][0]
tok = Tokenizer.from_file("E:/ninfer-all/models/inkling2/hf/tokenizer.json")
model = Inkling2Reference("E:/AI_Models/LLM/nokeepai/Inkling-2-Mini-Unbound-v2-Q4_K_M.gguf")
ids = e["prompt_token_ids"] + e["continuation_token_ids"]
gen = model.generate(ids, 8)
print("FORCED:", tok.decode(ids)[:80] + "||" + tok.decode(gen[len(ids):]))
gen2 = model.generate(e["prompt_token_ids"], 16)
print("GREEDY:", tok.decode(gen2[len(e['prompt_token_ids']):]))
gen3 = model.generate(e["prompt_token_ids"] + e["continuation_token_ids"][:8], 8)
print("PREF8 :", tok.decode(gen3[len(e['prompt_token_ids']) + 8:]))
