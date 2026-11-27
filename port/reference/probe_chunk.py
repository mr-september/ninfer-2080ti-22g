"""Probe 2: single chunk prefill of ~300 corpus tokens; compare the reference's
argmax next token against the server's first continuation token on the same text.
If these disagree, the bug is general (chunk path too), not decode-state-specific."""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference import Inkling2Reference
from tokenizers import Tokenizer

ids = [json.loads(l)["id"] for l in open("../m0/parity/corpus.jsonl")][:300]
tok = Tokenizer.from_file("E:/ninfer-all/models/inkling2/hf/tokenizer.json")
text = tok.decode(ids)

model = Inkling2Reference("E:/AI_Models/LLM/nokeepai/Inkling-2-Mini-Unbound-v2-Q4_K_M.gguf")
logits = model(ids)
ref_tok = int(logits.argmax())

req = {"model": "inkling-2-mini", "prompt": text, "temperature": 0.0, "max_tokens": 4, "echo": False}
r = urllib.request.urlopen("http://127.0.0.1:8191/v1/completions", json.dumps(req).encode(), timeout=300)
out = json.load(r)
srv = out["choices"][0]["text"]
srv_ids = tok.encode(srv, add_special_tokens=False).ids
print("prompt tokens:", len(ids), "| server usage:", out["usage"]["prompt_tokens"])
print("REF next:", ref_tok, repr(tok.decode([ref_tok])), "| SERVER next:", srv_ids[0] if srv_ids else None,
      repr(srv[:30]), "| match:", bool(srv_ids) and srv_ids[0] == ref_tok)
