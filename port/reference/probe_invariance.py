"""Chunk-vs-per-token invariance of the reference (weights shared)."""
import sys, json
sys.path.insert(0, ".")
import reference as R

base = json.load(open('../m0/parity/reference_baseline.json'))['entries']
ids = [x for x in base if x['id'] == 'factual-1'][0]['prompt_token_ids']
model = R.Inkling2Reference("E:/AI_Models/LLM/nokeepai/Inkling-2-Mini-Unbound-v2-Q4_K_M.gguf")

chunk = model(ids).clone()                      # prefill path, pos advances to 5
model.reset()
model.pos = 0                                   # rewind to feed one at a time
last = None
for i, tok in enumerate(ids):
    last = model([tok])
d = (chunk - last).abs().max()
print("chunk argmax", int(chunk.argmax()), "pertok argmax", int(last.argmax()),
      "max|diff|", float(d), "rel", float(d / chunk.abs().max()))
