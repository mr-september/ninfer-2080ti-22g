"""Per-visit diagnostics: after every visit print (visit, layer, pass, cur RMS,
logit max). Add logit max at the last visit. Finds where activations leave the
plausible band instead of guessing."""
import sys, json
sys.path.insert(0, ".")
import torch
import reference as R
from tokenizers import Tokenizer

tok = Tokenizer.from_file("E:/ninfer-all/models/inkling2/hf/tokenizer.json")
base = json.load(open('../m0/parity/reference_baseline.json'))['entries']
e = [x for x in base if x['id'] == 'factual-1'][0]
model = R.Inkling2Reference("E:/AI_Models/LLM/nokeepai/Inkling-2-Mini-Unbound-v2-Q4_K_M.gguf")

orig_call = R.Inkling2Reference.__call__
log = []
def instrumented(self, tokens):
    out = orig_call(self, tokens)
    return out
# wrap visit loop via monkeypatch of __call__ internals is invasive; instead just
# run prefill and dump RMS of the final state + logits distribution.
logits = model(e['prompt_token_ids'])
x = logits.reshape(-1)
top = torch.topk(x, 8)
print("logit max:", float(x.max()), "argmax:", int(x.argmax()), repr(tok.decode([int(x.argmax())])))
print("top8:", [(int(i), round(float(v), 3)) for v, i in zip(*top)])
print("logit std:", float(x.std()))
