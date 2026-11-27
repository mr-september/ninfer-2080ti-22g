"""Reference top-8 at position 6 (prompt + space token) — how far is Canberra?"""
import sys, json
sys.path.insert(0, ".")
import torch
import reference as R
from tokenizers import Tokenizer

tok = Tokenizer.from_file("E:/ninfer-all/models/inkling2/hf/tokenizer.json")
base = json.load(open('../m0/parity/reference_baseline.json'))['entries']
e = [x for x in base if x['id'] == 'factual-1'][0]
model = R.Inkling2Reference("E:/AI_Models/LLM/nokeepai/Inkling-2-Mini-Unbound-v2-Q4_K_M.gguf")
model(e['prompt_token_ids'])
lg = model([220])
top = torch.topk(lg, 10)
print([(int(i), round(float(v), 2), repr(tok.decode([int(i)]))) for v, i in zip(*top)])
print("Canberra logit:", float(lg[40935]), "rank:", int((lg > lg[40935]).sum()) + 1)
