"""M2 length-bisect vs the fp16-KV oracle (port 8192).
Finds the anchor length where the fp32 reference first diverges from a
same-family engine with MATCHED KV coding. Greedy, single-step, forced prefix."""
import json, sys, time, urllib.request
sys.path.insert(0, ".")
import reference as R
from tokenizers import Tokenizer

TOK = Tokenizer.from_file("E:/ninfer-all/models/inkling2/hf/tokenizer.json")
URL = "http://127.0.0.1:8192/v1/completions"

def wait_ready(minutes=10):
    end = time.time() + minutes * 60
    while time.time() < end:
        try:
            urllib.request.urlopen("http://127.0.0.1:8192/v1/models", timeout=3)
            return
        except Exception:
            time.sleep(5)
    raise RuntimeError("oracle not ready")

def server_text(anchor):
    req = urllib.request.Request(URL, data=json.dumps(
        {"model": "inkling-2-mini", "prompt": anchor, "max_tokens": 1, "temperature": 0}
    ).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return r["choices"][0]["text"]

def main():
    wait_ready()
    base = json.load(open('../m0/parity/reference_baseline.json'))['entries']
    model = R.Inkling2Reference("E:/AI_Models/LLM/nokeepai/Inkling-2-Mini-Unbound-v2-Q4_K_M.gguf")
    for pid in ["factual-1", "factual-2"]:
        e = [x for x in base if x['id'] == pid][0]
        ids, cont = e['prompt_token_ids'], e['continuation_token_ids']
        plen = len(ids)
        for L in [plen, plen + 4, plen + 8, plen + 12, plen + 16, plen + 20, plen + 26, plen + 32]:
            anchor = ids + cont[:L - plen]
            st = server_text(anchor)
            lg = model(anchor)
            rt = TOK.decode([int(lg.argmax())])
            print(f"{pid} len={len(anchor)} server={st!r} ref={rt!r} {'MATCH' if st == rt else 'MISS'}", flush=True)
            model.reset(); model.pos = 0

if __name__ == "__main__":
    main()
