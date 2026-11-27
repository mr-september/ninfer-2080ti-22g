import json, hashlib
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('E:/ninfer-all/models/inkling2/hf')
src = 'E:/LLM-Isomorph/data/calib/wikitext-2-raw.txt'
with open(src, 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()
print('source chars', len(text))
TARGET = 60000
ids = []
# tokenize in chunks to bound memory
CHUNK = 20000
for i in range(0, len(text), CHUNK):
    chunk = text[i:i+CHUNK]
    ids.extend(tok(chunk, add_special_tokens=False)['input_ids'])
    if len(ids) >= TARGET:
        break
ids = ids[:TARGET]
print('token count', len(ids))
with open('corpus.jsonl', 'w', encoding='utf-8') as f:
    for i, tid in enumerate(ids):
        f.write(json.dumps({'i': i, 'id': tid}) + '\n')
h = hashlib.sha256()
with open('corpus.jsonl','rb') as f:
    for b in f: h.update(b)
print('sha256', h.hexdigest())
