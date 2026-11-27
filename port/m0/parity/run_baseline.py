import json, time, urllib.request
from transformers import AutoTokenizer

BASE = 'http://127.0.0.1:8191/v1'
MODEL = 'inkling-2-mini'
tok = AutoTokenizer.from_pretrained('E:/ninfer-all/models/inkling2/hf')
prompts = json.load(open('prompts.json'))

def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

out = []
for p in prompts:
    prompt = p['prompt']
    prompt_ids = tok(prompt, add_special_tokens=False)['input_ids']
    resp = post('/completions', {'model': MODEL, 'prompt': prompt, 'max_tokens': 64,
                                  'temperature': 0, 'echo': True})
    full_text = resp['choices'][0]['text']
    full_ids = tok(full_text, add_special_tokens=False)['input_ids']
    # continuation = full minus prompt prefix
    if full_ids[:len(prompt_ids)] == prompt_ids:
        cont_ids = full_ids[len(prompt_ids):]
    else:
        # fallback: tokenize continuation text directly
        cont_text = full_text[len(prompt):]
        cont_ids = tok(cont_text, add_special_tokens=False)['input_ids']
    out.append({
        'id': p['id'], 'category': p['category'], 'prompt': prompt,
        'prompt_token_ids': prompt_ids,
        'continuation_token_ids': cont_ids,
        'continuation_text': full_text[len(prompt):],
        'logprobs': None,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'usage': resp.get('usage'),
        'finish_reason': resp['choices'][0].get('finish_reason'),
    })
    print(p['id'], 'prompt_tokens', len(prompt_ids), 'cont_tokens', len(cont_ids), 'finish', resp['choices'][0].get('finish_reason'))

with open('reference_baseline.json', 'w') as f:
    json.dump({'model': MODEL, 'base_url': BASE, 'temperature': 0, 'max_tokens': 64,
               'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'entries': out}, f, indent=2)
print('saved reference_baseline.json')
