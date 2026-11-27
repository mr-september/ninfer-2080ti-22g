# Capabilities probe — http://127.0.0.1:8191/v1 (inkling-2-mini, NInfer 2026-11-22.5+719119a)

Probe date: 2026-11-23. All requests below were actually run.

## /v1/tokenize — 404 Not Found
```
POST /v1/tokenize
{"model":"inkling-2-mini","input":"Hello world"}
-> HTTP/1.1 404 Not Found (empty body)
```

## /v1/detokenize — 404 Not Found
```
POST /v1/detokenize
{"model":"inkling-2-mini","tokens":[15496,995]}
-> HTTP/1.1 404 Not Found (empty body)
```

## /v1/completions — works; echo works; logprobs NOT supported
```
POST /v1/completions
{"model":"inkling-2-mini","prompt":"The capital of France is","max_tokens":8,"temperature":0,"echo":true,"logprobs":1}
-> {"error":{"code":"logprobs_not_supported","message":"logprobs requires per-token log probabilities in the response, which NInfer does not provide","param":"logprobs","type":"invalid_request_error"}}
```
```
POST /v1/completions
{"model":"inkling-2-mini","prompt":"The capital of France is","max_tokens":8,"temperature":0,"echo":true}
-> {"choices":[{"finish_reason":"length","index":0,"logprobs":null,"text":"The capital of France is Paris.\n\nA. Noun\n\n"}],
    "usage":{"completion_tokens":8,"prompt_tokens":5,"total_tokens":13}, ...}
```
Echo returns the FULL text (prompt + continuation). `logprobs` is always null.

## /v1/chat/completions — works; per-token logprobs NOT returned
```
POST /v1/chat/completions
{"model":"inkling-2-mini","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":16,"temperature":0,"logprobs":true,"top_logprobs":1}
-> {"choices":[{"finish_reason":"length","index":0,"logprobs":{"content":[],"refusal":null},
    "message":{"content":"","reasoning_content":"The user asks a simple arithmetic question: What is 2+2","refusal":null,"role":"assistant"}}], ...}
```
`logprobs.content` is an empty array — no per-token logprobs. Chat also returns reasoning_content; not used for the baseline.

## /v1/models — works
```
GET /v1/models -> {"data":[{"id":"inkling-2-mini","architecture":{...},"context_length":81920,"max_model_len":81920,...}]}
```

## Summary
- Tokenization must use the HF tokenizer (`E:/ninfer-all/models/inkling2/hf`), not the server.
- Echo on /v1/completions returns prompt+continuation as text; token IDs are derived by re-tokenizing with the HF tokenizer.
- No per-token logprobs anywhere; `logprobs` in baseline entries is null.
