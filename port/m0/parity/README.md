# Inkling-2-Mini parity harness — server side (M0)

Reference server: `http://127.0.0.1:8191/v1`, model id `inkling-2-mini` (NInfer).
This directory is the SERVER-SIDE half of the parity harness: it pins the reference behavior so a future C++ port can be checked against it.

## Files
- `CAPABILITIES.md` — what the server supports (token ids NOT returned; echo works; no logprobs).
- `corpus.jsonl` — 60,000 token IDs (one JSON object per line: `{"i": n, "id": token_id}`), from `E:/LLM-Isomorph/data/calib/wikitext-2-raw.txt` (real corpus, reused for continuity). sha256: `9ff6aa603cb6958b70d3bf097e95aafc0b7eef13cd5c6d347319476c75282c57`.
- `prompts.json` — 20 short prompts (factual/code/math/instruction-following/multilingual), temperature 0.
- `reference_baseline.json` — greedy 64-token continuation per prompt, captured 2026-11-23.
- `build_corpus.py`, `run_baseline.py` — reproducible builders.

## Contract for the port
- Input: token IDs (from the HF tokenizer at `E:/ninfer-all/models/inkling2/hf`, `add_special_tokens=False`) — reference IDs, not server-internal IDs (the server returns none; re-tokenized boundaries can differ from the server's actual segmentation).
- Output: greedy continuation TEXT (temperature 0, max_tokens 64). Comparison is on decoded text first; re-tokenized token IDs are a convenience view, not authoritative.
- These continuations are behavioral evidence (a smoke test), not numerical qualification. They establish gross behavioral agreement; they cannot establish mathematical correctness.

## First-divergent-token report format
```
prompt_id: <id>
first_divergent_token_index: <k>   # 0-based index into continuation
reference_id: <int>
port_id: <int>
prefix_match_length: <k>
```
If the port diverges at token k, all tokens before k must match; tokens after k are reported but not required to match.

## KV coding is part of the parity contract (established 2026-11-24)

The pinned baseline was recorded on the production server (`serve-inkling-unbound.ps1`,
`--kv-dtype int8`). The engine's KV codings are not interchangeable, and comparisons are
only valid at matched coding. Source facts, from `E:/ninfer-all/repo/src/models/inkling2`:

- `--kv-dtype bf16` (the default) and `fp16` both select the plain **F16** cache
  (`kv_format.cpp:13`); the cache write is `__float2half(k * kf)` / `__float2half(v * vn)`
  in `kv_post_kernel` — K clipped and V normed *before* storage.
- `int8` selects Int8G64 planes, and **only the full-attention visits store planes**
  (`kv_format.h:8`); window-layer attention stays F16. Int8G64 per 64-dim group:
  scale = FP16-RNE(absmax/127), code = RN(x/scale) clamp ±127, decode = code*scale
  (`ops/kv_cache/int8_g64_codec.cuh`).
- Consequence measured: with the reference storing exact (F16) planes, the step-anchored
  probe against the int8 server missed at the first anchored token on 3/3 prompts —
  a coding gap, not chaos and not a formula bug. The reference now mirrors the engine's
  full-layer Int8G64 round-trip (`INK2_KV_CODING=int8`, default).

Engine behaviors still NOT mirrored in the reference: q8_1 decode-row activation
quantization and fixed-point expert sums. These are the only remaining suspects for
anchored misses at matched coding.

## Sensitivity bound on greedy agreement (measured 2026-11-24)

The model is chaotic at the activation level: in the FP32 torch reference, running the same 6-token prompt as one chunk vs prefill-plus-decode (structurally identical math, differing only in matmul tiling, ~1e-7 relative) diverges from visit 2 by 2.9e-4 in state and decorrelates the final logits by ~1e5 amplification across the 48 visits (bf16 rounding edges set the per-visit growth ~2.5x). Two trajectories that differ by any activation-scale perturbation (quantized or not) therefore flip greedy choices early without any semantic difference. Consequence: the first-divergent-token index against this baseline is a sample from the model's sensitivity, not a numerical qualification gate. Use it to catch semantic bugs (divergence accompanied by incoherent text is a bug; early divergence with coherent text is not).

## How to re-run
```
E:/k2-heretic/.venv314/Scripts/python.exe run_baseline.py   # re-capture baseline (server must be up)
E:/k2-heretic/.venv314/Scripts/python.exe build_corpus.py   # rebuild corpus (deterministic)
```

## What "matched effective weights" means
Two weight sets match when their **decoded tensor values are equal** (element-wise, after dequantization to a common FP type), independent of any generated output. Greedy agreement neither defines nor proves weight matching. Because this port requantizes the source GGUF into groupwise-int formats, effective weights differ from the server's by construction; exact numerical comparisons therefore require a reference with demonstrably matched effective weights and identical token inputs (e.g. a FP32 torch reference fed the same decoded values). The port's numerical qualification is the repository's own route — independent mathematical references per Op plus perplexity on the pinned corpus; the 20-prompt baseline stays a behavioral smoke test.

## Known limitations
- No per-token logprobs from the server (NInfer does not provide them); `logprobs` in baseline entries is null.
- Token IDs are derived by re-tokenizing echo text with the HF tokenizer, not from server-internal tokenization.
- Corpus is the first 60,000 tokens of the source file (not a stratified sample).
