# M3 — artifact binding semantics (resolved 2026-11-25)

Source: `tools/convert/recipe.py` L534–565 + live artifact `inkling_2_mini.ninfer`.

- `bindings[name]` = `{"object": id}` OR `{"parts": [{"object", "range":[first,last)} …]}` — flat element ranges into coalesced objects, logical reshape by the caller. `artifact::Binder::parameter(name, Shape, {input})` resolves either form: the load layer passes only NAME + LOGICAL SHAPE; per-expert and qkv-gate row-ranges are the binder's job. No custom view code needed.
- `uses` input names (load `b.parameter` demand list must match): per-layer `mixer_input`, `attention/gated_output`, `controller_input`, `controller_state`, `ffn_input`, `moe/experts/E/product`; top-level `output_head ← text/final_hidden`. All recorded `A16Only`.
- Layer-conditional structure: full-attention layers = 5,11,17,23 (`query/key/value_input_norm` + `*_convolution_projection`); looped layers = 6–17 (controllers, conv projections); `input_norm` on 20/24; `residual_query` on 23/24 (absent layer 0). Stored blocks 24 run as 48 visits (prelude 0–5, core 6–17 ×3, coda 18–23); bindings address the 24 stored sets once.

## Non-matrix shapes (from object table; matrices follow (out,in) convention)

| name | shape | notes |
|---|---|---|
| embedding_norm, final_norm, final_residual_query, input_norm, ffn_residual_query | {2304} | direct FP32 |
| attention/query/key/value_input_norm | {2304} | direct FP32; full-attention layers only (executor falls back to `input_norm`) |
| residual_query (full layers 5/11/17/23) | {3,2304} pre-tiled, DISTINCT rows (measured weight/000170) | direct FP32 |
| attention/query_clip | {24,128} | direct |
| attention/relative_clip | {24,16} | direct |
| attention/relative_bias | {1024,16} | learned rel-bias (16 taps) |
| attention/relative_projection | {384,2304} | matrix (24×16) |
| attention/sink_projection | {24,2304} | per-head sink; direct FP32 (v2 artifact — executor `gemm_f32` reads raw rows) |
| attention/key_convolution, value_convolution | {1024,4} | direct |
| attention/post_convolution, ffn_post_convolution | {2304,4} | direct |
| attention/key_convolution_projection, value_convolution_projection | {32,2304} = kv_heads × conv_taps | full-attention layers only; direct FP32 (v2 artifact) |
| moe/router | {128,2304}; router_bias {128} | direct bias |
| controller/message_query | {4,2304} | |
| controller/{q,k,v,ffn}_gain, _forget | {2304,64} | rank 64 |
| controller/{q,k,v,ffn}_down | {64,4608} | reads [message;state] concat |
| moe experts gate/up | {1152,2304} | 128/layers via binder ranges |
| moe experts down | {2304,1152} | " |
| qkv_gate coalesced | {8192,2304} | q3072‖k1024‖v1024‖gate3072 |

Controller uses (measured from the artifact `uses` table, 2026-11-25 — supersedes the earlier
reverse-engineered split): `*_down` ← `controller_input` (q_down {64,4608} reads the
[message;state] concat); `*_gain`/`*_forget` ← `controller_state`; `message_query` has NO
`uses` entry (direct FP32 tensor, bound with no input demand). The binder validates demands
at load — mismatch fails loudly.
