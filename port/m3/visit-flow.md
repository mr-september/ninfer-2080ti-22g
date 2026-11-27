# M3 — donor visit() flow (transcribed 2026-11-25 from executor.cpp:2114–2530)

Authoritative per-visit order. Port replaces GGUF `product()`/`gguf::moe_*` table machinery with engine `ops::linear` (q8_g64/q4_g64) — the sanctioned storage rewrite; kernel-call sequence and state semantics copy 1:1.

1. **Bank ops**: `save_prelude` snapshots block-residual buffer state; `restart` restores prelude snapshot into `bank` (pass-2/3 message source), marks inputs not ready.
2. **Control mix** (looped layers, before inputs): `kernels::mix` over `bank` states + `bank_last` (count = bank+1), `queries = lp.loop_queries` (4 rows), raw = `c.messages`, tokens = t.
3. **Attention input**: layer 0 → `rms_norm(buffer, input_norm)` from embeddings; else `mix(attention_mix(c,v))` unless fused from previous visit.
4. **Control apply**: `control(c, lp, 3, {x0,x1,x2}, messages+{0,1,2}·plane, norms = full?{q,k,v norms}:{input_norm×3})` → rewritten per-input activations + bf16/q8 variants. `distinct = control||full ? 3 : 1` picks k/v input indices ki/vi.
5. **Projections**: `qgr` rows = query block (+gate at +q_floats, +relative at 2·q_floats; stride `query_rows`) from input 0; `kv` = [k | v] (stride 2·kvd) from inputs ki/vi; `sinks` = gemm(query_heads); full layers: `taps` = k-taps rows + v-taps rows (kv_heads·kTaps each, stride 2·kv_heads·kTaps).
6. **query_post** (QueryArgs): clips (`query_clip`, `relative_clip` per-layer; `qk_clip`, `rel_clip` config), `log_scale=full` with alpha/floor, positions → `q`, `r`.
7. **kv_post** (KvArgs): conv histories at `site(v,0)`, `site(v,kvd)`; k/v conv weights; taps; qk_clip; eps. Decode: writes into seq planes (`vp.k/v.p`, seq_stride/head_stride/ring). Chunked-prefill staging + quantized mxfp8/staged paths are prefill/quant variants — decode driver uses the plain f16 branch.
8. **keep_history** (chunk only) — skip in decode driver.
9. **attention** (AttentionArgs): table = `lp.distances`, extent = full? relative_extent : sliding_window, window = full? 0 : window, scale 1/√128, splits (driver: 1), gate = qgr + q_floats (stride query_rows), out bf16.
10. **output product** → y; if `vp.start`: push block buffer, acquire cur.
11. **residual** (ResidualArgs): conv = output_conv, history `site(v,2·kvd)`, `mix = ffn_mix(c,v)`.
12. **control apply (ffn side)**: `control(c, lp, 1, {c.ffn_raw}, messages+3·plane, {ffn_norm})`.
13. **experts**: router gemm f32 → `route` (bias, config.route_scale, per-layer gate_up/down scales, pair_scale; sorted route for short calls); gate/up vector product → `situ` (pair_scale) → down product into **fixed-point `add_fixed` sums** (exact order-independent accumulation) → `residual` (conv = moe_conv, history `site(v, 2·kvd+hidden)`), mix = next visit's `attention_mix` when `fuses_next(v)` (sets inputs_ready) else `output_mix_args(c)` after last visit.

State per call: `blocks` (block-residual buffers) + `cur`, `prelude_*`, `bank`+`bank_last`; conv/keep history rings indexed by `site(v, offset)` per visit (per-visit conv state — looped visits of the same layer have SEPARATE history slots); KV planes per layer (vp.k/v).

Decode-driver simplifications (correctness-first): t=1; no chunk/staging/keep paths; splits=1; bf16 activations (no q8 activation path); f16 KV (quantized variants deferred to M4 int8-KV decision).
