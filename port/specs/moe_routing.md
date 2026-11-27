# Inkling-2-Mini — MoE routing + activation mathematics (from basalt reference)

Reference clone (read-only): `E:/ninfer/port/basalt-reference`
Files used: `src/models/inkling2/session.cpp`, `pack.cpp`, `model.cpp`, `cpu_experts.cpp`, `prompt.cpp`, `kernels/glue.cu`, `kernels/kernels_parity.cpp`, `include/basalt/kernels/router.hpp`, `src/kernels/cuda/router.cu`, `include/basalt/kernels/cpu/swiglu.hpp`, `include/basalt/kernels/cpu/native_expert.hpp`, `src/kernels/cpu/native_expert.cpp`, `include/basalt/kernels/iq_kernels.hpp`, `include/basalt/prefill/moe_mmq.hpp`, `src/prefill/moe_mmq.cu`, `tools/convert/checkpoint_inkling2.py`, `src/models/inkling2/kernels.hpp`.
Skipped: dspark (draft model), loop controllers, residual mixing (per brief).

## Constants (from geometry.json metadata + code)
- `n_expert = 128` (metadata `expert_count`), `n_used = 8` (`expert_used_count`), `n_embd = 2304` (`embedding_length`), `n_ff_exp = 1152` (`expert_feed_forward_length`), `expert_weights_scale = 8.0` (`expert_weights_scale`), `gate_softcap = 4.0` (`expert_gate_softcap`), `up_softcap = 25.0` (`expert_up_softcap`), `router_gate = Sigmoid` (hard-coded in session.cpp, not from metadata), `k = 8` (`n_used`).
- `final_logit_softcapping = 30.0` (`final_logit_softcapping`) — NOT part of the routing/activation path (it caps the final logits; see UNRESOLVED).
- `activation.situ_beta = 4.0` and `activation.situ_linear_beta = 25.0` (geometry.json) — NOT read by basalt anywhere; see UNRESOLVED.

## 1. Router (gate matrix + bias + top-k + weights)

**Inputs:** `x` = the MLP input (the post-attention, post-norm fp32 row, `nrm`), one row per token; `w.router` = the router matrix (F32, row-major `[n_expert][n_embd]` = `[128][2304]`), `w.router_bias` = per-block `exp_probs_b.bias` (F32, `[128]`).

**Wiring (session.cpp:963-971):**
```cpp
bk::RouterArgs ra;
ra.n_expert = g_.n_expert;      // 128
ra.n_embd   = E;                // 2304
ra.k        = g_.n_used;        // 8
ra.weight_scale = g_.expert_scale;   // 8.0
ra.gate     = bk::RouterGate::Sigmoid;
bk::router(nrm, w.router, w.router_bias, rows, ra, x.rlog_, x.ids_, x.rw_, k.stream);
```
`ra` is also used for the next-visit guess (session.cpp:972-977, `ga.k = kGuessExperts / rows`).

**Router matrix load (model.cpp:295-296):** `w.router = S("ffn_gate_inp.weight", n_expert * n_embd)` — F32, flat `[128][2304]`; the GGUF tensor `blk.N.ffn_gate_inp.weight` is shape `[2304, 128]` (GGUF ne0=2304 fastest, ne1=128 experts), i.e. memory layout = `[128 experts][2304 embd]` row-major — matches. `w.router_bias = S("exp_probs_b.bias", n_expert)` (F32 `[128]`).

**RouterArgs defaults do NOT match inkling2** (router.hpp:16-26: `n_expert=384`, `n_embd=5120`, `k=6`, `weight_scale=1.5f`, `SqrtSoftplus`) — all fields are overridden in session.cpp; only `weight_scale` and `gate` are set there, the rest come from `g_` (see UNRESOLVED: no metadata key for the gate choice; sigmoid is hard-coded).

**Router math (router.cu):**
- Logits: `logits[t][e] = sum_c x[t][c] * gate[e][c]` — one block per gate row, one warp per token-column group; fp32, `router_logits_kernel` (router.cu:63-80) or the 4-rows-per-block `router_logits_rows_kernel<3,4>` for prompts with ≥64 tokens (router.cu:82-110, used when `n_tokens >= 64 && n_embd == 4*3*256 && n_expert % 4 == 0` — i.e. 2304 and 128, so the prompt path uses it).
- Score: `s = score(logit, gate)`; for Sigmoid (gate=1): `s = 1 / (1 + expf(-logit))` (router.cu:114-115: `score()`; `sqrt_softplus` for gate=0 is `sqrt(softplus(x))`, unused here).
- Biased score: `b = s + bias[e]` (router.cu:140-141); with `prior > 0 && held.held(e)`: `b += prior` (router.cu:142, `--route-prior`, off by default).
- Selection: rank experts by `b` descending (ties: lower index wins); the top-`k` (k=8) are chosen (router.cu:144-176, `router_select_token`).
- Weights: `weights[r] = s_r / (sum_{r=0..k-1} s_r + 1e-20) * weight_scale` — the **unbiased** scores `s` (not the biased ones), normalized by their sum plus 1e-20, times `weight_scale` (8.0) (router.cu:170-175).
- Ids are the chosen expert indices, in rank order (router.cu:162-167).

**Prompt path (prompt.cpp:523):** `bk::router_select(b().rlog, w.router_bias, rows, ra, b().ids, b().rw, strm())` — the same selection/weights from logits computed another way.

**Router bias is per-block** (`exp_probs_b.bias`, 24 blocks × `[128]`, dtype 0/F32) and is added to the biased score only for the rank — not to the weights.

## 2. Expert blob layout (per layer, per expert)

From `checkpoint_inkling2.py` (lines 175-260) and `experts.layout` (pack.cpp:140-159):
- The checkpoint's `w13_weight` has `2*ff = 2304` rows; **even rows = gate, odd rows = up** (`c13[e, 0::2]` gate, `c13[e, 1::2]` up — checkpoint_inkling2.py:257-258). So basalt stores **gate-then-up per expert**, not interleaved: gate rows at blob offset 0, up rows at `up_off = ff * row_bytes(h)`, down rows at `down_off = 2 * ff * row_bytes(h)`, then the 16-byte FP32 tail `{s_gate, s_up, s_down, 0}` (checkpoint_inkling2.py:199, 216-218, 257-260; `experts.layout` line: `layer gu_type d_type offset blob_bytes`).
- NVFP4 blob: `2*ff * nvfp4_row_bytes(h) + h * nvfp4_row_bytes(ff) + 16` (checkpoint_inkling2.py:199); the tail is `{s_gate, s_up, s_down, 0}` = amax/denominator per expert (checkpoint_inkling2.py:256-260), one copy per expert.
- `NativeExpertLayout` (iq_kernels.hpp:50-61): `gu_type`/`d_type` (NVFP4 for inkling2), `gu_row`/`d_row`/`up_row` (bytes per row), `up_off`/`down_off` (byte offsets inside the blob), `bytes` (whole blob, NVFP4: with its 16-byte FP32 scale tail), `swiglu_limit` (0 for inkling2), `gate_softcap`/`up_softcap` (>0: the softcapped SwiGLU).
- `native_expert_layout(gu_type, d_type, n_embd, n_ff, swiglu_limit=0, up_type=-1)` (iq_kernels.hpp:67-68) builds it; session.cpp:985-988 sets `L.gate_softcap = g_.gate_softcap` (4.0) and `L.up_softcap = g_.up_softcap` (25.0).

**GGUF shapes (geometry.json tensor_inventory):**
- `blk.N.ffn_gate_up_exps.weight` dtype 12 [2304, 2304, 128], 24 blocks — per expert `[2304, 2304]` (ne0=2304 h, ne1=2304 = 2*ff rows, ne2=128 experts); the pack tool reads it as `(E, 2*ff, h)` (checkpoint_inkling2.py:193) and splits even/odd rows.
- `blk.N.ffn_down_exps.weight` dtype 6 [1152, 2304, 128] (12 blocks) + dtype 8 [1152, 2304, 128] (12 blocks) — per expert `[1152, 2304]` (ne0=1152 ff, ne1=2304 h, ne2=128).
- `blk.N.ffn_gate_inp.weight` dtype 0 [2304, 128], 24 blocks — router gate matrix (see §1).
- `blk.N.exp_probs_b.bias` dtype 0 [128], 24 blocks — router bias (see §1).

## 3. Expert computation (gate/up → SwiGLU → down)

**GPU path (session.cpp:989):** `bk::native_expert_grouped(L, ...)` — grouped experts in the native format: group g's blob at `grp_ptr[g]`; entries read token `ent_tok[e]`'s q8_1 activation and write row `ent_dst[e]` of `out` (iq_kernels.hpp:84-88). The q8_1 activation is the MLP input quantized (`x.xq_[0]`).

**CPU path (cpu_experts.cpp):** `CpuExperts::Impl::run` — per-expert jobs over RAM-resident experts; `cpu::native_quant_act` quantizes the activation, `pool->run_split_multi_native(fmt, jobs, n)` runs the gate/up/down rows on the pool; `f.gate_softcap = g.gate_softcap; f.up_softcap = g.up_softcap` (cpu_experts.cpp:108-109). `CpuExperts::make` sets `p.n_expert = g.n_expert`, `p.k = g.n_used`, `p.job_of.assign(g.n_expert, -1)` (cpu_experts.cpp:94-96).

**SwiGLU expression (both paths):** `h = gate_softcap * tanh(g / gate_softcap) * sigmoid(g) * up_softcap * tanh(u / up_softcap)` where `g` = gate dot, `u` = up dot, the sigmoid sees the **raw** gate (swiglu.hpp:17-19, native_expert.hpp:50-61, moe_mmq.cu:105/117, iq_kernels.hpp:60-61). For NVFP4 the gate/up are first scaled by the tail's `s_gate`/`s_up` (moe_mmq.cu:114-117: `g *= s.x; u *= s.y;` then the softcapped expression, then `* s.z` (s_down) is applied after the down product — actually the tail's `s.z` is applied in the down pass).

**Exact kernel order (GPU prompt path, moe_mmq.cu:96-119, 338-345):** `swiglu_kernel<CAP=true>` — `g = row[2k]` (interleaved) or `row[k]` (non-interleaved), `u = row[2k+1]` or `row[n_ff+k]`; if `tail != nullptr`: `g *= s.x; u *= s.y;` then `h = cg tanh(g/cg) sigmoid(g) cu tanh(u/cu) * s.z` (s.z = s_down applied here); else `h = cg tanh(g/cg) sigmoid(g) cu tanh(u/cu)`. The non-softcapped branch (`CAP=false`): `h = g / (1 + expf(-g)) * u`.

**CPU native path (native_expert.cpp:164-165):** only NVFP4 rows take the softcapped SwiGLU; a softcapped layer of another type takes the ggml dot. The softcapped branch is `h = cg tanh(g/cg) sigmoid(g) uc` with `uc = cu tanh(u/cu)` (swiglu.hpp:17-19).

**Limited SwiGLU (unused for inkling2):** `limit > 0`: `g = min(g, limit); u = clamp(u, -limit, limit)` then `h = silu(g) * u` (swiglu.hpp:21-25). `swiglu_limit = 0` for inkling2 (native_expert.hpp:38, iq_kernels.hpp:59).

**Down projection:** `out[t][r] = down_r . hq[t]` (NVFP4: times `s_down`) — native_expert.hpp:70-73, moe_mmq.cu:117 (`* s.z`).

**Weights applied at the merge (glue.cu:600):** `moe_finish_kernel` — `x = sum_{s=0..k-1} w[(r-j)*k + s] * rows[((r-j)*k + s) * kWidth + c]` for each of 4 conv taps j (current row + 3 before it), then `y = (sum_j taps[j] * x_j + x_0) * rsqrt(4+1) + resid` (glue.cu:599-612). `mf.w = x.rw_` (session.cpp:994) — the router weights (already scaled by 8.0).

## 4. CPU/GPU consistency
- Both paths apply the same `gate_softcap`/`up_softcap` (4.0/25.0) with the same expression (swiglu.hpp:17-19 vs moe_mmq.cu:105/117).
- GPU: `native_expert_grouped` with `L.gate_softcap = g_.gate_softcap` (session.cpp:985-988). CPU: `f.gate_softcap = g.gate_softcap` (cpu_experts.cpp:108-109).
- kernels_parity.cpp:482-483 sets `L.gate_softcap = 4.f; L.up_softcap = 25.f` and reports "experts (NVFP4, softcapped SwiGLU)" (kernels_parity.cpp:540) — the same values.

## UNRESOLVED / inferred
- **`activation.situ_beta` (4.0) / `activation.situ_linear_beta` (25.0)** (geometry.json) are NOT read by basalt anywhere (grep for "situ" in the source tree: no hits). They are inferred to be the same as `expert_gate_softcap`/`expert_up_softcap` (4.0/25.0) by value match, but the mapping is not explicit in the code. Basalt reads `expert_gate_softcap`/`expert_up_softcap` from the pack's meta section (pack.cpp:76-77), written by `checkpoint_inkling2.py` from the checkpoint's `mlp_gate_softcap`/`mlp_up_softcap` (checkpoint_inkling2.py:502-503).
- **Router gate choice** (Sigmoid) is hard-coded in session.cpp:967 — no metadata key; the `RouterArgs` default (SqrtSoftplus) is overridden.
- **Router bias** (`exp_probs_b.bias`) is used only for the rank, not the weights; no metadata key for it (it is a tensor, not a scalar).
- **`final_logit_softcapping` (30.0)** is read by pack.cpp:78 as `g.logit_softcap` but is NOT part of the routing/activation path (it caps final logits).
- **`activation.situ_beta` semantics** (whether it is the gate softcap or the up softcap) — inferred by value match only.
