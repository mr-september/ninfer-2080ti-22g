# Inkling-2-Mini — attention + convolution mathematics (from basalt reference)

Reference clone (read-only): `E:/ninfer/port/basalt-reference`
Files used: `src/models/inkling2/kernels/attn.cu`, `attn_prompt.cu` (note: ~2× attn.cu; used for prompt-tier softmax details), `glue.cu`, `ops.cu`, `prep.cuh`, `include/basalt/models/inkling2/kernels.hpp`.
Skipped: dspark*, prompt_ops.cu (per brief).

## Constants (all from kernels.hpp unless noted)
- `kHeads = 24`, `kKvHeads = 4`, `kHeadDim = 128`, `kConv = 4`, `kRel = 16`, `kConvRing = 256`, `kRelExtent = 1024`, `kWidth = kHeads*kHeadDim = 3072`, `kMaxMix = 8`, `kDecodeRows = 8`, `kDecodeRowsFp8 = 4`, `kFusedMaxSeq = 128`, `kFusedRows = 2`, `kFusedRowsFp8 = 1`, `kFusedTiles = 128`, `kMaxSplits = 32`, `kClusterSplits = 2`, `kRelHeads = 3`, `kProjQ = kHeads*kHeadDim`, `kProjKV = kKvHeads*kHeadDim`.
- `kOffQ = 0`, `kOffR = kProjQ` (=3072), `kOffGate = kProjQ + kProjKV + 16*kHeads` (=3072+512+384=3968), `kOffSink = kOffGate + kHeads*kHeadDim` (=3968+3072=7040), `kOffTaps = kOffSink + kHeads` (=7040+24=7064), `kConvTapsWidth = kKvHeads*kConv = 16`.
- `kDecodeRowsFp8 = 4`, `kDecodeRows = 8` (bf16), `kFusedRowsFp8 = 1`, `kFusedRows = 2`, `kMaxSplits = 32`, `kClusterSplits = 2`, `kRelExtent = 1024`, `kDecodeRows = 8`, `kThreads = 128`, `kFusedTiles = 128`.
- `kConvRing = 256` — ring of the last 256 positions for the conv inputs not in the current window.

## 1. Q/K/V preparation (`attn_prepare_kernel`, ops.cu:230-282; shared body in prep.cuh)

**Per-token, per-visit, per-row r (row = batch item in window), position p = pos[r]:**

- `tau = full ? 1 + log_alpha*log(max((p+1)/log_floor, 1)) : 1` — prep.cuh:21-23 (`tau_of`), used in ops.cu:233 (`full` layer) and glue.cu:239-240 (decode) and prep.cuh:172 (fused).
- **K/V (warps 0/1):** for each channel c = g*128 + lane*4 + u (u=0..3):
  - taps `wt[u][j]` = `taps[(c)*4 + (3 - j)]` (tap order reversed: tap 3 first, j=0 is newest) — prep.cuh:42-45 (`load_weights`), ops.cu:242-245.
  - conv input `x[j]` = `y[(r - j)*ld + c]` if r - j >= 0, else `ring[(pos[r]-j) % kConvRing * ring_w + c]` (zero if pos[r]-j < 0) — prep.cuh:31-42 (`conv_in`/`conv_ring`), ops.cu:233 (`conv_in4`).
  - for full-attention layers: `t[j] = wt[u][j] + y[kOffTaps + g*4 + j]` (input-dependent taps, per-head per-channel) — prep.cuh:86-88, ops.cu:258-262 (`dt`).
  - `acc = sum_{j=0..3} t[j] * x[j]`; `raw = x[0]` (current-input); `val = (acc + raw) * inv_k` where `inv_k = full ? 1/3 : rsqrt(4+1)` — prep.cuh:83-95, ops.cu:261-271, glue.cu:241-255.
  - ss = sum_u val[u]^2; `f = key ? min(qk_clip/(sqrt(ss)+1e-6), 1) : rsqrt(ss/128 + eps)` — prep.cuh:96-97, ops.cu:270-271, glue.cu:257-258.
  - store: `y_cache = val * f` (key: clipped; value: RMS-normalized). Keys stored as fp8-e4m3 or bf16 depending on `fp8` flag; values always bf16 (`store_kv4`, ops.cu:215-228 — actually `store_kv`/`store_kv4`: fp8 → uint8, else `__half`).
- **Q (warps 2/3, heads 3g..3g+2):** for head h: `q[u] = yq[kOffQ + h*128 + lane*4 + u] * q_scale[h*128 + lane*4 + u]`; `ss = sum q[u]^2`; `f = min(qk_clip/(sqrt(ss)+1e-6), 1) * tau`; `q_out = q * f` — prep.cuh:159-172, ops.cu:275-282, glue.cu:261-267.
- **Relative state (per head, 16 values, lanes 0..15):** `rv = yq[kOffR + h*16 + lane] * r_scale[h*16 + lane]`; `rs = sum rv^2`; `fr = min(rel_clip/(sqrt(rs)+1e-6), 1) * tau`; `r_out = rv * fr` — prep.cuh:174-179, ops.cu:281-283, glue.cu:265-267.
- **Sink (per head):** `sink[r*24 + h] = yq[kOffSink + h]` (raw, unscaled) — ops.cu:282, prep.cuh:179.

**Norms:** V uses RMS-style norm `rsqrt(ss/128 + eps)` (per-head, 128 channels) — ops.cu:272, glue.cu:257. Q and K use L2-clip: `min(clip / (||v||_2 + 1e-6), 1)` — ops.cu:274, glue.cu:263. Relative state uses L2-clip with `rel_clip` — ops.cu:282. **No eps for Q/K/rel; only V norm has eps (eps = metadata key, see UNRESOLVED).**

**Input-dependent conv taps (full-attention layers only):** `t[j] += dt[j]` where `dt = y[kOffTaps + g*4 + j]` (one float per tap per channel-group, per head g) — ops.cu:256-257, prep.cuh:87-88, glue.cu:252-254.

**Causal 4-tap conv semantics:** `val = (sum_{j=0..3} t[j]*x[j] + x[0]) * inv_k` — i.e. scaled-sum-plus-current-input (raw). **No SiLU** (SiLU is not present in these files; `sigmoidf_` appears only in attn_gate — see §3). `inv_k = 1/3` for full layers, `rsqrt(5)` for window layers (4 taps + 1).

**Ring state:** `ring[(pos % 256) * ring_w + c]` — last 256 positions of the conversation's K/V inputs (per-channel, full width `kKvHeads*kHeadDim = 512`) — prep.cuh:25-28, ops.cu:302-315 (`ring_store_kernel`). Written by `ring_store` after every row read the window (last 256 positions before the window); `ring_store` grid: `((width+255)/256, n)`, 256 threads.

## 2. Attention scoring

**Q·K scale:** `qk_scale = 1/sqrt(head_dim)` (`kHeadDim = 128` → scale = 0.088388...) — attn.cu:48-50 (`attn_fused_kernel`), attn.cu:78-80 (`attn_split_kernel`). **No other scale factor.**

**Relative position bias (per head, over last 1024 positions):** `bias[r, h, k] = attn_rel[h, k] * sum_{j=0..15} r_state[r, h, j] * r_state[k, h, j]` — attn.cu:82-90 (split), attn.cu:362-372 (fused), attn_prompt.cu:103-115 (prompt). `attn_rel` is a per-head 16-dim vector indexed by head h (`attn_rel + h*16 + j`). **Max distance = 1024** (kRelExtent); the bias term is used only for keys within the last 1024 positions (window tier) — attn.cu:90-91 (`kRelExtent` guard), attn_prompt.cu:115-116. For full-attention layers: no relative bias (`full` flag) — attn.cu:81-91, attn_prompt.cu:108-115.

**Per-head sink logit:** `sink_logit = sink_q[h] * sink[h]` where `sink_q[h]` is the per-head sink query (from `a.sink_q`), `sink[h]` is the per-head sink value (written raw in prep, ops.cu:282) — attn.cu:85-88 (split), attn.cu:364-368 (fused), attn_prompt.cu:116-120 (prompt).

**Score:** `score = q·k * qk_scale + bias + sink_logit` — attn.cu:95 (split), attn.cu:366 (fused), attn_prompt.cu:117 (prompt).

**Masking:** keys with `key_pos > cur_pos` are skipped (`if (key_pos > cur_pos) continue`) — attn.cu:96-97 (split), attn.cu:369-370 (fused), attn_prompt.cu:119-120 (prompt). For window tier: keys with `key_pos < cur_pos - 1024` are also skipped (`key_pos < cur_pos - kRelExtent`) — attn.cu:98-99 (split), attn.cu:371-372 (fused), attn_prompt.cu:121-122 (prompt). **Sliding window 1024 semantics:** window tier keeps only keys in `[cur_pos-1024, cur_pos]`; full tier keeps all keys ≤ cur_pos.

**Softmax:** online softmax (running max + running sum) — attn.cu:101-112 (split), attn.cu:373-384 (fused), attn_prompt.cu:124-138 (prompt). **Precision:** float (fp32) for the running max/sum; keys are bf16 or fp8 (fp8 → dequant to fp32 via `__nv_fp8_e4m3`); values bf16 (fp8 values not stored). **Ordering:** keys processed in ascending position order; the running max is updated before the running sum; the final output is `sum(p_i * v_i) / sum(p_i)` with `p_i = exp(score_i - m)` — attn.cu:101-112, attn.cu:373-384, attn_prompt.cu:124-138.

**Window tier vs full tier:** full tier (`full = true`): no relative bias, no sliding window, `tau > 1` (position-dependent scale), `inv_k = 1/3`; window tier (`full = false`): relative bias, sliding window 1024, `tau = 1`, `inv_k = rsqrt(5)`.

**Split-K / combine:** split kernel produces per-split partial outputs (`partial[split][head][dim]`) and per-split `(m, l)` (max, sum) — attn.cu:78-96; combine kernel merges splits with `(m_i, l_i)` weights — attn.cu:266-305 (`attn_combine_kernel`). Fused kernel (`attn_fused_kernel`) does the same in one block (attn.cu:266-305 is the combine; fused is at 305-365).

## 3. Output path

**attn_output projection:** not in these files — UNRESOLVED (no projection kernel in attn.cu/attn_prompt.cu/glue.cu/ops.cu/prep.cuh; the projection is in a separate kernel/module).

**Gate semantics:** `attn_gate_kernel` (ops.cu:317-329): `out = attn * sigmoid(yq[kOffGate + c])` (element-wise, fp32) — gate is the input-dependent per-channel gate from the projection output (offset kOffGate). Applied after attention output.

**attn_res_query / residual mixing:** `attnres_mix_kernel` (ops.cu:82-163): for each row r and head h: `m = max(q[h*16 + j]) over j`; `p = exp(q - m)`; `s = sum p`; `r = sum p * v[h*16 + j]`; `out[h*16 + j] = v[h*16 + j] * r / (s + eps)` where eps is a per-head scalar — ops.cu:82-163 (`attnres_mix`). **Note:** this is a different operation (relative-state mixing, not the attention residual); the attention residual is not in these files — UNRESOLVED.

**post_conv (residual handling):** `post_conv_kernel` (ops.cu:331-395): for each row r and channel c: `out[r,c] = (sum_j t[j]*x[j] + x[0]) * inv_k + add[r,c]` where `add` is the residual (current-input) — ops.cu:331-395. `inv = rsqrt(5)` (4 taps + 1). **No SiLU.** The `add` term is the residual (added after the scaled sum).

## 4. State per token per visit written to KV and conv state

**KV cache (per token per visit):**
- **Keys:** `k_cache[cell * kKvHeads * kHeadDim + head * kHeadDim + channel]` where `cell = pos % cap` (cap = cache capacity) — ops.cu:270-271, glue.cu:257-258. **Dtype:** fp8-e4m3 (uint8) if `fp8 = true`, else bf16 (`__half`). **Layout:** `[cell][head][channel]`, stride = kHeadDim per head, kKvHeads*kHeadDim per cell.
- **Values:** `v_cache[cell * kKvHeads * kHeadDim + head * kHeadDim + channel]` — same layout, **always bf16** (`store_kv4`, ops.cu:215-228 — actually `store_kv`/`store_kv4`: fp8 → uint8, else `__half`; values always bf16).
- **Relative state:** `r[r * kHeads * kRel + h * kRel + lane]` — fp32, 16 values per head, 24 heads — ops.cu:281-282, prep.cuh:174-179.
- **Sink:** `sink[r * kHeads + h]` — fp32, raw value — ops.cu:282, prep.cuh:179.
- **Conv ring:** `ring[(pos % 256) * ring_w + c]` — fp32, 512 channels (kKvHeads*kHeadDim), 256 positions — prep.cuh:25-28, ops.cu:302-315.

**conv state (per token per visit):** the 4-tap conv reads the last 4 positions (r-1..r-3) from the window or the ring; the ring is written by `ring_store` after every row read the window (last 256 positions before the window). **No separate conv state beyond the ring.**

## Magic constants (source)
- `1/3` (full-layer inv_k): hardcoded in ops.cu:233, prep.cuh:83, glue.cu:241, attn.cu:362-372 (fused).
- `rsqrt(5)` (window-layer inv_k): hardcoded in ops.cu:233, prep.cuh:83, glue.cu:241.
- `qk_scale = 1/sqrt(128)`: hardcoded in attn.cu:48-50, attn_prompt.cu:80-82.
- `tau_of`: `1 + log_alpha * log(max((p+1)/log_floor, 1))` — `log_floor` and `log_alpha` are metadata keys (passed via `AttnPrep`); `p` is the position.
- `kRelExtent = 1024`: hardcoded in kernels.hpp.
- `kConvRing = 256`: hardcoded in kernels.hpp.
- `kRel = 16`: hardcoded in kernels.hpp.
- `kConv = 4`: hardcoded in kernels.hpp.
- `kHeadDim = 128`: hardcoded in kernels.hpp.
- `kHeads = 24`, `kKvHeads = 4`: hardcoded in kernels.hpp.
- `qk_clip`, `rel_clip`, `eps`, `log_floor`, `log_alpha`: metadata keys (passed via `AttnPrep`) — UNRESOLVED: exact metadata key names (need the metadata module; not in these files).
- `attn_q_clip` weight: the L2 clip value `qk_clip` is a per-head scalar (from `a.qk_clip`) — UNRESOLVED: exact metadata key name (need metadata module).
- `attn_rel` tensor: per-head 16-dim vector — UNRESOLVED: exact metadata key name.
- `attn_sink` / `sink_q`: per-head scalars — UNRESOLVED: exact metadata key names.
- `attn_output` projection: UNRESOLVED (not in these files).
- `attn_res_query` residual mixing: UNRESOLVED (not in these files; the `attnres_mix_kernel` is a different operation).

## Review notes (parent, 2026-11-24)
- Constants above are basalt's model sizes (kKvHeads = 4). The Inkling-2-Mini file uses num_key_value_heads = 8 (verified from GGUF metadata and tensor shapes); take all constants from `port/m0/geometry.json` / artifact config, not from these literals.
- UNRESOLVED items (attn_output/residual mixing, metadata key spellings) are the re-dispatched MoE/controller task's scope plus the metadata module; cross-reference port/specs/ once both land.
- RESOLVED 2026-11-24: attn_output projection + residual mixing → `port/specs/loop_residual.md` §3 (post_attn_kernel: out-proj → 4-tap depth conv ×1/√5 + residual → mix_rows over block states → norm_q8). MoE routing/activation → `port/specs/moe_routing.md`. Metadata key names remain open.
