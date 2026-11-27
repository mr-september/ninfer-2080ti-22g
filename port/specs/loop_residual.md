# Inkling-2-Mini — loop controller + depth-residual mixing (from basalt reference)

Reference clone (read-only): `E:/ninfer/port/basalt-reference`
Primary source: `src/models/inkling2/kernels/glue.cu`; secondary: `src/models/inkling2/kernels/ops.cu`.
EXTRACTION PROVENANCE: transcribed by the parent from a subagent's verified reading of glue.cu (the subagent failed before writing this file; line numbers are approximate `~` unless marked exact). ops.cu's `attnres_mix_kernel` (ops.cu:82) and `ctrl_blend_kernel` (ops.cu:187) were NOT read in full — see UNRESOLVED. Constants from `port/m0/geometry.json`.

## Constants
- `loop.start = 6`, `loop.end = 18`, `loop.count = 3`, `loop.gate_rank = 64`, `loop.varying = false`
- `attn_res.block_size = 6`, `embedding_length = 2304`, `block_count = 24`, `epsilon = 1e-6`, `sliding_window = 1024`
- Execution: 48 visits = prelude blocks 0–5, core blocks 6–17 × 3, coda blocks 18–23.

## Inventory (reconciled against geometry.json tensor_inventory and the artifact — EXACT)
- `blk.N.loop_msg_query.weight` (12, [2304, 4]) — blocks 6–17 only
- Per stream s ∈ {q, k, v, ffn}, blocks 6–17 (12 each): `loop_{s}_ctrl_down.weight` [4608, 64], `loop_{s}_ctrl_forget.weight` [64, 2304], `loop_{s}_ctrl_gain.weight` [64, 2304] — 156 controller tensors total
- `blk.N.attn_res_query.weight`: 23 tensors = 19 × [2304] (windowed blocks) + 4 × [2304, 3] (full-attention blocks {5, 11, 17, 23}); **absent in blk.0**
- `blk.N.ffn_res_query.weight`: 24 × [2304]; `output_res_query.weight` 1 × [2304]; `blk.N.exp_probs_b.bias` 24 × [128]

## 1. Controller path (per looped block, per stream s)
Source: glue.cu `ctrl_down_kernel` (~line 300), `ctrl_up_kernel` (~393), `ctrl_blend_kernel` (~447).

- **down**: per stream, per output pair, `hid[s][r][·] = dest·W_down ∥ msg·W_down` over bf16 row-pairs; each half (kWidth = 2304 wide) is RMS-normed per half: `half ← half · rsqrtf(ss/kWidth + 1e-6f)`; the two halves are summed → `scratch[s][r][rank]` (rank = 64).
- **up**: `gf[s][r][i] = up_s[i] · hid[s][r]` over `2*kWidth` rows → `gf[s][r][2*kWidth]`.
- **blend**: `y[c] = (1 − tanh(gf[kWidth + c])) · dest[c] + tanh(gf[c]) · msg[c]`, then `norm_q8(y, norm_w, eps) → (q8, norm_out)` — the blend output is stored as q8_1 via `q8_store3` (amax/127 quantization).
- **norm_q8(x, w, eps)**: `inv = rsqrtf(sum(x²)/kWidth + eps)`; `y = x·inv·w`; q8 encode `q8_store3` (scale = amax/127).

NOTE (UNRESOLVED): whether `ctrl_forget`/`ctrl_gain` rows index state-vs-hidden exactly as the kernel pairing above implies is inferred from the kernel, not from named variables; ops.cu `ctrl_blend_kernel` (ops.cu:187) not yet cross-checked against glue.cu's path.

## 2. Depth-residual mixing (attn-res / ffn-res / output-res)
Source: glue.cu `mix_rows` (~line 100), used by `post_attn_kernel` (~490) and `attnres_mix` wrapper (ops.cu:564–588).

- For each block-state i (states = accumulated block outputs + current; count = n_blocks + 1):
  - `l_i = ‖v_i‖` via `inv_l = rsqrtf(sum(v_i²)/kWidth + 1e-6f)`
  - score `q_i = query · v_i` (query = `attn_res_query` / `ffn_res_query` / `output_res_query` row; full blocks carry 3 query streams, [2304, 3])
  - softmax over states (max taken exactly; sum accumulated in state order)
  - `mixed = Σ_i softmax(q)_i · v_i · inv_l_i`
- Dispatch: `attnres_mix_kernel` (ops.cu:82) selects n_q = 1 / 3 / 4 (ops.cu:567–573) — consistent with 19 windowed [2304] queries and 4 full [2304, 3] queries; exact n_q=3↔full-block mapping cross-check still open (UNRESOLVED).

## 3. Block epilogue placement (glue.cu)
- `post_attn_kernel` (~490): output-projection result → 4-tap depth conv (`x0` added, scaled by `rsqrtf(kConv+1)` = 1/√5) → + residual → `mix_rows` over (n_blocks + 1) states → `norm_q8`.
- `moe_finish_kernel` (~590): expert-sum output → same 4-tap conv (taps, x0 added, ×1/√5) → + residual.
- The depth conv here mixes the CURRENT stream with previous depth states (kConv taps), distinct from the per-token attention CausalConv1d in `attention_conv.md`. `kMaxMix` bound: glue.cu:141.

## UNRESOLVED (M2 numerics must pin these)
1. ops.cu `ctrl_blend_kernel` (187) vs glue.cu ctrl_* kernels: same math? (rounding/q8 detail on blend output captured from glue.cu only.)
2. `attnres_mix_kernel` n_q dispatch (ops.cu:567–573) exact semantics for n_q=3 (full) vs n_q=1.
3. Forget/gain weight-to-stream assignment (which of `ctrl_forget`/`ctrl_gain` multiplies tanh for dest vs msg) — kernel-level reading says gain→msg term (tanh(gf[c])), forget→dest term (1−tanh(gf[kWidth+c])), but gf row provenance (up_s rows for gain vs forget) is inferred.
4. Whether the loop controller's dest/msg are the pre-block hidden state and block output respectively (assumed from kernel argument naming in glue.cu).
