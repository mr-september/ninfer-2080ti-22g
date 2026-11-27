# Inkling-2-Mini sm_75 port — memory ledger

Source: `E:/AI_Models/LLM/nokeepai/Inkling-2-Mini-Unbound-v2-Q4_K_M.gguf` (17,782,517,280 bytes, GGUF v3, 704 tensors, 63 kv).
Adapter under inspection: `C:/Users/Larry/AppData/Local/hermes/cache/scratch/inkling_port/basalt/src/models/inkling2/meta.hpp`.

## Key-prefix mismatch (measured, not assumed)

- `meta.hpp` reads keys with prefix `inkling2.*` (`kArch = "inkling2"`) and requires `general.architecture == "inkling2"`.
- The file stores keys with prefix `inkling2_loop.*` (e.g. `inkling2_loop.block_count`) and `general.architecture == "inkling2_loop"`.
- Consequence: `validate_meta()` as written returns `"architecture is 'inkling2_loop', this adapter requires 'inkling2'"` — it never reaches the per-key checks. UNVERIFIED by execution (no C++ build run in this session); the string mismatch is directly read from both files.
- `describe_meta()` would return all-zero `n_layers`/`n_embd`/`n_expert`/`n_ff_exp`/`context_length` (all keys missing under `inkling2.*`).
- Conversion JSON `E:/ninfer-all/models/Inkling-2-Mini-Unbound-v2-Q4_K_M-ninfer-v3.ninfer.conversion.json` confirms `model_type: "inkling2"` for the ninfer artifact (different namespace than the GGUF's `inkling2_loop`).

## Measured geometry (from GGUF, not assumed)

| Field | Value | Source |
|---|---|---|
| block_count | 24 | `inkling2_loop.block_count` |
| context_length | 1,048,576 | `inkling2_loop.context_length` |
| embedding_length | 2304 | `inkling2_loop.embedding_length` |
| vocab_size | 201,024 | `inkling2_loop.vocab_size` |
| vocab_size_unpadded | 200,058 | `inkling2_loop.vocab_size_unpadded` |
| expert_count | 128 | `inkling2_loop.expert_count` |
| expert_used_count | 8 | `inkling2_loop.expert_used_count` |
| expert_feed_forward_length | 1152 | `inkling2_loop.expert_feed_forward_length` |
| head_count | 24 (all layers) | `inkling2_loop.attention.head_count` |
| head_count_kv | 8 (all layers) | `inkling2_loop.attention.head_count_kv` |
| key_length / value_length | 128 | `inkling2_loop.attention.key_length` / `.value_length` |
| sliding_window | 1024 | `inkling2_loop.attention.sliding_window` |
| sliding_window_pattern | True=sliding; False=full at layers 5, 11, 17, 23 | `inkling2_loop.attention.sliding_window_pattern` |
| attn_res.block_size | 6 | `inkling2_loop.attn_res.block_size` |
| loop.start / loop.end / loop.count | 6 / 18 / 3 | `inkling2_loop.loop.start` / `.end` / `.count` |
| loop.gate_rank | 64 | `inkling2_loop.loop.gate_rank` |
| loop.varying | False | `inkling2_loop.loop.varying` |
| shortconv.l_cache | 4 | `inkling2_loop.shortconv.l_cache` |
| rope.dimension_count | 0 | `inkling2_loop.rope.dimension_count` |
| relative_length / relative_extent | 16 / 1024 | `inkling2_loop.attention.relative_length` / `.relative_extent` |
| qk_clip | 22.627416610717773 | `inkling2_loop.attention.qk_clip` |
| relative_clip | 8.0 | `inkling2_loop.attention.relative_clip` |
| log_scale_floor / log_scale_alpha | 8192 / 0.1 | `inkling2_loop.attention.log_scale_floor` / `.log_scale_alpha` |
| final_logit_softcapping | 30.0 | `inkling2_loop.final_logit_softcapping` |
| activation.situ_beta / situ_linear_beta | 4.0 / 25.0 | `inkling2_loop.activation.situ_beta` / `.situ_linear_beta` |
| expert_weights_scale | 8.0 | `inkling2_loop.expert_weights_scale` |
| rms_norm_epsilon | 9.999999974752427e-07 | `inkling2_loop.attention.layer_norm_rms_epsilon` |
| file_type | 15 | `general.file_type` |
| quantization_version | 2 | `general.quantization_version` |
| sampling | top_k=50, top_p=1.0, min_p=0.0, temp=1.0 | `general.sampling.*` |

Full-attention layers (measured): 5, 11, 17, 23 (4 layers). Sliding-window-1024 layers: the other 20.

## Weight bytes by group (measured, from GGUF reader)

| Group | Tensors | Elements | Bytes | Dtypes |
|---|---|---|---|---|
| expert_moe (gate_up + down) | 48 | 24,461,180,928 | 16,307,453,952 | Q4_K (12), Q5_0 (6), Q8_0 (8) |
| attention_matrices (q/k/v/gate/output) | 120 | 622,854,144 | 661,782,528 | Q8_0 (8) |
| lm_head (output.weight) | 1 | 463,159,296 | 379,935,360 | Q6_K (14) |
| embedding (token_embd.weight) | 1 | 463,159,296 | 260,527,104 | Q4_K (12) |
| loop_controller (k/q/v/ffn gain/forget/down + msg_query) | 156 | 28,422,144 | 113,688,576 | F32 (0) |
| ffn_gate_inp | 24 | 7,077,888 | 28,311,552 | F32 (0) |
| attn_rel (rel + rel_bias + rel_clip) | 72 | 21,636,096 | 13,553,664 | F32 (0), Q4_K (12) |
| conv_weights (k_conv, v_conv, post_conv attn+ffn, k/v_conv_proj) | 104 | 1,228,800 | 3,182,592 | F32 (0), Q8_0 (8) |
| attn_sink (attn_sink_proj) | 24 | 1,327,104 | 746,496 | Q4_K (12) |
| attn_q_clip | 24 | 73,728 | 294,912 | F32 (0) |
| attn_res_query | 23 | 71,424 | 285,696 | F32 (0) |
| ffn_norm | 24 | 55,296 | 221,184 | F32 (0) |
| ffn_res_query | 24 | 55,296 | 221,184 | F32 (0) |
| attn_norm | 20 | 46,080 | 184,320 | F32 (0) |
| attn_inp_norm (k/v/q inp_norm) | 12 | 27,648 | 110,592 | F32 (0) |
| output_aux (output_norm + output_res_query) | 2 | 4,608 | 18,432 | F32 (0) |
| exp_probs_bias | 24 | 3,072 | 12,288 | F32 (0) |
| norm_vectors (token_embd_norm) | 1 | 2,304 | 9,216 | F32 (0) |
| **TOTAL** | **704** | **26,070,385,152** | **17,770,539,648** | |

Note: "Q4_K_M" in the filename is a label; the actual file mixes Q4_K/Q5_0/Q8_0/Q6_K/F32 per tensor. Effective bpw is NOT a single number — do not assume 4.5 bpw. Q4_K effective bpw ≈ 4.5 is an estimate only.

## KV bytes per token (formula, constants substituted)

Per layer per token:
- K: 8 kv heads × 128 head_dim × 2 bytes (BF16) = **2,048 bytes**
- V: 8 kv heads × 128 head_dim × 2 bytes (FP16) = **2,048 bytes**
- Total per layer per token = **4,096 bytes**

Sliding-window tier (20 layers, window=1024):
- Fixed cap: 20 × 1024 × 4,096 = **83,886,080 bytes (80.0 MiB)**

Full-attention tier (4 layers, context C):
- 4 × C × 4,096 = **16,384 × C bytes**

## Conv / controller state per token (across 48 visits)

Visits: prelude 0-5 (6) + core 6-17 × 3 passes (36) + coda 18-23 (6) = **48 visits**.

Conv state per visit (per layer): attn_k_conv (4×1024) + attn_v_conv (4×1024) + attn_post_conv (4×2304) + ffn_post_conv (4×2304) = 26,624 elements.
- At BF16 (2 bytes, ASSUMPTION — not read from file): 53,248 bytes per visit.
- Across 48 visits: **2,555,904 bytes (2.44 MiB)** — fixed, not growing with context.

Controller state per core visit: gate_rank 64 × 2 bytes (BF16, ASSUMPTION) = 128 bytes.
- Across 36 core visits: **4,608 bytes (~4.5 KiB)** — fixed.

Note: conv/controller state is a fixed-size rolling buffer per visit, not per-token growing. "Per token" here means "for each token generated" (the state is reused across tokens).

## Totals at context 8K / 32K / 64K / 80K

| Context | Full-tier KV | Sliding-tier KV | Total KV | Weights | Total (weights + KV + state) |
|---|---|---|---|---|---|
| 8K (8192) | 134,217,728 (128.0 MiB) | 83,886,080 (80.0 MiB) | 218,103,808 (208.0 MiB) | 17,770,539,648 (16.55 GiB) | ~17.99 GiB |
| 32K (32768) | 536,870,912 (512.0 MiB) | 83,886,080 (80.0 MiB) | 620,756,992 (592.0 MiB) | 17,770,539,648 (16.55 GiB) | ~17.78 GiB |
| 64K (65536) | 1,073,741,824 (1024.0 MiB) | 83,886,080 (80.0 MiB) | 1,157,627,904 (1104.0 MiB) | 17,770,539,648 (16.55 GiB) | ~18.63 GiB |
| 80K (81920) | 1,342,177,280 (1280.0 MiB) | 83,886,080 (80.0 MiB) | 1,426,063,360 (1360.0 MiB) | 17,770,539,648 (16.55 GiB) | ~18.88 GiB |

Conv+controller state (fixed): ~2,560,512 bytes (2.44 MiB), included in totals above.

## Assumptions vs measured

Measured (from GGUF): all geometry constants, weight bytes by group, tensor counts, dtype assignments, sliding/full layer split, loop boundaries.

Assumed (not in file, marked):
- KV storage dtype: K=BF16, V=FP16 (per task spec; not read from GGUF).
- Conv state dtype: BF16 (2 bytes) — ASSUMPTION.
- Controller state dtype: BF16 (2 bytes) — ASSUMPTION.
- Controller state size: gate_rank 64 per core layer per visit — ASSUMPTION (the file stores the weight matrices, not the state size).
- sm_75 port target: weights stay in GGUF's Q4 block tier (mixed Q4_K/Q5_0/Q8_0/Q6_K/F32) — per task spec.
- "Per-visit conv state" interpretation: each of the 48 visits holds its own conv buffer — ASSUMPTION (alternative: shared across visits of the same layer, which would be 24 layers × 53,248 = 1,278,? bytes; not computed).

## Not in GGUF (separate files)

- Vision encoder (`Inkling2VisionModel`, hidden 1024, 23 layers, 16 heads, patch 20, adapter_merge 2, projection_hidden_dim 4096, posemb_grid 32, text_hidden 2304, intermediate 4096, rope_theta 10000, max_image_tokens 4096, image_token_id 200054): in conversion JSON + `E:/ninfer-all/models/inkling2/mmproj-Inkling-2-Mini-Q8_0.gguf` (345,910,944 bytes). NOT in the main GGUF (no vision tensors).
- DSpark (`Inkling2DSpark`, 3 layers, 24 heads, 8 kv heads, head_dim 128, intermediate 9216, block_size 8, markov_rank 256, sliding_window 1024, rope_theta 1e6, layer_types sliding/full/sliding, target_layer_ids 5/23/41/48, mask_token_id 0): in conversion JSON + `E:/ninfer-all/models/inkling2/dspark-Inkling-2-Mini-Q8_0.gguf` (402,278,560 bytes). NOT in the main GGUF.
- `E:/ninfer-all/models/inkling2/hf/` — not examined (not needed for text-only geometry).

---

## Orchestrator addendum (verified corrections, 2026-11-23)

1. KV must be counted PER VISIT (48 visits), not per stored layer. Full-attention tier is visited 8 times per token (layers 5 and 23 once; layers 11 and 17 three times each): 8 x 4,096 B x C = 2,560 MiB at C=81,920. Window tier: 40 visits x 4,096 B x min(C,1024) = 160 MiB fixed. Total KV at 80K: 2.66 GiB (the per-layer figure above, 1.33 GiB, is superseded). Engine behavior corroborates per-visit scaling: the live ninfer-all server KV delta 64K->80K was 512 MiB = 32,768 B/token = 8 full-tier visits x 4,096 B.
2. Exact total weight bytes (summed n_bytes): 17,770,539,648 B = 16.55 GiB. By dtype: Q4_K 8.80, Q8_0 4.65, Q5_0 2.61, Q6_K 0.35, F32 0.14 GiB. Experts (gate_up Q4_K + down split Q5_0/Q8_0) = 15.19 GiB; attention matrices Q8_0 = 0.62 GiB.
3. Port memory verdict at 80K BF16-K/FP16-V: 16.55 (weights) + 2.66 (KV) + ~0.6 (workspace/conv/graphs est) ~= 19.8 GiB of 22.3 — fits, thin headroom. v1 consumes the mixed blocks as-is (no expert requant); Q5_0->Q4_K on down-proj is the reserved size knob.
4. meta.hpp prefix mismatch is real (kArch="inkling2" at meta.hpp:12 vs general.architecture="inkling2_loop", 36 inkling2_loop.* keys, zero inkling2.* keys). The port reads the inkling2_loop.* namespace; basalt adapter code cannot ingest this GGUF without renaming.
