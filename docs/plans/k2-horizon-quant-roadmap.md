# K2-Horizon port + rotated low-bit quantization on sm_75

Owner: Larry (product decision), OWL (spec + acceptance harness). Implementation: coding agent(s) on this repo.
Status: scope change APPROVED (Larry, 2026-11-09). Revised 2026-11-09 after review against the reference implementation and the engine sources: model facts corrected, reuse map added, quantization order changed (quality screen in simulation before any quantization kernel), gates rewritten around one oracle and one scoring protocol. Each milestone is gated; stop after any gate that fails and report.

## 1. Goal

Add `K2HorizonForCausalLM` (`model_type: k2_horizon`, dense, 8.999 B parameters including both embeddings) as a second explicit model, and use it to test whether rotated, error-compensated 4-bit weights served by this engine beat the llama.cpp 4-bit deployments on quality and decode speed on one RTX 2080 Ti. The product line becomes "Qwen3.x-27B + K2-Horizon, single 2080 Ti"; update `AGENTS.md`/`README.md` wording in M1's final batch, not before.

Decode at batch 1 is weight-bandwidth bound, so decode speed is set by stored bits per weight, not by activation precision. INT8 activations (W8A8, W4A8) only pay off in compute-bound prefill, where Turing's INT8 tensor-core rate exceeds its FP16 rate. The decode win therefore has to come from better 4-bit weights; activation quantization is a separate, prefill-only question (M5).

## 2. Model facts

Source checkpoint (abliterated variant, Apache-2.0): `E:\k2-heretic\output\` — HF safetensors; `modeling_k2_horizon.py` + `configuration_k2_horizon.py` are the reference implementation (transformers 5.22.1, `trust_remote_code`). Published as `Flakily6416/K2-Horizon-7B-Uno-heretic`. M0 re-derives everything below from that code; nothing here is exempt from checking.

| item | value |
|---|---|
| layers / hidden / intermediate | 36 / 4096 / 12288, SwiGLU (silu), no biases |
| attention | 32 query heads, 8 KV heads, head_dim 128, scale 128^-0.5, full causal, no sliding window, `query_key_norm: false`, `attention_gate_func: null` |
| RoPE | default type, θ = 1e7, rotary dim 128 (= head_dim), `rotate_half` convention |
| normalization | **grouped RMSNorm**: `layernorm_num_groups: 4`, i.e. per-group RMS over 4 × 1024 channels, then a 4096-wide weight multiplied in fp32 before the cast; eps 1e-6; used for input, post-attention and final norms |
| embeddings | untied, vocab 250,624: `model.embed_tokens.weight`, `lm_head.weight` |
| tokens | BOS 0; EOS {1, 250019} (`generation_config.json`) |
| tensor names per layer | `input_layernorm`, `post_attention_layernorm`, `self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj` |
| max context | 524,288 (config); no YaRN/dynamic RoPE requested |

Per decoded token the engine reads all weights except the input embedding: 36 × 192.94 M (layers) + 1.026 B (LM head) ≈ 7.97 B weights. At the repo's formats ([tensor-formats.md](../maintainer/tensor-formats.md)): `q8_g32_fp16` (8.5 bit) ≈ 8.47 GB/token, `q5_g64_fp16` (5.25) ≈ 5.23 GB, `q4_g64_fp16` (4.25) ≈ 4.23 GB, Q4 body with Q8 LM head ≈ 4.78 GB. Against the 2080 Ti's ~616 GB/s peak these cap decode near 73, 118, 146 and 129 t/s; at the ~550 GB/s ninfer reaches on the 27B, near 65, 105, 130 and 115 t/s.

Frontend facts:

1. Tokenizer: BPE, ByteLevel, NFC normalizer, 626 added tokens. The split regex differs from Qwen's: digits in runs of 1–3 (`\p{N}{1,3}`), ZWNJ/ZWJ inside letter runs, and a different punctuation class. The Qwen tokenizer accepts only its exact pattern ([tokenizer.cpp](../../src/models/common/qwen/frontend/tokenizer.cpp)), so K2 needs its own pre-tokenizer matcher; BPE merge, ByteLevel and NFC plumbing are reusable.
2. Chat template (`chat_template.jinja`, ~990 lines): turns are `<|ifm|im_start|>role\n…<|ifm|im_end|>`. Reasoning uses one of three tag pairs, `<ifm|think>`, `<ifm|think_fast>`, `<ifm|think_faster>`, chosen by the effort setting; the generation prompt opens the tag. The template calls `raise_exception` for any assistant history message without a thinking field, so the server must supply an empty one for clients that do not echo reasoning. The template also defines a tool-calling format and a `tool` role.
3. Do not port: MoVA attention, MoE routing, the Uno diffusion drafter, the other family sizes, sliding-window paths. No speculative decoding (the checkpoint has no MTP head).

## 3. Reuse map (engine as of `dd80950f`)

Kernels in this engine are registered for exact geometries; "the Qwen path already does it" holds only where the geometry matches. K2's attention geometry (32/8/128, θ 1e7) happens to match the DFlash drafter's.

| component | exists for K2 geometry | work |
|---|---|---|
| RoPE 32/8, dim 128, θ 1e7 | yes ([dispatch.cu:50](../../src/ops/rope/apply/dispatch.cu)) | reuse |
| BF16 context attention 32/8/128 | yes, DFlash drafter ([context_softmax_attention.cpp](../../src/ops/softmax_attention/dense/context/context_softmax_attention.cpp)) | candidate base for prefill |
| causal paged-KV attention + KV write | no: fixed at 24/4/256 ([causal_softmax_attention.cpp:23-32](../../src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp)) | new 32/8/128 instances, BF16 KV (M1) then INT8 KV (M2) |
| grouped RMSNorm D4096 × 4 groups | no: Plain/Offset/gated, whole-row only | new Epilogue/instance, with oracle |
| linear projection/split/pair/swiglu/add, BF16 and Q8 | no: dispatched per (N, K), e.g. `n5120_k6144` | new shapes: q/o 4096×4096, k/v 1024×4096 (or fused qkv 6144×4096), gate/up 12288×4096, down 4096×12288; tuning per [linear-tuning.md](../maintainer/linear-tuning.md) |
| LM head / top-k 250624×4096 | no (Qwen vocab shapes only) | new instance |
| embedding gather d4096 | check in M0 | likely new instance |
| W8A8 / W4A8 GEMM | no: `mma_s8` is used only by INT8-KV attention | new kernel family (M5 only) |
| Jinja engine, unicode/NFC, BPE | yes ([src/text](../../src/text)) | reuse; confirm the template renders |
| reasoning split, tool-call parser, prompt layout | Qwen-only (`</think>` hard-coded in [frontend](../../src/models/common/qwen/frontend)) | K2 frontend |

## 4. Oracles and measurement protocol

- Mathematical oracle: the HF reference in `E:\k2-heretic\output\`, eager attention, run on the same token ids. BF16 fits the card (HF peak 16.8 GB measured); fp32 logits for short sequences on CPU where a tie-break needs it. Every new Op is qualified against its own independent oracle per [op-development.md](../maintainer/op-development.md) before model-level gates.
- Secondary reference: the f16 GGUF (`E:\k2-heretic\quant\gguf\k2-uno-heretic-f16.gguf`), an independent implementation of the same weights. Its distance from HF is the measured noise floor for "correct implementation" in G1.
- Token ids: tokenize the corpus once with the HF tokenizer and feed ids to every engine, so tokenizer differences cannot enter quality comparisons; the ninfer tokenizer is gated separately (G1).
- Scoring: one protocol for all arms, fixed context window and stride, defined in M0. Quality metrics are paired over identical scored positions: mean KL divergence to the reference distribution (primary), ΔNLL with its paired standard error, and PPL. Unpaired PPL standard errors (±0.14 on the 60k subset) are not the comparison error. [perplexity.md](../perplexity.md) applies: values from different protocols are never compared. If `llama-perplexity` cannot be made to score the same windows, the llama.cpp arms are scored with its `--kl-divergence` mode against the f16 GGUF under the matched window, and ninfer/torch arms against HF BF16; M1 establishes that the two references are close enough to make this valid (G1 item 2).
- Corpus: `E:\k2-heretic\quant\ppl_sub.txt` (~60k tokens, wikitext-2 test). Calibration data (M3/M4) must not overlap it.
- Speed: decode tg at fixed lengths and prefill pp512/pp8192, interleaved A/B on an idle card, per [turing-port.md](../turing-port.md#standard-measurements-and-acceptance-rules); llama.cpp arms with `llama-bench` under the same conditions. Report effective GB/s from the bytes/token in §2.
- Behaviour: `E:\k2-heretic\quant\bench_and_probes.py` against the ninfer server, slices and classifiers from `E:\k2-heretic\config.toml` unchanged, greedy, content-scored, fixed token budget. Thinking-overrun (empty-content count) is the reported discriminating axis; refusal and sycophancy hits are a sanity report, not a gate, because they did not resolve 4-bit damage.

## 5. Baselines (measured 2026-11-08/09 on this machine, greedy)

| arm | file | decode t/s | peak VRAM | wikitext-2 PPL (llama-perplexity, 60k subset) | thinking-overrun /78 |
|---|---|---|---|---|---|
| HF BF16 (transformers) | — | 14.1 | 16.8 GB | — | — |
| GGUF f16 | 18.0 GB | 32.9 | 18.6 GB | 9.61 ± 0.14 | 16 |
| GGUF Q8_0 | 9.57 GB | 56.4 | 11.5 GB | 9.57 ± 0.14 | 20 |
| GGUF IQ4_XS (imatrix) | 5.11 GB | 90.2 | 8.1 GB | 10.02 ± 0.15 | 26 |
| GGUF Q4_K_M | 5.59 GB | 82.1 | 8.3 GB | 10.21 ± 0.15 | 33 |

These PPL values use llama-perplexity's protocol and stay valid only for comparisons within this table. M0/M1 re-measure the arms that gates use (f16, Q8_0, IQ4_XS, Q4_K_M) under §4's protocol and the same speed conditions. Q8_0 at 56.4 t/s reads ≈ 8.5 GB/token, about 478 GB/s effective. Deployment status quo stays the llama.cpp Q8_0 GGUF until a ninfer arm passes G4.

## 6. Milestones

### M0 — Design check (no merges)

Read `AGENTS.md`, [engine-architecture.md](../maintainer/engine-architecture.md), [op-development.md](../maintainer/op-development.md), [qwen3_5-model.md](../maintainer/qwen3_5-model.md), [weight-conversion.md](../weight-conversion.md), [tensor-formats.md](../maintainer/tensor-formats.md) and the reference modeling code. Deliver a short design note in `.agent-tmp/k2/` covering:

1. Op map for one layer and the head, checked against §3: which instances exist, which are new, and the oracle for each new Op.
2. Memory plan at BF16 and Q8, at 8K and 32K context, with BF16 and INT8 KV (BF16 KV is 147,456 B/token), including both embeddings and LM-head logits workspace, against ~19 GB usable (22 GB minus desktop). State whether the input embedding can live in host memory and what that costs.
3. Artifact: whether V3 `.ninfer` expresses this architecture (untied embeddings, grouped norm parameter, second vocab) without a format change, and the conversion route.
4. Frontend structure: a separate explicit `k2_horizon` frontend (tokenizer matcher, prompt layout, reasoning split over the three tag pairs, history-reasoning injection), sharing only `src/text` utilities; no family base classes. Pin the llama.cpp mainline `k2_horizon` reasoning/tool parser version used as the behaviour reference.
5. Tools: implement the K2 tool-call format, or reject `tools` for K2 with a documented error. Recommend one.
6. The scoring protocol of §4, including how the llama.cpp arms are scored on identical windows.

### M1 — BF16 port

`k2_horizon` as a second explicit model under `src/models/`, registered in [registry.cpp](../../src/models/registry.cpp); converter to V3 `.ninfer`; the new Ops from M0 with their oracles; K2 frontend and serving (OpenAI/Anthropic schema tests and [serving.md](../serving.md) together). BF16 weights, BF16 KV.

**G1:**
1. Each new Op passes its oracle suite.
2. Logits versus HF BF16 on the protocol corpus and on the CLI example prompts: mean KL divergence and top-1 agreement no worse than the GGUF f16 achieves against the same HF logits on the same positions. Report the distribution and the worst positions.
3. Tokenizer: ninfer ids identical to HF ids on the corpus and on a Unicode/digit/whitespace stress set.
4. Chat template: rendered prompts byte-identical to HF `apply_chat_template` for multi-turn, each reasoning effort, missing-reasoning history and tool messages; reasoning split matches the pinned llama.cpp parser on recorded outputs.
5. Behaviour report from `bench_and_probes.py`.
6. Qwen equality gates (AGENTS.md "Behavior-preserving changes") on the final batch: any edit to a shared dispatch file must leave every surviving Qwen kernel's SASS and the Qwen logit-compare/perplexity results identical.

### M2 — Native groupwise-int baselines

Q8 (`q8_g32_fp16`), Q5 and Q4 (`q4_g64_fp16`, existing stored-scale conversion methods) linear instances for K2's shapes, INT8 KV instances for 32/8/128, and the LM head at Q8.

**G2:**
1. Quality per §4 for Q8, Q5, Q4 and Q4-with-Q8-head, paired against ninfer BF16 and against HF.
2. Decode speed, effective GB/s against §2's bytes/token, and the same for Q8_0/IQ4_XS/Q4_K_M under identical conditions. If ninfer Q8 decodes slower than llama.cpp Q8_0, report this as an engine-efficiency finding before starting M4.
3. INT8 KV versus BF16 KV paired quality.
4. Behaviour report.

### M3 — Quality screen in simulation (decides M4–M5)

PyTorch fake-quantization of the HF model on the GPU (sequential with other GPU jobs), in a repo tool or `.agent-tmp/k2/`. Arms, all at `q4_g64_fp16` (symmetric, static 64-groups, fp16 scale, so every arm is storable in the existing format) and optionally `q5_g64_fp16`:

| arm | weights |
|---|---|
| RTN | the engine's existing conversion method, reproduced exactly |
| GPTQ | GPTQ (arXiv:2210.17323), static groups |
| R + GPTQ | offline rotation (below), then GPTQ |
| R + GPTAQ | offline rotation, then GPTAQ asymmetric calibration (arXiv:2504.02692) |
| R + GPTAQ, A8 | as above with per-token INT8 activations (prefill-relevant only) |

Rotation design, forced by the architecture:
- Residual-stream rotation must be block-diagonal, four 1024×1024 orthogonal blocks aligned to the RMSNorm groups. Only then does the grouped norm commute with it. Fold norm weights into the following linears first. The rotation is then absorbed offline into the embedding, q/k/v/gate/up inputs, o/down outputs and the LM head; no runtime cost.
- A per-head 128×128 rotation between v_proj outputs and o_proj inputs is absorbable offline.
- A rotation of the down_proj input (12288 = 12 × 1024) cannot be absorbed through SwiGLU; it needs an online Hadamard (e.g. H12 ⊗ H1024) before down_proj. Screen with and without it; the "with" arm implies one new Op.
- Hadamard versus learned/optimized rotations (SpinQuant arXiv:2405.16406, OptRot arXiv:2512.24124) is a screen dimension; start with randomized Hadamard.

Calibration: wikitext-2 train, disjoint from the scoring corpus (512 × 1024-token samples as default); report sensitivity to a task-calibrated mix (harmless_alpaca) on the layers the abliteration edited.

**G3:** the simulator's RTN arm reproduces ninfer's measured M2 Q4 quality on the same positions (validates the simulator). Proceed to M4 only if some weight-only arm's paired quality is strictly better than IQ4_XS's under §4. Proceed to M5 only if the A8 arm's quality loss relative to its weight-only twin is small enough to keep it ahead of IQ4_XS. Otherwise stop and report the grid.

### M4 — Rotated, error-compensated weights in the existing formats

The best M3 weight-only arm as a new conversion method producing `q4_g64_fp16` (and `q5_g64_fp16` if screened) codes, in the sense of [tensor-formats.md](../maintainer/tensor-formats.md): new codes, same decoding contract, no new format or kernel family. Rotations are applied offline in the converter. If the chosen arm needs the online down_proj Hadamard, add it as an explicit Op with its own oracle. No runtime weight repacking.

**G4 (project acceptance):** under §4, the ninfer arm dominates IQ4_XS: decode t/s ≥ 90.2 (re-measured value) and paired quality no worse, one of the two strictly better; thinking-overrun reported for both. Also report the arm against ninfer Q4 RTN and Q8, and against Q4_K_M. The arm's measured quality matches its M3 simulation on the same positions; a mismatch is a defect to explain, not noise.

### M5 — INT8-activation prefill (conditional on G3 and G4)

W4A8 and/or W8A8 GEMMs on INT8 tensor cores for the prefill path only, with online per-token activation quantization, using M4's rotated weights. Cover the token counts the scheduler actually issues, including tails. Design note first.

**G5:** prefill speedup at pp512/pp8192 over the W4A16 path at equal decode speed, with paired quality inside the M3 A8 prediction.

### M6 — Trellis-coded weights (research, unscoped)

QTIP-style trellis codes (arXiv:2406.11235) at 3–5 bits with decoding fused into the GEMM. PRISM-Q (github.com/Nif00/PRISM-Q) is a diffusion-model ComfyUI pack that expands trellis tiles into a temporary INT8 weight per call; that design costs extra bandwidth at LLM decode and is a reference for the code construction only. Only after G4; design doc before code.

## 7. Hard rules

- No change to Qwen3.x behaviour in any milestone; Qwen equality gates run on each batch that touches shared files.
- No generic model graphs, family base classes, plugin systems, string-driven execution, hidden allocation or runtime weight repacking.
- `master` only, Conventional Commits, one concern per commit, heavy gates once per batch.
- No FP8/FP4 paths. No MoVA, MoE or speculative decoding for K2.
- Single GPU job at a time, including M3's simulation runs.
- Scratch, logs, calibration caches and simulated checkpoints under `.agent-tmp/k2/`; retained artifacts under `out/` by explicit path.

## 8. Open questions (answer in M0 unless noted)

1. Can the input embedding stay in host memory without violating the Engine's explicit-allocation contract, and is it needed to fit BF16 at the target context?
2. Fused q/k/v (6144×4096) and gate/up (2 × 12288 × 4096) projections versus separate ones: which matches the existing split/pair/swiglu Op contracts with the fewest new instances?
3. Tools for K2: implement or reject (§6 M0 item 5).
4. Does `llama-perplexity` allow scoring the windows the §4 protocol defines, or does the cross-engine comparison go through KL divergence against the two references?
5. (M3) Is the online down_proj Hadamard worth one new Op, judged by the with/without screen?
