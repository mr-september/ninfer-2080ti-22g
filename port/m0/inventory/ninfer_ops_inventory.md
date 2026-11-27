# Inkling-2-Mini reuse map — E:/ninfer (branch `larry`, sm_75 / RTX 2080 Ti only)

Repo: `E:/ninfer` — lean C++/CUDA fork. NVFP4/K8V4/FP8 kernels and Blackwell files removed. Qwen4Exp/PLE/NF4/MoE model-side code and FP8 formats stay, without Turing kernels — Ops without Turing kernels throw `"<op>: not available on the Turing (sm_75) target"` (see `src/ops/unsupported/turing_unsupported.cpp`). V3 `.ninfer` is the only product artifact.

Seven Inkling features, each mapped to what the tree already has. Verdicts: **REUSE AS-IS** (fits, no geometry change) / **NEW KERNEL SAME CONTRACT** (same op contract, new geometry/constants) / **NEW OP** (no scaffold).

---

## 1. Looped layers (weight-sharing across layer iterations)

**No scaffold.** `src/models/registry.cpp` resolves exactly one architecture per artifact (`Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM` / `Qwen4ExpForCausalLM`); `src/models/registry.h` `Architecture` enum has no loop concept. `src/artifact/schema.h` `Component { Json config; std::optional<std::string> target; std::map<std::string, ObjectHandle> resources; std::optional<Proposal> proposal; }` — loop metadata would have to live in `Component::config` (the free-form `Json`) and the `bindings` map would need per-loop-iteration parameter names (e.g. `layer.{i}.*` vs a shared `layer.*` binding). No loop/untied-layer concept exists in the artifact format.

- Path: `src/models/registry.cpp`, `src/artifact/schema.h`
- Verdict: **NEW OP** (frontend/artifact-level, not a kernel). Loop metadata must be expressed in `Component::config` JSON + binding naming convention; the loader (`src/artifact/schema.cpp` `parse_components`, `parse_bindings`) does not reject unknown config keys, so this is a format-convention choice, not a format change.

## 2. No-RoPE relative-bias attention with per-head sink + sigmoid output gate

**No scaffold.** RoPE exists (`src/ops/rope/`) but Inkling is no-RoPE, so `rope` is simply unused. Relative-bias attention: no op anywhere in `src/ops/` computes a per-position relative bias (checked: `softmax_attention/dense/{causal_cache,context}`, `sliding_window`, `qsa` — none take a bias tensor). Per-head sink + sigmoid gate: no op. The sigmoid/gating family is `elementwise` (check `src/ops/elementwise/` — generic) and `gated_residual` (BF16 norm/combine, not attention-gate).

- Path: `src/ops/rope/` (unused), `src/ops/softmax_attention/` (no bias support), `src/ops/elementwise/` (generic)
- Verdict: **NEW OP** — new attention kernel (or extend `softmax_attention/sliding_window`) that takes a relative-bias tensor + per-head sink + sigmoid gate. No existing contract to reuse.

## 3. Sliding-window 1024 + full attention layers

**Partial scaffold.** `src/ops/softmax_attention/sliding_window/sliding_window_attention.cpp` — computes paged sliding-window attention (BF16 K, FP16 V, `kHeadDim=128`, `kQHeads=32`, `kKVHeads=8`, `kExpectedScale=0.08838834764831844055f`). Fixed geometry: `registered_window(window) { return window == 2048; }` — **only 2048 is accepted today**; `require_profile` throws `std::invalid_argument(op + ": unsupported head geometry")` / `": supported window is 2048"`. `plan.h`: `kSlidingWindowMaxSplits=32`, `SlidingWindowAttentionPlan { route; tokens; warps; key_block; reduce_warps; split_capacity; max_context; window; }`, `select_sliding_window_attention_plan(window, tokens, batch, envelope)`. `launch.h`: `launch_sliding_window_attention(q, query_k, query_v, positions, valid_columns, state_slots, scale, context, plan, partial_acc, partial_m, partial_l, out, stream)`.

- Path: `src/ops/softmax_attention/sliding_window/{sliding_window_attention.cpp,plan.h,launch.h}`
- Verdict: **NEW KERNEL SAME CONTRACT** — parameterize `registered_window`/`require_profile` to accept 1024 (and a full-attention route), keep the contract (BF16 K / FP16 V / 128 head dim / 32 q heads / 8 kv heads). 1024 is a small constant change, not a new op.

## 4. Causal conv kernel-4 on K/V/attn-out/MLP-out

**Scaffold, near-exact.** `src/ops/linear_attention/causal_conv/causal_conv1d_silu.cpp` — computes causal depthwise conv1d (kernel-4, BF16) + SiLU. Fixed constants: weight shape `[C,4]` (`require_weight_shape`: `weight.ne[1]!=4` → throws `"weight must have shape [C,4]"`), conv_state `[C,3]` (`require_state_shape` → `"conv_state must have shape [C,3]"`), snapshot `[C,3,S]` with `S>=B*W`, `kMaximumBatch=8`, `kMaximumWidth=16`. Routes: `Decode` (tokens==1), `TokenParallel` (tokens<=32), `PrefillScalar` / `PrefillPairs` (alignment check on addresses), snapshot routes `Decode/TokenParallel/Sequence/BatchedParallel/BatchedSequence`. `plan.h`: `select_causal_conv_plan(x, weight, state_in, state_out, out)`, `select_causal_conv_split_plan(tokens)`, `select_causal_conv_snapshot_plan(tokens, batch, masked)`.

- Path: `src/ops/linear_attention/causal_conv/{causal_conv1d_silu.cpp,plan.h}`
- Verdict: **REUSE AS-IS** — kernel-4 is hard-coded and is exactly what Inkling wants; BF16 in/out; routes decode/prefill. No sm_75 unsupported-throw (it is a Turing-compiled kernel). Apply it to K/V/attn-out/MLP-out by instantiating per-tensor instances.

## 5. Block-attention-residual softmax mix

**Partial scaffold.** `src/ops/gated_residual/` — `operands.h`: `kWidth=2560`, `kStreams=4`, `kRank=320`, `kResidualWidth=kWidth*kStreams` (= 10240); `enum class Form { ExpandNorm, GroupedNorm, Combine, CombineNorm, AddNorm }`; `Operands { input; block; injection; weight; residual; normalized; columns; epsilon; }`; `validate(form, ...)` enforces `[kWidth, a, b, c]` BF16 shape (`check(t, dtype, kWidth, a, b, c, ...)`); `launch_norm_combine(form, operands, execution)`. `norm_combine.cpp`: `hc_grouped_rmsnorm` (GroupedNorm), `hc_combine` (Combine), `hc_combine_norm` (CombineNorm / AddNorm). `norm_combine.cpp` is BF16 and Turing-compiled.

`src/ops/moe/topk_softmax_swiglu_moe/` and `src/ops/gated_residual/` are separate. The "block-attention-residual softmax mix" (softmax over blocks of attention-residual) is **not** the same as `hc_combine` (which is a linear combine of residual + block + injection). `hc_mix` (the FP8 hyper-connection final mix) throws on sm_75: `unsupported("hc_mix")` → `"hc_mix: not available on the Turing (sm_75) target"` (`src/ops/unsupported/turing_unsupported.cpp`).

- Path: `src/ops/gated_residual/{operands.h,operands.cpp,norm_combine.cpp}`, `src/ops/unsupported/turing_unsupported.cpp`
- Verdict: **NEW OP** — the softmax-over-blocks mix is not in the tree. `gated_residual` is a useful reference for the BF16 residual-combine contract but does not compute softmax-mix. `hc_mix` is the closest concept but is FP8-only and not Turing-compiled.

## 6. Sigmoid top-8-of-128 MoE with softcapped SwiGLU

**Scaffold, wrong geometry + wrong format.** `src/ops/moe/` — `operands.h`: `validate_route_profile` requires `hidden==2560 && experts==512 && top_k==10` → throws `"topk softmax route: unsupported geometry"`; `validate_expert_profile` requires fp8 (`FP8_E4M3FN_BLOCK128X128_BF16` + `BlockScaleM128K128` + `AllowA8`) and `intermediate==640`, down==gate_up type/layout/policy → throws `"routed SwiGLU: unsupported block-scaled profile"`; `validate_shared_profile` requires `hidden==2560 && intermediate==640 && BF16 both && Contiguous both && A16Only both` → throws `"shared SwiGLU: unsupported BF16 profile"`; `validate_output` allows BF16 Write / FP32 Write / BF16 AddResidual → throws `"MoE: output requires FP32 Write, BF16 Write or BF16 AddResidual"`; `validate_interval` requires `first>0 && last>=first` → throws `"MoE workspace: invalid token interval"`; `validate_subset_kind` (All/None/MaskOnes/MaskZeros) → throws `"routed SwiGLU: invalid expert subset"`; `MoeOperands` ctor requires CUDA + `multiprocessor_count()>0` → throws `"MoE: invalid CUDA execution view"`. `routed_swiglu_profile` helper: `gu.region.rows%2==0 && gu.region.count==dn.region.count && gu.columns==dn.region.rows && gu.region.rows/2==dn.columns && top_k>0 && top_k<=gu.region.count && valid policies` → throws `"routed SwiGLU banks have inconsistent geometry or policy"`. `topk_softmax_swiglu_moe/dispatch.cpp`: `shape_dispatch` switches on `profile.routed.ffn.down_type` — only `FP8_E4M3FN_BLOCK128X128_BF16` → `fp8_h2560_i640_e512_k10()`; default throws `"unsupported MoE expert format"`. `topk_softmax_swiglu_moe/entry.cpp`: `topk_softmax_swiglu_moe_workspace_capacity_bytes` + `topk_softmax_swiglu_moe` — requires BF16 destination (`"complete MoE requires BF16 destination"`). `weights.cpp`: `swiglu_profile(gu, down)` requires `gu.n%2==0 && gu.n/2==down.k && gu.k==down.n` → throws `"SwiGLU matrices have inconsistent geometry or policy"`; `routed_swiglu_profile(gu, dn, top_k, ...)` same geometry contract.

**Crucially, the entire MoE path is FP8-only on Turing**: `src/ops/unsupported/turing_unsupported.cpp` throws `unsupported("launch_experts")` / `"launch_input"` / `"launch_vector"` → `"launch_experts: not available on the Turing (sm_75) target"` etc., and `fp8_h2560_i640_e512_k10` → `"fp8_h2560_i640_e512_k10: not available on the Turing (sm_75) target"`. So `topk_softmax_swiglu_moe` never runs on sm_75 today.

- Path: `src/ops/moe/{operands.h,operands.cpp,weights.cpp,topk_softmax_swiglu_moe/{entry.cpp,dispatch.cpp}}`
- Verdict: **NEW OP** — Inkling needs sigmoid gating (not softmax), top-8-of-128 (not top-10-of-512), softcapped SwiGLU (not plain SwiGLU), and BF16/dequant-friendly expert codes. The existing `moe/` scaffold is Qwen3.8-specific (2560/512/10/640) and FP8-only; none of its geometry or format fits. The `MoeOperands`/`validate_*` contract style is a useful template but the kernels are new.

## 7. NVFP4/MXFP8 expert weight codes needing dequant on Turing

**No scaffold.** NVFP4 and MXFP8 are not in the tree (NVFP4 removed; MXFP8 not present). The FP8 formats that remain are `FP8_E4M3FN_BLOCK128X128_BF16` (block-scaled, `BlockScaleM128K128` layout, `AllowA8` policy) — see `src/ops/moe/operands.h`, `src/artifact/formats.h`. Turing has no FP8 kernels (`src/ops/unsupported/turing_unsupported.cpp`). No dequant path exists for Turing.

- Path: `src/artifact/formats.h`, `src/ops/moe/operands.h` (FP8 profile), `src/ops/unsupported/turing_unsupported.cpp`
- Verdict: **NEW OP** — new dequant kernels (NVFP4 → BF16, MXFP8 → BF16) with a Turing-compatible path. The block-scaled FP8 profile style (`FP8_E4M3FN_BLOCK128X128_BF16` + `BlockScaleM128K128`) is a useful template for the MXFP8 path, but the kernels are new.

---

## Artifact format (.ninfer v3) — entry point and loop metadata

**Loader path** (read-only, for reference):
- `src/artifact/schema.h` — `Directory` { `components: map<string, Component>`, `objects: vector<Object>`, `object_index: map<string, ObjectHandle>`, `bindings: map<string, Binding>`, `uses: map<pair<string,string>, Use>`, `files: vector<FileRecord>`, `payload_bytes`, `metadata: Json`, `provenance: Json` }. `Component { Json config; optional<string> target; map<string, ObjectHandle> resources; optional<Proposal> proposal; }`. `TensorObject { id; shape; format; layout; offset; bytes; }`. `Binding { whole_object; vector<Part> parts; elements; }`. `Use { parameter; input; optional<ActivationPolicy> activation_policy; map<string, Binding> auxiliaries; }`. `FileRecord { optional<string> path; payload_bytes; logical_begin; }`.
- `src/artifact/schema.cpp` — `parse_directory(root, entry_name)`: `require_members(root, {"components","objects","bindings","uses","files"}, {"metadata","provenance"}, "directory")`; `parse_files`, `parse_objects`, `parse_components`, `parse_bindings` (each binding name validated via `require_id`), `parse_uses`; metadata/provenance optional but must be objects; `metadata.name` validated via `require_id`.
- `src/artifact/reader.h` / `reader.cpp` — `Reader` owns cold directory + lazily opened files. `segments(offset, bytes)` walks `files` via `std::upper_bound` on `logical_begin`, checks `offset > data.payload_bytes || bytes > data.payload_bytes - offset` → throws `ArtifactError("read range exceeds logical payload")`, returns `ReadSegment { file_index; file_offset; destination_offset; bytes; }` (index 0 → `impl_->entry_payload_start`, else `kPayloadAlignment`). `read_into` / `read_range` / `read_object` / `open_file`.
- `src/artifact/materializer.h` — `materialize(reader, plan, execution, observer)` → `MaterializedArtifact` (device/host/mapped parents, stats). `src/artifact/binder.h` — `bind_view(reference, materialized, device)` → `WeightView`. `src/artifact/views.h` — `bind_view`.
- `src/artifact/formats.h` — `QType`, `QuantLayout`, `ActivationPolicy`, `LinearPolicy`.
- `src/artifact/placement.h` — `MaterializationPlan`, `Placement`, `HostAccess`.

**Loop metadata / untied layers**: no field exists. The only free-form JSON is `Component::config` and top-level `metadata`. Untied embeddings / looped layers would have to be expressed as (a) additional `Component`s (e.g. `loop`, `untied_embed`) with `config` JSON carrying the loop count / untied flag, and (b) binding names in `bindings` (e.g. `layer.{i}.*` vs shared `layer.*`). The loader does not validate unknown config keys, so this is a convention, not a format change — but it is a new convention the converter must emit.

**Manifest note**: `model-cards/Qwen3.8-27B-NInfer/artifact-manifest.json` declares `cuda_architecture: sm_120a` (stale — repo is sm_75-only) and `container_version: 3`. Do not trust it for the sm_75 target.

---

## Summary table

| # | Inkling feature | Existing scaffold | Verdict |
|---|---|---|---|
| F1 | Looped layers | none (registry/artifact config only) | NEW OP (frontend/artifact convention) |
| F2 | No-RoPE relative-bias attn + per-head sink + sigmoid gate | rope/ (unused), softmax_attention (no bias) | NEW OP |
| F2/F3 | Sliding-window 1024 + full attn | `sliding_window_attention.cpp` (window 2048 only) | NEW KERNEL SAME CONTRACT |
| F3 | Causal conv kernel-4 | `causal_conv1d_silu.cpp` (kernel-4, BF16) | REUSE AS-IS |
| F4 | Block-attn-residual softmax mix | `gated_residual/` (BF16 combine, no softmax); `hc_mix` FP8-only | NEW OP |
| F5 | Sigmoid top-8-of-128 MoE + softcapped SwiGLU | `moe/` (2560/512/10/640, FP8-only) | NEW OP |
| F6 | NVFP4/MXFP8 dequant on Turing | none (FP8 formats stay, no Turing kernels) | NEW OP |
| F7 | Frontend + artifact v3 loop metadata | `src/models/common/qwen/frontend/` (Qwen-specific), `src/text` (shared), `src/artifact/schema.h` | NEW OP + NEW-TENSOR |

**Reusable as-is**: causal_conv1d_silu (kernel-4). **New kernel same contract**: sliding_window (parameterize window). Everything else is new.
