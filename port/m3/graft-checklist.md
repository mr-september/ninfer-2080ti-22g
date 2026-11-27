# Inkling-2 graft checklist — donor → lean port

Scope: map donor `E:\ninfer-all\repo\src\models\inkling2\` (executor.cpp ~172KB / 3307 lines, model.cpp 473, config.cpp 215, kv_format.cpp 40, kernels.cu/.h, situ.h, kv_codec.cuh, kv_format.h) against the port `E:\ninfer\src\models\inkling2\`. Read-only except this file.

## 0. Verified facts (checked this session)

- Vendor tree `execution/vendor/`: **24/25 donor files byte-identical** to donor `src/` (verified by `cmp`). The one differing file is `models/inkling2/kv_format.h` — only the `KvCacheStorage`→`VendoredKvCacheStorage` rename (4 lines); the enum itself lives in the new `ninfer/kv_cache_storage.h` (no donor, 12 values). `models/inkling2/kernels.cu` is **byte-identical** to donor.
- Port executor (`execution/executor.cpp`, 865 lines) is a hand-written transcription: T=1 decode only, no prefill chunk loop, no CUDA graphs, no sequences>1, no vision, no dspine/dspark, no expert delivery/parallelism. It includes vendored `models/inkling2/kernels.h`, `core/{arena,buffer,device,tensor,weight_view}.h`, `models/common/bindings.h`, and `ninfer/ops/{embedding,linear,weight_input}.h` (the latter three resolve to `E:\ninfer\include\ninfer\ops\*.h` — the port's engine ops surface, NOT the donor's `src/ops/`).
- Port `core/` is a **different engine** from donor `core/` (~40 files vs ~60; e.g. port has `cyclic_kv_cache`, `paged_kv_cache`, `host_kv_arena`, `nf4`, `expert_cache`; donor has `device_buffer.h`, `device_wait.h`, `growable_region.h`, `mapped_host_copy.h`, `pinned_memory.h`, `vmm.h`, `weight_view.h`, `expert_parallel.h`, `tensor_parallel.h`, `divided_head.h`, `expert_disk.h`, `decode_graph.h`). The vendored tree carries only 2 core files (`core/device_launch.h`, `core/paged_kv_page.h`) — both byte-identical to donor.
- Port `ops/` (`src/ops/`) is tuned for RTX 2080 Ti: `linear/projection/q4/shapes/n1152_k2304.cpp`, `q5/shapes/n2304_k1152.cpp`, `q6/shapes/n201024_k2304.cpp`, `q8/shapes/n1024_k2304.cpp`, `n2304_k3072.cpp`, `n3072_k2304.cpp`, `n384_k2304.cpp` (all inkling2-specific shapes); donor has the generic GGUF path (`ops/linear/gguf/gguf_linear.h`, `ggml_bridge.h`). The port's `ninfer/ops/linear.h` (`include/ninfer/ops/linear.h`) exposes `linear(Tensor, Weight, Tensor&, LinearPolicy, MemoryArena&, CudaExecutionView)` + `prepare_linear_weight` (`include/ninfer/ops/weight_input.h`). Donor executor.cpp uses `ops::prepare_linear_weight` / `ops::linear` / `ops::joins` / `ops::detail::gguf_project` / `ops::detail::gguf_type` / `ops::detail::gguf_project_workspace_bytes` (executor.cpp:66-82, 603-604, 654-655, 698, 738, 746, 1447-1454, 1751, 1771, 1777, 1789, 1800, 2605).
- Port's `load/text.cpp` is a **separate, port-native binder** (`kDirect = QType::FP32`; `direct()` for norms/clips/biases/convs; `parameter()` for matrices; full-attention layers store residual_query pre-tiled to `[3][hidden]` with three DISTINCT mix rows — measured on weight/000170, 2026-11-25; sliding layers one row). It produces `TextWeights` (port `weights.h`) + `ModelWeights`. It does NOT use the donor's `artifact::Binder`/`qwen3_5::loading::Bindings` path.
- Port `port_config.h` defines its own `Config` (hidden_size, vocab_size, num_hidden_layers, max_position_embeddings, num_attention_heads, num_key_value_heads, head_dim, sliding_window, conv_taps, attn_res_block_size, eos_token_id, rms_norm_eps, final_logit_softcapping, full_attention_layer[], AttentionConfig, LoopConfig, MoeConfig) — a **different struct** from donor `TextConfig` (which adds unpadded_vocab_size, relative_width, log_scale_alpha/floor, block_size, gate_cap/up_cap/route_scale/logit_softcap, core_start/core_end/num_passes/gate_rank, visits()/visit_count()).
- Port's executor is wired ONLY through `apps/logit_compare/inkling2_probe.cpp` (the `ninfer-inkling2-probe` binary). The `apps/serve` and `apps/cli` apps do NOT reference inkling2 (grep: no matches). The engine path (`src/runtime/engine`) is qwen3_5-based; inkling2 has no server wiring yet.
- Port `models/CMakeLists.txt:44-53` compiles `inkling2/execution/executor.cpp` + `inkling2/execution/vendor/models/inkling2/kernels.cu` into `ninfer_inkling2_execution` with the vendor dir first on the include path.

## 1. Dependency inventory — donor executor.cpp/executor.h + model.cpp/model.h + config.cpp + kv_format.cpp

| Donor symbol/file | Where | Port status | Evidence |
|---|---|---|---|
| `models/inkling2/kernels.h` (kMaxStates, MixArgs, gemm_f32, gemm_f32_batch, control_inputs/apply, query_post, kv_post, attention, residual, keep_products, keep_history, route, situ_rows, bf16_expert_rows, bf16_rows_f32, softcap_bf16, f32_to_bf16, place_rows, ring_copy, spark_mix/rotate/table/head_inputs/step, add_rows, silu_mul, add_fixed, rms_norm, mix, history) | vendored | **PRESENT** (byte-identical) | `execution/vendor/models/inkling2/kernels.h` |
| `models/inkling2/kernels.cu` | vendored | **PRESENT** (byte-identical) | `execution/vendor/models/inkling2/kernels.cu` |
| `models/inkling2/situ.h` (portable_tanh/sigmoid/situ) | vendored | **PRESENT** (byte-identical) | `execution/vendor/models/inkling2/situ.h` |
| `models/inkling2/kv_codec.cuh` | vendored | **PRESENT** (byte-identical) | `execution/vendor/models/inkling2/kv_codec.cuh` |
| `models/inkling2/kv_format.h` (`VendoredKvCacheStorage`, `kv_format()`) | vendored | **PRESENT** (renamed, 4 lines differ) | `execution/vendor/models/inkling2/kv_format.h` vs donor |
| `models/inkling2/config.h` (`TextConfig`, `VisionConfig`, `DSparkConfig`, `kMixEps`, `parse_text_config`, `visits()`, `visit_count()`) | vendored | **PRESENT** (byte-identical) | `execution/vendor/models/inkling2/config.h` |
| `models/inkling2/config.cpp` (`parse_text_config`, `parse_vision_config`, `parse_dspark_config`) | NOT vendored | **MISSING** in port (port has `port_config.cpp` instead) | port `port_config.cpp` is a separate parser |
| `models/inkling2/model.h` (`Model`, `TextWeights`, `AttentionWeights`, `MoeWeights`, `ControllerWeights`, `LoopWeights`, `LayerWeights`, `ExpertBanks`, `VisionWeights`, `DSparkWeights`, `LoadOptions`, `FrontendResources`, `InstanceInfo`, `is_inkling2`, `has_vision`, `has_dspark`, `load_model`, `default_expert_split`) | NOT vendored | **MISSING** in port (port has `weights.h` instead) | port `weights.h` is a separate struct set |
| `models/inkling2/model.cpp` (473 lines; artifact::Binder/Reader, qwen3_5::loading::Bindings, core/tensor_split.h, models/expert_parallel.h, qwen3_5::frontend::Tokenizer) | NOT vendored | **MISSING** in port | port loader is `load/text.cpp` |
| `models/inkling2/kv_format.cpp` (40 lines; includes `models/kv_storage_name.h`) | NOT vendored | **MISSING** in port (kv_format.h vendored but .cpp not; port has no `kv_format.cpp`) | port has no `kv_format.cpp` |
| `models/inkling2/vision.h/.cpp` (VisionTower) | NOT vendored | **MISSING** — drop for text-only server | executor.cpp:2047-2057, 2064-2068, 2087 (all guarded by `options.vision_max_patches > 0`) |
| `models/inkling2/cpu_experts.h/.cpp` (CpuExperts) | NOT vendored | **MISSING** — drop for text-only server | executor.cpp:255, 1262-1267, 2479-2484 (guarded by `options.expert_dma_share`) |
| `models/inkling2/kv_planes.h` (24 lines) | NOT vendored | **MISSING** — drop (port has no kv_planes; uses per-session planes) | executor.cpp:18 include; port replaces with `f16_plane()` in executor.cpp:42 |
| `models/inkling2/kernels.h` entry points above | vendored | **PRESENT** | — |
| `core/device_launch.h` | vendored | **PRESENT** (byte-identical) | — |
| `core/paged_kv_page.h` | vendored | **PRESENT** (byte-identical) | — |
| `core/qtype.h` (QType, is_gguf) | NOT vendored | **MISSING** in port (port has `core/dtype.h` instead) | donor model.cpp:5, executor.cpp:603-604, 654-655, 1751, 1800, 2605 |
| `core/device.h` (DeviceContext) | port | **PRESENT** (port's own) | port `core/device.h` |
| `core/tensor.h` (Tensor, DType) | port | **PRESENT** (port's own) | port `core/tensor.h` |
| `core/weight_view.h` (WeightView) | port | **PRESENT** (port's own) | port `core/weight_view.h` |
| `core/arena.h` (MemoryArena) | port | **PRESENT** (port's own) | port `core/arena.h` |
| `core/buffer.h` (DeviceBuffer) | port | **PRESENT** (port's own) | port `core/buffer.h` |
| `core/decode_graph.h` (CUDA graphs) | NOT vendored | **MISSING** — drop (port executor has no graphs) | executor.cpp:3, 2791 (`if (!options.cuda_graphs)`) |
| `core/device_buffer.h` | NOT vendored | **MISSING** — drop (port has `core/buffer.h`) | executor.cpp:4 |
| `core/device_wait.h` (`ninfer::wait_for_stream`) | NOT vendored | **MISSING** — drop (port executor uses `cudaStreamSynchronize` directly) | executor.cpp:1385, 1397, 1401, 2080, 2797, 2966 |
| `core/expert_parallel.h` (ExpertDelivery, ExpertDeliveryLayer, ExpertResidency, ExpertRouteCounts) | NOT vendored | **MISSING** — drop (port has no expert delivery) | executor.cpp:6, 255-256, 1236-1269, 2474-2484 |
| `core/growable_region.h` | NOT vendored | **MISSING** — drop (port has no growing caches) | executor.cpp:7, 101, 384-388 |
| `core/mapped_host_copy.h` | NOT vendored | **MISSING** — drop (port has no host-mapped copy) | executor.cpp:8 |
| `core/pinned_memory.h` | NOT vendored | **MISSING** — drop (port has no pinned memory) | executor.cpp:9 |
| `core/vmm.h` (vmm::granularity) | NOT vendored | **MISSING** — drop (port has no VMM) | executor.cpp:12, 318 |
| `core/tensor_parallel.h` | NOT vendored | **MISSING** — drop (port has no tensor parallelism) | executor.cpp:11 |
| `models/divided_head.h` (models::DividedHead) | NOT vendored | **MISSING** — drop (port has no expert parallelism) | executor.cpp:14, 265, 1405, 1499, 1555, 1587, 2592 |
| `models/dspine/runtime.h` (dspine::Runtime, dspine::ring_capacity, dspine::Target, dspine::RuntimeOptions) | NOT vendored | **MISSING** — drop (text-only server) | executor.cpp:15, 288, 358-359, 388, 536, 681-757, 810 |
| `models/expert_profile.h` (ExpertRouteCounts) | NOT vendored | **MISSING** — drop | executor.h:43, executor.cpp:2969 |
| `models/expert_parallel.h` (ExpertResidency, ExpertDelivery, ExpertDeliveryLayer) | NOT vendored | **MISSING** — drop | executor.h:28, executor.cpp:255-256, 1236-1269, 2474-2484 |
| `models/inkling2/cpu_experts.h` (CpuExperts) | NOT vendored | **MISSING** — drop | executor.cpp:16, 255, 1262-1267, 2479-2484 |
| `models/inkling2/vision.h` (VisionTower) | NOT vendored | **MISSING** — drop | executor.cpp:19, 2047-2057, 2064-2068, 2087 |
| `models/inkling2/kv_planes.h` | NOT vendored | **MISSING** — drop | executor.cpp:18 |
| `models/qwen3_5/frontend/tokenizer.h` (Tokenizer, TokenizerResources) | NOT vendored | **MISSING** — drop (port has no tokenizer; probe takes token ids) | model.cpp:8, 38-43 |
| `models/qwen3_5/load/bindings.h` (qwen3_5::loading::Bindings, resolve_weights, ParameterPiece) | NOT vendored | **MISSING** — drop (port has `models/common/bindings.h`) | model.cpp:9, 20, 300, 435 |
| `models/kv_storage_name.h` (kv_storage_name) | NOT vendored | **MISSING** — drop (port has `ninfer/kv_cache_storage.h`) | kv_format.cpp:5 |
| `artifact/binder.h` (artifact::Binder, host_object, resource, device_tail, geometry, device_tail) | NOT vendored | **MISSING** — drop (port has `artifact/binder.h`) | model.cpp:4, 30-43, 300, 435 |
| `artifact/reader.h` (artifact::Reader, geometry, MaterializedArtifact, MaterializationStats) | NOT vendored | **MISSING** — drop (port has `artifact/reader.h`) | model.cpp:5, 60-75, 202, 433 |
| `artifact/schema.h` (artifact::Json, Shape, Residency, ParameterReference, ArtifactError) | vendored | **PRESENT** (byte-identical) | `execution/vendor/artifact/schema.h` |
| `core/tensor_split.h` (tensor_split) | NOT vendored | **MISSING** — drop (port has no tensor split) | model.cpp:6 |
| `core/expert_disk.h` (ExpertDiskReads) | NOT vendored | **MISSING** — drop | executor.h:28, executor.cpp:1236 |
| `ops/linear/gguf/ggml_bridge.h` (gguf::GgmlType, gguf::vector_product, gguf::dequantize_rows) | NOT vendored | **MISSING** — drop (port has no GGUF; uses engine `ops::linear`) | executor.cpp:603, 654, 1751, 1800, 2605 |
| `ops/linear/gguf/gguf_linear.h` (gguf_type, gguf_project, gguf_project_workspace_bytes) | NOT vendored | **MISSING** — drop (port has no GGUF) | executor.cpp:603-604, 654-655, 698, 738, 746, 1447-1454 |
| `ninfer/ops/linear.h` (ops::linear, ops::prepare_linear_weight, ops::joins) | port | **PRESENT** (port's own, different API) | `include/ninfer/ops/linear.h`, `include/ninfer/ops/weight_input.h` |
| `ninfer/ops/embedding.h` (ops::embedding) | port | **PRESENT** (port's own) | `include/ninfer/ops/embedding.h` |
| `ninfer/ops/weight_input.h` (ops::WeightInput, ops::prepare_linear_weight) | port | **PRESENT** (port's own) | `include/ninfer/ops/weight_input.h` |
| `ninfer/ops/linear_policy.h` (LinearPolicy) | port | **PRESENT** (port's own) | `include/ninfer/ops/linear.h` |
| `ninfer/ops/weight_input.h` (WeightInput) | port | **PRESENT** (port's own) | `include/ninfer/ops/weight_input.h` |
| `cuda_fp16.h` (`__half`, `__nv_bfloat16`) | system | **PRESENT** | — |
| `cuda_runtime.h` | system | **PRESENT** | — |

**Bottom line:** the vendored tree covers exactly the kernel/math surface the executor needs (`kernels.h/.cu`, `situ.h`, `kv_codec.cuh`, `kv_format.h`, `config.h`, `artifact/schema.h`, `core/device_launch.h`, `core/paged_kv_page.h`, `ops/common/*`, `ops/kernel/paged_kv_address.cuh`, `ops/kv_cache/*`, `ops/linear/nvfp4/*`). Everything else — the donor's executor.cpp/.h, model.cpp/.h, config.cpp, kv_format.cpp, vision, cpu_experts, kv_planes, dspine, dspark, expert delivery, CUDA graphs, VMM, tensor parallelism, device_wait, growable_region, mapped_host_copy, pinned_memory, divided_head, expert_profile, expert_parallel, tensor_split, qwen3_5 bindings/tokenizer, artifact binder/reader — is NOT in the port and must be either grafted or replaced.

## 2. Minimal file set to graft

**Graft as-is (no adaptation):** nothing beyond the vendored tree. The vendored `kernels.h/.cu`, `situ.h`, `kv_codec.cuh`, `kv_format.h`, `config.h`, `artifact/schema.h`, `core/device_launch.h`, `core/paged_kv_page.h`, `ops/common/*`, `ops/kernel/paged_kv_address.cuh`, `ops/kv_cache/*`, `ops/linear/nvfp4/*` are already in the port and byte-identical.

**Graft with adaptation (the real work):**
1. `models/inkling2/executor.h` — port's `execution/executor.h` is the **wrong API** (single-sequence, no prefill, no graphs). For a production server you need: `ExecutorOptions` (max_context, prefill_chunk, sequences, cuda_graphs, kv_cache, draft_tokens, checkpoints, dspine, dspark, vision_max_patches, expert_cache_bytes, expert_profile, expert_dma_share, expert_cpu_threads, expert_disk), `InsufficientMemory`, `ExecutorMemory`, `SequenceSnapshot`, `SequenceImage`, `ImageRows`, `Executor` (constructor, forward, forward_branches, forward_prompt_branches, decode, logits, head_stream, draft_tokens, verify, commit, draft, dspark_block, snapshot, restore, rewind, vision, encode_image). For a text-only decode+prefill server, drop: dspine, dspark, vision, checkpoints, expert_disk, expert_profile, expert_dma_share, divided_head, cuda_graphs (or keep graphs), and multi-sequence (keep sequences=1). Keep: prefill_chunk, kv_cache, draft_tokens=0, checkpoints=false.
2. `models/inkling2/executor.cpp` — 3307 lines. Adapt: (a) replace `ops::detail::gguf_type`/`gguf_project`/`gguf_project_workspace_bytes` with the port's `ops::linear`/`ops::prepare_linear_weight` (the port's projection layer is the tuned engine path); (b) replace `ninfer::wait_for_stream` with `cudaStreamSynchronize`; (c) drop vision tower (`plan_vision`, `encode_image`, `tower`), dspine (`spine`, `spine_options`, `make_spine`), dspark (`spark`, `plan_spark`, `spark_planned_bytes`, `dspark_block`, `draft`), cpu_experts (`cpu_experts`, `CpuExperts`), divided_head (`divided_head`), expert_parallel (`expert_parallel`, `expert_range`), growable_region (`back`, `growable`), mapped_host_copy (`transport.disk`), pinned_memory (`PinnedMemory`), VMM (`vmm::granularity`), tensor_parallel (`tensor_parallel`), decode_graph (`decode_graph`), device_buffer (`DeviceBuffer` → port's `core/buffer.h`), device_wait (`wait_for_stream`), expert_disk (`ExpertDiskReads`), expert_profile (`ExpertRouteCounts`), expert_parallel (`ExpertDelivery`), divided_head, kv_planes (`kv_planes.h` — port uses `f16_plane()` in executor.cpp:42), vision (`VisionTower`), cpu_experts (`CpuExperts`), dspine (`dspine::Runtime`), dspark (`DSparkConfig`, `DSparkWeights`, `Spark`).
3. `models/inkling2/model.h` — port's `weights.h` is the **wrong struct set**. The donor's `Model` class is the seam for the adapter (see §3). For a text-only server, drop: `VisionWeights`, `VisionLayerWeights`, `VisionPlacement`, `DSparkWeights`, `DSparkLayerWeights`, `DSparkConfig`, `VisionConfig`, `ExpertBanks`, `ExpertResidency`, `ExpertDeliveryLayer`, `FrontendResources`, `InstanceInfo`, `default_expert_split`, `is_inkling2`, `has_vision`, `has_dspark`, `load_model` (port has its own loader). Keep: `TextConfig`, `TextWeights`, `AttentionWeights`, `MoeWeights`, `ControllerWeights`, `LoopWeights`, `LayerWeights`, `Model` (constructor + accessors).
4. `models/inkling2/model.cpp` — 473 lines. Drop: artifact::Binder/Reader, qwen3_5::loading::Bindings, qwen3_5::frontend::Tokenizer, core/tensor_split.h, models/expert_parallel.h, models/kv_storage_name.h. The port's `load/text.cpp` is the replacement loader. The donor's `load_model` is NOT needed — the port's binder produces `TextWeights` directly.
5. `models/inkling2/config.cpp` — 215 lines. Drop: `parse_vision_config`, `parse_dspark_config`, `parse_text_config` (port has `port_config.cpp`). Keep: `TextConfig` struct (vendored) + `visits()`/`visit_count()` (vendored). The port's `port_config.cpp` is the replacement parser.
6. `models/inkling2/kv_format.cpp` — 40 lines. Drop: `kv_storage_name.h` include. The port has no `kv_format.cpp` (the enum is vendored but the .cpp is not). For a text-only server with `kv_cache=BFloat16`, the `kv_format()` function is only called for the quantized full-attention visits (executor.cpp:452-464, 901-913) — drop it or keep it with a stub.

**Droppable files (do not graft):** `vision.h/.cpp`, `cpu_experts.h/.cpp`, `kv_planes.h`, `kv_format.cpp` (or stub), `config.cpp` (port has its own), `model.cpp` (port has its own loader), `situ.h` (vendored, keep), `kernels.cu/.h` (vendored, keep), `kv_codec.cuh` (vendored, keep), `kv_format.h` (vendored, keep), `config.h` (vendored, keep), `artifact/schema.h` (vendored, keep), `core/device_launch.h` (vendored, keep), `core/paged_kv_page.h` (vendored, keep), `ops/common/*` (vendored, keep), `ops/kernel/paged_kv_address.cuh` (vendored, keep), `ops/kv_cache/*` (vendored, keep), `ops/linear/nvfp4/*` (vendored, keep).

**Droppable features (keep the structs, drop the call sites):** vision tower, dspine drafter, dspark drafter, cpu experts, expert parallelism, divided head, expert disk, expert profile, expert dma share, growable region, mapped host copy, pinned memory, VMM, tensor parallelism, CUDA graphs, checkpoints, multi-sequence (keep sequences=1), expert cache (keep device residency), expert split (keep device residency).

## 3. Adapter contract — port loader → donor Model

**Seam:** the port's `load/text.cpp` (`bind_text`) + `weights.h` (`TextWeights`, `ModelWeights`) + `port_config.h` (`Config`) must construct a donor-compatible `Model` (or a donor-compatible `TextWeights` + `TextConfig`) that the grafted executor can consume.

**What the donor's `load_model` expects** (model.cpp:179-473): `artifact::Reader`, `artifact::Binder`, `qwen3_5::loading::Bindings`, `core/tensor_split.h`, `models/expert_parallel.h`, `qwen3_5::frontend::Tokenizer`, `qwen3_5::loading::resolve_weights`, `artifact::materialize`. The donor's `Model` constructor takes `(TextConfig, TextWeights, std::vector<BoundWeight>, FrontendResources, InstanceInfo, artifact::MaterializedArtifact)` (model.h:214-215).

**What the port's loader produces** (load/text.cpp): `TextWeights` (port `weights.h`) with `WeightId` handles into a frozen weight array; `ModelWeights` wrapping `TextWeights`; `Config` (port `port_config.h`). The port's `WeightId` is `models/common/weights.h` — need to verify it is the same as the donor's `WeightId` (donor `model.h:33` uses `WeightId` from `models/common/weights.h`).

**Cleanest seam:** a **thin adapter struct** (`PortModel`) that wraps the port's `ModelWeights` + `Config` and exposes the donor's `Model` API surface (`config()`, `weights()`, `weight(WeightId)`, `input(WeightId)`, `weight_data()`). Estimated size: ~150-250 lines (a new `adapter.h/.cpp` in the port's inkling2 dir). The adapter does NOT re-implement the loader; it re-shapes the port's `TextWeights` into the donor's `TextWeights` (or the grafted executor consumes the port's `TextWeights` directly via a small `if` on the weight-id type).

**Alternative (simpler):** do NOT graft the donor's `Model` class at all. Keep the port's `weights.h`/`port_config.h`/`load/text.cpp` as-is, and graft ONLY the donor's `executor.cpp`/`executor.h` (adapted) + `kernels.h/.cu` (already vendored). The grafted executor takes the port's `TextWeights` + `Config` directly. This is the **recommended path** — it avoids the adapter entirely and keeps the port's tuned loader.

## 4. Verdict

**Estimated glue size:** loose-to-medium. ~800-1500 lines of adapted donor executor.cpp (3307 lines minus vision/dspine/dspark/cpu_experts/expert_parallel/divided_head/growable_region/mapped_host_copy/pinned_memory/VMM/tensor_parallel/decode_graph/device_buffer/device_wait/expert_disk/expert_profile/kv_planes/vision/cpu_experts/dspine/dspark/checkpoints/multi-sequence/expert_cache/expert_split/expert_disk/expert_profile/expert_dma_share — most of the 3307 lines are feature-specific). Plus ~150-250 lines of adapter (or 0 if the port's `TextWeights` is consumed directly). Plus ~50-100 lines of CMake/include-path glue.

**Top 3 graft risks:**
1. **Engine-layer mismatch (highest).** The donor executor.cpp uses `ops::detail::gguf_project`/`gguf_type`/`gguf_project_workspace_bytes` (executor.cpp:603-604, 654-655, 698, 738, 746, 1447-1454) for the expert-bank projections and the head projection. The port has no GGUF path. The port's `ops::linear` (`include/ninfer/ops/linear.h`) is a different API (`linear(Tensor, Weight, Tensor&, LinearPolicy, MemoryArena&, CudaExecutionView)`). The adapter must map donor's `ops::detail::gguf_project` calls to the port's `ops::linear` calls — this is the single largest adaptation (~10-15 call sites). Evidence: executor.cpp:603-604, 654-655, 698, 738, 746, 1447-1454, 1751, 1771, 1777, 1789, 1800, 2605.
2. **Weight-binding mismatch.** The donor's `Model::weight(WeightId)` returns `BoundWeight` (donor `models/common/weights.h`); the port's `TextWeights` uses `WeightId` (port `models/common/weights.h`). The grafted executor must consume the port's `WeightId`/`BoundWeight` or the adapter must translate. Evidence: donor model.h:202-204 (`weight(WeightId)`, `weight_data()`, `input(WeightId)`); port weights.h (`WeightId` from `models/common/weights.h`).
3. **Config struct mismatch.** The donor's `TextConfig` (vendored, byte-identical) has `unpadded_vocab_size`, `relative_width`, `log_scale_alpha/floor`, `block_size`, `gate_cap/up_cap/route_scale/logit_softcap`, `core_start/core_end/num_passes/gate_rank`, `visits()`, `visit_count()` — the port's `Config` (port_config.h) has a different field set (`attn_res_block_size`, `final_logit_softcapping`, `AttentionConfig`, `LoopConfig`, `MoeConfig`). The grafted executor must either consume the donor's `TextConfig` (and the port's `port_config.cpp` must be replaced by the donor's `config.cpp`) or the adapter must translate. Evidence: donor config.h:28-72; port port_config.h:28-66.

**Alternative: graft the donor model dir wholesale INCLUDING donor kernels.cu.** Simpler in one sense (no adapter, no translation) but the port's value-add is its tuned engine (core/ and ninfer/ops layer differences: q4/q5/q6/q8 projection shapes for inkling2, `ops/linear/projection/q8/shapes/n1024_k2304.cpp` etc.). Grafting wholesale would discard the port's tuned projection layer. **Recommendation: graft the donor's executor.cpp/.h + model.h (adapted) + config.h (vendored) + kernels.h/.cu (vendored) + situ.h (vendored) + kv_codec.cuh (vendored) + kv_format.h (vendored) + artifact/schema.h (vendored) + core/device_launch.h (vendored) + core/paged_kv_page.h (vendored) + ops/common/* (vendored) + ops/kernel/paged_kv_address.cuh (vendored) + ops/kv_cache/* (vendored) + ops/linear/nvfp4/* (vendored). Keep the port's `core/`, `ninfer/ops/`, `artifact/`, `load/text.cpp`, `weights.h`, `port_config.h`, `models/common/bindings.h`. Do NOT graft the donor's `config.cpp`, `model.cpp`, `kv_format.cpp`, `vision.h/.cpp`, `cpu_experts.h/.cpp`, `kv_planes.h`.**

**Engine-layer differences (vendor vs donor vs port):**
- Vendor vs donor: **24/25 byte-identical** (only `kv_format.h` differs by 4 rename lines). No divergence in kernels/math/kv_cache/ops/common/ops/linear/nvfp4.
- Vendor vs port: the vendored tree is a **subset** of the donor's ops/core surface (only 2 core files, 0 ops/linear/gguf, 0 ops/kernel/paged_kv_address.cuh beyond the vendored one, 0 ops/kv_cache beyond the vendored ones). The port's own `core/` and `ninfer/ops/` are **different engines** (different file sets, different APIs). The vendored tree is NOT a tuned copy — it is a byte-identical copy of the donor's kernel/math surface. The port's tuned engine is in `src/core/` and `src/ops/` (not in the vendor tree).

## 5. Integration points (port wiring)

- **Probe app:** `apps/logit_compare/inkling2_probe.cpp` (lines 1-60+) — the only inkling2 consumer today. It: (1) opens `artifact::Reader` on the artifact dir; (2) calls `models::inkling2::config_from_directory(reader.directory())` (port `port_config.h`); (3) calls `models::inkling2::loading::bind_text(bindings, config)` (port `load/text.cpp`); (4) materializes weights via `artifact::plan_materialization` + `artifact::materialize`; (5) calls `models::resolve_weights(std::move(bindings.weights), materialized, devices)` (port `models/common/bindings.h`); (6) constructs `models::inkling2::execution::Executor executor(config, bound, device, device.stream(), 4096)` (port `execution/executor.h`); (7) calls `executor.make_session()`, `executor.decode_step(*session, token, position)`, `executor.logits()`.
- **CMake:** `src/models/CMakeLists.txt:44-53` — `ninfer_inkling2_execution` target compiles `inkling2/execution/executor.cpp` + `inkling2/execution/vendor/models/inkling2/kernels.cu` with vendor dir first on the include path. `src/models/CMakeLists.txt:58` — `ninfer_model_runtime` links `ninfer_inkling2_execution`.
- **Registry:** `src/models/registry.cpp:29` — `Inkling2LoopForCausalLM` + `inkling2_loop_text` → inkling2.
- **No server wiring:** `apps/serve/main.cpp` and `apps/cli/main.cpp` do NOT reference inkling2 (grep: no matches). The inkling2 executor is only reachable via the probe binary. A production server needs a new `apps/serve` path (or a new `apps/inkling2-server`) that: (a) opens the artifact, (b) binds weights via the port's `bind_text`, (c) constructs the grafted executor, (d) drives `decode_step`/`logits` per request.
- **Integration contract for a grafted executor:** must accept the port's `Config` + `TextWeights` (or a donor-compatible `Model` via the adapter), must expose `make_session()`, `decode_step(Session&, token, position)`, `logits()`, and must compile against the port's `core/` + `ninfer/ops/` (not the donor's).

## 6. Recommended next steps (for the parent)

1. Decide: graft the donor's `executor.cpp/.h` (adapted) + keep the port's `weights.h`/`port_config.h`/`load/text.cpp` (recommended) vs graft the donor's `model.h`/`model.cpp` + adapter (simpler in one sense, but discards the port's tuned loader).
2. If grafting the donor's executor.cpp: do a line-by-line adaptation pass (replace GGUF calls with `ops::linear`, drop vision/dspine/dspark/cpu_experts/expert_parallel/divided_head/growable_region/mapped_host_copy/pinned_memory/VMM/tensor_parallel/decode_graph/device_buffer/device_wait/expert_disk/expert_profile/kv_planes, keep prefill_chunk + kv_cache + sequences=1).
3. If grafting the donor's model.h: write the adapter (`PortModel`) that wraps the port's `ModelWeights` + `Config` and exposes the donor's `Model` API surface (`config()`, `weights()`, `weight(WeightId)`, `input(WeightId)`, `weight_data()`).
4. Wire the grafted executor into `apps/serve` (or a new `apps/inkling2-server`) — the probe app is the only current consumer.
5. Build + run the probe (`ninfer-inkling2-probe`) against the artifact to verify the grafted executor produces the same logits as the port's current transcription.

## 7. Graft build session (2026-11-27, parent agent)

`ninfer_inkling2_graft` (donor `executor.cpp` + `kv_format.cpp`, verbatim, against `graft/include`
shims + vendor + port surface) **compiles and archives to a static library** on Windows/MSVC
(`build/windows/src/models/ninfer_inkling2_graft.lib`, 0 warnings). Method — the destination engine
owns every shared name (the vendor header's stated rule, applied throughout):

- Enums unified in the port's headers, donor values appended: `KvCacheStorage` (types.h) carries
  the vendor superset and `VendoredKvCacheStorage` is now an alias of it (vendor + graft shims);
  `SpeculativeBackend` gained DSpark/DSpine; `LinearPolicy` (ops/linear.h) carries the donor's
  7-value set and donor `valid_linear_policy`/`allows_a8` semantics; `QType` (core/weight.h) gained
  NVFP4=7 and the donor's T2/ggml formats appended at 34-58 (port keeps 9/10; nothing on disk
  stores these numbers — binder names formats from metadata).
- `graft/include` shadows the donor's view for grafted TUs only: `core/{arena,buffer,device,tensor,
  cyclic_kv_cache,layout,qtype}.h`, `ninfer/{model_enums,kv_cache_storage}.h`,
  `ninfer/ops/{linear_policy,weight_input}.h` (adds `ops::joins`), `artifact/materializer.h` (adds
  `MaterializeOptions`), `models/qwen3_5/config.h` (TensorShare). The graft arena keeps the donor's
  `DeviceArena`/`WorkspaceArena`; the graft `device.h` = donor's multi-rank DeviceContext + a
  graft `CudaExecutionView` (port layout, implicit from `DeviceExecutionView`/`cudaStream_t`).
- Not yet run: the LINK stage. The graft library declares but does not define donor functions
  (`ops::joins`, `DeviceArena`, donor `DeviceBuffer`/`Tensor`/`DeviceContext`/`wait_for_stream`
  methods, `gguf::*`) — a probe target must add the donor implementations or stubs, and the
  Tensor/DeviceContext **layout seam against port-compiled Ops** (checklist §4 risks 1-2) is live
  at that point, not before. ABI across the seam is name-compatible by construction (same class
  names) but layout-divergent (donor Tensor vs port MemorySpan Tensor).
