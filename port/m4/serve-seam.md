# Inkling 2 serve seam — Setup contract and minimal viable configuration

Ground truth read 2026-11-28: `src/runtime/engine/model_construction.cpp` (switch at
`construct_model`), `src/runtime/engine/model_startup.h` (full `build_model<Setup>` /
`initialize_program` flow), `src/models/qwen3_5/startup.cpp` (exemplar Setup),
`include/ninfer/types.h` (`ContextCacheOptions`).

## Setup contract (from build_model / initialize_program)

Required members of `struct Startup`:
- `Types` — ties `Model`, `Frontend`, `Program` together via `ModelInstance<Types>`.
- `ExecutionPlan` — `{sequence (SequencePlan), capacity (resolution), placement, chunk, extra}`.
- `static constexpr bool supports_graph` — Inkling 2 starts **false** (eager loop; skips
  `Calibration`, graph preparation, `prepare_program` machinery).
- ctor `(const artifact::Reader&, const EngineOptions&, Device)`.
- `placements()` → ordered `{WeightPlacement, device_bytes}` — Inkling 2: single
  `Resident` choice = materialized weight bytes (probe-measured, ~14.65 GiB for v3).
- `plan(device, options, placement, available)` → `optional<ExecutionPlan>`; must honor
  `KvCapacityMode::Explicit|Automatic` via `runtime::resolve_kv_capacity(policy, curve, available)`
  and a capacity `curve` derived from the executor's own sizing math (`executor.cpp` store
  layout: `2*kv_heads*slots*kHead` fp16 planes + `hist_floats` + per-site state — the planner
  curve must reuse those formulas, not re-derive them independently).
- `materialize(device, placement, observer)` → `unique_ptr<Model>` — wraps the existing
  `Binder → Bindings → bind_text → resolve_weights` path (as `inkling2_ppl.cpp` does);
  `Model` must expose `info()`, `storage_stats()`, `weight_data()`, `parameters`, `resources()`.
- `frontend(model, options)` → `qwen::make_frontend(model.resources(), {...}, default_sampling(...))`
  — needs a `resources()` provider (public_token_count = **unpadded_vocab_size**, eos ids,
  chat template passthrough; no vision in the first slice).
- `profile` member — `execution::describe_model` equivalent (hidden/layers/heads for the planner).
- `prefill_signature(model, preparation, chunk)` — stable string, e.g. `inkling2_lean_t1`.
- `architecture()` → `Architecture::Inkling2`.

## Program concept surface (what the engine actually calls)

From `causal_score_core.h`, `context_cache/prefix_index.h`, `context_cache/resource_manager.h`,
`engine_core.h ProgramCallScope`:
`memory_summary`, `reset_memory_peaks`, `causal_score(prompt, first_target)`,
`checkpoint_key/metadata`, `valid_checkpoint`, `inspect_source`, plus the generation
forward/decode/logit-row entry points (`qwen3_5/program/*` total ~5.7k lines — the checkpoint/
context-cache half is what Inkling 2 must satisfy or avoid).

**Avoidance strategy (first slice):** serve with `context_cache.enabled = false` (types.h:156)
and purpose=Generation or causal_score only; implement checkpoint surface as trivially-invalid
(`valid_checkpoint → false`, `checkpoint_metadata.frontier = 0`). Revisit only when prefix reuse
matters. KV stays contiguous per sequence (executor sessions own their slots; the program maps
engine sequence ids → executor sessions 1:1, capacity = `options.max_context × concurrency`).

## Known properties / risks

- The lean executor has **no chunked prefill**: forward(chunk of N tokens) = N decode_steps
  (24 ms each → 4096-token prompt ≈ 98 s). Functionally sufficient for the VRAM/int8-KV
  decision and serve smoke; the kernel-efficiency pass (matrix prefill + decode kernels toward
  the ~110 t/s graph ceiling) replaces it later.
- Memory: no absolute footprint gate (the "≤14.4 GB" was arbitrary, retired 2026-11-28 — see the
  Memory policy in `port/m4/plan.md`). v3 weights are 14.65 GiB / 15.74 GB; per-component
  quantization choices must each be justified by rules of thumb + first principles.
- Donor reference for semantics when the program does prefill properly later:
  donor `executor.cpp` vector path (t≤8) vs matrix path, `quantize_vector_activation` +
  fixed-point `moe_weighted_rows` (still unported; §46 — status UNSETTLED until the post-sync
  PPL gate lands, `report_sync.json`).

## Work list (dependency order)

1. `Model` wrapper + `resources()` provider over `bind_text` output (loads/inkling2 area).
2. Profile + capacity-curve helpers reusing executor sizing math.
3. `Startup` (eager, single placement) mirroring `qwen3_5/startup.cpp` structure.
4. `Program` adapter: sequence→session map, T=1 forward, logit rows, trivial checkpoints.
5. `model_construction.cpp`: `case Architecture::Inkling2:` + CMake wiring.
6. Serve smoke: `ninfer-serve` on v3 with `--context-cache off`, max-context sized to budget;
   record VRAM + int8-KV decision.
