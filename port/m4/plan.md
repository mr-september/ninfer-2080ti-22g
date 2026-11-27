# M4 — Gating and Serving

Goal: the ported engine must (a) match the oracle greedy tokens (M3 gate carried into the serve path), (b) beat the M1 baselines under the memory budget, and (c) leave a decided int8-KV position.

## Baselines (from M1, same card, same prompt style)
| System | Decode t/s | VRAM |
|---|---|---|
| llama.cpp IQ4_NL | 38.7 | ~22 GB |
| Strata (fp16 KV) | — | quality gates 92.35/96.68/100% same-word; KL 0.072/0.011/0.000 |
| Target | ≥ ~39 | ≤ 14.4 GB |

Oracle (ninfer-all donor server, port 8191) is the correctness reference, not a speed reference (it carries DSpark + full quantized-KV machinery this lean port defers).

## Sequence
1. **Token parity** (carries M3 closure): probe greedy ids == `port/m0/parity/reference_baseline.json` continuation ids on the 20-prompt set; bit-instability rule from M2 applies (differences only where the reference baseline itself is tie-fragile).
2. **Serve integration**: engine-level load.cpp (Reader→Binder→materialize→resolve_weights, mirroring qwen3_5/load.cpp), registry entry, Startup memory plan, Program/graph path if the direct-launch decode misses the gate.
3. **Speed gate**: 64-token greedy decode, warm, on the compute card (CUDA_VISIBLE_DEVICES=00000000:04:00.0). Measure t/s and nvidia-smi peak. Compare vs 38.7 / target ~39.
4. **int8-KV decision**: measure FP16-KV fit at the 80K ctx budget first. If weights+KV+arena ≤ 14.4 GB with FP16, int8-KV stays deferred (lean mandate); if not, implement via the vendored KvPlaneView quantized codings (already in kernels).
5. **Quality spot-check** (only if int8-KV lands): same-word + KL on the Strata corpus (sha256 9ff6aa…82c57) vs FP16-KV serve, same harness as M1.

## Risks
- 22 GB card vs 15.6 GiB weights + 2.66 GiB KV @80K + arena/activations: tight; Startup plan decides ctx size for the benchmark, report actual usable ctx separately from llama.cpp's 22-GB packing.
- bf16-rounded projection rows (engine `ops::linear`) vs donor FP32 rows: parity-tolerance question resolved empirically at step 1.
- CUDA graphs: decode step is ~600 kernels/visit × 48 visits — kernel-launch overhead may dominate; graph capture is the M4 escape hatch if under target.

## Measurements
- **2026-11-27 speed gate, step 3 run early (lean probe, direct launch).** artifact
  `inkling_2_mini_v3.ninfer`, prompt code-1, 64-token greedy, warm, compute card
  (04:00.0), two independent processes: **22.33 / 23.34 t/s** (median over 61 steps,
  first 2 steps excluded; probe now prints `[decode-timing]`). FAIL vs gate ~39.
  Roofline: ~1.87 GB/token active bytes -> ~320 t/s at 600 GB/s; measured 44 ms/token
  = ~42 GB/s effective = ~7 % of bandwidth. Verdict: launch/overhead-bound at this
  shape, not bandwidth-bound. Escape hatch as planned (step 2): graph path; plus the
  130+ target's Turing-specific work (port-tuned projection kernels replacing the
  ggml bridge on hot shapes). Profile in flight (nsys, 32 steps) to count launches.
- **2026-11-27 nsys census (32+64 T=1 steps, whole-process totals).** 149,226
  `cudaLaunchKernel` calls = **~1,550 launches/step (~32/visit)**. Host launch API time
  alone ≈ 21 ms/step (avg 13.6 us/launch — WDDM submit cost), vs **~9 ms/step GPU kernel
  busy**: the 44 ms step is launch-submission-bound, GPU is ~7 % busy. Kernel mix per
  step: q4 gemv ×336 (6.6 us avg), bf16_rows_f32 ×631 (1.3 us — pure launch-cost
  kernels), q5 ×168 (9.8 us), q8 attn/up ×126 (14.6 us), lm-head q6 0.82 ms/step.
  `cudaMemcpyAsync` 2.5 ms/step (route-ids DtoH + token upload).
  ⇒ **CUDA graphs is the gate lever** (planned escape hatch, step 2): removes ~35 ms of
  the 44; ceiling after graphs ≈ GPU-busy 9 ms ≈ ~110 t/s. Reaching 130+ then needs the
  kernel-efficiency pass (bf16_rows fan-in, q5/q8 SIMT → tuned sm_75 tiles), which also
  covers the 130+ target's port-tuned-kernel mandate.
