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
