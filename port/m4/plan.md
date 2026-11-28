# M4 — Gating and Serving

Goal: the ported engine must (a) match the oracle greedy tokens (M3 gate carried into the serve path), (b) beat the M1 baselines under the memory budget, and (c) leave a decided int8-KV position.

## Baselines (from M1, same card, same prompt style)
| System | Decode t/s | VRAM |
|---|---|---|
| llama.cpp IQ4_NL | 38.7 | ~22 GB |
| Strata (fp16 KV) | — | quality gates 92.35/96.68/100% same-word; KL 0.072/0.011/0.000 |
| Target | ≥ ~39 | see Memory policy below |

Oracle (ninfer-all donor server, port 8191) is the correctness reference, not a speed reference (it carries DSpark + full quantized-KV machinery this lean port defers).

## Sequence
1. **Token parity** (carries M3 closure): probe greedy ids == `port/m0/parity/reference_baseline.json` continuation ids on the 20-prompt set; bit-instability rule from M2 applies (differences only where the reference baseline itself is tie-fragile).
2. **Serve integration**: engine-level load.cpp (Reader→Binder→materialize→resolve_weights, mirroring qwen3_5/load.cpp), registry entry, Startup memory plan, Program/graph path if the direct-launch decode misses the gate.
3. **Speed gate**: 64-token greedy decode, warm, on the compute card (CUDA_VISIBLE_DEVICES=00000000:04:00.0). Measure t/s and nvidia-smi peak. Compare vs 38.7 / target ~39.
4. **int8-KV decision**: measure FP16-KV fit at the 80K ctx budget first. If the model fits the 22 GB card with headroom under the Memory policy, int8-KV stays deferred (lean mandate); if not, implement via the vendored KvPlaneView quantized codings (already in kernels).
5. **Quality spot-check** (only if int8-KV lands): same-word + KL on the Strata corpus (sha256 9ff6aa…82c57) vs FP16-KV serve, same harness as M1.

## Memory policy (revised 2026-11-28 — supersedes the "≤ 14.4 GB" gate)

The 14.4 GB absolute was arbitrary and is RETIRED (violated the quantitative-gate rule: no basis,
invented into the roadmap). The actual criterion: each component's quantization (per-tile expert
tiers, attention q8-class, embeddings/head, conv/norm fp32 keeps, KV dtype) is chosen by known
rules of thumb plus first-principles reasoning — smallest size with minimal behavior
degradation/divergence for that component's sensitivity class — and each choice must be
individually considered and justifiable, with the overall table coherent. Total footprint may land
±1 GB around ~15 GB; that variance is uninteresting. No overly-fine-tuned benchmark sweeps to
justify components; the PPL/behavior gates already in this plan are the qualification.

### Artifact pass (after the engine line is clean/integrated) — two options

Soft size basis: the Q4_K_M artifact we work against (GGUF 16.56 GiB / ninfer artifact 17.27 GiB);
~15 GB with no extra degradation is a commendable outcome. No vendor supplies NVFP4/MXFP8 for the
Unbound lineage (it is community-abliterated) and sm_75 has no FP4/FP8 tensor cores — the earlier
"vendor NVFP4" suggestion was wrong and is withdrawn.

1. **Training-free mixed quant ladder**: source = the FULL bf16 Unbound checkpoint (not the GGUF),
   so the Q4_K_M ceiling does not apply — training-free calibrated methods (GPTQ/AWQ/imatrix-style:
   calibration corpus only, no post-training) are on the table and can in principle BEAT the
   community Q4_K_M at equal size. Best-known methods for the best
   quality ladder; formats with direct sm_75 hardware support first (int8/int4 weight-only, fp16;
   W8A16/W4A16 dequant-then-fp16-GEMM styles where sensible — Turing INT8 tensor cores). Reserved
   knob from the GGUF path (down-proj → gate_up tier, ≈ −2.5 GiB) is subsumed by this: from bf16 we
   choose every component's tier fresh under the Memory policy.
2. **Strata-style runtime uncensor** — mechanism ESTABLISHED 2026-11-28 (`strata-lut-research.md`,
   primary sources read): not a weight-delta LUT. The uncensor edit is rank-1 along one refusal
   direction — for Flash-Next, Strata applies `h = h − (h·r)r` at runtime from a 480 KB control
   vector (weights byte-unchanged); for Inkling-2-Mini, SC117 bakes the same class of edit
   (`late_24_47_orth, k=1`, s_nonloop 4.0 / s_loop 2.5) into only 20/1,244 tensors
   (`layers.4..23.attn.wo_ud`) on TOP of the official Strata packs (NVFP4/MXFP8 experts preserved;
   ΔPPL +0.70 %, refusal 98→9/100, SQ-15 = 13.36 GiB). Reframes the vendor point: no vendor ships
   Unbound in native formats (still true), but official-quantized + rank-1 uncensor = uncensored
   from best-quantized weights. Build-time (our ladder from official bf16 + direction applied
   pre-quant) or runtime (port implements the projection; loop-aware per-visit semantics are the
   porting risk). Discriminator before choosing: diff Unbound v2 vs official bf16 — if Unbound is
   itself rank-1-on-wo_ud, the two routes converge. Lineage caveat: SC117 direction derived from
   the official base, not Unbound; equivalence UNMEASURED.

## Risks
- 22 GB card vs 14.65 GiB weights (= 15.74 GB decimal; the "15.6 GiB" here was a units slip) + 2.66 GiB KV @80K + arena/activations: fits with headroom per the Memory policy; Startup plan decides ctx size for the benchmark, report actual usable ctx separately from llama.cpp's 22-GB packing.
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
