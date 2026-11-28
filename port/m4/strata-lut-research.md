# Strata-style runtime uncensor — research consolidation (2026-11-28)

Question (Larry): Strata reportedly has an `-uncensored`-style toggle that alters weights at init
from a small stored LUT, giving uncensored behavior from the best quantized checkpoint. Establish
mechanism and history; assess adoption for Inkling-2-Mini (diff an official pre-quantized checkpoint
vs Unbound and ship the delta).

All findings verified this session by reading the linked primary pages; nothing below was run
locally. Sources are the authoritative cards, not press paraphrases.

## 1. Mechanism — it is NOT a weight-delta LUT (Qwen3.8-Flash-Next case)

Source: https://huggingface.co/alesha-pro/Qwen3.8-Flash-Next-abliterated-GSQ-RCO-Strata-GGUF +
https://github.com/Niko1221/Strata

- Huihui's abliteration of Qwen3.8-Flash-Next = a **rank-one edit along one unit direction `r`**
  applied to 144 residual-writing tensors: `W_new = W − r rᵀW` (strength 1.0). The edit is
  therefore fully described by `r`.
- Strata's "uncensored" feature does NOT store weight deltas. It loads a **480 KB control-vector
  sidecar** (`Huihui-…-refusal-direction-r.gguf`) and applies, after each steered layer, the
  runtime projection
  `h = h − (h·r) r`
  via engine flags `--control-vector-scaled <file>:1.0 --control-vector-layer-range 4 44
  --cvec-mode project --cvec-dir per-layer`. The official quantized weights stay byte-unchanged;
  per-request `"experimental_speed_projection": false` reverts to the censored model.
- Documented differences vs the baked weight edit: the runtime projection removes the `r`
  component from embedding/n-gram contributions too, and layer 0 cannot be steered.
- Measurements on the card (RTX 3090, IQ3_S): refusals 10/10 → 0/10; KL to BF16-Huihui reference
  0.074 with the vector vs 0.083 without; top-1 agreement 93 %; perplexity 2.527 vs reference
  2.565; agent coding tasks 5/5; the vector itself moves output only KL 0.005–0.023/window.
  Layer ranges 4–44 and 1–47 both tested working.
- History note: the user's recollection "don't touch layers 33 and beyond or catastrophic
  degradation" was NOT found in any source read. What IS documented: layer 0 exclusion (steering
  it is impossible in the runtime scheme) and steered ranges 4–44/1–47. Treat "layer 33" as
  unverified.

## 2. Inkling-2-Mini specifically — the analog already exists, and abliteration is SMALL

Source: https://huggingface.co/SC117/Inkling-2-Mini-abliterated-Strata (+ Strata README:
https://huggingface.co/strata-engine/Inkling-2-Mini-Strata)

- SC117 built abliterated Inkling packs **on top of the official strata-engine packs** (base:
  thinkingmachines/Inkling-2-Mini — so the vendor DOES supply the official quantized lineage;
  Strata preserves NVFP4/MXFP8 experts losslessly, "SF4+MX8").
- The abliteration: rank-1 orthogonal subspace projection (`late_24_47_orth, k=1`) applied ONLY to
  `layers.4..23.attn.wo_ud.weight` — **exactly 20 of 1,244 tensors**, with distinct strengths for
  loop vs non-loop layers (`s_nonloop=4.0`, `s_loop=2.5` — loop-aware). Baked by tensor transplant
  into the pack (byte-exact), not runtime — for Inkling the delta is small enough to just ship.
- Measured cost (their verification.json): KLD 0.0136 / same-top 95.63 % / **ΔPPL +0.70 %**;
  semantic refusal EN 98/100 → 9/100; ZH 50/50 → 4/50. Experts, DSpark, vision untouched.
- Pack sizes: SQ-20 18.02 GiB (lossless), **SQ-15 13.36 GiB (= 14.35 GB, the 16-GiB-GPU pack;
  wo_ud at 4 BF16 + 20 Q8_ROW)**, SQ-10 9.75 GiB. RTX 20-series supported; 4080S decode ~75–105
  t/s with DSpark (not our hardware).
- Lineage caveat: this direction was derived from the OFFICIAL base weights, not from nokeep's
  Unbound. Unbound (our current required uncensor) is a different, separately-evaluated edit
  (conv-/loop-/FP4-aware per [[Projects/Inkling-2-Mini]] verdict). SC117-vs-Unbound behavioral
  equivalence is plausible from the refusal/PPL numbers but UNMEASURED.

## 3. Consequences for our artifact pass (port/m4/plan.md Memory policy §Artifact pass)

1. Option 2's "LUT" generalizes to something cheaper: for both Flash-Next and Inkling the
   uncensor edit is **rank-1 on a small tensor set**. Two viable forms:
   a. *Build-time*: apply/extract the rank-1 direction on our own bf16 source before quantizing
      (Path 1 pipeline untouched; zero engine work).
   b. *Runtime*: implement `h = h − (h·r)r` after steered layers in the port (engine feature, like
      Strata's `--control-vector-scaled`; ~480 KB sidecar; censored/uncensored one-flag switch on
      ONE artifact). For the looped architecture, per-visit strength (`s_nonloop`/`s_loop`) and the
      3× loop-core projection semantics must be mirrored exactly — that is the porting risk.
2. It reframes the vendor point: no vendor ships *Unbound* in NVFP4/MXFP8 (still true), but the
   official NVFP4/MXFP8 base + a rank-1 uncensor gets us uncensored behavior **from
   best-quantized-official weights**, which is exactly the Strata method's win — and SC117's
   published ΔPPL (+0.70 %) is a sane prior for what the edit costs.
3. Decision input for when the engine line is clean: choose between (i) Unbound-bf16 → our ladder
   (Larry's Option 1), and (ii) official-bf16 → our ladder + rank-1 uncensor (built-time or
   runtime). A cheap discriminator first: diff nokeep Unbound v2 vs official bf16 and check
   whether Unbound is also rank-1-on-wo_ud-shaped (then (i) and (ii) converge) or a wider edit
   (then (ii) needs SC117's method re-derived, and the Unbound-vs-SC117 behavioral A/B becomes
   the gate).
4. Open: "layer 33" claim unverified (§1); SC117 method details live in SC117's repos/results;
   runtime-projection semantics on a LOOPED model (project once per visit vs per physical layer)
   should be read from Strata's `strata-inkling` notes before we implement form (b).
