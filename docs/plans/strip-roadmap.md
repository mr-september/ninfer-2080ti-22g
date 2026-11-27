# Strip roadmap: what was done, what is left

Revised 2026-10-25 at `master` `81b4888c` (`v0.2.0` published). Untracked working plan, not product documentation. The full original plan (decisions, inventory, stage texts) and the orchestrator prompt are archived in `.agent-tmp/strip-run/archive/`; durable rules live in [AGENTS.md](../../AGENTS.md) ("Behavior-preserving changes") and [turing-port.md](../turing-port.md) ("Standard measurements and acceptance rules", "Cut-down to one product").

## 1. Goal

Delete what a 22 GB RTX 2080 Ti running dense Qwen3.8-27B (`groupwise-int` Q4/Q5/Q6/Q8 and BF16, `sm_75`) does not use, without changing a bit of what the product computes. Keep DFlash/DFlash2, Qwen3.6-27B, vision, MTP, CLI, serve, perplexity, logit-compare, Windows plus Linux/Docker, and the Qwen4Exp/PLE/NF4/MoE model-side code and FP8 formats (for the Qwen4 series).

## 2. Done (log)

### 2.1 Decisions in force

| ID | Decision |
|---|---|
| D1, D2 | Keep DFlash/DFlash2 and Qwen3.6-27B support. DFlash2 and Qwen3.6-27B run on the 2080 Ti but are not yet in the gates: the published 27B artifact (20,437,521,664 bytes) includes the DFlash2 weights, while the local copy used for every baseline is an older 18,210,750,120-byte revision without them; no Qwen3.6-27B artifact has been converted yet. DFlash v1's only drafter targets the 35B-A3B, which has no kernels here |
| D3 | Linux verification in the Docker `dev` stage; do not touch the damaged WSL distro |
| D4, D5 | `NINFER_SM75` folded; `NINFER_TARGET` deleted; presets `release`/`dev`/`ops`/`windows`/`windows-dev`/`windows-ops` |
| D6 to D10 | eval harness, `.codex/`, FUNDING.yml, four of five model cards deleted; `eval/corpora`, tools, PR template, trimmed `multi-gpu.md` kept |
| D11 | FP8 formats, enums, reader/binder and declarations kept; FP8 kernels removed; NVFP4 and K8V4 removed completely |
| D12 | MoE, Qwen4Exp, PLE, NF4 model-side code kept; hwloc and the Qwen3.5-MoE host code kept (Stage 5) |
| D13 | QSA, hyper-connection mix, MoE FP8/NVFP4/CPU kernels stay deleted (recoverable from `v0.1.0`) |
| Stage 8 | Option B: prune the 35B-A3B (hidden 2048) dense shapes; keep every K = 2560 shape (Qwen4Exp) |

### 2.2 Stages

| Stage | Result |
|---|---|
| 0 Upstream cut-over | 2026-10-21, `8c6f8b67`: two cherry-picks, `upstream` remote removed |
| 1 Baseline and gate harness | 2026-10-21, baseline `92c77431` in `.agent-tmp/strip-baseline/`; harness passes against itself twice and catches a planted 1-ulp change |
| 2 `dgx_spark` and never-built code | 2026-10-21: 332 files deleted, compiled-source list identical |
| 3 Blackwell-only numerics | 2026-10-21, `334807a5`: NVFP4/K8V4 and FP8 kernels removed; G6 exact for 3,557 kernels with the K = 2560 Q8 GDN family excluded (nvcc 12.9 nondeterminism, unreached by 27B workloads) |
| 4 Target scheme, `NINFER_SM75` fold | 2026-10-22, `f0566a17`: 38 commits, `rtx5090/` trees flattened, 94 guards folded; Docker dev CTest 141/141 |
| 5 MoE/CPU-executor leftovers | 2026-10-22, `ab6a3546`: dead manifests and Kimi delta attention removed |
| 6, 7 | dropped (D12, D1) |
| 8 Shape pruning | 2026-10-24, `8e8b69f6`: 15 35B-A3B shape files deleted; G6 moved to object level (device-link shared-memory layout shifts executable hashes) |
| 9 Tests, docs, eval, cards, CI | 2026-10-24: docs only, link check clean; then the 5090 residue pass (context-cost preset, help text, SM120A traits, fixtures, benches of stubbed Ops) and the TTFT campaign port to 22 GB / groupwise-int |
| 10 Product identity | 2026-10-24: presets renamed, NOTICE, README, AGENTS, CONTRIBUTING, docs map |
| 11 Verification and release | 2026-10-25, `81b4888c`: all exact gates pass against `v0.1.0`; `v0.2.0` published (Windows zip) |

### 2.3 Outcome (from turing-port.md)

| Metric | `v0.1.0` | `v0.2.0` |
|---|---|---|
| Tracked files | 2,426 | 1,959 |
| Code lines (`src include apps bench tests`) | 267,039 | 225,488 |
| Compiled TUs, dev tree | 785 | 761 |
| Build steps, release / dev | 906 / 1,670 | 884 / 1,626 |
| Executable size | 713 to 715 MB | 517 to 519 MB |
| Release zip | 1.56 GiB | 1.03 GiB |
| CTest (Windows) | 138 | 134 |

The compiled set shrank by 3%: most deletions were never-built code. Build time was not measured comparably (10.6 min at full parallelism against 19.8 min at `-j4`), so no build-time gain is established. Device code in `ninfer_ops` at `v0.2.0`: 3,368 kernels in 158 objects, 385 MB of `.cu.obj` (release tree, `.agent-tmp/stage4/objhash_s11.json`).

## 3. Status after v0.3.0 (2026-10-30)

Batches A to D are done, gated and released as `v0.3.0` (pushed through `34f4b343`): 34 removal and build commits `81b4888c..4418c9a6`, then the docs commit. Gates against the `v0.2.0` results on the 2080 Ti: release and dev builds clean; `ninfer-logit-compare` (int8 and bf16 KV), greedy ids, CLI transcripts, protocol-B perplexity (4.796799), CLI and `ninfer-serve` startup memory lines identical; Windows CTest 134/134 and the real 27B loading test; object-level SASS identical for every surviving kernel (153 objects, 2,529 kernels; 3,368 at `v0.2.0`), apart from the nondeterministic `q5_a16_sliced_k_mma` compile (recompile before reporting a difference; the K = 2560 Q8 GDN exclusion is gone with those kernels); Docker dev CTest 141 tests (119 pass, 22 skip, 0 fail), 50 repeats of `ninfer_linear_swiglu_q4_test` and the Linux loading test. Measured: cold release build 9.0 min at `-j4` (19.8 at `v0.2.0`), executables 128-130 MB (517-519), zip 0.28 GiB (1.03). The A/B smoke against `v0.1.0` stayed inconclusive (about 14% session noise); device code is identical, so it is not a gate.

What was removed, by batch (details in git history and `docs/turing-port.md`):
- A: orphan launchers and aliases, `launch_column_jobs`, TMA/mbarrier helpers, `mma_fp8_e4m3`, `quant/fp8.cuh`, stale 5090 comments and docs.
- B: groupwise-int MoE kernels and the h2048 dispatchers, the Q8 GDN 12288x2048 kernels, the GDN control `n64_k2048` kernel with the fused-norm plan.
- C: `--kv-dtype fp8` and the FP8 KV-write and embedding kernels; FP8 oracles stay (D11).
- D: Qwen4Exp `K = 2560` linear, split and GDN shapes and the Q8 GDN input path; 35B-A3B 16/2 attention, RMSNorm, RoPE, embedding, split-conv, replay folds, the 4096 window and the 35B weight shape; `-lineinfo` made optional (`NINFER_CUDA_LINEINFO`, off in release; SASS identical).
- Kept deliberately: the 27B DFlash2 drafter shapes (window 2048, rmsnorm_rope 32/8, grouped conv, Q8 K=25600/17408), DFlash v1 code (about 170 lines, no kernels), PLE ops, hc `norm_combine`, block-scaled MoE and route h2560, BF16 d2560 embedding, `weight_input` `{48,2560}`, `test_recurrence.cpp` 35B head counts, `tests/test_gdn_replay_records.cpp` layout-only 30-layer cases, kv append tests with a 4096 window. No DFlash2 or Qwen3.6-27B artifact is run; their shapes were derived from configs and confirmed not removed.

### Open, none blocking

Done after the release (2026-10-30, `ef0624d7..ef3d3f94`): the `block-mix` and `final-mix` ops and fixture of `ninfer_gated_residual_bench`, the README section of the deleted `ninfer_qwen3_5_moe_execution_bench`, and `ninfer_qwen3_5_moe_real_test` with its offload variant.

- Fused-norm `Normalize`/`Capacity` branches of `gdn_control/kernels/mma.cuh` and `mma_launch.cuh`: kept on purpose. They are never instantiated, but `mma_kernel` takes `norm_weight`, `normalized_x` and `norm_eps` before the live parameters, so removing them shifts the kernel parameter-bank offsets and changes the SASS of the kept `n96_k5120` kernels, which the exact-equality gate forbids.
- Documentation pass done (2026-10-30, `a6ff25b0..`): the 35B-A3B DFlash commands are gone from `docs/cli.md`, `docs/serving.md` and `tests/README.md`; the paths that cannot run (`--spec dflash`, `dflash7`, the 35B DFlash round bench, the 35B recipe) are marked; the maintainer design records (`expert-weight-cache.md`, `replayssm-gdn.md`, `qwen3_5-model.md`) carry a status line; stale suites and test names (`qwen3_6_35b_a3b` linear suite, `ninfer_cpu_tensor_test`, the Qwen4Exp reclaim test) are removed; the lost backslashes in `turing-port.md` are restored. All Markdown links resolve and every `ninfer_*_test`/`_bench` name in the docs maps to a CMake target apart from names generated by macros.
- The Qwen3.5-MoE host code, planning and model tests that assume op support for 35B profiles fail at the op-capacity query by design (no GPU run touches them).
- Closed: the single failure of `ninfer_linear_swiglu_q4_test` in the Stage 11 Docker CTest. Static analysis (2026-10-30) found no defect: the folded Q4 MMA kernel and the sliced-K MMA kernel have a barrier between every shared-memory write and read of a reused stage, the staging/partial union is separated by `cp_wait<0>` and `__syncthreads`, reductions use a fixed order with no atomics, accumulators are zero-initialised and the test's data, oracle and tolerance are deterministic. The `cp_async` emulation on Turing is a synchronous load/store, so no pending-copy hazard exists. All of that is consistent with the 141-test Docker runs, 76 clean repeats and an environmental cause (host or WSL2 memory pressure killing the process, or an uncaught `cudaMalloc` failure in `run_case`, both of which leave no output). Latent only: the non-full sliced-K scale load reads a 4-byte pair that can extend 2 bytes past a row when the group count is odd; every kept Q4 shape has an even group count.
- Behavior change in unreachable code: the h2560 router's allocation boundaries went from `{2, 33}` to `{2}`.
- Not planned: a shared library or single executable for the shipped executables (download size is irrelevant at about 130 MB each).

## 4. How to gate the remaining work

Unchanged from the strip (AGENTS.md "Behavior-preserving changes"):
- **Per commit:** preprocessed device-pass identity of the affected TUs (`.agent-tmp/stage4/ppident.py`), the compile-database comparison (`ccdiff.py`), `git show -M --stat` for moves, and the targeted unit test for host logic. One concern per commit (delete, move or edit).
- **Per batch:** release build, then the dev build with CTest; `ninfer-logit-compare` bit-identical (int8 and bf16 KV); object-level G6 against `.agent-tmp/stage4/objhash_s11.json` (`objcmp.py`): the only allowed differences are the kernels the batch deletes on purpose; G5 startup plan; G9 grep. Perplexity protocol B (4.796799) and the A/B smoke once at the end; the Docker Linux CTest at the end.
- Batch A deletes only unreachable code and B only shapes no kept model can request, so G3 and G4 must stay identical and every kept kernel's hash unchanged. A batch that removes a feature also removes its docs, help text, tests and CMake references.
- Report sizes against the `v0.2.0` row of section 2.3: kernel count, `.cu.obj` bytes, executable size, zip size, build steps.

## 5. Evidence files

`.agent-tmp/strip-baseline/` (baseline, `gate.py`, `g6_family_excluded.txt`), `.agent-tmp/stage4/` (static-identity and object-hash tooling, `objhash_s11.json`), `.agent-tmp/strip-run/` (`LOG.md`, `STATE.md`, per-stage G7 traces, `stage3/trace/` nsys kernel traces, `archive/`), `.agent-tmp/strip-inventory/` (original inventory), `.agent-tmp/audit/` (2026-10-25 audit scripts).
