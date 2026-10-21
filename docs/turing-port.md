# Turing (sm_75) port

This fork builds NInfer for NVIDIA Turing (RTX 2080 Ti 22 GB, `CMAKE_CUDA_ARCHITECTURES=75`) as the only build target of the engine. The engine, the Qwen3.5-family runtime, frontend, serving and scheduling are unchanged; only the CUDA layer differs. This page records what differs and why. Device code that differs on sm_75 sits behind `NINFER_SM75`.

## Status of this tree

`master` is the Turing port rebased onto upstream `dev` `438ae67c` (merge `7923b62f` keeps the earlier fork history). Everything on this page marked "previous base" was measured before that port, on the fork's earlier sources, and its absolute figures are not current; [Current performance](#current-performance) and [Test status](#test-status-on-this-machine-native-windows-sm_75) are. The port's own gate results, including the greedy-output difference and how it was accepted, are under [Port onto upstream dev](#port-onto-upstream-dev-2026-10-18).

## Supported surface

| Area | sm_75 |
|---|---|
| Artifacts | `groupwise-int` (Q4/Q5/Q6/Q8 + BF16 tensors): Qwen3.8-27B is the only tested model. Qwen3.6-27B and Qwen3.6-35B-A3B are untested (op-level kernels only; the 21.2 GiB 35B-A3B artifact does not fit 22 GB) |
| NVFP4 / FP8 weights | unsupported; the routes link throwing definitions |
| KV cache | `bf16`, `int8`; `fp8`, `nvfp4`, `k8v4` unsupported |
| Speculation | MTP (tested); DFlash is untested and tied to 35B-A3B |
| Kimi delta attention | not built into the sm_75 test set (needs more than the 64 KiB CTA limit) |

Unsupported formats are not stubbed one symbol at a time inside upstream sources. `src/ops/CMakeLists.txt` filters the Blackwell-only sources (FP8 and the TMA kernels) out of the source manifests, and `src/ops/unsupported/turing_unsupported.cpp` defines the declarations the shared wrappers still name.

## Hardware differences and how each is handled

| Turing limit | Handling |
|---|---|
| no `cp.async` | `core/arch_traits.cuh`, `ops/common/cuda/memory.cuh`: synchronous copies, `commit`/`wait` are no-ops |
| no PDL | `core/pdl.cuh`: plain launches, `griddepcontrol` calls removed |
| no BF16 / TF32 tensor core | `mma_bf16`: BF16 -> FP16 operands on the FP16 tensor core (exact in range, saturating beyond ±65504); `mma_bf16_simt` is the exact FP32 reference; `mma_tf32`: SIMT emulation |
| FP16 MMA is k8, INT8 MMA is m8n8k16 | `mma_f16`, `mma_s8`: decomposed into the native shapes (A registers are `{row g K-lo, row g+8 K-lo, row g K-hi, row g+8 K-hi}`) |
| 64 KiB shared memory per CTA | `core::kMaxCtaSharedBytes`; schedules fit by single-buffering, halving K warps (Q8), smaller tiles (attention) |
| no `redux`, no `cvt.rn.bf16x2.f32` | `warp_reduce_max_u32` butterfly; `pack_bf16x2` via intrinsics |
| 68 SMs | launch plans already derive from `DeviceExecutionView::multiprocessor_count`; cooperative residency is queried with the occupancy calculator |

`tests/ops/test_mma_shim.cu` checks every tensor-core wrapper against naive math in the PTX fragment layouts; run it first when touching `ops/common/mma.cuh`.

## Prefill tiles and token embedding (previous base, one RTX 2080 Ti 22 GB)

**Prefill tiles.** About 95% of prefill time is the dequantizing GEMMs. Upstream sizes their tiles for the RTX 5090 (two `cp.async` stages, about 46 KB of shared memory), which on Turing leaves one 4-warp CTA per SM: profiled at 9.8% tensor-pipe and 11% issue-slot utilization, stalled on global loads with no `cp.async` to hide them. Single-stage tiles use about 27 KB, so two CTAs share each SM and one tile's MMA overlaps the other's decode and loads. Applied under `NINFER_SM75` only (the other architectures keep upstream's tables): Q5 plain and residual-add (`linear_add`) projections, Q4 plain projections, the folded Q4 gate/up SwiGLU (`MmaPairC128W8S1B2`), and the grouped GDN and attention input projections. Routes were chosen per shape by timing every instance across 128 to 2048 tokens.

| Operation (1024 tokens) | Before | After |
|---|---|---|
| Q5 MLP down 5120x17408 | 19.4 ms | 8.9 ms |
| Q5 down with residual add | 18.4 ms | 9.8 ms |
| Q4 gate/up 34816x5120, plain | 32.6 ms | 16.2 ms |
| Q4 gate/up folded with SwiGLU | 21.7 ms | 13.2 ms |
| GDN grouped input projection | 12.1 ms | 6.8 ms |

End-to-end prefill of the current tree is in [Current performance](#current-performance). Kernels at 5 to 8 tokens (MTP depths 4 and 5) were already within about 10% of the best candidate, so draft depth 3 stays the default: a 5-token step costs about 1.4x a 4-token step because the cheap SIMT kernels stop at 4 tokens.

**Token embedding.** The embedding is gathered one row per token, so it lives in pinned host memory that kernels read over PCIe (`Residency::PinnedHost`): 1.26 GiB less device memory, output byte-identical, no measurable decode change. If page-locked memory is unavailable (a host short of free RAM), the binder falls back to device residency at bind time. Effect: a 64k-token prompt now fits with MTP (it did not before); the largest context that starts is about 97k to 99k with MTP (it moves with how much VRAM the desktop and other programs hold) and between 110k and 125k without it.

**Output head format.** Re-encoding `text/output_head` from Q8 to Q6 (`python -m tools.artifact.requantize_tensor SRC OUT --object text/output_head --format q6_g64_fp16`, about 6 minutes) is nearly free in quality and saves memory but not time. Perplexity protocol B (fixed corpus, quick set, 4096 context, INT8 KV), previous base: overall 4.7944 to 4.7950 (+0.014%); Chinese +0.023%, long-form English +0.035%, English reference +0.001%, code -0.002%. The artifact is 341 MiB smaller and planned device memory drops 0.4 GiB, but decode is not faster (23.5 against 23.8 tok/s): the saved bandwidth is spent unpacking Q6.

**Not done: a keep-list output head.** Dropping rarely used rows from the main output head (and gathering logits back to the full vocabulary) would save about 0.6 GiB and a few percent of decode bandwidth. The repository's own frequency ranking shows the cost: its top 131,072 tokens (the proposal-head list) cover 99.1 to 99.4% of real tokens overall, 99.6% for English, 99.2 to 99.5% for Chinese and code, and only 97.5 to 98.2% for other languages, so up to a few percent of tokens could never be generated. With the embedding already on the host and a Q6 head available, that trade was not taken.

A bottom-up variant (keep every special, template and byte token and every real word; cut only tokens with no legitimate use) was audited on the Qwen3.8 tokenizer and is not worth building: of 248,320 head rows only 243 are padding, and the junk-shaped tokens (long symbol runs, control characters, replacement characters, mojibake, hex runs, unused code points) number about 450 rows, 1.7 MiB at Q6. One head row costs 4,000 bytes at Q6, so 0.1 GiB needs about 27,000 rows, and those would be real words: a large share of the tokens never chosen in the ranking corpus are ordinary Korean, Thai, Vietnamese and code-identifier words. Frequency alone cannot separate artifacts from rare words.

**MTP depth and the proposal head.** `--lm-head-draft` is free speed: +3% decode on average (+1.7%, +2.6%, +4.7% in paired runs), identical acceptance and byte-identical output, so the README examples use it. Depth 3 is best for code (decode tok/s at depth 1 to 5: 31.8, 37.0, 39.3, 31.4, 32.9; acceptance 92% down to 65%). For prose depth 1 or 2 is better (28.3, 28.0, 26.4, 21.4, 20.0; acceptance 69% down to 31%). The depth is fixed at startup because the draft count shapes the captured CUDA graphs and round buffers (51 references), so an adaptive depth would be a larger change than a tuning step.

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j
```

CUDA 12.8+ is enough (no 13.x-only features are used). The Q8 split-K translation units are slow to compile for sm_75 (minutes each); a cold build of the release preset took 10.6 minutes on 16 logical cores at full parallelism (native Windows, warm vcpkg cache, measured on the previous base; the default parallelism can run nvcc out of memory on a machine with little free RAM, so cap it with `CMAKE_BUILD_PARALLEL_LEVEL=4` or `-j4`). `ninja -t deps | grep '#deps 0,'` finds objects whose header dependencies were lost by an interrupted build; delete those objects, otherwise header edits silently do not rebuild them.

## Native Windows build

The same sources build with MSVC 14.4 and `nvcc` 12.9 (`cmake --preset turing-windows`; see the [README](../README.md#build) for prerequisites). Linux/WSL2 builds are unchanged; nothing selects between them at runtime, the environment you build in is the mode. What differs:

| Area | Handling |
|---|---|
| process id, local/UTC time, tty, console width, console setup | `src/core/os.h`; `init_console()` enables UTF-8, ANSI escapes and untranslated `\n` on stdout/stderr |
| 128-bit products (`unsigned __int128` is not MSVC) | `src/core/wide_math.h` (`mul_wide`); the host-context budget parser doubles its decimal fraction instead |
| artifact reads | `src/artifact/file_io.cpp`: POSIX `pread`/`O_DIRECT`, or Windows overlapped `ReadFile` with `FILE_FLAG_NO_BUFFERING` for the pinned-buffer weight upload |
| sockets | TCP keep-alive options also on Windows (no `TCP_USER_TIMEOUT`); Winsock started before name resolution; `ws2_32` via `ninfer::platform` |
| FFmpeg, libcurl | pkg-config on Linux, vcpkg packages on Windows, both exposed as `ninfer::ffmpeg` / `ninfer::curl` |
| argv/path encoding, long paths | `cmake/windows_utf8.manifest` embedded in the three executables |
| `localtime_r` in the vendored Jinja engine | `cmake/windows_compat.h`, force-included for that target only |
| `<windows.h>` min/max macros | `NOMINMAX` and `WIN32_LEAN_AND_MEAN` for every Windows target |

Windows-specific runtime cautions: keep the artifact on NTFS; the card is in WDDM mode, so the desktop and other GPU programs (browsers, image-generation tools) take VRAM that `--kv-capacity auto` can no longer use; if the NVIDIA driver allows CUDA allocations to spill into system memory, disable that ("CUDA - Sysmem Fallback Policy") for these executables so a full card fails instead of slowing down. `ninfer-serve` pins about 3 GiB of host memory by default (a 2 GiB context budget plus state images, and the embedding table); on a desktop with little free RAM pass a smaller `--host-context-mib`. FFmpeg needs the `zlib` feature for PNG; see the README.

### Native Windows versus WSL2

Measured on the previous base (2026-10-16) and not repeated since. Same machine, same 2080 Ti 22 GB, Qwen3.8-27B `groupwise-int`, INT8 KV, greedy, MTP off, runs interleaved and gated on an otherwise idle GPU (another program intermittently used this card, which invalidated some runs; those were discarded). Each environment used its own build and its own copy of the artifact.

| Prompt | Metric | Native Windows | WSL2 |
|---|---|---|---|
| 122 tokens, 512 generated | decode | 24.4, 24.5, 24.0 tok/s | 24.1, 24.0 tok/s |
| 122 tokens | prefill | 201.8, 204.9, 196.6 tok/s | 215.3, 213.5 tok/s |
| ~8k tokens | prefill | 253.0, 236.0, 258.9 tok/s | 259.5, 258.6, 262.0 tok/s |

Decode is at parity. Prefill is about 6% lower natively on a short prompt and about 2.5% on an 8k prompt (medians), consistent with a small fixed per-launch cost under WDDM; the kernels are identical. Absolute figures vary by around 15% between sessions with GPU clocks and background load, so only same-session interleaved comparisons mean anything. Native vision output (image, image+video) and the OpenAI/Anthropic server behavior match WSL2; the CUDA fault-injection part of the artifact-materialization test is Linux-only (it needs GNU `ld --wrap`).

## Default chat template

The former `--chat-style sharp-v22.1` option is now the vendored Sharp v22.5.0 Jinja template `tools/chat_templates/qwen3_8_sharp_v22_5.jinja`, which is embedded in `ninfer` and `ninfer-serve` at build time and used by default. `--artifact-chat-template` selects the artifact's own template and `--chat-template FILE` another file:

```bash
./build/turing-release/apps/ninfer model.ninfer --artifact-chat-template --prompt "..."
```

## Current performance

Measured 2026-10-19 on the release tree (`master`, built with `scriptsuild.ps1`; the benchmark executable comes from the same sources built with `scriptsuild.ps1 -Dev`). One RTX 2080 Ti 22 GB (SM clock 1350 MHz idle, 2.3 GiB of VRAM held by the desktop), native Windows 11, driver 610.88, CUDA 12.9, MSVC 14.44, Qwen3.8-27B `groupwise-int` (`qwen3_8_27b.ninfer`, 18.2 GB), INT8 KV, CUDA Graph decode, `ninfer_bench` on the repository token corpus (`bench/fixtures/bench_corpus.ids`, random token ids). Each command takes `--weights M`.

| Run | Options | Prefill (tok/s) | Decode (tok/s) |
|---|---|---|---|
| bench-short | `-p 512 -n 256 -r 3 --warmup 1 --kv-dtype int8 --max-ctx 16384` | pp512 615.99 +- 0.70 | tg256 27.03 +- 0.11 (MTP off) |
| bench-mtp | bench-short plus `--spec mtp --draft-tokens 3 --lm-head-draft` | pp512 594.56 +- 4.31 | tg256 29.58 +- 0.05, acceptance 0.377 |
| bench-long | `-p 8192,65536 -n 32 -r 2 --warmup 2 --kv-dtype int8 --max-ctx 70000` | pp8192 568.63 +- 7.59, pp65536 457.51 +- 7.26 | tg32 25.91 +- 0.01 (MTP off, `--max-ctx 70000`) |

The corpus is random token ids, so MTP acceptance and the MTP decode figure are far below real text. On real prompts (`ninfer`, `examples/cli/messages`, `--greedy --max-new 256 --no-thinking`, default flags against `--spec none`), decode was 47.5 against 26.3 tok/s for `scenario_code_python` and 28.8 against 26.0 tok/s for `scenario_story_en_mystery`. Through `ninfer-serve` (one request, 174-token prompt, thinking on, MTP) the first token arrived in 0.71 s and decode ran at 40.0 tok/s with 62% of drafted tokens accepted.

Memory at the default configuration (64k-token context, MTP): 15.4 GiB of weights on the device (14.7 GiB without MTP), a 2.60 GiB runtime (64k INT8 KV, workspace and graphs), 1.26 GiB of embedding table and 0.8 to 3.2 GiB of host context pinned in RAM (the server halves the default host context when free RAM is short). The 22 GiB card had 1.97 GiB left with the desktop running.

Session-to-session variation is several percent (the 8k prefill figure was 612 tok/s in an earlier session and 568.6 here; a same-session comparison against the pre-port build gave 566 and 568), so compare only runs taken in one session. Perplexity (`ninfer-perplexity`, protocol B: `--corpus eval/corpora/perplexity-1m/manifest.json --quick --kv-dtype int8`, 261,223 scored tokens): overall 4.796799, the same to the last digit on the release tree and on the dev tree.

### Performance model

Derived from the figures above, not measured separately. Decode without MTP reads the 14.66 GiB (15.7 GB) of device weights once per token, so 27.0 tok/s is about 425 GB/s, roughly 69% of the card's 616 GB/s peak. Prefill at 616 tok/s (512 tokens) and 569 tok/s (8k) is about 33 and 31 TFLOPS of dense-equivalent throughput, taking two FLOPs per parameter per token for a 27-billion-parameter model.

## Standard measurements and acceptance rules

Use these commands unless a change calls for something else. `M` is the artifact path.

| Name | Command | Reports |
|---|---|---|
| bench-short | `ninfer_bench --weights M -p 512 -n 256 -r 3 --warmup 1 --kv-dtype int8 --max-ctx 16384` | pp512, tg256 with MTP off |
| bench-long | `ninfer_bench --weights M -p 8192,65536 -n 32 -r 2 --warmup 2 --kv-dtype int8 --max-ctx 70000` | long prefill with MTP off (two warmups: one warmup once gave a 26 tok/s outlier at 8k) |
| bench-mtp | bench-short plus `--spec mtp --draft-tokens 3 --lm-head-draft` | decode with MTP (random-token corpus, so acceptance is low) |
| ppl-quick | `ninfer-perplexity M --corpus eval/corpora/perplexity-1m/manifest.json --quick --kv-dtype int8` | protocol B perplexity, overall and per domain ([perplexity.md](perplexity.md#reference-values)) |
| logit-compare | `ninfer M --messages examples/cli/messages/<file> --greedy --max-new 256 --spec none --kv-dtype bf16 --print-token-ids` on the baseline build, then score those ids with `ninfer-logit-compare` on both builds ([perplexity.md](perplexity.md#compare-two-builds-token-by-token)) | first-token candidate gap and per-token absolute log-probability differences |

Rules that have held up:

- Compare before and after in one session, interleaved, on an idle card. A gain smaller than the run-to-run spread counts as no change; session-to-session drift is several percent.
- Use a tolerance gate only where the arithmetic is allowed to change. The tolerance gate used for the port onto upstream `dev`: log-probability limits set before measuring (median absolute difference <= 0.02, maximum <= 0.2) and no top-2 logit gap moving by more than 3 BF16 output steps (0.375 at magnitude 16 to 32). A change that must not alter the arithmetic (removing code, moving files) is gated on exact equality instead: `ninfer-logit-compare` bit-identical, protocol B perplexity unchanged, full CTest.
- Perplexity and the op oracle suites, not greedy text, decide whether a port is numerically right: a wrong port once produced plausible text with perplexity 6.66 against 5.17.

## Baseline before the tuning (2026-10-17, previous base)

Reference numbers for the kernel tuning described below (the labels P1 to P7 and C1 in the headings name those items), measured on the previous base with the product `ninfer_bench` (Windows, MSVC 19.44, CUDA 12.9, driver 610.88, RTX 2080 Ti 22 GB, Qwen3.8-27B `groupwise-int`, INT8 KV, CUDA Graph decode, the repository's token-id corpus). The desktop was running (about 2.6 GiB of VRAM held by other programs, 0% GPU utilization, idle SM clock 1350 MHz before each run); each run starts with the warmup repetitions shown.

| Run | Command (after `--weights M`) | Result (mean, sample sd over repetitions) |
|---|---|---|
| `bench-short` | `-p 512 -n 256 -r 3 --warmup 1 --kv-dtype int8 --max-ctx 16384` | pp512 494.88 tok/s (sd 0.39); tg256 26.45 tok/s (sd 0.04) |
| `bench-long` | `-p 8192,65536 -n 32 -r 2 --warmup 1 --kv-dtype int8 --max-ctx 70000` | pp65536 394.51 tok/s (sd 0.08); tg32 26.20 tok/s (sd 0.00); pp8192: see below |
| pp8192 rerun | `-p 8192 -n 32 -r 4 --warmup 2 --kv-dtype int8 --max-ctx 70000` | pp8192 513.73 tok/s (sd 0.18); tg32 26.35 tok/s (sd 0.07) |
| `bench-mtp` | `bench-short` plus `--spec mtp --draft-tokens 3 --lm-head-draft` | pp512 492.88 tok/s (sd 0.18); tg256 29.76 tok/s (sd 0.04), acceptance 0.371 |

Exception: in the `bench-long` run the two measured pp8192 repetitions were 26.0 and 512.8 tok/s. The first was a transient of unknown cause (not reproduced in the rerun, where all four repetitions were within 0.4 tok/s), so the rerun is the pp8192 baseline and `bench-long` needs at least two warmups to be trusted at 8k. The benchmark corpus is random token ids, so MTP acceptance (0.371) and the MTP decode figure are far below the 38-44 tok/s seen on real text and are only comparable with each other.

## FP16 activation range (C1)

On Turing every BF16 operand is converted to FP16 before the tensor core, and values beyond +-65504 saturate. The debug build option `NINFER_SM75_RANGE_PROBE` (`cmake --preset turing-windows -B build-probe -DNINFER_SM75_RANGE_PROBE=ON`, sm_75 only) records the largest magnitude that reaches that conversion in any kernel, and how many elements exceed 16384, 32768 and 65504; `ninfer`, `ninfer-perplexity` and the benchmarks print one `range probe | ...` line when the Engine shuts down. It is a global maximum over all call sites and kernels, which bounds every individual site.

Measured on Qwen3.8-27B `groupwise-int`, INT8 KV, 2026-10-17:

| Workload | max abs value | elements > 16384 | clamped (> 65504) |
|---|---|---|---|
| `ppl-quick` (261,223 tokens: English reference, English long-form, Chinese, code; 4096 context) | 516.0 | 0 | 0 |
| one 65,536-token window of long-form English (`--text pg19/00.txt --context 65536`) | 438.0 | 0 | 0 |
| short chat prompt, 64 greedy tokens | 304.0 | 0 | 0 |

Every maximum is below 16384, so there is at least 4x headroom to the 16384 threshold set for this check (and more than 100x to the FP16 limit), nothing was clamped, and no pre-scaling is added. This covers these corpora and prompts; the SwiGLU output feeding the Q5 down projection, the site most at risk, is included in the global maximum. Decode at 1 to 4 tokens uses FP32 SIMT, so the two paths agree on these inputs. The ppl-quick result of the probe build (overall 4.794356) equals the production build's.

## Route sweep tool (P1)

`ninfer_turing_route_sweep` (`bench/ops/turing_route_sweep.cu`, built only for sm_75 with `NINFER_BUILD_BENCHMARKS=ON`) forces every Q4/Q5/Q6/Q8 launcher of one shape, drops the ones that throw or whose output differs from the production route's by more than 5% of the output RMS, times the rest with cold-cache CUDA Graph replay, and prints the ranking per token count plus a `TokenRoute` table to paste into the shape selector. The candidate list comes from the `q*_launch.h` headers at configure time. It includes private `ops/linear` headers, so it is a fork-only developer tool and not part of any product or CI target.

```bash
./build/turing-windows-dev/bench/ninfer_turing_route_sweep --qtype q5 --n 5120 --k 17408 --t 1,2,3,4,1024 --top 8
```

First result (previous base, Q5 `n5120_k17408`, the MLP down projection, T=1024, three runs within 0.2%): the production route `launch_q5_a16_mma_r64_t128_a1` took 8.41 ms (21.7 TFLOPS) and `launch_q5_a16_mma_r128_t64_s1_a1` 8.04 ms (22.7 TFLOPS), 4.6% faster; that result drove the route table. Current tree (2026-10-19, same command with `--t 1024`): the production route ranks first at 5.58 ms (32.7 TFLOPS), ahead of `launch_q5_a16_mma_r128_t64_s1_a1` at 8.10 ms; of the 32 Q5 launchers, 22 were legal at that shape and 10 were rejected.

## Decode kernel breakdown and the first P2 change (P2)

One decode step is 37.6 ms (Nsight Systems, CUDA Graph nodes traced, MTP off, 103 steps): kernel time equals step time. The card's sustained read ceiling, measured with `tools/hbm_bandwidth_probe.cu` built for sm_75, is 584.2 GB/s (94.8% of the 616 GB/s peak). Per step before the change:

| Kernel (op) | share | per call | bytes read | rate | of 584 GB/s |
|---|---|---|---|---|---|
| Q4 gate/up GEMV (`linear_swiglu`, 34816x5120) | 31.2% | 183 us | 94.7 MB | 517 GB/s | 88% |
| Q5 direct SIMT, K=17408 (MLP down, `linear_add`) | 22.0% | 130 us | 58.5 MB | 451 GB/s | 77% |
| Q5 staged GEMV, value+z rows (GDN input, T=1) | 15.4% | 124 us | 41.3 MB | 332 GB/s | 57% |
| Q5 direct SIMT, K=6144 (output projections, `linear_add`) | 8.2% | 48 us | 20.6 MB | 428 GB/s | 73% |
| Q8 output head (plain `linear`) | 6.4% | 2425 us | 1351 MB | 557 GB/s | 95% |

The MLP down and output projections run through `linear_add`, which has its own route tables (`src/ops/linear_add/q5/q5_linear_add_plan.cpp`); the plain-linear shape files under `src/ops/linear/*/shapes/` serve only the output head and the vision tower on the text path, so tuning them does not change decode.

**Change.** On Turing the GDN input projection's T=1 Q5 launch (`launch_q5_t1` in `src/ops/linear_attention/gated_delta_net/input_conv/q4_q5/convolution.cu`) now uses the direct SIMT kernel that the T>=2 path already used, instead of the staged shared-memory GEMV. Turing has no `cp.async`, so the staged GEMV cannot overlap its loads. That kernel went from 124.4 us to 92.5 us (446 GB/s, 76%).

Measured interleaved in one session (the GPU ran about 3% slower than in the morning baseline, so only same-session pairs are compared), `ninfer_bench`, INT8 KV, `-n 256 -r 3 --warmup 1`:

| Run | before (tok/s) | after (tok/s) |
|---|---|---|
| MTP off, round 1 / 2 / 3 | 24.92 (sd 1.03), 26.01, 25.84 | 26.68, 26.68, 26.49 |
| MTP off, medians | 25.84 | 26.68 (+3.3%) |
| MTP 3 draft tokens, round 1 / 2 | 29.61, 29.46 | 29.65, 29.59 (acceptance 0.371 both) |

Paired by round the MTP-off gain is +7.1% (the first baseline run was noisy), +2.6% and +2.5%. `greedy-identity` holds: `long_decode_aime26_01.json`, `--greedy --max-new 256 --spec none`, identical output text before and after (and identical across two baseline runs, so decoding is deterministic).

Remaining decode headroom, by the table above: the K=17408 and K=6144 `linear_add` Q5 kernels (77% and 73%), then the Q4 GEMV (88%).

**Q5 `linear_add` direct-kernel schedule sweep (P2 step 4, no change made).** `Q5A16DirectSimtSchedule<1, T, W, G, MinBlocks, K, exact>` in `q5_linear_add_gemm_simt.cu` (W warps per row, G groups per warp tile, K tile per iteration W*G*64) was swept with `ninfer_q5_linear_add_bench` (graph replay, warmup 200, repeat 200) over (W,G,MinBlocks) = (2,8,16) current, (2,4,16), (4,4,10), (4,8,8), (2,12,16), (2,8,8). For K=17408 the current schedule is best or tied at every T (T=1 about 115 us, 510 GB/s in isolation); the only gain is (4,4,10) at T=1, 2.5%, and it is far worse at T>=2 (T=3: 305 us against 184 us). For K=6144 all variants are within the run-to-run spread (the T=1 median alternates between about 43 us and 49 us across identical runs). Larger W, G or MinBlocks values and W=1 are slower. No schedule change is justified. (An earlier sweep with short warmups was discarded: its dev-shell invocations accumulated PATH until builds failed silently.)

**Decode is power-capped.** While decoding, the card sits at its 250 W cap (`clocks_throttle_reasons.active` = software power cap, 0x4): measured over 75 busy samples, SM clock 1455-1635 MHz (mean 1511), power 237-252 W (mean 246), memory clock fixed at 7000 MHz, temperature at most 44 C. Short micro-benchmarks draw less power and run faster, which is why the in-situ kernels run slower than the same kernels in isolation (the K=17408 `linear_add` kernel is about 115 us alone and 130-135 us inside decode). Gains from kernels that issue fewer ALU instructions per weight byte may therefore be larger in decode than in isolation, but load-width or occupancy changes alone are not expected to close the gap.

## INT8-KV attention tile 64x32 (P3)

`Int8KvTiledInstance` (`src/ops/softmax_attention/dense/causal_cache/int8/instances.h`) on Turing changes from `<32, 32, 128>` to `<64, 32, 128>`. The 64x32 tile stages 55,296 bytes of shared memory (54 KiB, under the 64 KiB limit; 64x64 would need 92 KiB), uses 16 warps per CTA instead of 8, and reuses each Q tile over twice as many query rows.

Measured interleaved in one session (`ninfer_bench`, INT8 KV, MTP off, `--max-ctx 70000`; 8k: 4 repetitions after 2 warmups, 64k: 2 repetitions after 1 warmup; two rounds of before then after):

| Prompt | before (tok/s) | after (tok/s) | change |
|---|---|---|---|
| 8192, round 1 | 507.17 (sd 1.10) | 513.44 (sd 2.38) | +1.2% |
| 8192, round 2 | 496.74 (sd 15.68; one repetition at 473.4) | 517.46 (sd 1.37) | +4.2% (+2.4% against the other three repetitions) |
| 65536, round 1 | 390.74 (sd 0.33) | 427.04 (sd 2.76) | +9.3% |
| 65536, round 2 | 389.47 (sd 0.57) | 424.13 (sd 1.46) | +8.9% |

Checks: the attention, KV-cache and sliding-window tests pass (`ninfer_softmax_attention_test`, `ninfer_kv_cache_test`, `ninfer_kv_cache_append_test`, `ninfer_sliding_window_attention_test`, `ninfer_causal_conv1d_silu_test`). `ppl-quick` (INT8 KV, 4096 context) with the new tile against the baseline 4.794356: overall 4.794641 (+0.006%); Chinese 6.750531 against 6.746396 (+0.061%), long-form English 7.179776 against 7.178846 (+0.013%), English reference 6.458972 against 6.461225 (-0.035%), code 1.673260 against 1.673525 (-0.016%). The tile shape changes the order in which key blocks are merged, so bit-identical output is not expected; the overall change is within the 0.05% limit and the Chinese domain alone is slightly above it, with the other domains moving the other way.

## Converting sliced-K activations once (P4): tried, regressed, not kept

The sliced-K kernels (T of about 5 to 160) stage BF16 activations raw and convert each B fragment to FP16 inside every MMA (`mma_integer_weights`), once per row tile. The change tried for Q5 (`q5_a16_sliced_k_mma.cuh`): stage the activations with `stage_activation_operand_zfill`, which converts BF16 to FP16 once while writing shared memory, and call `mma_operands` (FP16 MMA on Turing) in the loop. The Q5 linear, `linear_add`, `linear_pair`, `attn_input_proj` and GDN-input op tests all passed with it.

Measured interleaved in one session (`ninfer_bench`, INT8 KV, `--spec mtp --draft-tokens 5 --lm-head-draft`, `-p 512 -n 256 -r 3 --warmup 1`, three rounds of before then after):

| | before | after |
|---|---|---|
| pp512 (tok/s) | 497.0, 496.8, 496.6 | 494.7, 494.6, 497.0 |
| MTP-5 tg256 (tok/s) | 22.17, 22.18, 22.19 | 21.21, 21.22, 21.07 (-4.4%) |
| acceptance | 0.231 | 0.231 |

The decode step at MTP depth 5 (T=6) is consistently 4.4% slower, so the gate for this change ("`bench-mtp` with `--draft-tokens 5` not worse") fails and the change was reverted. A likely cause is that Turing's emulated `cp.async` is an ordinary load followed by a store: converting in between lengthens that dependent chain once per staged vector, while for the 16-row blocks used at these token counts the per-MMA conversions it removes were not more numerous than the staged conversions it adds. The other kernels considered (Q4, Q6, Q8, grouped Q8, `linear_topk`) were not changed.

## Register-prefetch pipelining of the prefill GEMMs (P5)

Prefill is almost entirely three dequantizing GEMM kernels. Nsight Systems on a 4096-token prefill (INT8 KV, MTP off, kernel time 98% of the total): `q5_a16_mma_kernel` (MLP down and output projections through `linear_add`, 37.8%), `q4_linear_swiglu_mma_split_half_pair_kernel` (MLP gate/up, 37.0%) and `rowsplit_grouped_mma_kernel` (GDN and attention input projections, 18.6%).

Turing has no `cp.async`, so these kernels' staging is a load from global memory followed by a store to shared memory, issued one item at a time after the MMAs of the previous tile: each k-tile waited out several serialized global-memory latencies. The change (Turing only, `q5_a16_mma.cuh`, single-buffered-activation schedules such as `MmaR64T128A1`, which is what `linear_add` and the plain-linear prefill routes use): the next activation tile is loaded into registers (all of a thread's loads issued together) before the current tile's MMAs and stored to shared memory after them. Other targets keep the asynchronous copy.

Kernel level, `ninfer_turing_route_sweep` (cold graph replay) on the `MmaR64T128A1` launcher, before, batched loads only, and with prefetch across the MMAs:

| Shape, T | before | batched loads | prefetch across MMAs (kept) |
|---|---|---|---|
| Q5 n5120 k17408, T=512 | 5026 us (18.2 TFLOPS) | 3463 us (-31%) | 3166 us (-37%, 28.8 TFLOPS) |
| Q5 n5120 k17408, T=1024 | 8413 us (21.7) | 5816 us (-31%) | 5542 us (-34%, 32.9) |
| Q5 n5120 k6144, T=1024 (old production route `r128_t64_s1_a1`: 2823 us) | 2900 us | 2065 us | 1903 us (-33% against the old production route) |
| Q5 n7168 k5120, T=1024 (old production route: 3347 us) | 3438 us | 2431 us | 2279 us (-32% against the old production route) |

After the change this launcher is the fastest Q5 route at every T from 512 up in those sweeps, ahead of the `r128_t64_s1_a1` route that the plain-linear selector uses above 1536 tokens.

End to end, interleaved in one session (`ninfer_bench`, INT8 KV, MTP off; pp512: 3 repetitions after 1 warmup, pp8192: 4 after 2, pp65536: 1 after 1), two rounds of before then after:

| Prompt | before (tok/s) | after (tok/s) | change |
|---|---|---|---|
| 512, round 1 / 2 | 497.76 / 494.44 | 580.92 / 575.64 | +16.7% / +16.4% |
| 8192, round 1 / 2 | 523.27 / 519.91 | 579.16 / 577.37 | +10.7% / +11.1% |
| 65536, round 1 / 2 | 435.36 / 432.50 | 471.06 / 470.62 | +8.2% / +8.8% |

Checks: `ninfer_linear_q5_a16_test` and `ninfer_linear_add_q5_a16_test` pass; greedy text of `long_decode_aime26_01.json` (341-token prompt, which runs through the changed route) is byte-identical before and after (only the load order changed). This is the Q5 family only (37.8% of prefill); the gate for the whole change (8k prefill +15%) needs the Q4 SwiGLU and grouped-projection kernels as well.

**Same change in the folded Q4 gate/up SwiGLU kernel** (`q4_linear_swiglu_gemm_mma.cuh`, the single-stage `TuringPrefillCfg`): the next tile's activation vectors, code vectors and scale words are loaded into registers before the MMAs and stored after them. Interleaved in one session against the build with only the Q5 change (same method as above, two rounds, before then after; greedy text identical, `ninfer_linear_swiglu_q4_a16_test` passes):

| Prompt | before (tok/s) | after (tok/s) | change |
|---|---|---|---|
| 512, round 1 / 2 | 583.06 / 575.69 | 603.89 / 599.44 | +3.6% / +4.1% |
| 8192, round 1 / 2 | 586.15 / 578.78 | 599.09 / 597.29 | +2.2% / +3.2% |
| 65536, round 1 / 2 | 473.64 / 471.31 | 483.52 / 483.32 | +2.1% / +2.5% |

This kernel (37% of prefill) gains less than the Q5 one because it does more ALU work per loaded byte and already had two resident CTAs overlapping its loads.

**Same change in the grouped input-projection kernel** (`rowsplit_grouped_mma.cuh`, GDN and attention input projections, single-stage `TuringPrefillCfg`; activation, code, high-bit and scale loads prefetched into registers). Interleaved against the build with the Q5 and SwiGLU changes (two rounds; greedy text identical; the GDN-input, attention-input, snapshot and record tests pass):

| Prompt | before (tok/s) | after (tok/s) | change |
|---|---|---|---|
| 512, round 1 / 2 | 609.33 / 601.34 | 614.76 / 610.81 | +0.9% / +1.6% |
| 8192, round 1 / 2 | 605.51 / 598.40 | 612.40 / 609.18 | +1.1% / +1.8% |
| 65536, round 1 / 2 | 487.14 / 483.58 | 491.71 / 491.05 | +0.9% / +1.5% |

**P5 total.** Each step was measured against the build before it, in its own session; the sessions drift by a few percent, so the steps are not additive on paper. Whole-item numbers in the first session's terms (before any P5 change: pp512 497.8 / 494.4, pp8192 523.3 / 519.9, pp65536 435.4 / 432.5): after all three kernel families, a later session measured pp8192 612.40 and 609.18 against a same-session predecessor, so the 8k prefill is about 17% above the pre-P5 build (523.3 to 612.4 and 519.9 to 609.2: +17.0% and +17.2%), which meets the gate (8k prefill +15%). Remaining kernels in the planned list (Q4 and Q6 and Q8 plain MMA, `linear_pair`, the attention and GDN gating projections) were not changed: none of them is more than a few percent of prefill time.

## Output head Q6 against Q8, re-measured (P6)

The earlier conclusion that a Q6 head saves memory but not time (23.5 against 23.8 tok/s) sat inside the session noise. Re-measured with five interleaved rounds, the Q8 artifact (`qwen3_8_27b.ninfer`) against a copy with `text/output_head` re-encoded to `q6_g64_fp16` (`python -m tools.artifact.requantize_tensor ... --object text/output_head --format q6_g64_fp16`, 17,853,160,360 bytes, 341 MiB smaller; about 17 minutes on the CPU here), same binary, `-p 512 -n 256 -r 3 --warmup 1 --kv-dtype int8 --max-ctx 16384`, MTP off:

| Round | Q8 head tg256 (tok/s) | Q6 head tg256 (tok/s) |
|---|---|---|
| 1 | 27.08 | 26.93 |
| 2 | 27.11 | 26.88 |
| 3 | 27.05 | 26.96 |
| 4 | 27.06 | 26.99 |
| 5 | 27.07 | 26.97 |
| median | 27.07 | 26.96 (-0.4%) |

pp512 is the same within noise (about 613 to 615 tok/s for both). Planned device memory is 358 MB lower with the Q6 head (16.105 against 16.463 GB). Every Q6 run is slower than every Q8 run: the Q6 head saves memory and costs about 0.4% of decode speed, so the Q8 head stays the recommended artifact. The Q8 head read is 1.35 GB at about 95% of the card's sustained bandwidth, so a smaller head has little time to gain, and the Q6 kernel's extra decode work shows.

## Test status on this machine (native Windows, sm_75)

`ctest --test-dir build/turing-windows-dev` on `master` (2026-10-19, `PYTHONUTF8=1`): 138 tests, 0 failed, 25 skipped. The skips are the FP8, NVFP4 and K8V4 cases (their routes are stubs on Turing) and the real-model tests, which need `NINFER_TEST_ARTIFACT` and were not run here. The Python-driven tests (chat templates, the schema oracles, the PLE disk tests) need Python 3.10 or newer with `jinja2` and `jsonschema`; on Windows set `PYTHONUTF8=1` first, because they pipe UTF-8 text to child processes through the console code page otherwise. `ninfer_device_sync_empty_test` is registered only outside Windows (CTest on Windows drops an environment entry with an empty value).

## Fusing small decode projections (P7): measured ceiling, not pursued

Per decode step (Nsight Systems, CUDA Graph nodes traced, MTP off, 103 steps, after the P2 change): 711 kernel launches and 37.23 ms of kernel time. The kernels whose average duration is under 20 us (norms, RoPE, GDN gating and its small recurrent update, SiLU-mul, the causal-convolution and attention-merge helpers: 18 kinds) account for 390 launches and 2.00 ms per step, 5.4% of kernel time. None of them is a projection: the projections are all among the 321 launches that take 20 us or more (35.2 ms), and the small ones proposed for fusing are already fused (`attn_input_proj` carries Q, gate, K and V; `gdn_input_proj` carries the Q4 query/key and Q5 value/z sides; the gating projection is fused with the GDN norm). The gap between the kernel time and the step time (37.2 ms against 1/26.4 s = 37.9 ms) is about 0.7 ms, so the missing PDL costs under 2% of a step. Even a fusion that removed every small kernel would be bounded by about 7%, and a realistic one (folding norms and gating into their neighbours) by 1 to 2%, at the cost of new kernels whose numerics must be requalified. Since the item is optional and its ceiling is that small next to the changes already made, it was not implemented.

## Prefill tuning result (2026-10-18, previous base)

`ninfer_bench`, INT8 KV, CUDA Graph decode, native Windows, RTX 2080 Ti, Qwen3.8-27B `groupwise-int`. The comparison is the build after the P2 decode change (but before P3 and P5) against the current build, interleaved in one session, two rounds each:

| Metric | before P3-P5 (round 1 / 2) | current (round 1 / 2) | change |
|---|---|---|---|
| pp512, tok/s | 498.3 / 498.5 | 613.9 / 613.1 | +23.2% / +23.0% |
| pp8192, tok/s | 516.4 / 516.2 | 611.9 / 612.0 | +18.5% / +18.6% |
| pp65536, tok/s | 397.0 / 396.9 | 492.8 / 493.1 | +24.1% / +24.2% |
| tg256, MTP off, tok/s | 27.08 / 26.99 | 27.02 / 27.07 | unchanged (no decode kernel changed after P2) |
| tg256, MTP 3 draft tokens, tok/s | 29.96 / 29.91 | 29.97 / 29.95 | unchanged (acceptance 0.371 both) |

Perplexity protocol B (INT8 KV, 4096 context): overall 4.794641 on the build after these changes, identical in every digit to the build after P3; the P5 kernels change only the order of loads. The scoring rate of that run went from 247.5 to 305.9 tok/s.

Against the first measurements of the baseline above (different session, so the session-to-session drift of a few percent applies): pp512 494.9, pp8192 513.7 (rerun), pp65536 394.5, tg256 26.45, MTP-3 tg256 29.76 tok/s, i.e. prefill about 19 to 25% faster, decode about 2% faster (the interleaved P2 measurement of the decode change was +3.3%) and MTP decode within 1%.

What each item contributed: P2 (GDN T=1 Q5 launch) +3.3% decode; P3 (64x32 attention tile) +9% at 64k and +1 to 4% at 8k; P4 (convert activations once) tried and reverted, -4.4% at MTP depth 5; P5 (register prefetch in the Q5, Q4 SwiGLU and grouped input-projection kernels) +17% at 8k and +16% at 512 on its own; P6 (Q6 head) not recommended, -0.4% decode; P7 (fusing small kernels) not pursued, ceiling 1 to 2%. C1 (FP16 range) closed with no change needed.

## Docker image

Verified on this machine (Windows 11, Docker Desktop 28.3.2 with the WSL2 backend, 24 GB VM, driver 610.88): `docker build -t ninfer-fork .` took 19.3 minutes (the build stage compiles for sm_75, the CMake default; the runtime image is 3.17 GB) and `docker run -d --gpus all -v E:\AI_Models\LLM\neroued\Qwen3.8-27B-NInfer:/models -p 18095:8080 ninfer-fork` started `ninfer-serve /models/qwen3_8_27b.ninfer --host 0.0.0.0` with the default configuration (64k-token context, MTP, 3.15 GiB host context pinned in 3.7 s, 1.26 GiB pinned embedding). The default embedded chat template worked inside the container with no source tree. One chat completion through the published port returned the expected reply (171 prompt tokens, 27 output tokens, TTFT 688 ms, prefill 252 tok/s, decode 49 tok/s with MTP).

## Port onto upstream dev (2026-10-18)

The fork now lives on top of upstream's `dev` branch as the `turing` build target (`NINFER_TARGET=turing`, sources composed from the `rtx5090` trees with `NINFER_SM75` defined, native Windows through `turing-windows*` presets) instead of a layer re-applied on each sync. The port is the commit series on branch `s1-turing-port` (ten commits on upstream `438ae67c`): build and toolchain, Windows I/O layer, shared CUDA primitives, dequantizing kernels, route tables, throwing definitions for unbuilt routes, product layer, tests, benchmarks, docs.

Verification on this machine (RTX 2080 Ti 22 GB, native Windows, CUDA 12.9, Visual Studio 2022 Build Tools), against the fork's last pre-port build (`bde50a4e`, `build-windows`), Qwen3.8-27B `groupwise-int`:

| Check | Pre-port | Port | Note |
|---|---|---|---|
| Perplexity, protocol A: 20 237 tokens of this repository's docs (a one-off text, not kept), BF16 KV, context 2048 stride 1024. Not comparable with the next row | 5.171412 | 5.169092 | -0.045%, same corpus and flags |
| Perplexity, protocol B: `ninfer-perplexity --corpus eval/corpora/perplexity-1m/manifest.json --quick --kv-dtype int8` (261,223 scored tokens, context 4096 stride 2048), final series tree (2026-10-19, after the read-queue fix) | 4.794641 | 4.796799 | +0.045% overall; pass limit set before the run was 0.1% |
| Decode, greedy, INT8 KV, context 4096, 128 new tokens, tok/s, three interleaved rounds | 25.0 / 24.7 / 24.6 | 24.5 / 24.9 / 24.7 | mean 24.77 against 24.70 (-0.3%), inside the round-to-round spread |
| Default launch (65 536 context, explicit KV) | starts, 18.0 GiB planned | starts, 18.9 GiB planned | the port plans 1 GiB of KV headroom by default (upstream's policy), the old build 0 B |

Protocol B per domain (pre-port / port): chinese_reference 6.750531 / 6.767452 (+0.25%), english_long_form 7.179776 / 7.178341 (-0.02%), english_reference 6.458972 / 6.454913 (-0.06%), ninfer_code 1.673260 / 1.673458 (+0.01%). The 0.1% limit applies to the overall figure; the Chinese domain alone moved by more. Quote 5.169092 only with protocol A and 4.7968 only with protocol B.

`ninfer_bench`, the Phase 2 procedure (INT8 KV, CUDA Graph decode, MTP depth 3 with the optimized head for the MTP row), pre-port build `bench_final_new` against the port, interleaved in one session, two rounds (round 1 / round 2):

| Metric | pre-port | port | change |
|---|---|---|---|
| pp512, tok/s | 568.1 / 569.0 | 581.5 / 575.7 | +2.4% / +1.2% |
| pp8192, tok/s | 566.1 / 555.6 | 568.3 / 557.4 | +0.4% / +0.3% |
| pp65536, tok/s | 457.4 / 447.7 | n/a / 448.1 (a separate run: 449.0) | 0% (round 1 returned no value, see below) |
| tg256 decode, tok/s | 25.12 / 24.75 | 25.16 / 24.02 | +0.2% / -2.9% |
| MTP3 decode, tok/s (acceptance) | 27.57 / 27.60 (0.371) | 28.02 / 27.72 (0.377) | +1.6% / +0.4% |

Pooling all decode samples taken in this session (these rounds plus three interleaved CLI runs of 128 tokens: 25.0 / 24.7 / 24.6 against 24.5 / 24.9 / 24.7) gives 24.83 tok/s before and 24.66 after (-0.7%); the worst single pair is -2.9% (tg256, round 2) and was not reproduced in the other rounds, so it is reported and not attributed. The round-1 pp65536 value of the port is missing because the benchmark returned no result in that run, at a time another process held GPU memory (nvidia-smi showed 5 GB in use by other applications); the same command run alone afterwards gave 449.0 tok/s.

Noise bound for decode: eight further interleaved pairs in one session (pre-port `ninfer_bench` against the port's `ninfer_bench`, `-p 512 -n 256 -r 3 --warmup 1 --kv-dtype int8 --max-ctx 16384`, MTP off, pre first in each pair): pre 27.22 / 27.21 / 27.18 / 27.16 / 27.19 / 27.13 / 27.14 / 27.08, port 27.25 / 27.22 / 27.14 / 27.19 / 27.14 / 27.13 / 27.14 / 27.13 tok/s. Per-pair differences +0.12, +0.03, -0.14, +0.10, -0.16, -0.01, 0.00, +0.18%; mean +0.01%, standard deviation 0.12%, round-to-round standard deviation 0.045 tok/s for both builds. The earlier -2.9% pair was not reproduced in any of these eight pairs and its cause is not known; the eight pairs bound the pair-to-pair spread under these conditions at about 0.2%.

A first benchmark of the port showed prefill 8 to 16% slower (pp512 476 / 508, pp8192 499 / 520, pp65536 404 / 428 tok/s). A kernel-level profile of a 2048-token prefill showed 192 launches of the grouped input-projection kernel using the 5090 tile (about 11.7 ms each) where the pre-port build used the single-stage Turing tile (5.2 ms). Upstream now serves the GDN input projection through the split-projection shape `n4096_n12288_k5120`, not the plan the fork had patched; that shape now selects the Turing tile above 64 columns. After it, total kernel time of the profile is 7199 ms against 7243 ms before.

Greedy identity, re-checked on eight fixed prompts (`examples/cli/messages`: long_decode_aime26_01/15/30, reasoning_jacobian_counterexample_3d, scenario_code_cuda/python/typescript, scenario_story_en_mystery; `--greedy --max-new 256 --spec none --kv-dtype bf16 --max-context 8192`, pre-port `build-windows` against the port): the generated text is byte-identical for three prompts (aime_01, aime_15, jacobian) and diverges for five: aime_30 at the first token ("We need" against "Let me"), the others after 324 to 1,614 characters. The divergence is deterministic and not caused by prefill chunking (the pre-port build gives the same text with `--prefill-chunk` 512, 1024 and 2048, and so does the port). Sampling the aime_30 first token 60 times at temperature 1 with the same seeds gives We 38 / Let 21 / I 1 on the pre-port build and We 21 / Let 38 / I 1 on the port, so on that prompt the two builds order the same two candidate tokens differently by roughly the size of their gap (both distributions are split close to 40/60). Perplexity (5.169092 against 5.171412) says the two builds compute the same function up to rounding. A search of upstream's log between the fork's old base and `dev` found no commit that changes the arithmetic of a route the Turing build compiles: the sampling and argmax change (`2e098d5b`), the GDN control and recurrence changes (`38ce3b49`, `22470b06`) and the 10240-wide rmsnorm (`569861a3`) are Spark-only or leave the result unchanged. What did change is the accumulation order of the Turing projections themselves: upstream moved the GDN input projection and the attention projections into the independent split-projection shapes (`90d25edd`, `4eb2c042`), which this port tiles differently from the pre-port grouped kernel. That is the likely cause, but it is a hypothesis: the divergence was not traced to a single kernel, and no upstream commit that changes the arithmetic was found. The logit-level comparison below measures the size of the difference instead.

**Logit-level comparison (2026-10-19).** Method: `ninfer-logit-compare` ([perplexity.md](perplexity.md#compare-two-builds-token-by-token)), the `CausalScoring` Engine path that `ninfer-perplexity` uses, BF16 KV, context 8192, the default chat template. The greedy ids of each build's own `ninfer --greedy --max-new 256 --spec none --print-token-ids` run were taken for the pre-port build, and those 256 tokens were scored on both builds after the rendered prompt (the rendered prompt length matched `Engine::count_tokens` on every prompt on both builds). Pre-port is master's `build-windows` (commit `bde50a4e`), the port is this series.

aime_30, first token, log-probability after the prompt:

| build | log p(We) | log p(Let) | We - Let | greedy first token |
|---|---|---|---|---|
| pre-port | -0.6405 | -0.7655 | +0.125 | We |
| port | -0.8336 | -0.5836 | -0.250 | Let |

The difference moves by -0.375 nats. The final logits are BF16, whose step is 0.125 at magnitude 16 to 32, so the We - Let difference can only move in multiples of 0.125: the shift is three output steps on a candidate pair that was one step apart.

Per prompt, |delta log p| of the pre-port greedy tokens between the two builds, from the first generated token up to and including the first token where the builds' greedy outputs differ (all 256 where they do not):

| prompt | first differing token | tokens scored | median | max (token index) |
|---|---|---|---|---|
| long_decode_aime26_01 | none | 256 | 0.0000 | 0.0905 (4) |
| long_decode_aime26_15 | none | 256 | 0.0000 | 0.0915 (151) |
| long_decode_aime26_30 | 0 | 1 | 0.1931 | 0.1931 (0) |
| reasoning_jacobian_counterexample_3d | none | 256 | 0.0000 | 0.0819 (122) |
| scenario_code_cuda | 194 | 195 | 0.0000 | 0.0753 (158) |
| scenario_code_python | 80 | 81 | 0.0000 | 0.0635 (43) |
| scenario_code_typescript | 227 | 228 | 0.0000 | 0.0806 (201) |
| scenario_story_en_mystery | 138 | 139 | 0.0000 | 0.0536 (113) |
| pooled | | 1,412 | 0.0000 | 0.1931 (aime_30, 0) |

Over all 2,048 scored tokens (no truncation at the divergence) the pooled median is also 0.0000 and the maximum is the same 0.1931; the largest values after it are 0.0924 (code_typescript, token 233), 0.0915 and 0.0905. The median is exactly zero: most scored tokens have the same log-probability on both builds. At the other four divergences (tokens 80 to 227) the pre-port token's log-probability moved by at most 0.0924 anywhere in those four prompts.

**Decision and rule.** The log-probability limits were set before measuring (median |delta| <= 0.02, max <= 0.2) and passed: 0.0000 and 0.1931. Only the pre-set first-token limit failed (the aime_30 We - Let difference must shift by less than 0.3 nats; it shifted by 0.375). That limit ignored the BF16 output step, under which 0.375 is three steps on a near-tie, so the owner replaced it afterwards, and accepted the greedy gap, with: no top-2 logit gap moves by more than 3 BF16 output steps (0.375 at magnitude 16 to 32). The aime_30 first token meets this exactly (three steps). Together with perplexity agreeing to 0.045% (5.169092 against 5.171412), the port is accepted as computing the same function up to BF16 rounding. Log-probabilities are not quantized to the logit step, so no limit is stated in steps for them; the log-probability limits above stay as measured.

Qualification:

- Linear oracle suites (projection, split, add, pair, swiglu, topk) pass for every format Turing runs (BF16, Q4, Q5, Q6, Q8). The first port attempt failed them (Q8: 592 cases) because a bulk edit had also converted weight-code and scale copies to FP16 activation staging, and a perplexity of 6.66 against 5.17 was caught by the full-model check, not by greedy output on a short prompt.
- `ctest` of the Ops configuration (80 tests): all pass. `ninfer_device_sync_empty_test` is now registered only outside Windows: CTest on Windows drops an environment entry with an empty value, so the test saw the variable unset and failed (it failed the same way on the previous base, before the Turing tuning). Launching `ninfer_device_test --invalid-sync` with a genuinely empty `NINFER_CUDA_SYNC` through `subprocess` returns 0, so the engine rejects the empty value on Windows too; Linux and WSL2 run the registered test. FP8, NVFP4 and K8V4 cases are reported as skipped through the suite (a case whose route is a Turing stub counts as skipped, not failed).
- Product-layer tests (CLI and serve options, context fitting, KV capacity, frontend, vision workspace, request log, schemas, HTTP transport, loading, state image): all 25 registered tests pass (the real-model ones skip without their artifact variable) once the Qwen4Exp state test, which needs the sparse QSA attention that Turing does not build, is excluded on this target.
- The new Windows read queue (I/O completion port) passes the shared behavioral test; the io_uring fault-injection part of that test stays Linux-only. The NF4 exact-table oracle passes once its binary stdio is set to binary mode.

Not carried over, and why:

- CUDA Graph memory allowances per profile (upstream now calibrates graph memory at startup), the cooperative GDN gating projection (removed upstream), and the device-memory typed-error classes (upstream has its own; the fork's `DeviceMemoryError` remains only as the type the context-fitting retry catches).
- The pinned-host embedding is carried over as a platform capability (`NINFER_HOST_MAPPED_EMBEDDING`, registered host memory read in place); the earlier fallback to a device copy when pinning fails is not, so a host short of RAM fails at startup with `HostMemoryError` instead of loading with the embedding on the device.
- Qwen4Exp, Kimi delta attention, sparse QSA attention, the hyper-connection mix, block-scaled MoE and CPU expert offload are not built for Turing.

Linux, packaging and Docker (verified on the same tree):

- **Linux build and `ctest` (WSL2 Ubuntu 24.04, CUDA 12.9, GCC 13, `turing-dev`, benchmarks off):** 145 tests; all pass or skip (28 skipped: FP8/NVFP4/K8V4 cases, and real-model tests that need their artifact). The seven Python-driven tests (artifact writer interop, chat templates, the two schema oracles, three PLE disk tests) pass with a Python that has `jinja2` and `jsonschema`; the distro's default Python 3.11 could not run them because its standard library file `platform.py` is corrupt (see below). This ran after fixes that only the Linux build exposed: six inline-assembly strings in `pipeline.cuh` had lost their `\n` escapes (the Windows front end accepted the broken literals, GCC rejects them), the CUDA 12 signature of the batch-copy interposer in `tests/expert_cache_cuda_errors.cpp`, and a race in `ninfer_context_store_test`, which seeds device memory with a legacy-stream `cudaMemcpy` and then migrates it on non-blocking streams (it failed 4 of 5 runs on both platforms and passes 10 of 10 with a `cudaDeviceSynchronize` after the seeding; the test is upstream's and the race is in its setup). `ninfer_state_store_test` failed once on its first Linux run and passed in every later run (10 of 10 on Windows, and on Linux); that single failure was not reproduced or explained.
- **Windows package:** `scripts\package.ps1` on the new layout writes a 1.6 GB zip (executables plus the vcpkg DLLs and `serve.ps1`) whose `ninfer.exe` starts from an empty directory.
- **Docker:** `docker build` of the Dockerfile (CUDA 13.1.2 devel image, `NINFER_TARGET=turing`) compiles the Linux Release binaries in about 10 minutes. Running the first image failed because upstream links shared libraries through a relative RPATH and the runtime stage copied only the executables; the runtime stage now ships `apps/` and `lib/` together. The next failure was "direct read ended before the required payload": the model is bind-mounted from the Windows drive and a direct (`O_DIRECT`) read there may return fewer bytes than requested, which upstream's `read_direct` treated as fatal. It now continues until the request is satisfied or the file ends. With both fixes `docker run --gpus all -v <models>:/models -p 18095:8080` loads the model in 2 min 4 s (153 MiB/s through the mount; 15.4 GiB of weights on the device, the 1.26 GiB embedding pinned, 3.15 GiB of host context pinned) and answers a chat completion correctly (177 prompt tokens, 13 output tokens, TTFT 724 ms, prefill 248 tok/s, decode 44 tok/s with MTP over those 13 tokens; the pre-port image measured 688 ms, 252 and 49).
- **The WSL2 Ubuntu distro used for the Linux run has filesystem damage:** its ext4 root remounted read-only several times during builds, `/usr/include/linux/falloc.h` and `futex.h` return "Bad message" (EBADMSG) on access, and the standard-library files `platform.py` (Python 3.11) and a `numpy` file fail to read. The build and the tests were run from the Windows drive, with the kernel UAPI headers taken from a conda package into an include directory outside the distro and the temporary directory on the Windows drive. The distro needs `e2fsck` or a rebuild; nothing in it was repaired or deleted.

### Verification of `master` (2026-10-19 and 2026-10-20)

Everything here was run on the sources of `master` after the port (`7923b62f`, plus later commits that only touch documentation and scripts), on one RTX 2080 Ti 22 GB with Qwen3.8-27B `groupwise-int`.

- **Windows:** release build with `scriptsuild.ps1`, development build with `scriptsuild.ps1 -Dev`; `ctest`: 138 tests, 0 failed, 25 skipped; protocol-B perplexity 4.796799; benchmarks in [Current performance](#current-performance).
- **Linux / WSL2** (Ubuntu 24.04, GCC 13.3, CUDA 12.9.86, `cmake --preset turing-dev` at `4f4e6059`, built from the Windows drive): `ctest` 145 tests, all pass, 28 skipped (FP8/NVFP4/K8V4 cases and real-model tests). The four product executables built and `ninfer` answered a real prompt from the 27B artifact (decode 40.8 tok/s with MTP, prefill 303.6 tok/s). Nineteen benchmark targets did not compile in that distro, every one for the same reason: its CUDA install has no `cuda_profiler_api.h`; they build on Windows. The distro's filesystem damage described above is unchanged, so the kernel UAPI headers still come from `linux-libc-dev` (`apt-get download`, `dpkg -x`) on `CPATH`.
- **Docker:** `git archive HEAD | docker build` built the image; `docker run --gpus all -v <models>:/models -p 18120:8080` started the server with the default configuration (15.4 GiB of weights, 3.15 GiB host context pinned in 4.4 s, engine ready in 2 min 14 s through the bind mount) and a chat request returned HTTP 200 (174 prompt and 88 output tokens, TTFT 729 ms, decode 40.1 tok/s, 60 of 87 drafted tokens accepted).
- **Windows package:** the first package check failed: the executables need `cudart64_12.dll`, because the build links the shared CUDA runtime (upstream's CMake setting), and the zip did not ship it. `scripts\package.ps1` now copies it from the CUDA Toolkit. The zip, extracted into an empty directory and started with `serve.ps1` under a `PATH` of only `System32` and Windows PowerShell, answered a chat request with HTTP 200 (TTFT 670 ms). The download branch of `serve.ps1` (no model present) was not run.

## Maintainer notes

Facts and decisions that outlive any one change:

- **Where Turing code lives.** The build targets Turing only. Behaviour that exists only for Turing sits behind `NINFER_SM75`, and the unsupported Blackwell-only routes are throwing definitions in `src/ops/unsupported/turing_unsupported.cpp`.
- **Only sm_75 is supported.** The build requires `CMAKE_CUDA_ARCHITECTURES=75` (`cmake/NinferPlatform.cmake`). An Ampere (sm_86) build is not supported and was never tested: the tree contains TMA kernels (`cp.async.bulk.tensor` in `src/ops/common/cuda/pipeline.cuh` and others), which need sm_90, and the Turing builds avoid them through `NINFER_SM75` guards. Supporting another architecture means adding a no-TMA path under its own guard and a preset.
- **FP16 range.** On Turing every BF16 operand becomes FP16 before a tensor-core GEMM at 5 or more tokens, so values beyond 65504 saturate. Measured maxima (see the FP16 range section) leave at least 4x headroom to 16384 and no pre-scaling exists. If a new model or workload reaches 16384 at some GEMM input, add an exact power-of-two pre-scale while staging and fold it back into the FP32 epilogue, then require perplexity not to get worse.
- **Decode depth and MTP.** MTP depth is fixed at startup because the draft count shapes the captured CUDA graphs and round buffers (51 references), so an adaptive depth is a larger change than tuning; depth 3 is the default (best for code, 1 to 2 for prose).
- **Output head.** A keep-list output head would make a few percent of tokens impossible to generate (see "Not done: a keep-list output head" above); a Q6 head saves memory but not time.
- **Multi-GPU.** Pipeline parallelism across two GPUs stays a design note ([multi-gpu.md](multi-gpu.md)) until there is a concrete second card.
- **Small inefficiencies left alone** (re-verify before acting): the BF16-KV attention kernel (`src/ops/softmax_attention/kernels/causal/bf16/grouped_mma.cuh`) converts the query fragment on every key tile; Turing's `cp_async` ignores `Cache::cg`, so streamed weights could use `ld.global.cg` to avoid filling L1; the SIMT TF32 emulation in `src/ops/common/cuda/mma.cuh` costs 32 shuffles per call in the GDN chunk recurrence but is under 5% of prefill.
- **Lessons from the port onto upstream `dev`.** (1) A bulk regex over kernel files once converted weight and scale copies to FP16 activation staging; check each replacement site against the previous version with a whitespace-normalized diff. (2) Upstream moves routes between ops (the GDN input projection now runs through the split-projection shape), so profile a prefill kernel by kernel against the old build. (3) A missing pinned embedding made the default 64k context fail on a 22 GB card. (4) Verify on Linux and in Docker as well: the Windows front end accepted corrupted inline-assembly strings that GCC rejects, and Docker needed its shared libraries and short-read handling for bind mounts. (5) An intermittent test failure can be a real bug (the Windows read queue counted a read complete before `poll` delivered it); run a flaky test a hundred times and dump the data before blaming the test.
