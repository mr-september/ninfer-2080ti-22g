# Turing (sm_75) port

This fork builds NInfer for NVIDIA Turing (RTX 2080 Ti 22 GB, `CMAKE_CUDA_ARCHITECTURES=75`) on top of the upstream engine. The engine, the Qwen3.5-family runtime, frontend, serving and scheduling are unchanged; only the CUDA layer differs. This page records what differs and why, so the next upstream sync can be redone deliberately instead of by merge.

## Supported surface

| Area | sm_75 |
|---|---|
| Artifacts | `groupwise-int` (Q4/Q5/Q6/Q8 + BF16 tensors): Qwen3.8-27B is the only tested model. Qwen3.6-27B and Qwen3.6-35B-A3B are untested (op-level kernels only; the 21.2 GiB 35B-A3B artifact does not fit 22 GB) |
| NVFP4 / FP8 weights | unsupported; the routes link throwing definitions |
| KV cache | `bf16`, `int8`; `fp8`, `nvfp4`, `k8v4` unsupported |
| Speculation | MTP (tested); DFlash is untested and tied to 35B-A3B |
| Kimi delta attention | not built into the sm_75 test set (needs more than the 64 KiB CTA limit) |

Unsupported formats are not stubbed one symbol at a time inside upstream sources. `src/ops/CMakeLists.txt` filters the FP8/NVFP4/K8V4 sources out of the upstream manifests for every target other than `120a`, and `src/ops/unsupported/sm120_formats_unsupported.cpp` defines the declarations the shared wrappers still name. Upstream manifests therefore stay byte-identical.

## Hardware differences and how each is handled

| Turing limit | Handling |
|---|---|
| no `cp.async` | `core/arch_traits.cuh`, `ops/common/memory.cuh`: synchronous copies, `commit`/`wait` are no-ops |
| no PDL | `core/pdl.cuh`: plain launches, `griddepcontrol` calls removed |
| no BF16 / TF32 tensor core | `mma_bf16`: BF16 -> FP16 operands on the FP16 tensor core (exact in range, saturating beyond ±65504); `mma_bf16_simt` is the exact FP32 reference; `mma_tf32`: SIMT emulation |
| FP16 MMA is k8, INT8 MMA is m8n8k16 | `mma_f16`, `mma_s8`: decomposed into the native shapes (A registers are `{row g K-lo, row g+8 K-lo, row g K-hi, row g+8 K-hi}`) |
| 64 KiB shared memory per CTA | `core::kMaxCtaSharedBytes`; schedules fit by single-buffering, halving K warps (Q8), smaller tiles (attention) |
| no `redux`, no `cvt.rn.bf16x2.f32` | `warp_reduce_max_u32` butterfly; `pack_bf16x2` via intrinsics |
| 68 SMs | launch plans already derive from `DeviceExecutionView::multiprocessor_count`; cooperative residency is queried with the occupancy calculator |

`tests/ops/test_mma_shim.cu` checks every tensor-core wrapper against naive math in the PTX fragment layouts; run it first when touching `ops/common/mma.cuh`.

## Prefill tiles and token embedding (measured on one RTX 2080 Ti 22 GB)

**Prefill tiles.** About 95% of prefill time is the dequantizing GEMMs. Upstream sizes their tiles for the RTX 5090 (two `cp.async` stages, about 46 KB of shared memory), which on Turing leaves one 4-warp CTA per SM: profiled at 9.8% tensor-pipe and 11% issue-slot utilization, stalled on global loads with no `cp.async` to hide them. Single-stage tiles use about 27 KB, so two CTAs share each SM and one tile's MMA overlaps the other's decode and loads. Applied under `NINFER_SM75` only (the other architectures keep upstream's tables): Q5 plain and residual-add (`linear_add`) projections, Q4 plain projections, the folded Q4 gate/up SwiGLU (`MmaPairC128W8S1B2`), and the grouped GDN and attention input projections. Routes were chosen per shape by timing every instance across 128 to 2048 tokens.

| Operation (1024 tokens) | Before | After |
|---|---|---|
| Q5 MLP down 5120x17408 | 19.4 ms | 8.9 ms |
| Q5 down with residual add | 18.4 ms | 9.8 ms |
| Q4 gate/up 34816x5120, plain | 32.6 ms | 16.2 ms |
| Q4 gate/up folded with SwiGLU | 21.7 ms | 13.2 ms |
| GDN grouped input projection | 12.1 ms | 6.8 ms |

End to end, native Windows, greedy, MTP off (prefill tokens/s): 122-token prompt 201 to 247, 8k prompt 253 to 479, 64k prompt about 205 to 372. Kernels at 5 to 8 tokens (MTP depths 4 and 5) were already within about 10% of the best candidate, so draft depth 3 stays the default: a 5-token step costs about 1.4x a 4-token step because the cheap SIMT kernels stop at 4 tokens.

**Token embedding.** The embedding is gathered one row per token, so it lives in pinned host memory that kernels read over PCIe (`Residency::PinnedHost`): 1.26 GiB less device memory, output byte-identical, no measurable decode change. If page-locked memory is unavailable (a host short of free RAM), the binder falls back to device residency at bind time. Effect: a 64k-token prompt now fits with MTP (it did not before); the largest context that starts is about 97k to 99k with MTP (it moves with how much VRAM the desktop and other programs hold) and between 110k and 125k without it.

**Output head format.** Re-encoding `text/output_head` from Q8 to Q6 (`python -m tools.artifact.requantize_tensor SRC OUT --object text/output_head --format q6_g64_fp16`, about 6 minutes) is nearly free in quality and saves memory but not time. Fixed-corpus perplexity, quick set, 4096 context, INT8 KV: overall 4.7944 to 4.7950 (+0.014%); Chinese +0.023%, long-form English +0.035%, English reference +0.001%, code -0.002%. The artifact is 341 MiB smaller and planned device memory drops 0.4 GiB, but decode is not faster (23.5 against 23.8 tok/s): the saved bandwidth is spent unpacking Q6.

**Not done: a keep-list output head.** Dropping rarely used rows from the main output head (and gathering logits back to the full vocabulary) would save about 0.6 GiB and a few percent of decode bandwidth. The repository's own frequency ranking shows the cost: its top 131,072 tokens (the proposal-head list) cover 99.1 to 99.4% of real tokens overall, 99.6% for English, 99.2 to 99.5% for Chinese and code, and only 97.5 to 98.2% for other languages, so up to a few percent of tokens could never be generated. With the embedding already on the host and a Q6 head available, that trade was not taken.

A bottom-up variant (keep every special, template and byte token and every real word; cut only tokens with no legitimate use) was audited on the Qwen3.8 tokenizer and is not worth building: of 248,320 head rows only 243 are padding, and the junk-shaped tokens (long symbol runs, control characters, replacement characters, mojibake, hex runs, unused code points) number about 450 rows, 1.7 MiB at Q6. One head row costs 4,000 bytes at Q6, so 0.1 GiB needs about 27,000 rows, and those would be real words: a large share of the tokens never chosen in the ranking corpus are ordinary Korean, Thai, Vietnamese and code-identifier words. Frequency alone cannot separate artifacts from rare words.

**MTP depth and the proposal head.** `--lm-head-draft` is free speed: +3% decode on average (+1.7%, +2.6%, +4.7% in paired runs), identical acceptance and byte-identical output, so the README examples use it. Depth 3 is best for code (decode tok/s at depth 1 to 5: 31.8, 37.0, 39.3, 31.4, 32.9; acceptance 92% down to 65%). For prose depth 1 or 2 is better (28.3, 28.0, 26.4, 21.4, 20.0; acceptance 69% down to 31%). The depth is fixed at startup because the draft count shapes the captured CUDA graphs and round buffers (51 references), so an adaptive depth would be a larger change than a tuning step.

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j
```

CUDA 12.8+ is enough (no 13.x-only features are used). The Q8 split-K translation units are slow to compile for sm_75 (minutes each); a cold build of the release preset took 10.6 minutes on 16 logical cores (native Windows, warm vcpkg cache). `ninja -t deps | grep '#deps 0,'` finds objects whose header dependencies were lost by an interrupted build; delete those objects, otherwise header edits silently do not rebuild them.

## Native Windows build

The same sources build with MSVC 14.4 and `nvcc` 12.9 (`cmake --preset windows`; see the [README](../README.md#build) for prerequisites). Linux/WSL2 builds are unchanged; nothing selects between them at runtime, the environment you build in is the mode. What differs:

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

Same machine, same 2080 Ti 22 GB, Qwen3.8-27B `groupwise-int`, INT8 KV, greedy, MTP off, runs interleaved and gated on an otherwise idle GPU (another program intermittently used this card, which invalidated some runs; those were discarded). Each environment used its own build and its own copy of the artifact.

| Prompt | Metric | Native Windows | WSL2 |
|---|---|---|---|
| 122 tokens, 512 generated | decode | 24.4, 24.5, 24.0 tok/s | 24.1, 24.0 tok/s |
| 122 tokens | prefill | 201.8, 204.9, 196.6 tok/s | 215.3, 213.5 tok/s |
| ~8k tokens | prefill | 253.0, 236.0, 258.9 tok/s | 259.5, 258.6, 262.0 tok/s |

Decode is at parity. Prefill is about 6% lower natively on a short prompt and about 2.5% on an 8k prompt (medians), consistent with a small fixed per-launch cost under WDDM; the kernels are identical. Absolute figures vary by around 15% between sessions with GPU clocks and background load, so only same-session interleaved comparisons mean anything. Native vision output (image, image+video) and the OpenAI/Anthropic server behavior match WSL2; the CUDA fault-injection part of the artifact-materialization test is Linux-only (it needs GNU `ld --wrap`).

## Default chat template

The former `--chat-style sharp-v22.1` option is now the vendored Sharp v22.5.0 Jinja template `tools/chat_templates/qwen3_8_sharp_v22_5.jinja`, which is embedded in `ninfer` and `ninfer-serve` at build time and used by default. `--artifact-chat-template` selects the artifact's own template and `--chat-template FILE` another file:

```bash
./build/apps/ninfer model.ninfer --artifact-chat-template --prompt "..."
```

## Measured on one RTX 2080 Ti 22 GB

Qwen3.8-27B `groupwise-int`, INT8 KV, greedy, long warmups (the GPU clock ramps).

| Metric | Result |
|---|---|
| decode, no MTP | 24-29 tok/s |
| decode, MTP3 | 38-44 tok/s |
| prefill (end to end), 122 / 8k / 64k-token prompts | 247 / 479 / 372 tok/s |
| prefill GEMM, Q4 34816x5120, T=512/2048 | 12.4 / 12.6 TFLOPS |
| prefill GEMM, Q5 5120x6144, T=512/2048 | 11.2 / 11.5 TFLOPS |
| prefill GEMM, Q8 5120x6144, T=512/2048 | 18.1 / 19.1 TFLOPS |

Single-prompt greedy comparisons are noisy for MTP (near-tie flips change the acceptance rate); compare decode with MTP off. The 35B-A3B and DFlash routes build and pass the op tests but were not run end to end here (no local artifact).

## Roadmap baseline (2026-10-17, commit `4bedd2e5`)

Reference numbers for the performance items in `docs/plans/fork-roadmap.md`, measured with the product `ninfer_bench` (Windows, MSVC 19.44, CUDA 12.9, driver 610.88, RTX 2080 Ti 22 GB, Qwen3.8-27B `groupwise-int`, INT8 KV, CUDA Graph decode, the repository's token-id corpus). The desktop was running (about 2.6 GiB of VRAM held by other programs, 0% GPU utilization, idle SM clock 1350 MHz before each run); each run starts with the warmup repetitions shown.

| Run | Command (after `--weights M`) | Result (mean, sample sd over repetitions) |
|---|---|---|
| `bench-short` | `-p 512 -n 256 -r 3 --warmup 1 --kv-dtype int8 --max-ctx 16384` | pp512 494.88 tok/s (sd 0.39); tg256 26.45 tok/s (sd 0.04) |
| `bench-long` | `-p 8192,65536 -n 32 -r 2 --warmup 1 --kv-dtype int8 --max-ctx 70000` | pp65536 394.51 tok/s (sd 0.08); tg32 26.20 tok/s (sd 0.00); pp8192: see below |
| pp8192 rerun | `-p 8192 -n 32 -r 4 --warmup 2 --kv-dtype int8 --max-ctx 70000` | pp8192 513.73 tok/s (sd 0.18); tg32 26.35 tok/s (sd 0.07) |
| `bench-mtp` | `bench-short` plus `--spec mtp --draft-tokens 3 --lm-head-draft` | pp512 492.88 tok/s (sd 0.18); tg256 29.76 tok/s (sd 0.04), acceptance 0.371 |

Exception: in the `bench-long` run the two measured pp8192 repetitions were 26.0 and 512.8 tok/s. The first was a transient of unknown cause (not reproduced in the rerun, where all four repetitions were within 0.4 tok/s), so the rerun is the pp8192 baseline and `bench-long` needs at least two warmups to be trusted at 8k. The benchmark corpus is random token ids, so MTP acceptance (0.371) and the MTP decode figure are far below the 38-44 tok/s seen on real text and are only comparable with each other.

## FP16 activation range (roadmap C1)

On Turing every BF16 operand is converted to FP16 before the tensor core, and values beyond +-65504 saturate. The debug build option `NINFER_SM75_RANGE_PROBE` (`cmake --preset windows -B build-probe -DNINFER_SM75_RANGE_PROBE=ON`, sm_75 only) records the largest magnitude that reaches that conversion in any kernel, and how many elements exceed 16384, 32768 and 65504; `ninfer`, `ninfer-perplexity` and the benchmarks print one `range probe | ...` line when the Engine shuts down. It is a global maximum over all call sites and kernels, which bounds every individual site.

Measured on Qwen3.8-27B `groupwise-int`, INT8 KV, 2026-10-17:

| Workload | max abs value | elements > 16384 | clamped (> 65504) |
|---|---|---|---|
| `ppl-quick` (261,223 tokens: English reference, English long-form, Chinese, code; 4096 context) | 516.0 | 0 | 0 |
| one 65,536-token window of long-form English (`--text pg19/00.txt --context 65536`) | 438.0 | 0 | 0 |
| short chat prompt, 64 greedy tokens | 304.0 | 0 | 0 |

Every maximum is below 16384, so there is at least 4x headroom to the threshold the roadmap set (and more than 100x to the FP16 limit), nothing was clamped, and no pre-scaling is added. This covers these corpora and prompts; the SwiGLU output feeding the Q5 down projection, the site the roadmap singled out, is included in the global maximum. Decode at 1 to 4 tokens uses FP32 SIMT, so the two paths agree on these inputs. The ppl-quick result of the probe build (overall 4.794356) equals the production build's.

## Route sweep tool (roadmap P1)

`ninfer_turing_route_sweep` (`bench/ops/turing_route_sweep.cu`, built only for sm_75 with `NINFER_BUILD_BENCHMARKS=ON`) forces every Q4/Q5/Q6/Q8 launcher of one shape, drops the ones that throw or whose output differs from the production route's by more than 5% of the output RMS, times the rest with cold-cache CUDA Graph replay, and prints the ranking per token count plus a `TokenRoute` table to paste into the shape selector. The candidate list comes from the `q*_launch.h` headers at configure time. It includes private `ops/linear` headers, so it is a fork-only developer tool and not part of any product or CI target.

```bash
./build-windows/bench/ninfer_turing_route_sweep --qtype q5 --n 5120 --k 17408 --t 1,2,3,4,1024 --top 8
```

First result (Q5 `n5120_k17408`, the MLP down projection, T=1024, three runs within 0.2%): the production route `launch_q5_a16_mma_r64_t128_a1` takes 8.41 ms (21.7 TFLOPS); `launch_q5_a16_mma_r128_t64_s1_a1` takes 8.04 ms (22.7 TFLOPS), 4.6% faster. Of the 32 Q5 launchers, 22 were legal at that shape and 10 were rejected.

## Decode kernel breakdown and the first P2 change (roadmap P2)

One decode step is 37.6 ms (Nsight Systems, CUDA Graph nodes traced, MTP off, 103 steps): kernel time equals step time. The card's sustained read ceiling, measured with `tools/hbm_bandwidth_probe.cu` built for sm_75, is 584.2 GB/s (94.8% of the 616 GB/s peak). Per step before the change:

| Kernel (op) | share | per call | bytes read | rate | of 584 GB/s |
|---|---|---|---|---|---|
| Q4 gate/up GEMV (`linear_swiglu`, 34816x5120) | 31.2% | 183 us | 94.7 MB | 517 GB/s | 88% |
| Q5 direct SIMT, K=17408 (MLP down, `linear_add`) | 22.0% | 130 us | 58.5 MB | 451 GB/s | 77% |
| Q5 staged GEMV, value+z rows (GDN input, T=1) | 15.4% | 124 us | 41.3 MB | 332 GB/s | 57% |
| Q5 direct SIMT, K=6144 (output projections, `linear_add`) | 8.2% | 48 us | 20.6 MB | 428 GB/s | 73% |
| Q8 output head (plain `linear`) | 6.4% | 2425 us | 1351 MB | 557 GB/s | 95% |

The MLP down and output projections run through `linear_add`, which has its own route tables (`src/ops/linear_add/q5/q5_linear_add_plan.cpp`); the plain-linear shape files under `src/ops/linear/*/shapes/` serve only the output head and the vision tower on the text path, so tuning them does not change decode.

**Change.** On Turing the GDN input projection's T=1 Q5 launch (`launch_q5_t1` in `q4_q5_gdn_input_conv_snapshot.cu`) now uses the direct SIMT kernel that the T>=2 path already used, instead of the staged shared-memory GEMV. Turing has no `cp.async`, so the staged GEMV cannot overlap its loads. That kernel went from 124.4 us to 92.5 us (446 GB/s, 76%).

Measured interleaved in one session (the GPU ran about 3% slower than in the morning baseline, so only same-session pairs are compared), `ninfer_bench`, INT8 KV, `-n 256 -r 3 --warmup 1`:

| Run | before (tok/s) | after (tok/s) |
|---|---|---|
| MTP off, round 1 / 2 / 3 | 24.92 (sd 1.03), 26.01, 25.84 | 26.68, 26.68, 26.49 |
| MTP off, medians | 25.84 | 26.68 (+3.3%) |
| MTP 3 draft tokens, round 1 / 2 | 29.61, 29.46 | 29.65, 29.59 (acceptance 0.371 both) |

Paired by round the MTP-off gain is +7.1% (the first baseline run was noisy), +2.6% and +2.5%. `greedy-identity` holds: `long_decode_aime26_01.json`, `--greedy --max-new 256 --spec none`, identical output text before and after (and identical across two baseline runs, so decoding is deterministic).

Remaining decode headroom, by the table above: the K=17408 and K=6144 `linear_add` Q5 kernels (77% and 73%), then the Q4 GEMV (88%).
