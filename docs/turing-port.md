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

**Q5 `linear_add` direct-kernel schedule sweep (P2 step 4, no change made).** `Q5A16DirectSimtSchedule<1, T, W, G, MinBlocks, K, exact>` in `q5_linear_add_gemm_simt.cu` (W warps per row, G groups per warp tile, K tile per iteration W*G*64) was swept with `ninfer_q5_linear_add_bench` (graph replay, warmup 200, repeat 200) over (W,G,MinBlocks) = (2,8,16) current, (2,4,16), (4,4,10), (4,8,8), (2,12,16), (2,8,8). For K=17408 the current schedule is best or tied at every T (T=1 about 115 us, 510 GB/s in isolation); the only gain is (4,4,10) at T=1, 2.5%, and it is far worse at T>=2 (T=3: 305 us against 184 us). For K=6144 all variants are within the run-to-run spread (the T=1 median alternates between about 43 us and 49 us across identical runs). Larger W, G or MinBlocks values and W=1 are slower. No schedule change is justified. (An earlier sweep with short warmups was discarded: its dev-shell invocations accumulated PATH until builds failed silently.)

**Decode is power-capped.** While decoding, the card sits at its 250 W cap (`clocks_throttle_reasons.active` = software power cap, 0x4): measured over 75 busy samples, SM clock 1455-1635 MHz (mean 1511), power 237-252 W (mean 246), memory clock fixed at 7000 MHz, temperature at most 44 C. Short micro-benchmarks draw less power and run faster, which is why the in-situ kernels run slower than the same kernels in isolation (the K=17408 `linear_add` kernel is about 115 us alone and 130-135 us inside decode). Gains from kernels that issue fewer ALU instructions per weight byte may therefore be larger in decode than in isolation, but load-width or occupancy changes alone are not expected to close the gap.

## INT8-KV attention tile 64x32 (roadmap P3)

`Int8KvTiledInstance` (`src/ops/softmax_attention/dense/causal_cache/int8/instances.h`) on Turing changes from `<32, 32, 128>` to `<64, 32, 128>`. The 64x32 tile stages 55,296 bytes of shared memory (54 KiB, under the 64 KiB limit; 64x64 would need 92 KiB), uses 16 warps per CTA instead of 8, and reuses each Q tile over twice as many query rows.

Measured interleaved in one session (`ninfer_bench`, INT8 KV, MTP off, `--max-ctx 70000`; 8k: 4 repetitions after 2 warmups, 64k: 2 repetitions after 1 warmup; two rounds of before then after):

| Prompt | before (tok/s) | after (tok/s) | change |
|---|---|---|---|
| 8192, round 1 | 507.17 (sd 1.10) | 513.44 (sd 2.38) | +1.2% |
| 8192, round 2 | 496.74 (sd 15.68; one repetition at 473.4) | 517.46 (sd 1.37) | +4.2% (+2.4% against the other three repetitions) |
| 65536, round 1 | 390.74 (sd 0.33) | 427.04 (sd 2.76) | +9.3% |
| 65536, round 2 | 389.47 (sd 0.57) | 424.13 (sd 1.46) | +8.9% |

Checks: the attention, KV-cache and sliding-window tests pass (`ninfer_softmax_attention_test`, `ninfer_kv_cache_test`, `ninfer_kv_cache_append_test`, `ninfer_sliding_window_attention_test`, `ninfer_causal_conv1d_silu_test`). `ppl-quick` (INT8 KV, 4096 context) with the new tile against the baseline 4.794356: overall 4.794641 (+0.006%); Chinese 6.750531 against 6.746396 (+0.061%), long-form English 7.179776 against 7.178846 (+0.013%), English reference 6.458972 against 6.461225 (-0.035%), code 1.673260 against 1.673525 (-0.016%). The tile shape changes the order in which key blocks are merged, so bit-identical output is not expected; the overall change is within the 0.05% limit and the Chinese domain alone is slightly above it, with the other domains moving the other way.

## Converting sliced-K activations once (roadmap P4): tried, regressed, not kept

The sliced-K kernels (T of about 5 to 160) stage BF16 activations raw and convert each B fragment to FP16 inside every MMA (`mma_integer_weights`), once per row tile. The change tried for Q5 (`q5_a16_sliced_k_mma.cuh`): stage the activations with `stage_activation_operand_zfill`, which converts BF16 to FP16 once while writing shared memory, and call `mma_operands` (FP16 MMA on Turing) in the loop. The Q5 linear, `linear_add`, `linear_pair`, `attn_input_proj` and GDN-input op tests all passed with it.

Measured interleaved in one session (`ninfer_bench`, INT8 KV, `--spec mtp --draft-tokens 5 --lm-head-draft`, `-p 512 -n 256 -r 3 --warmup 1`, three rounds of before then after):

| | before | after |
|---|---|---|
| pp512 (tok/s) | 497.0, 496.8, 496.6 | 494.7, 494.6, 497.0 |
| MTP-5 tg256 (tok/s) | 22.17, 22.18, 22.19 | 21.21, 21.22, 21.07 (-4.4%) |
| acceptance | 0.231 | 0.231 |

The decode step at MTP depth 5 (T=6) is consistently 4.4% slower, so the roadmap's gate ("`bench-mtp` with `--draft-tokens 5` not worse") fails and the change was reverted. A likely cause is that Turing's emulated `cp.async` is an ordinary load followed by a store: converting in between lengthens that dependent chain once per staged vector, while for the 16-row blocks used at these token counts the per-MMA conversions it removes were not more numerous than the staged conversions it adds. The other kernels named in the roadmap (Q4, Q6, Q8, grouped Q8, `linear_topk`) were not changed.

## Register-prefetch pipelining of the prefill GEMMs (roadmap P5)

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

Checks: `ninfer_linear_q5_a16_test` and `ninfer_linear_add_q5_a16_test` pass; greedy text of `long_decode_aime26_01.json` (341-token prompt, which runs through the changed route) is byte-identical before and after (only the load order changed). This is the Q5 family only (37.8% of prefill); the roadmap's whole-item gate (8k prefill +15%) needs the Q4 SwiGLU and grouped-projection kernels as well.

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

**P5 total.** Each step was measured against the build before it, in its own session; the sessions drift by a few percent, so the steps are not additive on paper. Whole-item numbers in the first session's terms (before any P5 change: pp512 497.8 / 494.4, pp8192 523.3 / 519.9, pp65536 435.4 / 432.5): after all three kernel families, a later session measured pp8192 612.40 and 609.18 against a same-session predecessor, so the 8k prefill is about 17% above the pre-P5 build (523.3 to 612.4 and 519.9 to 609.2: +17.0% and +17.2%), which meets the roadmap gate (8k prefill +15%). Remaining kernels in the roadmap's list (Q4 and Q6 and Q8 plain MMA, `linear_pair`, the attention and GDN gating projections) were not changed: none of them is more than a few percent of prefill time.

## Output head Q6 against Q8, re-measured (roadmap P6)

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

`ctest --preset windows-dev` after the roadmap's Phase 0 to 2 changes: 131 tests, 5 fail, 13 real-model tests report "Skipped" (they need artifacts or environment variables that are not set up), the rest pass. The five failures:

- `ninfer_chat_templates_test`, `ninfer_json_schema_oracle_test`, `ninfer_qwen3_5_tool_schema_oracle_test`: the Python oracles pipe UTF-8 text to a child process through the Windows ANSI code page (`UnicodeEncodeError: 'charmap' codec can't encode characters`). They pass with `PYTHONUTF8=1` in the environment.
- `ninfer_device_sync_empty_test` ("invalid sync setting did not fail before CUDA initialization") and `ninfer_gdn_gating_proj_test` (`qwen3_6_27b T=4097 g: reduction criterion failed at index 122 actual=-2.20374 reference=-2.20374`): both fail identically at the fork's commit before this roadmap (`a97bbc8d`, built in a separate worktree), so neither comes from the Phase 0 to 2 changes. They are not investigated here.

## Fusing small decode projections (roadmap P7): measured ceiling, not pursued

Per decode step (Nsight Systems, CUDA Graph nodes traced, MTP off, 103 steps, after the P2 change): 711 kernel launches and 37.23 ms of kernel time. The kernels whose average duration is under 20 us (norms, RoPE, GDN gating and its small recurrent update, SiLU-mul, the causal-convolution and attention-merge helpers: 18 kinds) account for 390 launches and 2.00 ms per step, 5.4% of kernel time. None of them is a projection: the projections are all among the 321 launches that take 20 us or more (35.2 ms), and the small ones the roadmap names are already fused (`attn_input_proj` carries Q, gate, K and V; `gdn_input_proj` carries the Q4 query/key and Q5 value/z sides; the gating projection is fused with the GDN norm). The gap between the kernel time and the step time (37.2 ms against 1/26.4 s = 37.9 ms) is about 0.7 ms, so the missing PDL costs under 2% of a step. Even a fusion that removed every small kernel would be bounded by about 7%, and a realistic one (folding norms and gating into their neighbours) by 1 to 2%, at the cost of new kernels whose numerics must be requalified. Since the item is optional and its ceiling is that small next to the changes already made, it was not implemented.

## Roadmap Phase 2 result (2026-10-18)

`ninfer_bench`, INT8 KV, CUDA Graph decode, native Windows, RTX 2080 Ti, Qwen3.8-27B `groupwise-int`. The comparison is the build after the P2 decode change (but before P3 and P5) against the current build, interleaved in one session, two rounds each:

| Metric | before P3-P5 (round 1 / 2) | current (round 1 / 2) | change |
|---|---|---|---|
| pp512, tok/s | 498.3 / 498.5 | 613.9 / 613.1 | +23.2% / +23.0% |
| pp8192, tok/s | 516.4 / 516.2 | 611.9 / 612.0 | +18.5% / +18.6% |
| pp65536, tok/s | 397.0 / 396.9 | 492.8 / 493.1 | +24.1% / +24.2% |
| tg256, MTP off, tok/s | 27.08 / 26.99 | 27.02 / 27.07 | unchanged (no decode kernel changed after P2) |
| tg256, MTP 3 draft tokens, tok/s | 29.96 / 29.91 | 29.97 / 29.95 | unchanged (acceptance 0.371 both) |

`ppl-quick` (INT8 KV, 4096 context): overall 4.794641 on the current build, identical in every digit to the build after P3; the P5 kernels change only the order of loads. The scoring rate of that run went from 247.5 to 305.9 tok/s.

Against the first measurements of the roadmap's morning baseline (different session, so the session-to-session drift of a few percent applies): pp512 494.9, pp8192 513.7 (rerun), pp65536 394.5, tg256 26.45, MTP-3 tg256 29.76 tok/s, i.e. prefill about 19 to 25% faster, decode about 2% faster (the interleaved P2 measurement of the decode change was +3.3%) and MTP decode within 1%.

What each roadmap item contributed: P2 (GDN T=1 Q5 launch) +3.3% decode; P3 (64x32 attention tile) +9% at 64k and +1 to 4% at 8k; P4 (convert activations once) tried and reverted, -4.4% at MTP depth 5; P5 (register prefetch in the Q5, Q4 SwiGLU and grouped input-projection kernels) +17% at 8k and +16% at 512 on its own; P6 (Q6 head) not recommended, -0.4% decode; P7 (fusing small kernels) not pursued, ceiling 1 to 2%. C1 (FP16 range) closed with no change needed.

## Docker image (roadmap U4)

Verified on this machine (Windows 11, Docker Desktop 28.3.2 with the WSL2 backend, 24 GB VM, driver 610.88): `docker build -t ninfer-fork .` took 19.3 minutes (the build stage compiles for sm_75, the CMake default; the runtime image is 3.17 GB) and `docker run -d --gpus all -v E:\AI_Models\LLM\neroued\Qwen3.8-27B-NInfer:/models -p 18095:8080 ninfer-fork` started `ninfer-serve /models/qwen3_8_27b.ninfer --host 0.0.0.0` with the default configuration (64k-token context, MTP, 3.15 GiB host context pinned in 3.7 s, 1.26 GiB pinned embedding). The default embedded chat template worked inside the container with no source tree. One chat completion through the published port returned the expected reply (171 prompt tokens, 27 output tokens, TTFT 688 ms, prefill 252 tok/s, decode 49 tok/s with MTP).
