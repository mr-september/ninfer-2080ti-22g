# Turing (sm_75) port

This fork builds NInfer for NVIDIA Turing (RTX 2080 Ti 22 GB, `CMAKE_CUDA_ARCHITECTURES=75`) on top of
the upstream engine. The engine, the Qwen3.5-family runtime, frontend, serving and scheduling are
unchanged; only the CUDA layer differs. This page records what differs and why, so the next upstream
sync can be redone deliberately instead of by merge.

## Supported surface

| Area | sm_75 |
|---|---|
| Artifacts | `groupwise-int` (Q4/Q5/Q6/Q8 + BF16 tensors): Qwen3.8-27B is the only tested model. Qwen3.6-27B and Qwen3.6-35B-A3B are untested (op-level kernels only; the 21.2 GiB 35B-A3B artifact does not fit 22 GB) |
| NVFP4 / FP8 weights | unsupported; the routes link throwing definitions |
| KV cache | `bf16`, `int8`; `fp8`, `nvfp4`, `k8v4` unsupported |
| Speculation | MTP (tested); DFlash is untested and tied to 35B-A3B |
| Kimi delta attention | not built into the sm_75 test set (needs more than the 64 KiB CTA limit) |

Unsupported formats are not stubbed one symbol at a time inside upstream sources. `src/ops/CMakeLists.txt`
filters the FP8/NVFP4/K8V4 sources out of the upstream manifests for every target other than
`120a`, and `src/ops/unsupported/sm120_formats_unsupported.cpp` defines the declarations the shared
wrappers still name. Upstream manifests therefore stay byte-identical.

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

`tests/ops/test_mma_shim.cu` checks every tensor-core wrapper against naive math in the PTX fragment
layouts; run it first when touching `ops/common/mma.cuh`.

## Prefill tiles and token embedding (measured on one RTX 2080 Ti 22 GB)

**Prefill tiles.** About 95% of prefill time is the dequantizing GEMMs. Upstream sizes their tiles for
the RTX 5090 (two `cp.async` stages, about 46 KB of shared memory), which on Turing leaves one 4-warp CTA
per SM: profiled at 9.8% tensor-pipe and 11% issue-slot utilization, stalled on global loads with no
`cp.async` to hide them. Single-stage tiles use about 27 KB, so two CTAs share each SM and one tile's MMA
overlaps the other's decode and loads. Applied under `NINFER_SM75` only (the other architectures keep
upstream's tables): Q5 plain and residual-add (`linear_add`) projections, Q4 plain projections, the folded
Q4 gate/up SwiGLU (`MmaPairC128W8S1B2`), and the grouped GDN and attention input projections. Routes were
chosen per shape by timing every instance across 128 to 2048 tokens.

| Operation (1024 tokens) | Before | After |
|---|---|---|
| Q5 MLP down 5120x17408 | 19.4 ms | 8.9 ms |
| Q5 down with residual add | 18.4 ms | 9.8 ms |
| Q4 gate/up 34816x5120, plain | 32.6 ms | 16.2 ms |
| Q4 gate/up folded with SwiGLU | 21.7 ms | 13.2 ms |
| GDN grouped input projection | 12.1 ms | 6.8 ms |

End to end, native Windows, greedy, MTP off (prefill tokens/s): 122-token prompt 201 to 247, 8k prompt 253
to 479, 64k prompt about 205 to 372. Kernels at 5 to 8 tokens (MTP depths 4 and 5) were already within
about 10% of the best candidate, so draft depth 3 stays the default: a 5-token step costs about 1.4x a
4-token step because the cheap SIMT kernels stop at 4 tokens.

**Token embedding.** The embedding is gathered one row per token, so it lives in pinned host memory that
kernels read over PCIe (`Residency::PinnedHost`): 1.26 GiB less device memory, output byte-identical, no
measurable decode change. If page-locked memory is unavailable (a host short of free RAM), the binder
falls back to device residency at bind time. Effect: a 64k-token prompt now fits with MTP (it did not
before); the largest context that starts is about 97k to 99k with MTP (it moves with how much VRAM the
desktop and other programs hold) and between 110k and 125k without it.

**Output head format.** Re-encoding `text/output_head` from Q8 to Q6 (`python -m
tools.artifact.requantize_tensor SRC OUT --object text/output_head --format q6_g64_fp16`, about 6
minutes) is nearly free in quality and saves memory but not time. Fixed-corpus perplexity, quick set,
4096 context, INT8 KV: overall 4.7944 to 4.7950 (+0.014%); Chinese +0.023%, long-form English +0.035%,
English reference +0.001%, code -0.002%. The artifact is 341 MiB smaller and planned device memory drops
0.4 GiB, but decode is not faster (23.5 against 23.8 tok/s): the saved bandwidth is spent unpacking Q6.

**Not done: a keep-list output head.** Dropping rarely used rows from the main output head (and gathering
logits back to the full vocabulary) would save about 0.6 GiB and a few percent of decode bandwidth. The
repository's own frequency ranking shows the cost: its top 131,072 tokens (the proposal-head list) cover
99.1 to 99.4% of real tokens overall, 99.6% for English, 99.2 to 99.5% for Chinese and code, and only 97.5
to 98.2% for other languages, so up to a few percent of tokens could never be generated. With the
embedding already on the host and a Q6 head available, that trade was not taken.

A bottom-up variant (keep every special, template and byte token and every real word; cut only tokens with no legitimate use) was audited on the Qwen3.8 tokenizer and is not worth building: of 248,320 head rows only 243 are padding, and the junk-shaped tokens (long symbol runs, control characters, replacement characters, mojibake, hex runs, unused code points) number about 450 rows, 1.7 MiB at Q6. One head row costs 4,000 bytes at Q6, so 0.1 GiB needs about 27,000 rows, and those would be real words: a large share of the tokens never chosen in the ranking corpus are ordinary Korean, Thai, Vietnamese and code-identifier words. Frequency alone cannot separate artifacts from rare words.

**MTP depth and the proposal head.** `--lm-head-draft` is free speed: +3% decode on average (+1.7%,
+2.6%, +4.7% in paired runs), identical acceptance and byte-identical output, so the README examples use
it. Depth 3 is best for code (decode tok/s at depth 1 to 5: 31.8, 37.0, 39.3, 31.4, 32.9; acceptance 92%
down to 65%). For prose depth 1 or 2 is better (28.3, 28.0, 26.4, 21.4, 20.0; acceptance 69% down to
31%). The depth is fixed at startup because the draft count shapes the captured CUDA graphs and round
buffers (51 references), so an adaptive depth would be a larger change than a tuning step.

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j
```

CUDA 12.8+ is enough (no 13.x-only features are used). The Q8 split-K translation units are slow to
compile for sm_75 (minutes each); expect a cold build of an hour on 16 cores.
`ninja -t deps | grep '#deps 0,'` finds objects whose header dependencies were lost by an interrupted
build; delete those objects, otherwise header edits silently do not rebuild them.

## Native Windows build

The same sources build with MSVC 14.4 and `nvcc` 12.9 (`cmake --preset windows`; see the
[README](../README.md#native-windows) for prerequisites). Linux/WSL2 builds are unchanged; nothing
selects between them at runtime, the environment you build in is the mode. What differs:

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

Windows-specific runtime cautions: keep the artifact on NTFS; the card is in WDDM mode, so the desktop
and other GPU programs (browsers, image-generation tools) take VRAM that `--kv-capacity auto` can no
longer use; if the NVIDIA driver allows CUDA allocations to spill into system memory, disable that
("CUDA - Sysmem Fallback Policy") for these executables so a full card fails instead of slowing down.
`ninfer-serve` pins about 9 GiB of host memory by default; on a desktop with little free RAM pass a
smaller `--host-context-mib`. FFmpeg needs the `zlib` feature for PNG; see the README.

### Native Windows versus WSL2

Same machine, same 2080 Ti 22 GB, Qwen3.8-27B `groupwise-int`, INT8 KV, greedy, MTP off, runs
interleaved and gated on an otherwise idle GPU (another program intermittently used this card, which
invalidated some runs; those were discarded). Each environment used its own build and its own copy of
the artifact.

| Prompt | Metric | Native Windows | WSL2 |
|---|---|---|---|
| 122 tokens, 512 generated | decode | 24.4, 24.5, 24.0 tok/s | 24.1, 24.0 tok/s |
| 122 tokens | prefill | 201.8, 204.9, 196.6 tok/s | 215.3, 213.5 tok/s |
| ~8k tokens | prefill | 253.0, 236.0, 258.9 tok/s | 259.5, 258.6, 262.0 tok/s |

Decode is at parity. Prefill is about 6% lower natively on a short prompt and about 2.5% on an 8k
prompt (medians), consistent with a small fixed per-launch cost under WDDM; the kernels are
identical. Absolute figures vary by around 15% between sessions with GPU clocks and background load,
so only same-session interleaved comparisons mean anything. Native vision output (image, image+video)
and the OpenAI/Anthropic server behavior match WSL2; the CUDA fault-injection part of the
artifact-materialization test is Linux-only (it needs GNU `ld --wrap`).

## Custom chat style

The former `--chat-style sharp-v22.1` option is the Jinja template
`tools/chat_templates/qwen3_8_sharp_v22_1.jinja`:

```bash
./build/apps/ninfer model.ninfer --chat-template tools/chat_templates/qwen3_8_sharp_v22_1.jinja --prompt "..."
```

## Measured on one RTX 2080 Ti 22 GB

Qwen3.8-27B `groupwise-int`, INT8 KV, greedy, ~125-token prompts, long warmups (the GPU clock ramps).

| Metric | Result |
|---|---|
| decode, no MTP | ~29 tok/s |
| decode, MTP3 | ~44 tok/s |
| prefill (end to end) | 225-240 tok/s |
| prefill GEMM, Q4 34816x5120, T=512/2048 | 12.4 / 12.6 TFLOPS |
| prefill GEMM, Q5 5120x6144, T=512/2048 | 11.2 / 11.5 TFLOPS |
| prefill GEMM, Q8 5120x6144, T=512/2048 | 18.1 / 19.1 TFLOPS |

Single-prompt greedy comparisons are noisy for MTP (near-tie flips change the acceptance rate); compare
decode with MTP off. The 35B-A3B and DFlash routes build and pass the op tests but were not run
end to end here (no local artifact).
