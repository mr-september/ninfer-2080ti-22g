# Turing (sm_75) port

This fork builds NInfer for NVIDIA Turing (RTX 2080 Ti 22 GB, `CMAKE_CUDA_ARCHITECTURES=75`) on top of
the upstream engine. The engine, the Qwen3.5-family runtime, frontend, serving and scheduling are
unchanged; only the CUDA layer differs. This page records what differs and why, so the next upstream
sync can be redone deliberately instead of by merge.

## Supported surface

| Area | sm_75 |
|---|---|
| Artifacts | `groupwise-int` (Q4/Q5/Q6/Q8 + BF16 tensors): Qwen3.6-27B, Qwen3.8-27B, Qwen3.6-35B-A3B |
| NVFP4 / FP8 weights | unsupported; the routes link throwing definitions |
| KV cache | `bf16`, `int8`; `fp8`, `nvfp4`, `k8v4` unsupported |
| Speculation | MTP; DFlash (35B-A3B) |
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

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j
```

CUDA 12.8+ is enough (no 13.x-only features are used). The Q8 split-K translation units are slow to
compile for sm_75 (minutes each); expect a cold build of an hour on 16 cores.
`ninja -t deps | grep '#deps 0,'` finds objects whose header dependencies were lost by an interrupted
build; delete those objects, otherwise header edits silently do not rebuild them.

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
