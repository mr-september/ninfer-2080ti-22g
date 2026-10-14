# NInfer for Turing (RTX 2080 Ti 22 GB)

> Selected checkpoints. Maximum single-GPU inference performance on modified 22 GB RTX 2080 Ti
> (`sm_75`) cards.

This fork re-targets the CUDA layer of [Neroued/ninfer](https://github.com/Neroued/ninfer) to NVIDIA
Turing. The Engine, runtime, frontend, scheduling, CLI and OpenAI/Anthropic serving are upstream's;
only the kernels differ. It tracks upstream by re-applying a small `sm_75` layer on each sync, not by
merging (see [docs/turing-port.md](docs/turing-port.md)).

## What runs here

| Area | Status on `sm_75` |
|---|---|
| Artifacts | v3 `groupwise-int` (Q4/Q5/Q6/Q8 + BF16): Qwen3.6-27B, Qwen3.8-27B, Qwen3.6-35B-A3B |
| NVFP4 / FP8 weights | not available (need Blackwell); the NVFP4 artifacts do not load |
| KV cache | `bf16`, `int8`; `fp8`, `nvfp4`, `k8v4` unavailable |
| Text, thinking, prefix reuse, CUDA Graph decode | yes |
| Image / video Vision | yes (`--vision`) |
| MTP | yes (`--spec mtp --draft-tokens N`) |
| DFlash (35B-A3B) | builds and passes op tests; not run end to end (no local artifact) |
| OpenAI Chat/Responses, Anthropic Messages, streaming | yes |
| Kimi delta attention | not built (needs more than the 64 KiB CTA shared memory limit) |

Checked on one 2080 Ti 22 GB with Qwen3.8-27B `groupwise-int`: CLI text, image and image+video
prompts; `ninfer-serve` with MTP3 and `--vision` answering OpenAI chat, SSE streaming, Anthropic
messages and two concurrent requests; and the full op test suite (119 tests).

## Build

Requires 64-bit Linux (WSL2 works), CUDA 12.8 or newer, CMake 3.28+, Ninja, a C++20 compiler,
`pkg-config`, FFmpeg development libraries (`libavformat`, `libavcodec`, `libavutil`, `libswscale`)
and `libcurl >= 7.85`.

```bash
git clone https://github.com/mr-september/ninfer-2080ti-22g.git
cd ninfer-2080ti-22g
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j
```

A cold build takes about an hour on 16 cores (the Q8 split-K translation units are slow for
`sm_75`). Tests and benchmarks are off by default: add `-DBUILD_TESTING=ON` and
`-DNINFER_BUILD_BENCHMARKS=ON` (see [build organization](docs/maintainer/build-system.md)). There is no
install target; run from the build tree.

## Models

Use v3 `.ninfer` artifacts, for example the upstream
[Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) or
[Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) `groupwise-int` downloads:

```bash
hf download neroued/Qwen3.8-27B-NInfer qwen3_8_27b.ninfer --local-dir models
```

Existing v2 artifacts can be [upgraded locally](docs/weight-conversion.md#upgrade-an-existing-v2-artifact);
you can also [convert your own weights](docs/weight-conversion.md).

## Quick start

One-shot CLI request (INT8 KV, MTP3):

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 8192 --max-new 1024 \
  --kv-dtype int8 --kv-capacity auto \
  --spec mtp --draft-tokens 3
```

OpenAI/Anthropic server with two lanes and image/video input:

```bash
./build/apps/ninfer-serve models/qwen3_8_27b.ninfer \
  --max-context 8192 --kv-capacity auto --kv-dtype int8 \
  --max-concurrency 2 --spec mtp --draft-tokens 3 --vision

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 64}'
```

Answer content goes to stdout; diagnostics and the timing/memory report go to stderr. Use
`--messages FILE` and `--vision` for structured image/video input; see the [CLI guide](docs/cli.md)
and [committed examples](examples/cli/). `--kv-capacity auto` sizes the shared KV pool from the memory
left after weights (1 GiB headroom). On multi-GPU hosts select a card with `--device N`
([multi-GPU notes](docs/multi-gpu.md)).

## Performance (one RTX 2080 Ti 22 GB)

Qwen3.8-27B `groupwise-int`, INT8 KV, greedy, ~125-token prompts after a long warmup (the GPU clock
ramps), single request:

| Metric | Result |
|---|---|
| decode, no MTP | ~29 tok/s |
| decode, MTP3 | ~44 tok/s |
| prefill, end to end | 225-240 tok/s |
| prefill GEMM Q4 / Q5 / Q8 | ~12.5 / ~11.4 / ~19 TFLOPS |

These are Turing numbers from this fork; the RTX 5090 tables in [docs/performance.md](docs/performance.md)
are upstream's and do not apply here. Details and caveats are in [docs/turing-port.md](docs/turing-port.md).

## Capabilities and limits

- thinking and non-thinking prompts; image, multi-image and video messages;
- chunked prefill, exact-batch CUDA Graph decode, one to eight execution lanes with bounded FIFO ingress;
- exact-prefix reuse with Device/Host state and KV retention, and pressure-driven pause/resume;
- model-aware sampling defaults with explicit overrides;
- OpenAI Responses, Chat Completions and Anthropic Messages, including streaming, tools, response
  state and token counting;
- one resident model per Engine; no priority/QoS, weight offload, or distributed serving.

Run the relevant `--help` for the exact current option contract.

## Documentation

- [Documentation index](docs/README.md)
- [Turing port notes](docs/turing-port.md)
- [CLI](docs/cli.md) and [HTTP serving](docs/serving.md)
- [Multi-GPU](docs/multi-gpu.md)
- [Weight conversion](docs/weight-conversion.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Contributing](CONTRIBUTING.md)

Based on [Neroued/ninfer](https://github.com/Neroued/ninfer); see the upstream repository for the
original project and its support options.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
