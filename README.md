# NInfer for Turing (RTX 2080 Ti 22 GB)

> Selected checkpoints. Maximum single-GPU inference performance on modified 22 GB RTX 2080 Ti (`sm_75`) cards. Runs on Windows natively or on Linux/WSL2.

This fork re-targets the CUDA layer of [Neroued/ninfer](https://github.com/Neroued/ninfer) to NVIDIA Turing. The Engine, runtime, frontend, scheduling, CLI and OpenAI/Anthropic serving are upstream's; the kernels, a Windows port and a few memory and tuning changes are this fork's. It tracks upstream by re-applying a small `sm_75` layer on each sync, not by merging (see [docs/turing-port.md](docs/turing-port.md)).

## What runs here

| Area | Status on `sm_75` |
|---|---|
| Artifacts | v3 `groupwise-int` (Q4/Q5/Q6/Q8 + BF16): Qwen3.8-27B (tested) |
| NVFP4 / FP8 weights | not available (need Blackwell); the NVFP4 artifacts do not load |
| KV cache | `int8` (default), `bf16`; `fp8`, `nvfp4`, `k8v4` unavailable |
| Text, thinking, prefix reuse, CUDA Graph decode | yes |
| Image / video Vision | yes (`--vision`) |
| MTP speculation with the optimized proposal head | yes, on by default |
| Qwen3.6-27B, Qwen3.6-35B-A3B, DFlash | untested; the kernels build and pass op tests, so they may work in theory, but nothing is guaranteed. The 35B-A3B artifact (21.2 GiB) cannot fit a 22 GB card |
| OpenAI Chat/Responses, Anthropic Messages, streaming | yes |
| Kimi delta attention | not built (needs more than the 64 KiB CTA shared memory limit) |

Checked on one 2080 Ti 22 GB with Qwen3.8-27B `groupwise-int`, natively on Windows: CLI text, image and image+video prompts, a 64k-token needle retrieval, and `ninfer-serve` answering OpenAI chat, SSE streaming, Responses, Anthropic messages, tool calls and two concurrent requests. The CTest suite (130 entries) was run on Linux/WSL2: everything passes except the real-model tests that exercise FP8 KV, which this build does not support, and two real-model tests (`prefix_real`, `agent_continuation_real`) that fail on artifact-template and request-validation checks not yet traced. The Python-oracle tests need `jinja2` and `jsonschema`, and the real-model tests need `NINFER_TEST_ARTIFACT`. See [Tests](tests/README.md).

## Quick start

Download the artifact (or [convert your own](#models-formats-and-converting-your-own-weights)), build (next section), then run:

```bash
# one request: answer on stdout, diagnostics on stderr
./build/apps/ninfer models/qwen3_8_27b.ninfer --prompt "Explain prefill and decode in two sentences."

# OpenAI- and Anthropic-compatible server on http://127.0.0.1:8080
./build/apps/ninfer-serve models/qwen3_8_27b.ninfer
```

On native Windows the binaries are `build-windows\apps\ninfer.exe` and `ninfer-serve.exe`, for example:

```bat
build-windows\apps\ninfer.exe E:\path\to\qwen3_8_27b.ninfer --prompt "Hello"
```

Add `--vision` for image and video input and `--messages FILE` for structured input; see the [CLI guide](docs/cli.md), [HTTP serving](docs/serving.md) and [committed examples](examples/cli/).

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 64}'
```

## Defaults

Those two commands already use the best-tested configuration for this GPU. Written out in full, the CLI default is

```bash
./build/apps/ninfer models/qwen3_8_27b.ninfer --prompt "..." \
  --max-context 65536 --kv-capacity 65536 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --max-new 128 --prefill-chunk 1024 --device 0 \
  --chat-template tools/chat_templates/qwen3_8_sharp_v22_5.jinja
```

and the server default is

```bash
./build/apps/ninfer-serve models/qwen3_8_27b.ninfer --port 8080 \
  --max-context 65536 --kv-capacity 65536 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --max-concurrency 1 --prefill-chunk 1024 \
  --chat-template tools/chat_templates/qwen3_8_sharp_v22_5.jinja
```

(`--lm-head-draft` is applied automatically when the artifact carries a proposal head, as the official ones do.) Each default is an ordinary flag, so an ablation changes exactly one of them:

| To get | Pass |
|---|---|
| BF16 KV cache (twice the KV memory) | `--kv-dtype bf16` |
| no speculative decoding | `--spec none` |
| MTP with the full output head for proposals | `--no-lm-head-draft` |
| a different draft window | `--draft-tokens 1..5` (3 is best for code, 1 to 2 for prose) |
| a shorter or longer context | `--max-context N` (the KV pool follows it) |
| the largest KV pool that fits | `--kv-capacity auto` |
| image and video input | `--vision` |
| more retained-prefix memory in RAM | `--host-context-mib N` (default 2 GiB plus eight state images) |
| the artifact's own chat template instead of Sharp v22.5.0 | `--artifact-chat-template` |
| no CUDA Graph decode | `--no-cuda-graph` |
| another GPU | `--device N` ([multi-GPU notes](docs/multi-gpu.md)) |

The default is a 64k-token context with MTP. That needs most of the 22 GB (the embedding table lives in host memory to make room), and the limit with MTP is about 97k tokens (110k without it). The card is also a desktop GPU: other programs that hold VRAM reduce what is available, and if startup reports out of memory, pass `--max-context 32768` or `--kv-capacity auto`.

## Build

Pick the environment you want to run in; the sources are the same. Linux/WSL2 is the original, longer-tested path; native Windows avoids carving memory and disk out for a WSL VM.

### Linux or WSL2

Requires 64-bit Linux, CUDA 12.8 or newer, CMake 3.28+, Ninja, a C++20 compiler, `pkg-config`, FFmpeg development libraries (`libavformat`, `libavcodec`, `libavutil`, `libswscale`) and `libcurl >= 7.85`.

```bash
git clone https://github.com/mr-september/ninfer-2080ti-22g.git
cd ninfer-2080ti-22g
cmake --preset release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j
```

Binaries are `build/apps/ninfer`, `ninfer-serve` and `ninfer-perplexity`.

### Native Windows

In short: install the prerequisites below, open an **"x64 Native Tools Command Prompt for VS 2022"** (plain PowerShell or cmd will not find the compiler), then run the `cmake` commands at the end of this section. Run `vcpkg install` once beforehand.

Requires Windows 10/11 x64, the NVIDIA driver, CUDA 12.8+ (validated with 12.9), Visual Studio 2022 or Build Tools with the "Desktop development with C++" workload (MSVC v143 and a Windows SDK), CMake 3.28+, Ninja and [vcpkg](https://github.com/microsoft/vcpkg). Install the two native libraries once into your vcpkg checkout (FFmpeg compiles from source; allow about 20 minutes). Run this from the vcpkg directory, or give `vcpkg.exe` by full path:

```bat
cd /d C:\path\to\vcpkg
vcpkg install "ffmpeg[core,avcodec,avformat,swscale,zlib,dav1d]:x64-windows" "curl[ssl]:x64-windows"
```

(`zlib` provides the PNG decoder and `dav1d` the AV1 decoder; without `zlib` image prompts fail with "media codec is not supported".)

Then, from an "x64 Native Tools Command Prompt for VS 2022" (it provides `cl`, `cmake` and `ninja`; set `VCPKG_ROOT` to the same vcpkg directory):

```bat
set VCPKG_ROOT=C:\path\to\vcpkg
cd /d C:\path\to\ninfer
cmake --preset windows
cmake --build build-windows -j
```

Binaries are `build-windows\apps\ninfer.exe`, `ninfer-serve.exe` and `ninfer-perplexity.exe`; the FFmpeg and curl DLLs are copied next to them. Keep the `.ninfer` artifact on an NTFS drive (reading it through `\\wsl$` is slow). Use `--preset windows-dev` for tests and benchmarks.

Page-locked memory has to be free RAM: the server pins about 3 GiB by default (the 2 GiB host-context budget plus state images), the embedding table another 1.3 GiB. If startup reports `cudaMallocHost ... out of memory`, close memory-hungry programs or lower `--host-context-mib`.

A cold build compiles the Q8 split-K kernels and takes tens of minutes on either platform, plus the one-time FFmpeg build on Windows. Tests and benchmarks are off by default (`BUILD_TESTING`, `NINFER_BUILD_BENCHMARKS`; see [build organization](docs/maintainer/build-system.md)). There is no install target; run from the build tree.

## Models, formats and converting your own weights

An artifact is a `.ninfer` v3 container: the model configuration, the encoded weights, logical bindings and the tokenizer and chat-template resources. This fork changes none of the format; files converted before or after its kernel changes are interchangeable with upstream's. The weight formats this GPU runs are:

| Format | Layout | Used for |
|---|---|---|
| Q4, Q5, Q6 | signed integer codes, one FP16 scale per 64 weights | most projections; Q6 optionally for the vocabulary head |
| Q8 | signed integer codes, one FP16 scale per 32 weights | vocabulary tensors, MTP block |
| BF16 | plain | norms and small tensors |

NVFP4 and FP8 are Blackwell-only, so the `nvfp4` artifacts and recipes do not apply. The kernels also exist only for the registered Qwen 27B-class shapes: converting some other model into the format would not make it runnable.

**Download.** The official `groupwise-int` artifacts need no conversion:

```bash
hf download neroued/Qwen3.8-27B-NInfer qwen3_8_27b.ninfer --local-dir models
```

**Convert from the original checkpoint.** Start from the original BF16 Hugging Face checkpoint (a local directory of safetensors). The converter quantizes once with group-wise absolute-max rounding; it reads safetensors and already-quantized NVFP4/FP8 checkpoints, and has no GGUF reader (converting a quantized file would round twice anyway). It needs Python 3.10 or newer with PyTorch and NumPy and runs the same on native Windows and on Linux/WSL2 (use `python` instead of `python3` on Windows); it uses CUDA by default and `--device cpu` otherwise. Windows CPU-only setup: `python -m pip install torch numpy safetensors`, with a CUDA build of PyTorch from pytorch.org for GPU conversion. From the repository root:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --recipe qwen3_8_27b \
  --components text,vision,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal \
  --name qwen3.8-27b \
  --out models/qwen3_8_27b.ninfer
```

`--proposal` adds the proposal head the default speculative decoding uses. The recipes `qwen3_6_27b` and `qwen3_8_27b` select the Q4/Q5 mixture; part of a recipe can be overridden, see [Weight conversion](docs/weight-conversion.md) (including the v2 to v3 upgrade tool).

**Shrink an existing artifact.** Re-encoding the output head from Q8 to Q6 is near-lossless (perplexity +0.014%) and frees 0.4 GiB of VRAM, without needing the source checkpoint:

```bash
python3 -m tools.artifact.requantize_tensor models/qwen3_8_27b.ninfer models/qwen3_8_27b.q6head.ninfer \
  --object text/output_head --format q6_g64_fp16
```

The Windows and Linux paths produce the same weights: re-encoding the Qwen3.8-27B head natively on Windows gave tensor data byte-identical to the WSL run (only the artifact ID and provenance metadata differ).

## Performance (one RTX 2080 Ti 22 GB)

Qwen3.8-27B `groupwise-int`, INT8 KV, greedy, single request, native Windows (WSL2 matches within a few percent). Clocks and other programs move absolute figures by around 15% between sessions, so only same-session comparisons are meaningful.

| Metric | Result |
|---|---|
| prefill, 122-token prompt | ~247 tok/s |
| prefill, 8k-token prompt | ~479 tok/s |
| prefill, 64k-token prompt | ~372 tok/s |
| decode, no MTP | 24-29 tok/s |
| decode, MTP3 with the proposal head (default) | 38-44 tok/s |
| longest context that starts | 64k+ with MTP (about 97k); about 110k without |

Prefill is about 1.9x faster on long prompts than before this fork's Turing tile tuning (8k: 253 to 479, 64k: about 205 to 372 tok/s) and 1.2x on a short one (122 tokens: 201 to 247). These are Turing numbers; the RTX 5090 tables in [docs/performance.md](docs/performance.md) are upstream's and do not apply here. Details and caveats are in [docs/turing-port.md](docs/turing-port.md).

## Capabilities and limits

- thinking and non-thinking prompts; image, multi-image and video messages;
- chunked prefill, exact-batch CUDA Graph decode, one to eight execution lanes with bounded FIFO ingress;
- exact-prefix reuse with Device/Host state and KV retention, and pressure-driven pause/resume;
- model-aware sampling defaults with explicit overrides;
- OpenAI Responses, Chat Completions and Anthropic Messages, including streaming, tools, response state and token counting;
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

Based on [Neroued/ninfer](https://github.com/Neroued/ninfer); see the upstream repository for the original project and its support options.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). These source repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files under `third_party/`.
