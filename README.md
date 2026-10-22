# NInfer for Turing (RTX 2080 Ti 22 GB)

Fast single-GPU inference of Qwen3.8-27B on the 22 GB modified GeForce RTX 2080 Ti (Turing, `sm_75`). It runs natively on Windows or on Linux/WSL2/Docker and serves OpenAI- and Anthropic-compatible HTTP APIs or a local CLI.

NInfer is a from-scratch C++/CUDA inference engine: Qwen3.5-family text, image and video prompts with native MTP speculative decoding, one resident model, one to eight execution lanes. This repository is a fork of [Neroued/ninfer](https://github.com/Neroued/ninfer) at its `dev` branch, commit `438ae67c`. The Engine, runtime, frontend, scheduling, CLI and serving are upstream's; this fork adds the `turing` build target (CUDA kernels and tile tuning for `sm_75`), a native Windows build, one-command Windows setup and a release package. See [docs/turing-port.md](docs/turing-port.md) for what differs and how it was measured.

## Download a release

Prebuilt Windows binaries are on the [Releases page](https://github.com/mr-september/ninfer-2080ti-22g/releases): `ninfer-windows.zip` (`ninfer.exe`, `ninfer-serve.exe`, `ninfer-perplexity.exe`, their DLLs including `cudart64_12.dll`, and `serve.ps1`) with its SHA-256 in the release notes. Extract it, then:

```powershell
.\serve.ps1                       # serves http://127.0.0.1:8080/v1
.\serve.ps1 -Model D:\models\qwen3_8_27b.ninfer -Port 9000
```

The machine needs only an NVIDIA driver (no CUDA Toolkit and no Visual Studio). `serve.ps1` uses `models\qwen3_8_27b.ninfer` beside the script; if that file is missing and Python 3 is on `PATH`, it downloads the artifact from Hugging Face (about 18 GB). Without Python, download it yourself:

```bash
hf download neroued/Qwen3.8-27B-NInfer qwen3_8_27b.ninfer --local-dir models
```

## Requirements

| | |
|---|---|
| GPU | GeForce RTX 2080 Ti with 22 GB (`sm_75`). The weights alone take 15.4 GiB on the device (14.7 GiB without MTP), so 11 GB cards cannot run this model |
| Driver | the binaries use the CUDA 12.9 runtime, so use a driver that supports it (576.02 or newer on Windows); only 610.88 was tested |
| Host RAM | the defaults pin about 1.3 GiB for the embedding table and 0.8 to 3.2 GiB of host context (the server halves its default host context when free RAM is short, as it did here); tested with 32 GB. Page-locked memory has to be free RAM |
| Disk | 18.2 GB for the model; the release zip is 1.6 GB |
| Windows | Windows 10/11 x64 |
| Linux | 64-bit Linux, WSL2 or Docker with the NVIDIA container toolkit |

The card is usually also the display adapter: programs that hold VRAM (browsers, other models) reduce what is free. When the default 64k context does not fit in the free VRAM, startup retries at 32768 and then 16384 tokens and logs `max-context reduced to N to fit free VRAM`. An explicit `--max-context` is never lowered.

## Quick start from source (Windows)

Install the NVIDIA driver, CUDA 12.8+ (12.9 was tested) and Visual Studio 2022 or Build Tools with "Desktop development with C++", then from a PowerShell prompt in the clone:

```powershell
scripts\build.ps1    # configures with vcpkg.json and builds; the first run also builds FFmpeg
scripts\serve.ps1    # serves http://127.0.0.1:8080/v1 (downloads the model if needed)
```

`scripts\package.ps1` packs the build into `dist\ninfer-windows.zip`, the same file as the release. Then:

```bat
rem one request: answer on stdout, diagnostics on stderr
build\turing-windows\apps\ninfer.exe models\qwen3_8_27b.ninfer --prompt "Explain prefill and decode in two sentences."

rem the server (scripts\serve.ps1) answers on port 8080
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\": \"qwen3.8-27b\", \"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"max_tokens\": 64}"
```

On Linux the binaries are `build/turing-release/apps/ninfer` and `ninfer-serve` (see [Build](#build)); the commands are the same. Add `--vision` for image and video input and `--messages FILE` for structured input; see the [CLI guide](docs/cli.md), [HTTP serving](docs/serving.md) and the [committed examples](examples/cli/).

With Docker and the NVIDIA container toolkit, the [Dockerfile](Dockerfile) builds the server for Turing and serves `qwen3_8_27b.ninfer` from a mounted folder:

```bash
docker build -t ninfer .
docker run --gpus all -v /path/to/models:/models -p 8080:8080 ninfer
```

## Defaults

The commands above already use the best-tested configuration. Written out in full, the CLI default is

```bash
ninfer models/qwen3_8_27b.ninfer --prompt "..." \
  --max-context 65536 --kv-capacity 65536 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --max-new 128 --prefill-chunk 1024 --device 0
```

with the vendored Sharp v22.5.0 chat template built into the executable (`--chat-template FILE` or `--artifact-chat-template` replace it), and the server default is

```bash
ninfer-serve models/qwen3_8_27b.ninfer --host 127.0.0.1 --port 8080 \
  --max-context 65536 --kv-capacity 65536 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --max-concurrency 1 --prefill-chunk 1024
```

(`--lm-head-draft` is applied automatically when the artifact carries a proposal head, as the official one does.) Each default is an ordinary flag, so an ablation changes exactly one of them:

| To get | Pass |
|---|---|
| BF16 KV cache (twice the KV memory) | `--kv-dtype bf16` |
| no speculative decoding | `--spec none` |
| MTP with the full output head for proposals | `--no-lm-head-draft` |
| a different draft window | `--draft-tokens 1..5` (in earlier measurements 3 was fastest on code and 1 to 2 on prose; see [turing-port.md](docs/turing-port.md)) |
| a shorter or longer context | `--max-context N` (the KV pool follows it) |
| the largest KV pool that fits | `--kv-capacity auto` |
| image and video input | `--vision` |
| more retained-prefix memory in RAM | `--host-context-mib N` (the server default is 2 GiB plus eight state images) |
| the artifact's own chat template | `--artifact-chat-template` |
| no CUDA Graph decode | `--no-cuda-graph` |
| another GPU | `--device N` ([multi-GPU notes](docs/multi-gpu.md)) |

The default is a 64k-token context with MTP, which needs most of the 22 GB; the embedding table lives in host memory to make room.

## Performance (one RTX 2080 Ti 22 GB)

Measured 2026-10-19 on `master`: native Windows 11, driver 610.88, CUDA 12.9, Qwen3.8-27B `groupwise-int`, INT8 KV, CUDA Graph decode, single request, desktop idle (2.3 GiB of VRAM held by other programs). Absolute figures move by several percent between sessions, so only same-session comparisons are meaningful.

| Metric | Result | How |
|---|---|---|
| prefill, 512-token prompt | 616 tok/s | `ninfer_bench -p 512 -n 256 -r 3 --warmup 1 --kv-dtype int8 --max-ctx 16384` |
| prefill, 8k-token prompt | 569 tok/s | `ninfer_bench -p 8192,65536 -n 32 -r 2 --warmup 2 --kv-dtype int8 --max-ctx 70000` |
| prefill, 64k-token prompt | 458 tok/s | same run |
| decode, MTP off | 27.0 tok/s (25.9 with the 70,000-token KV allocation of the long run) | the two runs above |
| decode, MTP3 with the proposal head, benchmark corpus | 29.6 tok/s | bench-short plus `--spec mtp --draft-tokens 3 --lm-head-draft` |
| decode, MTP3, real code prompt | 47.5 tok/s (26.3 with `--spec none`) | `ninfer --messages examples/cli/messages/scenario_code_python.json --greedy --max-new 256 --no-thinking` |
| decode, MTP3, real prose prompt | 28.8 tok/s (26.0 with `--spec none`) | the same with `scenario_story_en_mystery.json` |

The benchmark corpus is random token ids, so its MTP acceptance (0.377) is far below real text; the two CLI rows show the range on real prompts. The full table, methods and per-change measurements are in [docs/turing-port.md](docs/turing-port.md#current-performance). The RTX 5090 tables in [docs/performance.md](docs/performance.md) are upstream's and do not apply here.

## What runs here

| Area | Status on `sm_75` |
|---|---|
| Artifacts | v3 `groupwise-int` (Q4/Q5/Q6/Q8 + BF16): Qwen3.8-27B (tested) |
| FP8 weights | not available (they need Blackwell) |
| KV cache | `int8` (default), `bf16`; `fp8` unavailable |
| Text, thinking, prefix reuse, CUDA Graph decode | yes |
| Image / video input | yes (`--vision`) |
| MTP speculation with the optimized proposal head | yes, on by default |
| OpenAI Chat/Responses, Anthropic Messages, streaming | yes |
| Qwen3.6-27B, Qwen3.6-35B-A3B, DFlash | untested; the kernels build and pass op tests, nothing is guaranteed. The 35B-A3B artifact (21.2 GiB) cannot fit a 22 GB card |
| Qwen4Exp (Flash Next), CPU expert offload, QSA sparse attention, block-scaled MoE and other Blackwell-only Ops | not available: the Ops are stubs that throw on `turing` |

Checked on one 2080 Ti 22 GB with Qwen3.8-27B `groupwise-int` on native Windows (2026-10-19): the full CTest suite (138 tests: 0 failed, 25 skipped, the skips being FP8 cases and real-model tests that need `NINFER_TEST_ARTIFACT`), protocol-B perplexity, the release zip started from an empty directory with a stripped `PATH`, and a chat request through `ninfer-serve`. Earlier checks on the same kernels covered CLI text, image and image+video prompts, a 64k-token needle retrieval, and `ninfer-serve` with OpenAI chat, SSE streaming, Responses, Anthropic messages, tool calls and two concurrent requests. On Linux/WSL2 the same commit built and passed its CTest suite (145 tests, 28 skipped) and answered a real prompt, and the Docker image built, started with the model bind-mounted and answered a chat request ([details](docs/turing-port.md#verification-of-master-2026-10-19-and-2026-10-20)). See [Tests](tests/README.md).

## Known limitations

- Turing only. FP8 weights and KV are unavailable.
- Qwen3.8-27B is the only tested model.
- Greedy output is not bit-identical to the fork's build from before the port onto upstream `dev`: on 5 of 8 test prompts the text diverges at a near-tied token, while log-probabilities agree to 0.19 nats at most and perplexity to 0.045% ([details](docs/turing-port.md#port-onto-upstream-dev-2026-10-18)).
- The 64k default context leaves little VRAM headroom; other programs using the GPU can force a smaller context at startup.
- Page-locked host memory must be free RAM; on a machine short of RAM the server starts with a smaller host context, and an explicit `--host-context-mib` that cannot be pinned fails at startup.
- Reading the model through `\\wsl$` is slow; keep it on an NTFS drive (Windows) or a Linux filesystem (WSL2).

## Build

On native Windows, `scripts\build.ps1` does everything below. The two platforms side by side (the sources are the same):

| | Linux / WSL2 | Windows native |
|---|---|---|
| **Shell** | any bash | PowerShell (`scripts\build.ps1` enters the Visual Studio developer shell itself) or "x64 Native Tools Command Prompt for VS 2022" |
| **Needs** | 64-bit Linux, CUDA 12.8+, CMake 3.28+, Ninja, a C++20 compiler, `pkg-config`, FFmpeg dev libraries (`libavformat`, `libavcodec`, `libavutil`, `libswscale`), `hwloc >= 2.5`, `liburing >= 2.5`, `libcurl >= 7.85` | Windows 10/11 x64, NVIDIA driver, CUDA 12.8+ (validated with 12.9), Visual Studio 2022 or Build Tools with "Desktop development with C++" (MSVC v143 + a Windows SDK), CMake 3.28+ and Ninja (both ship with the VS workload), vcpkg (the copy bundled with VS 2022 17.6+ works) |
| **1. Libraries (once)** | Debian/Ubuntu: `sudo apt install pkg-config libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libcurl4-openssl-dev libhwloc-dev liburing-dev` | nothing to run: [`vcpkg.json`](vcpkg.json) lists FFmpeg, hwloc and curl, and the configure step installs them into the build folder (FFmpeg compiles from source the first time) |
| **2. Get the source** | `git clone https://github.com/mr-september/ninfer-2080ti-22g.git`<br>`cd ninfer-2080ti-22g` | the same clone |
| **3. Point at the libraries** | not needed (found through `pkg-config`) | `scripts\build.ps1` sets `VCPKG_ROOT` (the Visual Studio copy, else a clone in `.vcpkg`); otherwise `set VCPKG_ROOT=C:\path\to\vcpkg` |
| **4. Configure** | `cmake --preset turing-release` | `cmake --preset turing-windows` |
| **5. Build** | `cmake --build build/turing-release -j` | `cmake --build build/turing-windows -j` |
| **Binaries** | `build/turing-release/apps/ninfer`, `ninfer-serve`, `ninfer-perplexity`, `ninfer-logit-compare` | `build\turing-windows\apps\ninfer.exe`, `ninfer-serve.exe`, `ninfer-perplexity.exe`, `ninfer-logit-compare.exe` (the FFmpeg and curl DLLs are copied next to them) |
| **Tests and benchmarks** | `cmake --preset turing-dev` | `cmake --preset turing-windows-dev` (or `scripts\build.ps1 -Dev`) |

The `zlib` and `dav1d` features in `vcpkg.json` provide the PNG and AV1 decoders; without `zlib` image prompts fail with "media codec is not supported".

Presets are named `turing-release`, `turing-dev` (adds tests and benchmarks) and `turing-ops` (Core, Artifact and Ops only), and each builds in `build/<preset-name>/`; the Windows presets are `turing-windows`, `turing-windows-dev` and `turing-windows-ops`. The build requires `CMAKE_CUDA_ARCHITECTURES=75`. Machine-specific compiler and Python paths belong in the ignored `CMakeUserPresets.json`. On Linux the executables find the shared libraries in the same build's `lib/` directory through a relative RPATH, so keep it beside `apps/` when moving build outputs. See [build organization](docs/maintainer/build-system.md).

The Q8 split-K kernels make the build slow: it took 10.6 minutes on 16 logical cores at full parallelism with FFmpeg already in vcpkg's cache (measured on the previous base). On a machine with little free RAM the default parallelism can make `nvcc` run out of memory; set `CMAKE_BUILD_PARALLEL_LEVEL=4` (or pass `-j4`) and rerun, the build resumes.

## Models, formats and converting your own weights

An artifact is a `.ninfer` v3 container: the model configuration, the encoded weights, logical bindings and the tokenizer and chat-template resources. This fork changes none of the format; files converted before or after its kernel changes are interchangeable with upstream's. The weight formats this GPU runs are:

| Format | Layout | Used for |
|---|---|---|
| Q4, Q5, Q6 | signed integer codes, one FP16 scale per 64 weights | most projections; Q6 optionally for the vocabulary head |
| Q8 | signed integer codes, one FP16 scale per 32 weights | vocabulary tensors, MTP block |
| BF16 | plain | norms and small tensors |

FP8 is Blackwell-only. The kernels also exist only for the registered Qwen 27B-class shapes: converting some other model into the format would not make it runnable.

**Convert from the original checkpoint.** Start from the original BF16 Hugging Face checkpoint (a local directory of safetensors). The converter quantizes once with group-wise absolute-max rounding; it reads safetensors and already-quantized FP8 checkpoints, and has no GGUF reader. It needs Python 3.10 or newer with PyTorch and NumPy and runs the same on native Windows and on Linux/WSL2 (use `python` instead of `python3` on Windows); it uses CUDA by default and `--device cpu` otherwise. From the repository root:

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

**Shrink an existing artifact.** Re-encoding the output head from Q8 to Q6 is near-lossless (perplexity +0.014% on protocol B, measured on the previous base) and frees 0.4 GiB of VRAM, without needing the source checkpoint; decode is not faster:

```bash
python3 -m tools.artifact.requantize_tensor models/qwen3_8_27b.ninfer models/qwen3_8_27b.q6head.ninfer \
  --object text/output_head --format q6_g64_fp16
```

## Capabilities

- thinking and non-thinking prompts; image, multi-image and video messages;
- chunked prefill, exact-batch CUDA Graph decode, one to eight execution lanes with bounded FIFO ingress;
- exact-prefix reuse with Device/Host state and KV retention, and pressure-driven pause/resume;
- model-aware sampling defaults with explicit overrides;
- OpenAI Responses, Chat Completions and Anthropic Messages, including streaming, tools, response state and token counting;
- one resident model per Engine; no priority/QoS or distributed serving.

Run the relevant `--help` for the exact current option contract.

## Documentation

- [Documentation index](docs/README.md)
- [Turing port notes](docs/turing-port.md): what differs, measurements, verification
- [CLI](docs/cli.md) and [HTTP serving](docs/serving.md)
- [Perplexity and build-to-build comparison](docs/perplexity.md)
- [Multi-GPU](docs/multi-gpu.md)
- [Weight conversion](docs/weight-conversion.md)
- [Resource scheduling and context cache](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Contributing](CONTRIBUTING.md)

Based on [Neroued/ninfer](https://github.com/Neroued/ninfer); see the upstream repository for the original project and its support options.

## License

NInfer is licensed under the [Apache License 2.0](LICENSE).

The published artifacts are derived from [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). These source repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files under `third_party/`. The release zip includes `cudart64_12.dll` from the NVIDIA CUDA Toolkit, redistributed under NVIDIA's CUDA Toolkit license.
