# NInfer

> Selected checkpoints. Maximum single-GPU inference performance.

NInfer is a from-scratch C++/CUDA inference engine for explicitly registered Qwen checkpoints on a
single NVIDIA GeForce RTX 5090. It runs text, image, and video prompts through a local CLI or
OpenAI-/Anthropic-compatible HTTP APIs.

NInfer deliberately supports a closed set of model artifacts instead of acting as a general model
runtime:

| Model | Weights | NInfer artifact | Size | SHA-256 |
|---|---|---|---:|---|
| [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | `groupwise-int` | `qwen3_6_27b.ninfer` | 17,495,365,888 bytes (16.29 GiB) | `7b51600ffd10632b9660f56085efdd9b751d79733ad32036a652234b64bebe7b` |
| [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | 18,324,064,000 bytes (17.07 GiB) | `bce5f00d066c0f20f1317bf1fdcb458264cf95837c3b1f3fbec163694627893a` |
| [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | `groupwise-int` | `qwen3_8_27b.ninfer` | 18,210,531,328 bytes (16.96 GiB) | `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e` |
| [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | 21,492,695,040 bytes (20.02 GiB) | `bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32` |
| [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | 22,783,246,080 bytes (21.22 GiB) | `1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2` |

Qwen3.6-27B and Qwen3.8-27B each expose two registered weight profiles. The version-2 artifact
identity selects the profile without a separate runtime flag; Qwen3.8 uses target key
`qwen3_8_27b` while sharing the 27B execution package. The Qwen3.6 `nvfp4` profile uses W4A4 Tensor
Core MMA for prefill and A16 NVFP4 kernels for decode. The Qwen3.8 `nvfp4` profile preserves its
source's mixed allocation: NVFP4 MLP weights in Text layers 0–55 and row-scaled FP8 for the token
embedding, attention input/output projections, GDN Q/K/V/Z and output projections, output head, and
remaining MLP weights. All four 27B artifacts retain the same Text, Vision, MTP, prefix-reuse, CLI,
and serving routes.

## Chat-style support (`--chat-style`)

This fork adds `--chat-style default|sharp-v22.1` to the CLI and server, implementing a
[Sharp v22.1](https://huggingface.co/froggeric/Qwen3.8-27B-Instruct-v22.1-Q6_K.gguf) overlay as a
**compiled-C++ prompt modifier** rather than a Jinja template swap.

### What it does

The overlay appends Sharp's terseness instruction to the effective system content and changes
reasoning-effort defaults:
- **System content**: Sharp's `_terse` instruction is appended to any existing system/developer
  message, or becomes the system block itself when none is provided.
- **Reasoning effort**: Default effort shifts from `XHigh` (stock) to `Medium` (Sharp). Six
  discrete levels (`none`→`minimal`→`low`→`medium`→`high`→`xhigh`) are exposed; `none`
  disables thinking; `minimal`/`low` map to concise reasoning, `high`/`xhigh` to thorough.
- **Thinking-history rendering**: Historical turns without reasoning content omit the empty
  thinking wrapper, matching Sharp's `{% if message.reasoning_content or message.reasoning %}`.

All other rendering (tool format, per-turn formatting, generation prompt) is unchanged from stock
NInfer. The official `.ninfer` artifact's embedded template hash is not modified — the same
artifact loads with either style.

### Token-efficiency results (RTX 5090, Qwen3.8-27B NVFP4, INT8 KV)

| Metric | Value |
|--------|-------:|
| **Median completion-token reduction** (24 tasks, MTP3, medium effort) | **−42.2%** |
| Range (24 tasks) | −70.3% to +2.4% |
| **Median wall-time reduction** | **−22.6%** |
| **Raw decode TPS change** (median) | **−0.0%** |
| MTP acceptance Δ (mean) | −0.0 pp |

Sharp produces equivalent-quality answers on objective checks (math reasoning, JSON structure, code
correctness). The effect is prompt-driven (terse system instruction), not MTP-dependent — MTP0
shows the same 31–64% token reduction with similar TPS.

**MTP3 breakdown by reasoning effort** (matched config, system message present):

| Effort | Stock tokens | Sharp tokens | Token reduction | Wall reduction | Raw TPS (stock/sharp) |
|--------|------------:|------------:|----------------:|---------------:|----------------------:|
| medium | 1,184 | 609 | −48.6% | −26.8% | 142.6 / 142.1 |
| high | 1,084 | 511 | −52.9% | −41.5% | 130.3 / 129.6 |
| xhigh | 1,084 | 511 | −52.9% | −34.6% | 131.6 / 129.6 |

**Hard-subset token reduction** (out-of-distribution difficulty, MTP3): −7% to −59% (analytic
−58.7%, coding −33.5%).

### Usage

```bash
# CLI
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --chat-style sharp-v22.1 \
  --reasoning-effort medium \
  --messages chat.json

# Server (all requests use the configured style)
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --chat-style sharp-v22.1 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

The flag is server-wide; individual per-request override is not currently supported.

### Reproducing these results

To reproduce the byte-exact system-block validation and token-efficiency benchmark from scratch:

```bash
# 1. Clone and build
git clone https://github.com/mr-september/ninfer-sharp-5090.git
cd ninfer
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# 2. Download a Qwen3.8-27B NVFP4 model artifact
pip install huggingface-hub
hf download unsloth/Qwen3.8-27B-NVFP4 \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models

# 3. Download the Sharp v22.1 template for oracle validation
pip install jinja2
wget -O work/chat_template.jinja \
  https://huggingface.co/froggeric/Qwen3.8-27B-Instruct-v22.1-Q6_K.gguf/raw/main/chat_template.jinja

# 4. Build and run the Sharp overlay unit test
#    (compares system-block output against the real Sharp Jinja template)
cmake --build build --target ninfer_qwen3_6_chat_template_sharp_test
./build/tests/ninfer_qwen3_6_chat_template_sharp_test

# 5. Run a quick manual smoke test
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --chat-style sharp-v22.1 \
  --max-context 131072 --kv-dtype int8 --no-prefix-reuse \
  --greedy --seed 42 --default-max-tokens 256

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "system", "content": "You are a precise assistant."},
      {"role": "user", "content": "Explain virtual memory in one sentence."}
    ],
    "max_tokens": 256
  }'
```

A full A/B benchmark requires the task suite and analysis scripts in `work/` (included in the fork
at `work/ab_harness2.py`, `work/ab_analyze2.py`). Example run:

```bash
# Start two servers (default and sharp) on different ports
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --port 8100 --chat-style default --spec mtp --draft-tokens 3 --lm-head-draft \
  --max-context 131072 --kv-dtype int8 --greedy --seed 42

./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --port 8101 --chat-style sharp-v22.1 --spec mtp --draft-tokens 3 --lm-head-draft \
  --max-context 131072 --kv-dtype int8 --greedy --seed 42

# Run the benchmark harness
pip install httpx
python3 work/ab_harness2.py --default http://127.0.0.1:8100 --sharp http://127.0.0.1:8101

# Analyze results
python3 work/ab_analyze2.py
```

### Why not replace the Jinja template directly

The simplest-appearing approach — swap the Jinja template inside the `.ninfer` artifact and
repack — does not work with NInfer's architecture. NInfer does not interpret Jinja at runtime.
Its `chat_template.cpp` computes `sha256(template_source)` and matches it against a hardcoded
constexpr allowlist of exactly two digests: one for ThinkingToggle semantics and one for
ReasoningEffort. A third-party template (Sharp's SHA-256 `d1f22a89…`) is rejected at load.
Patching the allowlist would make the artifact loadable, but the runtime still invokes the
**compiled C++ renderer**, not the Jinja template — so Sharp's `reasoning_effort`, terse
instruction, tool formatting, and disabled-thinking behavior would remain inert.

Repacking Sharp's Jinja into the artifact would be the "correct" surface-level fix but is
**insufficient** without either:

- embedding a Jinja interpreter (heavy, fragile, defeats the purpose of a compiled renderer); or
- rewriting the C++ renderer to accommodate Sharp's semantics — which is what this `--chat-style`
  overlay already does.

The overlay approach (compiled C++ modifier on the existing renderer) was chosen because it:

- leaves the official artifact untouched (no digest repack);
- adds zero runtime overhead (a ~300-byte prompt append + enum dispatch);
- lets stock and Sharp share one code-path, reducing divergence surface area;
- exposes the configuration through a straightforward CLI flag.

### Limitations

1. **System-message dependency**: the overlay synthesizes a system block when none exists, but
   clients that intentionally suppress system content (transparent proxy layers) will not observe
   Sharp's effect. This matches Sharp's own Jinja behavior.
2. **Tool-formatting is NInfer-native, not Sharp-v22.1-format**: This is the main reason the
   overlay is recommended for **non-tool chat** when byte-exact Sharp compatibility matters.
   The overlay does not change NInfer's `<tool_call>` XML format to match Sharp's Jinja
   `_render_tool_schema` (`# Tools` header, `⚠️` error-escalation warnings, different JSON
   argument rendering). For tool-using clients, the rendered prompt is therefore stock NInfer's
   tool format with the Sharp terse overlay — not Sharp's exact template output. If your
   workflow does not use tool calling, or does not require byte-exact Sharp tool-format
   compatibility, the overlay's token savings still apply. Tool output is identical between
   `--chat-style default` and `--chat-style sharp-v22.1`.
3. **Server-wide style**: `--chat-style` is set at process startup and applies to all requests.
   Per-request override is not implemented.
4. **Qwen3.6/3.8 only**: the overlay targets the Qwen3.6/3.8 ReasoningEffort renderer (the C++
   `CompiledChatTemplate`). Other model families (future targets using ThinkingToggle or other
   semantics) are unaffected.

## Performance

The published measurements cover the three Qwen3.6 artifact profiles and the Qwen3.8-27B NVFP4
profile. The Qwen3.8-27B `groupwise-int` profile is supported by current NInfer builds but is not
yet included in a published benchmark campaign.

### Concurrent MTP3 decode

Saturated decode was measured on an RTX 5090 with INT8 group-64 KV cache, CUDA Graphs, MTP3, and
one 8,192-token generation per active request. The values below are aggregate committed decode
throughput from complete one-second intervals in which the actual decode batch remained equal to
the configured concurrency. MTP acceptance is aggregated over the complete request wave. Each
concurrency cell reports `decode tok/s / MTP acceptance`; profiles should be read independently.

| Model profile | C=1 tok/s / accept | C=2 tok/s / accept | C=4 tok/s / accept | C=8 tok/s / accept | C8 / C1 |
|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% | 2.88× |
| Qwen3.6-27B `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% | 5.67× |
| Qwen3.6-35B-A3B `groupwise-int` | 593.0 / 67.2% | 877.7 / 68.2% | 1,166.0 / 69.8% | 1,313.8 / 67.3% | 2.22× |
| Qwen3.8-27B `nvfp4` | 143.8 / 48.9% | 267.6 / 48.1% | 461.1 / 45.8% | 766.6 / 46.0% | 5.33× |

At C=8, Qwen3.6-35B-A3B reaches **1,313.8 aggregate decode tok/s**. Qwen3.6-27B NVFP4 reaches
**1,146.9 tok/s** and **5.67×** its C=1 throughput. Qwen3.8-27B NVFP4 has **45.8–48.9%** MTP
acceptance, versus **67.2–71.4%** across the other measured profiles, so aggregate committed
throughput reflects both execution performance and speculative acceptance.

### Single-request serving

The single-request corpus was measured on the same GPU with INT8 group-64 KV cache, CUDA Graphs,
and a 1,024-token prefill chunk. Each reported fixture uses five fixed seeds after server warm-up.
Targets and weight profiles are reported independently rather than as cross-target comparisons.
Requests were submitted serially to a persistent server. The Qwen3.8-27B NVFP4 MTP0 results use the
same dedicated serial corpus runner as the Qwen3.6 profiles; its MTP3 results come from the C=1 point
of the fixed concurrent-corpus campaign documented in [Performance](docs/performance.md).

**Qwen3.6-35B-A3B**

- MTP0 at a 7,680-token prompt: **15,544.3 prefill tok/s** and **271.1 decode tok/s**.
- MTP0 at a 260,096-token prompt: **5,157.1 prefill tok/s** and **188.2 decode tok/s**.
- MTP3 long reasoning: **620.3–726.2 decode tok/s** with **72.7–82.8% acceptance**.
- MTP3 structured output: **770.9 decode tok/s**, **89.1% acceptance**, and **3.67 tokens/round**.

**Qwen3.6-27B (`groupwise-int`)**

- MTP0 at a 7,680-token prompt: **3,218.1 prefill tok/s** and **77.6 decode tok/s**.
- MTP0 at a 260,096-token prompt: **1,614.8 prefill tok/s** and **54.8 decode tok/s**.
- MTP3 long reasoning: **161.9–175.4 decode tok/s** with **73.4–78.8% acceptance**.
- MTP3 structured output: **193.0 decode tok/s**, **88.7% acceptance**, and **3.66 tokens/round**.

**Qwen3.6-27B (`nvfp4`)**

- MTP0 at a 7,680-token prompt: **11,191.5 prefill tok/s** and **86.4 decode tok/s**.
- MTP0 at a 260,096-token prompt: **2,510.6 prefill tok/s** and **59.9 decode tok/s**.
- MTP3 long reasoning: **213.1–231.0 decode tok/s** with **76.3–81.1% acceptance**.
- MTP3 structured output: **252.2 decode tok/s**, **89.8% acceptance**, and **3.69 tokens/round**.
- Against groupwise-int on the same corpus and runtime options: **3.48× the 7,680-token prefill
  throughput**, **1.55× the 260,096-token prefill throughput**, and **30–32% higher MTP3 decode
  throughput**.

**Qwen3.8-27B (`nvfp4`)**

- MTP0 at a 7,680-token prompt: **8,340.4 prefill tok/s** and **71.2 decode tok/s**.
- MTP0 at a 260,096-token prompt: **2,203.1 prefill tok/s** and **52.9 decode tok/s**.
- MTP3 long reasoning: **151.4–195.2 decode tok/s** with **56.2–76.0% acceptance**.
- MTP3 structured output: **219.8 decode tok/s**, **90.8% acceptance**, and **3.72 tokens/round**.

See [Performance](docs/performance.md) for the full methodology, variability, reproduction command,
and per-fixture results.

## Evaluation

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP=3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond |
|---|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% |

Both Qwen3.8-27B profiles are supported but have not yet been added to this published evaluation
campaign.

These are single-sample results under that NInfer evaluation profile, not pass@k. See the model
cards and [full performance document](docs/performance.md) for correct/total counts and evaluation
notes.

## Requirements

NInfer currently requires:

- 64-bit Linux;
- NVIDIA GeForce RTX 5090 (`sm_120a`);
- NVIDIA driver support for CUDA 13.1 and the CUDA Toolkit 13.1 or newer;
- CMake 3.28 or newer and a C++20-capable host compiler;
- `pkg-config`;
- FFmpeg development libraries: `libavformat >= 60`, `libavcodec >= 60`,
  `libavutil >= 58`, and `libswscale >= 7`;
- `libcurl >= 7.85`;
- Ninja, when using the commands below.

The build rejects CUDA architectures other than `120a`. There is no install target or packaged
binary distribution; NInfer is run from its source build tree.

## Build

```bash
git clone https://github.com/Neroued/ninfer.git
cd ninfer

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The default configuration builds:

```text
build/apps/ninfer
build/apps/ninfer-serve
```

Tests, benchmarks, and maintainer tools are excluded from the default build.

## Docker

Build the runtime image on a 64-bit Linux host with an RTX 5090, a CUDA 13.1-compatible NVIDIA
driver, Docker, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker build --tag ninfer:local .
```

Download a model into `models/` as described below, then run the HTTP server:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_6_27b.ninfer \
  --host 0.0.0.0
```

Run the CLI from the same image:

```bash
docker run --rm \
  --gpus '"device=0"' \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer /models/qwen3_6_27b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-new 256
```

## Download a model

Use the Hugging Face CLI to download one of the registered artifacts:

```bash
hf download neroued/Qwen3.6-27B-NInfer \
  qwen3_6_27b.ninfer \
  --local-dir models

# Or the 27B NVFP4 weight variant:
hf download neroued/Qwen3.6-27B-nvfp4-NInfer \
  qwen3_6_27b_nvfp4.ninfer \
  --local-dir models

# Or Qwen3.8-27B:
hf download neroued/Qwen3.8-27B-NInfer \
  qwen3_8_27b.ninfer \
  --local-dir models

# Or Qwen3.8-27B NVFP4:
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models

# Or:
hf download neroued/Qwen3.6-35B-A3B-NInfer \
  qwen3_6_35b_a3b.ninfer \
  --local-dir models
```

Current NInfer builds accept only the version-2 artifact container, and all five downloads above
are version 2. Migration applies only to Qwen3.6 artifacts downloaded before their version-2
publication; both Qwen3.8-27B profiles were published directly as version 2. Migrate an older exact
local file in place:

```bash
python3 -m tools.artifact.migrate_v1_to_v2 models/qwen3_6_27b.ninfer
```

Use the same command with `qwen3_6_27b_nvfp4.ninfer` or `qwen3_6_35b_a3b.ninfer` for those
artifacts. The migration updates only container metadata; it does not rewrite the weight payload.
Alternatively, download the current version-2 file again from its Hugging Face repository.

Each `.ninfer` file contains the weights and frontend resources needed by NInfer. It is not a
Transformers checkpoint, Safetensors distribution, or GGUF file.

Each artifact is complete, while GPU residency is fixed at process startup. Speculative decoding is
disabled by default, so MTP/DFlash state and the optimized proposal head are not uploaded.
Vision is also disabled by default, so its weights, Vision scratch phase, and frozen
request-transient allocation are omitted. Add `--vision` to the CLI or server process that must
accept image or video input. Disabled capabilities cannot be enabled by a later request. DFlash is
available only for the 35B-A3B target and is text-only.

## Run the CLI

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Use `--messages FILE` instead of `--prompt` for chat history, images, or videos:

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --vision
```

Answer content is written to stdout. Loading progress, reasoning, timing, throughput, memory, and
speculative-decoding statistics are written to stderr. See the [CLI guide](docs/cli.md) and
[committed examples](examples/cli/) for structured input and runtime options.

## Run the HTTP server

```bash
./build/apps/ninfer-serve models/qwen3_6_27b.ninfer \
  --max-context 16384 \
  --kv-capacity auto \
  --max-concurrency 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

The public model ID defaults to the artifact's `identity.model_id`; use `--model-id` only to
publish a deployment-specific alias.

Then send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

The server also implements OpenAI Responses Core (typed Items, semantic SSE, local continuation
state, and function calls) plus Anthropic Messages, token counting, and multimodal input. See
[HTTP serving](docs/serving.md).

## Capabilities

All three registered model IDs support:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill and CUDA Graph decode;
- startup-bounded small-scale concurrent serving with true batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16 and INT8 group-64 KV cache;
- model- and thinking-mode-aware official sampling defaults, with explicit greedy, temperature,
  top-k, top-p, min-p, and presence/frequency-penalty overrides;
- compatible-prefix reuse;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming and
  usage accounting;
- prompt-rendered function tools and parsed tool calls.

The 35B-A3B target additionally supports text-only DFlash speculative decoding with draft windows
from one to fifteen.

## Current limits

- Only the five `(model_id, weights_id)` artifact identities listed above are accepted product
  identities.
- Execution is specialized for one RTX 5090 and one CUDA device.
- One Engine owns one resident model and supports a startup-fixed capacity of 1–8 active requests.
  Decode-ready requests are compacted at round boundaries and executed in one batched model
  traversal.
- NInfer does not provide large-scale or preemptive continuous batching, priority/QoS scheduling,
  multi-GPU execution, CPU/GPU offload, or distributed serving.
- `--max-context` is the logical ceiling of each sequence and is configurable up to the registered
  models' native 262,144-token limit. `--kv-capacity N` explicitly sizes the shared Main Text KV
  pool for all active and retained sequences, while `--kv-capacity auto` selects the largest usable
  capacity from the memory remaining after weights are loaded while preserving 1 GiB of sizing
  headroom. Omission defaults to one `--max-context` worth of pages. The resolved pool is fixed at
  startup and is not divided statically among request lanes.
- Tool calls are parsed and returned to the client; NInfer does not execute tools.
- The C++ headers are used by the in-tree applications and are not distributed as an installed SDK.

## Documentation

- [Contributing](CONTRIBUTING.md)
- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [CLI examples](examples/cli/)

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
