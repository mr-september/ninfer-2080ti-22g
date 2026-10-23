# NInfer CLI

`build/turing-release/apps/ninfer` (`build\turing-windows\apps\ninfer.exe` on native Windows) runs one request against one v3 `.ninfer` artifact. Build NInfer and download an artifact using the [project README](../README.md) before following this guide.

The examples use the Qwen3.8-27B `groupwise-int` artifact with this build's INT8 KV storage, which is the default on Turing, so no `--kv-dtype` flag is needed.

> **Turing fork.** Upstream's forms of these commands target an RTX 5090 (`--kv-dtype fp8`); FP8 is not available on a GeForce RTX 2080 Ti. The defaults of this build are INT8 KV, a 65,536-token context, and MTP with three draft tokens and the optimized proposal head when the artifact has one; see the [README](../README.md#defaults).

The CPU-expert-offload (`NINFER_VRAM_BUDGET_MIB`) examples below are upstream's. They need CPU expert offload or FP8 weights or KV, none of which the `turing` target provides (`NINFER_EXPERT_OFFLOAD` is 0 there), so they cannot run on a Turing GPU.

For Qwen3.6-35B-A3B, `NINFER_VRAM_BUDGET_MIB` caps the Engine's GPU allocation budget. If the
model cannot remain fully on GPU, Main MoE experts use CPU/GPU execution with segmented CUDA
Graph decode:

```bash
NINFER_VRAM_BUDGET_MIB=16384 ./build/turing-release/apps/ninfer out/qwen3_6_35b_a3b.ninfer \
  --prompt "Explain how a compiler works." --max-context 8192 --max-new 128
```

`--prefill-chunk N` sets a fixed positive multiple of 128. When omitted, the default is 4096 for
Qwen4Exp or CPU expert offload, and 1024 otherwise. Serving and offline scoring use the same Engine
defaults. See [expert execution](maintainer/expert-weight-cache.md) for ownership and scheduling.

## Text input

```bash
./build/turing-release/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Summarize the difference between prefill and decode." \
  --max-context 32768 \
  --max-new 8192 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Exactly one of `--prompt` and `--messages` is required. The CLI normally omits `--kv-capacity`, so the shared Main Text KV pool follows the example's 32,768-token `--max-context`.

Answer content is streamed to stdout. Tool calls supplied through `--messages` also stream to stdout as `{"name":"...","arguments":...}` records, with argument bytes appearing as they are generated. An interrupted record retains its partial JSON and reports the interruption on stderr; it is not repaired into a complete call. Human-readable startup milestones and runtime errors are written to stderr without service timestamps. Reasoning and the CLI result report (timings, throughput, GPU memory, token IDs when requested, and speculative-decoding statistics) also use stderr as unprefixed product output, so stdout can be redirected independently. On a terminal, weight materialization is one transient progress line followed by a compact Engine-ready summary. Redirected stderr contains persistent readable progress for long loads and no carriage returns or ANSI escapes. `--log-level debug` exposes every startup phase. Option and local prompt/message input failures remain direct command diagnostics:

```bash
./build/turing-release/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Return one sentence." \
  --max-context 4096 \
  --max-new 64 \
  > answer.txt 2> run.log
```

By default prompts are rendered with the vendored [Sharp v22.5.0 template](../tools/chat_templates/qwen3_8_sharp_v22_5.jinja), embedded in the executables at build time, so they run without the source tree (it adds about 150 prompt tokens to the system block and steers answers to be terse). `--chat-template FILE` uses a different local Jinja file, and `--artifact-chat-template` uses the template stored in the artifact; the two cannot be combined. Changes to a template file take effect after restarting NInfer:

```bash
./build/turing-release/apps/ninfer models/qwen3_8_27b.ninfer \
  --chat-template tools/chat_templates/qwen3_8.jinja --prompt "Hello"
./build/turing-release/apps/ninfer models/qwen3_8_27b.ninfer --artifact-chat-template --prompt "Hello"
```

The Sharp template takes `{"terse": false}` through `chat_template_kwargs` (server requests only) to drop its terse-answer instruction.

`--grammar-file FILE` constrains the answer with a GBNF grammar whose entry rule is `root`:

```bash
printf 'root ::= "yes" | "no"\n' > answer.gbnf
./build/turing-release/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Is 17 prime?" --no-thinking --grammar-file answer.gbnf --max-new 64
```

`--choice TEXT` selects a literal candidate; repeat the flag to supply the candidate set. `--regex PATTERN` constrains the complete answer to a regular expression:

```bash
./build/turing-release/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Classify this review: the service was excellent." --no-thinking \
  --choice positive --choice neutral --choice negative

./build/turing-release/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Return ticket TASK-0042." --no-thinking --regex '(BUG|TASK)-[0-9]{4}'
```

Candidates preserve exact text; `--choice ''` explicitly permits empty content. `--regex ''` permits only empty content. See the [language contract](maintainer/constrained-decoding.md#41-gbnf--regex--choice) for supported regex syntax.

`--json-object` constrains the answer to a JSON object. `--json-schema-file FILE` applies a JSON Schema. All output constraint options are mutually exclusive:

```bash
printf '%s\n' '{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"],"additionalProperties":false}' > answer.schema.json
./build/turing-release/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Return the answer to 6 times 7 as JSON." --no-thinking \
  --json-schema-file answer.schema.json --max-new 64
```

GBNF supports recursive rules, Unicode character classes and repetition. All constraint modes work with ordinary decoding, MTP, DFlash and DFlash2. Thinking may precede the constrained answer; an output limit or cancellation can leave it incomplete. JSON modes can accompany tools supplied in messages: the answer is either JSON or a tool-call sequence. GBNF, choice and regex require no active tools. Constraints reject custom stops and `--raw-output`. JSON uses compact separators and declared property order. Describe the desired content in the prompt; the schema is not added to it automatically. See the [supported schema subset](maintainer/constrained-decoding.md#42-json-与-schema-的执行合同).

Omitted thinking and effort options use the selected template's defaults. `--no-thinking` or `--reasoning-effort none` requests disabled thinking; other effort values cannot be combined with `--no-thinking`. The template interprets the selected effort. `--greedy` selects exact argmax decoding independently.

`--thinking-budget N` places a positive upper bound on accepted model-origin tokens while the new-turn Qwen thinking block remains open. If the model has not emitted `</think>` at that exact boundary, Engine appends [Qwen's canonical early-close guidance](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md) and `</think>` to the same resident sequence without sampling, publishes the guidance through the reasoning stream, then resumes ordinary generation from the updated context. A natural thinking close, stop condition, cancellation, or total output/context limit at the boundary takes priority and suppresses this insertion. The option cannot be combined with `--no-thinking`, but it can be combined with `--reasoning-effort`.

`--max-new` counts every committed generated token, including internally inserted control tokens. When the effective output capacity extends beyond the thinking budget, it must have room for the complete tokenizer-derived control suffix plus one post-close model token; an undersized request is rejected rather than truncating the suffix. Normal output sends the inserted guidance to stderr as reasoning. `--print-token-ids` includes the inserted IDs, while `--raw-output` preserves the raw control representation.

For example, this allows at most 512 model-origin thinking tokens while retaining enough total output capacity for the inserted suffix and the answer:

```bash
./build/turing-release/apps/ninfer models/qwen3_8_27b.ninfer \
  --prompt "Explain speculative decoding, then give a concise conclusion." \
  --max-context 4096 \
  --max-new 1024 \
  --thinking-budget 512 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

## Startup memory profile

GPU residency is frozen when the Engine starts:

- `--spec none` omits MTP/DFlash/DFlash2 weights and state and the optimized proposal head (the executable default is `--spec mtp --draft-tokens 3`);
- `--spec mtp`, `--spec dflash` (35B-A3B), and `--spec dflash2` (Qwen3.8-27B) load only the selected speculative backend;
- a speculative backend with the full proposal head omits the optimized proposal head;
- Vision is disabled by default, omitting its weights and Vision-specific unified-workspace extent;
- `--vision` loads the weights, expands the one Program workspace for Vision encode/handoff, and enables image/video input.
- the one-request CLI disables cross-request history and Host context backing, so it does not reserve an extra Device checkpoint StateImage or retain continuations.

The complete `.ninfer` inventory is still validated. These choices are not lazy loading: an Engine started without Vision rejects media and cannot enable Vision later. DFlash/DFlash2 and Vision may be enabled together; these backends apply to generated-text decode after multimodal prefill and does not accelerate Vision encode. The default speculative and Vision settings produce the smallest resident profile.

## Structured messages

`--messages` accepts either a non-empty JSON message array or an object containing `messages` and an optional `tools` array.

```json
[
  {
    "role": "system",
    "content": "Answer concisely."
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image",
        "image": "examples/cli/media/visual_chart.png"
      },
      {
        "type": "text",
        "text": "Describe the chart."
      }
    ]
  }
]
```

Run message files from the repository root when they contain repository-relative media paths:

```bash
./build/turing-release/apps/ninfer models/qwen3_8_27b.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --vision \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Supported roles are `system`, `developer`, `user`, `assistant`, and `tool`. The selected template formats these roles. The maintained Qwen templates keep system/developer messages at their input positions.

Message content may be a string or an ordered array containing:

| Content type | Source field | Accepted source |
|---|---|---|
| text | `text` | string |
| image / image_url | `image` or `image_url` | local path, HTTP(S) URL, or base64 data URI |
| video / video_url | `video` or `video_url` | local path, HTTP(S) URL, or base64 data URI |

`image_url` and `video_url` may be strings or objects containing a string `url`. Assistant history may include `reasoning_content` and `tool_calls`; a tool result uses role `tool` and `tool_call_id`.

See [`examples/cli/`](../examples/cli/) for committed text, image, video, mixed-media, thinking, long-decode, and long-context inputs.

## Speculative decoding

Speculative decoding defaults to MTP with three draft positions and, when the artifact carries one, the optimized proposal head; `--spec none` turns it off. MTP takes one to five draft positions; the alternatives are the
35B-A3B DFlash or Qwen3.8-27B DFlash2 backend with one to fifteen. Both masked-draft backends
may be combined with `--vision`.
`--lm-head-draft` selects the optimized proposal head when the architecture and artifact provide it,
and requires a selected backend.

Flash Next supports native MTP with the full head only; it does not support `--lm-head-draft`,
DFlash or DFlash2.

```bash
./build/turing-release/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 \
  --max-new 512 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

For DFlash:

```bash
./build/turing-release/apps/ninfer models/qwen3_6_35b_a3b.ninfer \
  --prompt "Write a short explanation of speculative decoding." \
  --max-context 16384 --max-new 512 \
  --spec dflash --draft-tokens 7 --lm-head-draft
```

For Qwen3.8-27B artifacts containing the DFlash2 companion weights, select `--spec dflash2 --draft-tokens 7`, optionally with `--lm-head-draft` and `--vision`. DFlash2 accepts every draft count from 1 through 15; seven is the checkpoint recommendation. The `groupwise-int` artifact uses the same Engine route, including CUDA Graph, concurrent requests, sampling penalties, and prefix reuse. An artifact without the companion weights reports a missing DFlash2 component when selected. Vision, MTP and DFlash follow the same rule: their weights are required only when that component is enabled at startup.

Only one speculative backend can be enabled per Engine. The measured configurations in [Turing port](turing-port.md#current-performance) use MTP with three draft tokens and DFlash with seven draft tokens (block length eight), both with the optimized proposal head. DFlash accepts one to fifteen draft tokens; seven forms the measured block length eight, while fifteen uses the maximum supported block length sixteen.

## Common options

The table lists the executable defaults of this build.

| Option | Meaning | Default |
|---|---|---:|
| `--max-context N` | per-sequence logical context ceiling; when omitted and the default does not fit in free VRAM, startup retries at `32768`, then `16384`, and says so; an explicit value is never lowered | `65536` |
| `--kv-capacity N\|auto` | explicit shared Main Text KV capacity, or maximize it from remaining GPU memory; omitted means `--max-context` | `65536` |
| `--prefill-chunk N` | positive text-prefill chunk, in multiples of 128 | `1024` |
| `--max-new N` | requested output-token limit | `128` |
| `--device N` | CUDA device index | `0` |
| `--kv-dtype bf16\|int8\|fp8` | KV-cache storage | `int8` on Turing builds, `bf16` otherwise |
| `--spec none\|mtp\|dflash\|dflash2` | speculative backend | `mtp` |
| `--draft-tokens N` | MTP `1..5`; DFlash/DFlash2 `1..15` (DFlash needs it explicitly) | `3` |
| `--lm-head-draft` / `--no-lm-head-draft` | force the optimized or the full proposal head | automatic: optimized when the artifact has one |
| `--vision` | enable image/video input and load Vision GPU allocations | off |
| `--vision-max-item-tokens N` | with `--vision`, largest single image or video item in merged tokens (`256..16384`); larger media is downscaled by the processor's resize rule, and a smaller value shrinks the Vision workspace | `16384` |
| `--no-cuda-graph` | disable CUDA Graph decode | graphs on |
| `--chat-template FILE` | use a local Jinja template | vendored Sharp v22.5.0 |
| `--artifact-chat-template` | use the template stored in the artifact | off |
| `--no-thinking` | disable thinking | template default |
| `--thinking-budget N` | positive model-origin thinking-token cap; omitted means unlimited | unset |
| `--reasoning-effort none\|minimal\|low\|medium\|high\|xhigh\|max` | pass an effort value to the selected template | template default |
| `--greedy` | exact argmax decoding | off |
| `--temperature F` | sampling temperature override | registered model/mode default |
| `--top-p F` | nucleus-threshold override | registered model/mode default |
| `--top-k N` | top-k-threshold override (`0..20`; zero selects the top-20 cap) | registered model/mode default |
| `--min-p F` | min-p-threshold override | registered model/mode default |
| `--presence-penalty F` | presence-penalty override | registered model/mode default |
| `--frequency-penalty F` | frequency-penalty override | registered model/mode default (`0`) |
| `--seed N` | sampling seed | `0` |

When a sampling flag is omitted, Engine selects the general-task preset for the loaded architecture and rendered prompt mode. The current official models use:

| Model | Prompt mode | Temperature | Top-p | Top-k | Min-p | Presence penalty |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.6-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.8-27B | thinking | `1.0` | `0.95` | `20` | `0` | `0` |
| Qwen3.8-27B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | thinking | `1.0` | `0.95` | `20` | `0` | `1.5` |
| Qwen3.6-35B-A3B | non-thinking | `0.7` | `0.80` | `20` | `0` | `1.5` |

Frequency penalty is `0` in every registered preset. Task-specific profiles such as Qwen's precise-coding profile use explicit sampling overrides.

Flash Next reads temperature, top-p, top-k and min-p from the artifact's `generation_config.json`,
using the same preset for thinking and non-thinking. Explicit sampling flags override these values.

Repeat `--stop-token-id`, `--stop`, or `--reasoning-stop` to add stop conditions. Use
`--raw-output` to expose the frontend's raw output stream and `--print-token-ids` to include
generated token IDs in diagnostics.

Run `./build/turing-release/apps/ninfer --help` for the exact option contract.

## CUDA synchronization

`NINFER_CUDA_SYNC` selects the CUDA device synchronization schedule at startup for both the CLI and HTTP server. When unset, it defaults to `spin`, prioritizing low synchronization latency at the cost of CPU usage while waiting for the GPU. Use `blocking` to let the waiting thread sleep; the decode performance cost depends on the host. `yield` yields the CPU while waiting, and `auto` uses CUDA's scheduling heuristic, not an automatic performance benchmark.

```bash
NINFER_CUDA_SYNC=blocking ./build/turing-release/apps/ninfer models/qwen3_8_27b.ninfer --prompt "Hello"
```

The Engine-ready log reports the selected mode. Empty or unrecognized values, or failure to apply the schedule, fail startup. This controls device scheduling (including stream synchronization); it does not override individual CUDA event creation flags.

## Context and memory

The official artifacts have a native context limit of 262,144 tokens. The practical allocation
on one GPU depends on the selected artifact, media workload, output budget, and KV-cache type.
The artifact describes its model configuration and weight representations;
`--kv-dtype` independently selects runtime KV storage. The prepared prompt must fit
`--max-context`; generation stops at the remaining context capacity when necessary.
`--kv-capacity N` controls the shared physical Main Text KV pool independently and is rounded up to
the 64-token page size. Startup plans the selected weights and complete enabled runtime layout
from artifact metadata before allocating their backing. With resident weights,
`--kv-capacity auto` chooses the largest legal page capacity within the available Device budget,
including the selected speculative backend, fixed sequence state, unified workspace and
measured CUDA Graph rebuilding cost. Both explicit and automatic capacity use 1 GiB of base startup
planning headroom.
Startup calibration releases its temporary storage before sizing the final pools;
retained driver memory is already included in the remaining free memory. With CPU expert
offload, `auto` reserves one full context and assigns the remaining budget to expert storage;
set an explicit capacity when more shared KV is needed. Graph calibration runs only at startup;
the pool stays fixed while serving requests. The single-request
CLI normally leaves the option omitted so it follows
`--max-context`; the distinction matters primarily to a concurrent Engine or server.

At Engine startup NInfer reserves model weights, persistent sequence state, one phase-reused
Program workspace, and CUDA Graph resources sized through calibration. With Vision enabled, that one
workspace contains a general execution prefix and a fixed item-output handoff region. Vision encode
may reuse the full backing before producing the output; Text/MTP/decode work remains inside the
general prefix while the handoff is live. The capacity is therefore the maximum legal simultaneous
extent, not the sum of Text, Vision scratch, and Vision output allocations. Text prefill uses
`min(--prefill-chunk,--max-context)`; Vision keeps the existing 32,768-token aggregate prompt budget
but plans Device execution for the single-item extent `min(--max-context, --vision-max-item-tokens)`, at most the registered 16,384 tokens (a 4096x4096 image). The encode peak scales with that extent, while the handoff and general prefix set a floor; lowering `--vision-max-item-tokens` trades maximum media resolution for Device memory. Media that would exceed the extent is downscaled before encoding, or rejected when the request sets `oversized_image` to `error`. Requests perform no
project-owned device allocation or growth. Context-cache capacity controls are intentionally absent
from this one-request interface; the persistent Engine and server routes own cross-request reuse and
optional Host backing.

All weight, sequence, workspace, and graph allocations are released when the Engine is destroyed.
