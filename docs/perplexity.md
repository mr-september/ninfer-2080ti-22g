# Perplexity evaluation

`ninfer-perplexity` measures the causal perplexity produced by a v3 `.ninfer` artifact.
It uses the artifact's tokenizer, Text model, selected Main KV representation, final normalization,
and main output head. It is an offline evaluator, not a serving endpoint or a logits-export API.
Only Text weights and resources are loaded; Vision and speculative components are not required.

## Run the fixed corpus

The repository includes `ninfer-ppl-1m-v1`, a fixed set of 16 independent UTF-8 streams covering
English reference text, English long-form text, Chinese reference text, and NInfer C++/CUDA code.
`full` selects all streams; `--quick` selects one stream from each domain.

```bash
./build/turing-release/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --corpus eval/corpora/perplexity-1m/manifest.json \
  --quick \
  --kv-dtype int8
```

The default evaluation uses a 4,096-token context and a 2,048-token stride. Use `--context` and
`--stride` to change that protocol, or score one UTF-8 file with `--text FILE`. The available Main
KV representations are `bf16`, `int8`, `fp8`, `nvfp4`, and `k8v4`; `fp8`, `nvfp4` and `k8v4` are not available on Turing builds.

```bash
./build/turing-release/apps/ninfer-perplexity models/qwen3_8_27b.ninfer \
  --text notes.txt \
  --context 16384 --stride 8192 \
  --kv-dtype int8
```

Run `./build/turing-release/apps/ninfer-perplexity --help` for the complete command surface. The evaluator loads
the model once, reads and tokenizes every selected stream before scoring, and writes readable
startup, corpus, scoring, and per-stream summaries to stderr. Interactive weight loading and
scoring use one transient progress line; redirected scoring emits persistent progress every ten
seconds. `--log-level debug` exposes internal startup and stream-begin detail. The final
domain/overall table remains product output on stdout; the independent full-precision machine
report is `report.json` under `profiles/perplexity/` unless `--output` supplies an empty directory.

For KV-format comparisons, the recommended long-context profile is the full corpus with
`--context 65536 --stride 32768` and without `--quick`.

## Compare two builds token by token

`ninfer-logit-compare` scores fixed token sequences through the same `CausalScoring` Engine path, so
two builds can be compared at the log-probability level where greedy text only shows the first flip.
It renders one user message (optionally after one system message) from a `--messages` file with the
default chat template and thinking enabled, checks the rendered token count against
`Engine::count_tokens`, and prints one JSON object: the log-probability of each `--candidates` first
token (comma-separated texts, each one token) after the prompt, and the log-probabilities of the
`--ids` continuation given the prompt.

```bash
./build/turing-release/apps/ninfer-logit-compare models/qwen3_8_27b.ninfer \
  --messages examples/cli/messages/long_decode_aime26_30.json \
  --candidates We,Let --ids "1596 1144 310" --context 8192 --kv-dtype bf16
```

Take the ids from `ninfer --greedy --print-token-ids` of one build, score them with both builds'
binaries, and compare the two `continuation_logprobs` arrays. `--kv-dtype` accepts `bf16` and `int8`.
The final logits are BF16, so a logit moves in whole steps (0.125 at magnitude 16 to
32) and a top-2 gap is best judged in steps, not in nats. The rule used to accept the Turing port is in
"Port onto upstream dev" in [turing-port.md](turing-port.md).

## Metric

For a stream `x[0..N)`, every token after `x[0]` is scored exactly once. A window `[b,e)` with target
suffix `[s,e)` contributes:

```text
log p(x[i] | x[b], ..., x[i-1])  for i in [s,e)
```

Each window starts from empty State and Main KV, so history before `b` is deliberately excluded.
The reported metric is therefore fixed-window, truncated-context causal perplexity:

```text
mean_nll = -sum(logprob) / scored_tokens
perplexity = exp(mean_nll)
```

The first window scores `[1,min(context,N))`. Each later window advances by `stride` targets while
retaining up to `context-stride` preceding tokens as local context. Streams never share history.

## Comparing runs

For a numerical comparison, keep the corpus, context, stride, and execution settings fixed except
the variable being measured. Compare KV formats with the same artifact and weight formats with the
same KV format.

The corpus name is a workload scale, not an exact token count. Exact input and scored-token counts
are runtime results from the current artifact tokenizer and are recorded in each report. Reports
contain unrounded NLL/PPL values for every window, stream, domain, and the token-weighted overall
aggregate.

The schema-v2 report identifies the artifact's architecture, public name, actual weight formats
and prefill signature alongside the workload and numerical results.
