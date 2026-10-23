# AGENTS.md

## Product and architecture

This repository is a fork of Neroued/ninfer, cut down to one product: a C++/CUDA inference engine for dense Qwen3.8-27B (`Qwen3_5ForCausalLM`, `groupwise-int` Q4/Q5/Q6/Q8 and BF16 weights) on one NVIDIA RTX 2080 Ti with 22 GB (Turing, `sm_75`), built natively on Windows 11 and on Linux/WSL2/Docker. It keeps vision input, MTP, DFlash and DFlash2 speculative decoding, Qwen3.6-27B support, the CLI, the OpenAI/Anthropic-compatible server, `ninfer-perplexity` and `ninfer-logit-compare`.

The build targets Turing only (`sm_75`); see [Build system](docs/maintainer/build-system.md). The tree is being stripped of everything this product does not use: NVFP4 and K8V4, the FP8 kernels and the never-built Blackwell kernel files. The Qwen4Exp, PLE, NF4 and MoE model-side code and the FP8 formats are kept, without GPU kernels for Turing, in anticipation of the Qwen4 series; they do not run on a 2080 Ti today. Until the strip finishes, files and docs may still describe the removed features; they are not part of the product, and new work must not extend them.

V3 `.ninfer` is the only C++ product artifact. Generation, offline CausalScoring, CLI, serving and inference benchmarks use the public Engine. Model data is immutable; every Program owns its mutable state and device allocations. Runtime owns execution and publication policy; product and serving own input acquisition and protocol translation.

Keep architecture, binding and execution explicit. Do not introduce generic model graphs, family base classes, plugin discovery, string-driven execution, hidden device allocation or runtime weight repacking.

This is a local, single-owner project with trusted local models, artifacts and workflows. Do not derive requirements from another deployment or trust model.

## Change policy

There is no more upstream sync. Project-owned contracts do not preserve backward compatibility: replace behavior completely and remove superseded aliases, fallbacks and transition branches, together with their tests and docs. Advertised OpenAI and Anthropic protocols are external contracts; update schema tests and serving documentation together. Update stable requirements in their existing authoritative document.

Work directly on `master` and push to `origin master`; never create branches or worktrees. Use Conventional Commit subjects with lowercase types. One concern per commit: a commit deletes, or moves/renames (`git mv`), or edits, never two of these.

## The strip rules

The strip deletes what the product does not use **without changing a bit of what it computes**. The plan lives in the untracked `docs/plans/strip-roadmap.md`. Every strip change is held to exact equality, never "within noise": the build compiles, CTest passes (minus deleted tests), protocol B perplexity and `ninfer-logit-compare` are bit-identical to the baseline, greedy CLI output and the startup memory plan are identical, and per-kernel SASS hashes of surviving kernels are identical, compared on the compiled device objects (the executable-level hashes also move when the device linker lays out shared memory for a smaller object set). The checks run on a cadence: every commit passes the cheap static checks (preprocessed device-pass TU identity of the affected translation units, the compile-database comparison, and `git show -M --stat` for moves; a commit that edits host logic also runs its targeted unit test); each sub-batch of a stage adds one release build and the full TU-identity sweep; each stage ends with one dev build and CTest, `ninfer-logit-compare` and the SASS hashes; perplexity, the A/B smoke and the Docker Linux CTest run only at the stages the roadmap names. Do not loosen a gate, re-baseline or call a difference noise; report it. A commit that removes a feature also removes every doc, help text, test and CMake reference to it.

## Verification and reporting

Before changing Ops, numerical or state semantics, read [Op development](docs/maintainer/op-development.md). Qualify affected production routes directly against the contract's independent mathematical oracle or specified exact reference. Kernel parity and plausible model output do not establish mathematical correctness; perplexity does.

Measure performance at the claimed scope, and establish the baseline, workloads, metrics and acceptance criteria before evaluating results. Report the baseline, hardware/toolchain, commands, run conditions, metrics, coverage, result distribution and worst changes. Include small and unexplained regressions. The standard measurements and acceptance rules are in [Turing port](docs/turing-port.md#standard-measurements-and-acceptance-rules).

## References

Read [Engine architecture](docs/maintainer/engine-architecture.md) before changing execution or ownership. [README](README.md) and executable `--help` define capabilities and exact commands. The [documentation map](docs/README.md) routes to detailed contracts; [Tests](tests/README.md) and [Benchmarks](bench/README.md) own their commands.

## Local environment

Windows 11, PowerShell, Visual Studio 2022 Build Tools, CUDA 12.9, vcpkg at `E:\vcpkg`. Build with `scripts\build.ps1` (release, `build\turing-windows`) or `scripts\build.ps1 -Dev` (tests and benchmarks, `build\turing-windows-dev`); set `CMAKE_BUILD_PARALLEL_LEVEL=4`, because nvcc runs out of memory at the default. Run tests with `PYTHONUTF8=1` and `ctest --test-dir build\turing-windows-dev`. Python is `C:\Users\Larry\AppData\Local\Python\pythoncore-3.10-64\python.exe`.

The usual local model is `E:/AI_Models/LLM/neroued/Qwen3.8-27B-NInfer/qwen3_8_27b.ninfer` (v3, `groupwise-int`). Select artifacts by explicit path, never glob order, modification time or "latest". The model needs about 18 GB of VRAM; check that the desktop holds under about 3 GB and never run two GPU jobs at once.

Store all agent-generated temporary files under the repository-root `.agent-tmp/<task>/` (ignored by Git; do not commit it): plans, experimental code and binaries, test data, logs, profiling captures and reports. Remove what is no longer needed when the task ends. Reserve `out/` for model weights, companion files and explicitly retained artifacts.
