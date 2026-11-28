// inkling2_ppl: teacher-forced perplexity over a token-id stream through the M3 executor.
//
// Usage: inkling2_ppl <artifact-dir> <ids-file> [--context N] [--stride N] [--output file]
// <ids-file> holds whitespace-separated token ids (no text tokenization here: the pinned
// parity corpus ids are fed directly, so port and donor runs score the identical stream).
// Windowing and scoring mirror apps/perplexity/evaluation.cpp exactly (shared source):
// first window scores tokens[1..context), later windows score only their stride-new suffix,
// each window is an independent sequence scored at LOCAL positions (donor engine.score_tokens
// semantics, ninfer-all apps/perplexity/main.cpp:607).
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/schema.h"
#include "core/device.h"
#include "models/common/bindings.h"
#include "models/inkling2/execution/executor.h"
#include "models/inkling2/load/bindings.h"
#include "models/inkling2/port_config.h"
#include "../perplexity/evaluation.h"

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;

static void stage(const char* s) { std::fprintf(stderr, "[ppl] %s\n", s); std::fflush(stderr); }

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    try {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: inkling2_ppl <artifact-dir> <ids-file> [--context N] [--stride N] "
                     "[--output file]\n");
        return 2;
    }
    const std::filesystem::path artifact = argv[1];
    std::uint32_t context                = 4096;
    std::uint32_t stride                 = 2048;
    std::uint32_t unpadded_override      = 0;
    std::uint64_t cap_targets            = 0;  // 0 = no cap (all windows)
    std::uint64_t trace_targets          = 0;  // print per-target nll for the first N
    std::string output;
    for (int i = 3; i < argc; ++i) {
        const std::string option = argv[i];
        const auto value         = [&] {
            if (i + 1 >= argc) { throw std::invalid_argument("missing value for " + option); }
            return std::string(argv[++i]);
        };
        if (option == "--context") { context = std::stoul(value()); }
        else if (option == "--stride") { stride = std::stoul(value()); }
        else if (option == "--unpadded-vocab") { unpadded_override = std::stoul(value()); }
        else if (option == "--cap-targets") { cap_targets = std::stoull(value()); }
        else if (option == "--trace") { trace_targets = std::stoull(value()); }
        else if (option == "--output") { output = value(); }
        else { throw std::invalid_argument("unknown option: " + option); }
    }

    std::ifstream in(argv[2]);
    std::vector<std::uint32_t> tokens;
    for (long long t; in >> t;) { tokens.push_back(std::uint32_t(t)); }
    if (tokens.size() < 2) { throw std::invalid_argument("ids stream shorter than two tokens"); }
    const auto windows = perplexity::plan_windows(tokens.size(), context, stride);
    std::fprintf(stderr, "[ppl] %zu tokens | %zu windows | context %u stride %u\n",
                 tokens.size(), windows.size(), context, stride);
    std::fflush(stderr);

    stage("device");
    DeviceContext device(0);
    device.bind_to_current_thread();
    stage("reader");
    artifact::Reader reader(artifact);
    // Transient shim: the v3 artifacts predate the converter's unpadded_vocab_size member, so
    // the PPL gate may supply the cut until the artifact is re-converted.
    models::inkling2::Config config = models::inkling2::config_from_directory(reader.directory());
    if (unpadded_override != 0) { config.unpadded_vocab_size = unpadded_override; }
    stage("binder/bind");
    artifact::Binder binder(reader);
    models::Bindings bindings(binder);
    const auto weights = models::inkling2::loading::bind_text(bindings, config);
    (void)weights;
    stage("materialize");
    std::vector<artifact::PlacementRequest> requests;
    for (const auto& pending : bindings.weights) {
        for (const auto& part : pending.reference.binding.parts) {
            requests.push_back({part.object, device.device(), artifact::HostAccess::CpuOnly});
        }
    }
    auto plan       = artifact::plan_materialization(reader, requests);
    auto materialized =
        artifact::materialize(reader, std::move(plan), device.transfer_view(), nullptr);
    std::vector<Device> devices(bindings.weights.size(), device.device());
    auto bound = models::resolve_weights(std::move(bindings.weights), materialized, devices);
    stage("executor");
    models::inkling2::execution::Executor executor(config, bound, device, device.stream(), context);

    // The donor head is cut to the unpadded vocabulary; executor.logits() honors that cut.
    std::vector<__nv_bfloat16> logits(std::size_t(config.unpadded_vocab_size));
    std::vector<float> scores;
    perplexity::ScoreAggregate aggregate;
    const auto started = std::chrono::steady_clock::now();

    for (std::size_t w = 0; w < windows.size(); ++w) {
        const auto& window = windows[w];
        auto session = executor.make_session();  // independent sequence per window
        scores.clear();
        // Step i consumes tokens[input_begin + i] and its logits predict the global token
        // input_begin + i + 1. The last input token's logits are never used.
        for (std::size_t i = window.input_begin; i + 1 < window.input_end; ++i) {
            executor.decode_step(*session, tokens[i], std::int64_t(i - window.input_begin));
            const std::size_t next = i + 1;
            if (next < window.target_begin || next >= window.target_end) { continue; }
            cudaMemcpy(logits.data(), executor.logits(), logits.size() * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
            float max_logit = -1e30f;
            for (const auto l : logits) { max_logit = std::max(max_logit, __bfloat162float(l)); }
            double sum_exp = 0.0;
            for (const auto l : logits) {
                sum_exp += std::exp(double(__bfloat162float(l)) - double(max_logit));
            }
            const double target = double(__bfloat162float(logits[tokens[next]]));
            const double nll    = -(target - double(max_logit) - std::log(sum_exp));
            if (aggregate.scored_tokens + scores.size() < trace_targets) {
                std::fprintf(stderr, "[trace] target %zu nll %.4f\n", next, nll);
            }
            // ScoreAggregate consumes logprobs (evaluation.h contract): push -nll.
            scores.push_back(float(-nll));
            if (cap_targets != 0 && aggregate.scored_tokens + scores.size() >= cap_targets) { break; }
        }
        aggregate.add(scores);
        const double done_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::fprintf(stderr,
                     "[ppl] window %zu/%zu | scored %llu tokens | running ppl %.4f | %.0f s\n",
                     w + 1, windows.size(), static_cast<unsigned long long>(aggregate.scored_tokens),
                     aggregate.ppl(), done_seconds);
        std::fflush(stderr);
        if (cap_targets != 0 && aggregate.scored_tokens >= cap_targets) { break; }
    }

    const double mean_nll = aggregate.mean_nll();
    const double ppl      = aggregate.ppl();
    std::printf("ppl %.6f  mean_nll %.6f  scored_tokens %llu\n", ppl, mean_nll,
                static_cast<unsigned long long>(aggregate.scored_tokens));
    if (!output.empty()) {
        std::ofstream out(output, std::ios::trunc);
        out << "{\n  \"ppl\": " << ppl << ",\n  \"mean_nll\": " << mean_nll
            << ",\n  \"scored_tokens\": " << aggregate.scored_tokens << ",\n  \"context\": " << context
            << ",\n  \"stride\": " << stride << "\n}\n";
    }
    return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ppl] EXCEPTION: %s\n", e.what());
        return 1;
    }
}
