// inkling2_probe: greedy-decode a token-id prompt through the lean M3 engine standalone.
//
// Usage: inkling2_probe <artifact-dir> <prompt-file> <steps> [--tie <file>]
// <prompt-file> holds whitespace-separated token ids; the last `steps` outputs (plus the
// first generated id) print one per line with the top logit. Oracle parity compares these ids.
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/schema.h"
#include "core/device.h"
#include "models/common/bindings.h"
#include "models/inkling2/execution/executor.h"
#include "models/inkling2/load/bindings.h"
#include "models/inkling2/port_config.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace ninfer;

static void stage(const char* s) { std::fprintf(stderr, "[probe] %s\n", s); std::fflush(stderr); }

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    try {
    if (argc < 4) {
        std::fprintf(stderr, "usage: inkling2_probe <artifact-dir> <prompt-file> <steps>\n");
        return 2;
    }
    const std::filesystem::path artifact = argv[1];
    const int steps                      = std::atoi(argv[3]);

    std::ifstream in(argv[2]);
    std::vector<std::uint32_t> prompt;
    if (argc >= 5 && std::string(argv[4]) == "--manifest") {
        // Batch mode: <prompt-file> is a manifest; each line "<id> <steps> <tok> <tok>...".
        // steps==0 on a line means: use argv[3] as the step count.
        struct Job { std::string id; int steps; std::vector<std::uint32_t> toks; };
        std::vector<Job> jobs;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') { continue; }
            std::istringstream ls(line);
            Job j; ls >> j.id >> j.steps;
            for (long long t; ls >> t;) { j.toks.push_back(std::uint32_t(t)); }
            if (j.toks.empty()) { std::fprintf(stderr, "manifest line without tokens: %s\n", j.id.c_str()); return 2; }
            if (j.steps <= 0) { j.steps = steps; }
            jobs.push_back(std::move(j));
        }
        if (jobs.empty()) { std::fprintf(stderr, "empty manifest\n"); return 2; }

        stage("device");
        DeviceContext device(0);
        device.bind_to_current_thread();
        stage("reader");
        artifact::Reader reader(artifact);
        const models::inkling2::Config config = models::inkling2::config_from_directory(reader.directory());
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
        auto plan = artifact::plan_materialization(reader, requests);
        auto materialized =
            artifact::materialize(reader, std::move(plan), device.transfer_view(), nullptr);
        std::vector<Device> devices(bindings.weights.size(), device.device());
        auto bound = models::resolve_weights(std::move(bindings.weights), materialized, devices);
        stage("executor");
        models::inkling2::execution::Executor executor(config, bound, device, device.stream(), 4096);

        std::vector<__nv_bfloat16> logits(std::size_t(config.vocab_size));
        for (const auto& job : jobs) {
            auto session = executor.make_session();   // fresh state per prompt
            for (std::size_t i = 0; i < job.toks.size(); ++i) {
                executor.decode_step(*session, job.toks[i], std::int64_t(i));
            }
            cudaStreamSynchronize(device.stream());
            std::printf("=== %s %zu\n", job.id.c_str(), job.toks.size());
            std::uint32_t next = 0;
            for (int s = 0; s < job.steps; ++s) {
                cudaMemcpy(logits.data(), executor.logits(), logits.size() * 2, cudaMemcpyDeviceToHost);
                int best = 0;
                float bestv = -1e30f;
                for (std::size_t i = 0; i < logits.size(); ++i) {
                    const float v = __bfloat162float(logits[i]);
                    if (v > bestv) { bestv = v; best = int(i); }
                }
                next = std::uint32_t(best);
                std::printf("%u %.4f\n", next, bestv);
                if (s + 1 < job.steps) {
                    executor.decode_step(*session, next, std::int64_t(job.toks.size() + s));
                    cudaStreamSynchronize(device.stream());
                }
            }
            std::fflush(stdout);
        }
        stage("done");
        return 0;
    }
    for (long long t; in >> t;) { prompt.push_back(std::uint32_t(t)); }
    if (prompt.empty()) { std::fprintf(stderr, "empty prompt\n"); return 2; }

    stage("device");
    DeviceContext device(0);
    device.bind_to_current_thread();

    stage("reader");
    artifact::Reader reader(artifact);
    const models::inkling2::Config config = models::inkling2::config_from_directory(reader.directory());

    stage("binder/bind");
    artifact::Binder binder(reader);
    models::Bindings bindings(binder);
    const auto weights = models::inkling2::loading::bind_text(bindings, config);
    (void)weights;  // the bind is the artifact-shape check; the executor resolves by name

    stage("materialize");
    std::vector<artifact::PlacementRequest> requests;
    for (const auto& pending : bindings.weights) {
        for (const auto& part : pending.reference.binding.parts) {
            requests.push_back({part.object, device.device(), artifact::HostAccess::CpuOnly});
        }
    }
    auto plan = artifact::plan_materialization(reader, requests);
    auto materialized =
        artifact::materialize(reader, std::move(plan), device.transfer_view(), nullptr);
    std::vector<Device> devices(bindings.weights.size(), device.device());
    auto bound = models::resolve_weights(std::move(bindings.weights), materialized, devices);

    stage("executor");
    models::inkling2::execution::Executor executor(config, bound, device, device.stream(), 4096);
    auto session = executor.make_session();

    std::printf("loaded %zu weights; %zu prompt tokens\n", bound.size(), prompt.size());

    stage("prefill");
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        executor.decode_step(*session, prompt[i], std::int64_t(i));
    }
    cudaStreamSynchronize(device.stream());

    std::vector<std::uint32_t> out;
    std::uint32_t next = 0;
    {
        std::vector<__nv_bfloat16> logits(std::size_t(config.vocab_size));
        cudaMemcpy(logits.data(), executor.logits(), logits.size() * 2, cudaMemcpyDeviceToHost);
        int best = 0;
        float bestv = -1e30f;
        for (std::size_t i = 0; i < logits.size(); ++i) {
            const float v = __bfloat162float(logits[i]);
            if (v > bestv) { bestv = v; best = int(i); }
        }
        next = std::uint32_t(best);
        std::printf("%u %.4f\n", next, bestv);
        out.push_back(next);
    }
    // Decode timing (M4 speed gate, port/m4/plan.md step 3): steady-clock around
    // decode_step + stream sync only; the logits copy + argmax below is harness work
    // and stays outside the measurement. Warmup: the first two steps (cold caches,
    // first-touch of late-allocated workspace) are excluded from the median.
    std::vector<double> step_ms;
    for (int s = 1; s < steps; ++s) {
        const auto t0 = std::chrono::steady_clock::now();
        executor.decode_step(*session, next, std::int64_t(prompt.size() + s - 1));
        cudaStreamSynchronize(device.stream());
        step_ms.push_back(std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count());
        std::vector<__nv_bfloat16> logits(std::size_t(config.vocab_size));
        cudaMemcpy(logits.data(), executor.logits(), logits.size() * 2, cudaMemcpyDeviceToHost);
        int best = 0;
        float bestv = -1e30f;
        for (std::size_t i = 0; i < logits.size(); ++i) {
            const float v = __bfloat162float(logits[i]);
            if (v > bestv) { bestv = v; best = int(i); }
        }
        next = std::uint32_t(best);
        std::printf("%u %.4f\n", next, bestv);
    }
    if (step_ms.size() > 4) {
        std::vector<double> warm(step_ms.begin() + 2, step_ms.end());
        std::sort(warm.begin(), warm.end());
        const double median = warm[warm.size() / 2];
        double mean = 0.0;
        for (const double v : warm) { mean += v; }
        mean /= double(warm.size());
        std::printf("[decode-timing] steps=%zu median_ms=%.3f mean_ms=%.3f -> %.2f t/s "
                    "(warmup: first 2 steps excluded)\n",
                    warm.size(), median, mean, 1000.0 / median);
    }
    stage("done");
    return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[probe] EXCEPTION: %s\n", e.what());
        return 1;
    }
}
