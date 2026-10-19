// Fork-only Turing (sm_75) developer tool. For one dequantizing linear shape it forces every
// Q4/Q5/Q6/Q8 launcher of the format, rejects the ones that throw or disagree with the production
// route's result, times the rest with cold-cache CUDA Graph replay, and prints the winner per
// token count plus a TokenRoute table that can be pasted into the shape selector.
//
// Upstream's linear_bench deliberately cannot force routes. This tool includes the private
// ops/linear launchers and is the reason it exists; it is not a product or CI target.
//
//   ./build-windows/bench/ninfer_turing_route_sweep --qtype q5 --n 5120 --k 17408 --t 1,2,3,4,1024
//   ./build-windows/bench/ninfer_turing_route_sweep --qtype q4 --n 34816 --k 5120 --t 512 --top 10
//
// The candidate list is generated at configure time from the q*_launch.h headers, so a new launcher
// is picked up without editing this file.

#include "core/device.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include "ops/linear/projection/q4/q4_dispatch.h"
#include "ops/linear/projection/q5/q5_dispatch.h"
#include "ops/linear/projection/q6/q6_dispatch.h"
#include "ops/linear/projection/q8/q8_dispatch.h"

#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

struct Candidate {
    std::string_view format; // "q4", "q5", "q6" or "q8"
    const char* name;
    Launch launch;
};

#define NINFER_ROUTE_CANDIDATE(fmt, fn) {#fmt, #fn, &ninfer::ops::detail::fn},
const Candidate kCandidates[] = {
#include "route_registry.inc"
};
#undef NINFER_ROUTE_CANDIDATE

struct Options {
    std::string qtype;
    std::int32_t n = 0;
    std::int32_t k = 0;
    std::vector<std::int32_t> tokens;
    int warmup = 3;
    int repeat = 20;
    int top    = 6;
    bool verbose = false;
};

[[noreturn]] void usage(const char* message) {
    if (message != nullptr) { std::fprintf(stderr, "error: %s\n", message); }
    std::fprintf(stderr,
                 "usage: ninfer_turing_route_sweep --qtype q4|q5|q6|q8 --n N --k K --t T[,T...]\n"
                 "       [--warmup N] [--repeat N] [--top N] [--verbose]\n");
    std::exit(2);
}

std::vector<std::int32_t> parse_list(const std::string& text) {
    std::vector<std::int32_t> values;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t end = text.find(',', begin);
        const std::string item =
            text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (item.empty()) { usage("empty token count in --t"); }
        values.push_back(static_cast<std::int32_t>(std::strtol(item.c_str(), nullptr, 10)));
        if (values.back() <= 0) { usage("--t values must be positive"); }
        if (end == std::string::npos) { break; }
        begin = end + 1;
    }
    return values;
}

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value      = [&]() -> std::string {
            if (i + 1 >= argc) { usage("missing value"); }
            return argv[++i];
        };
        if (arg == "--qtype") {
            o.qtype = value();
        } else if (arg == "--n") {
            o.n = static_cast<std::int32_t>(std::strtol(value().c_str(), nullptr, 10));
        } else if (arg == "--k") {
            o.k = static_cast<std::int32_t>(std::strtol(value().c_str(), nullptr, 10));
        } else if (arg == "--t") {
            o.tokens = parse_list(value());
        } else if (arg == "--warmup") {
            o.warmup = std::atoi(value().c_str());
        } else if (arg == "--repeat") {
            o.repeat = std::atoi(value().c_str());
        } else if (arg == "--top") {
            o.top = std::atoi(value().c_str());
        } else if (arg == "--verbose") {
            o.verbose = true;
        } else {
            usage(("unknown option " + arg).c_str());
        }
    }
    if (o.qtype.empty() || o.n <= 0 || o.k <= 0 || o.tokens.empty()) { usage(nullptr); }
    return o;
}

QType qtype_of(const std::string& name) {
    if (name == "q4") { return QType::Q4_G64_FP16; }
    if (name == "q5") { return QType::Q5_G64_FP16; }
    if (name == "q6") { return QType::Q6_G64_FP16; }
    if (name == "q8") { return QType::Q8_G32_FP16; }
    usage("--qtype must be q4, q5, q6 or q8");
}

Launch production_launch(const std::string& format, std::int32_t n, std::int32_t k,
                         std::int32_t t) {
    using ops::LinearPolicy;
    if (format == "q4") { return ops::detail::select_q4_launch(n, k, t, LinearPolicy::A16Only); }
    if (format == "q5") { return ops::detail::select_q5_launch(n, k, t, LinearPolicy::A16Only); }
    if (format == "q6") { return ops::detail::select_q6_launch(n, k, t, LinearPolicy::A16Only); }
    return ops::detail::select_q8_launch(n, k, t, LinearPolicy::A16Only);
}

const char* name_of(Launch launch) {
    for (const Candidate& candidate : kCandidates) {
        if (candidate.launch == launch) { return candidate.name; }
    }
    return "(unlisted)";
}

std::vector<float> read_output(const DeviceBuffer& buffer, std::size_t elements) {
    std::vector<std::uint16_t> raw(elements);
    CUDA_CHECK(cudaMemcpy(raw.data(), buffer.data(), elements * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    std::vector<float> values(elements);
    for (std::size_t i = 0; i < elements; ++i) {
        const std::uint32_t bits = static_cast<std::uint32_t>(raw[i]) << 16;
        std::memcpy(&values[i], &bits, sizeof(float));
    }
    return values;
}

struct Outcome {
    const Candidate* candidate = nullptr;
    double median_us           = 0.0;
};

} // namespace

int main(int argc, char** argv) {
    const Options options = parse(argc, argv);
    const QType qtype     = qtype_of(options.qtype);
    const std::int32_t max_t = *std::max_element(options.tokens.begin(), options.tokens.end());

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    bench::L2FlushBuffer flush(256ULL << 20);

    const std::int32_t padded_k = static_cast<std::int32_t>((options.k + 127) / 128 * 128);
    bench::PackedQuantizedWeight packed =
        bench::make_row_split_weight(qtype, options.n, options.k, padded_k, 501U);
    const double weight_gb = static_cast<double>(packed.model_weight_bytes()) * 1.0e-9;

    // Outputs get slack columns: a launcher built for a different token count may write past the
    // columns it was asked for, and that must not corrupt a neighbouring buffer.
    const std::size_t x_elements   = static_cast<std::size_t>(options.k) * max_t;
    const std::size_t out_elements = static_cast<std::size_t>(options.n) * (max_t + 256);
    const Device device = ninfer::bench::device_context().device();
    const CudaExecutionView cuda = ninfer::bench::device_context().execution_view(stream);
    DeviceBuffer x(device, x_elements * 2);
    DeviceBuffer out(device, out_elements * 2);
    DeviceBuffer reference(device, out_elements * 2);
    CUDA_CHECK(bench::fixture::fill_values(static_cast<__nv_bfloat16*>(x.data()), x_elements, 101U,
                                           -0.5F, 0.5F, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("%s n=%d k=%d weight %.3f GB, %d candidates, cold graph replay (warmup %d, repeat %d)\n",
                options.qtype.c_str(), options.n, options.k, weight_gb,
                static_cast<int>(std::size(kCandidates)), options.warmup, options.repeat);

    struct Winner {
        std::int32_t tokens;
        const char* name;
        double us;
        const char* production;
        double production_us;
    };
    std::vector<Winner> winners;

    std::vector<std::int32_t> tokens = options.tokens;
    std::sort(tokens.begin(), tokens.end());
    for (const std::int32_t t : tokens) {
        Tensor activation(x.data(), device, DType::BF16, {options.k, t});
        Tensor output(out.data(), device, DType::BF16, {options.n, t});
        Tensor reference_output(reference.data(), device, DType::BF16, {options.n, t});

        const Launch production = production_launch(options.qtype, options.n, options.k, t);
        CUDA_CHECK(cudaMemsetAsync(reference.data(), 0, reference.size(), stream));
        production(activation, packed.weight, reference_output, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::size_t elements = static_cast<std::size_t>(options.n) * t;
        const std::vector<float> expected = read_output(reference, elements);
        double square = 0.0;
        for (const float v : expected) { square += static_cast<double>(v) * v; }
        const double rms = std::sqrt(square / static_cast<double>(elements));

        std::vector<Outcome> legal;
        int rejected = 0;
        for (const Candidate& candidate : kCandidates) {
            if (candidate.format != options.qtype) { continue; }
            std::string reason;
            try {
                CUDA_CHECK(cudaMemsetAsync(out.data(), 0, out.size(), stream));
                candidate.launch(activation, packed.weight, output, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                CUDA_CHECK(cudaGetLastError());
                const std::vector<float> got = read_output(out, elements);
                double worst = 0.0;
                bool finite  = true;
                for (std::size_t i = 0; i < elements; ++i) {
                    if (!std::isfinite(got[i])) { finite = false; break; }
                    worst = std::max(worst, static_cast<double>(std::fabs(got[i] - expected[i])));
                }
                if (!finite) {
                    reason = "non-finite output";
                } else if (worst > 0.05 * rms) {
                    reason = "result differs from the production route (max |diff| " +
                             std::to_string(worst) + ", rms " + std::to_string(rms) + ")";
                }
            } catch (const std::exception& error) {
                reason = std::string("rejected: ") + error.what();
                (void)cudaGetLastError();
            }
            if (!reason.empty()) {
                ++rejected;
                if (options.verbose) {
                    std::printf("  T=%-5d skip %-50s %s\n", t, candidate.name, reason.c_str());
                }
                continue;
            }
            bench::TimedGraph graph;
            graph.capture(cuda, [&](CudaExecutionView c) {
                candidate.launch(activation, packed.weight, output, c.stream());
            });
            const bench::ColdTiming timing =
                bench::measure_cold_graph(graph, flush, cuda, options.warmup, options.repeat);
            legal.push_back({&candidate, timing.median_us});
        }
        if (legal.empty()) {
            std::printf("T=%d: no legal launcher (%d rejected)\n", t, rejected);
            continue;
        }
        std::sort(legal.begin(), legal.end(),
                  [](const Outcome& a, const Outcome& b) { return a.median_us < b.median_us; });
        double production_us = 0.0;
        for (const Outcome& outcome : legal) {
            if (outcome.candidate->launch == production) { production_us = outcome.median_us; }
        }

        std::printf("\nT=%d  (%zu legal, %d rejected)  production: %s\n", t, legal.size(),
                    rejected, name_of(production));
        const double flops = 2.0 * options.n * options.k * t;
        for (int rank = 0; rank < static_cast<int>(legal.size()) && rank < options.top; ++rank) {
            const Outcome& o = legal[static_cast<std::size_t>(rank)];
            std::printf("  %2d. %-48s %9.2f us  %6.2f TFLOPS  %7.1f GB/s%s\n", rank + 1,
                        o.candidate->name, o.median_us, flops / o.median_us * 1.0e-6,
                        weight_gb / (o.median_us * 1.0e-6),
                        o.candidate->launch == production ? "  <- production" : "");
        }
        if (production_us > 0.0 && legal.front().candidate->launch != production) {
            std::printf("  production %.2f us is %+.1f%% vs the winner\n", production_us,
                        (production_us / legal.front().median_us - 1.0) * 100.0);
        } else if (production_us == 0.0) {
            std::printf("  production route was not among the legal candidates\n");
        }
        winners.push_back({t, legal.front().candidate->name, legal.front().median_us,
                           name_of(production), production_us});
    }

    // A TokenRoute table: consecutive token counts with the same winner share one entry, ending at
    // the largest token count measured for it (the last entry becomes kAnyTokens).
    std::printf("\nstatic constexpr TokenRoute<Q%cLaunch> kRoutes[] = {\n", options.qtype[1]);
    for (std::size_t i = 0; i < winners.size(); ++i) {
        const bool last_of_run = i + 1 == winners.size() ||
                                 std::strcmp(winners[i + 1].name, winners[i].name) != 0;
        if (!last_of_run) { continue; }
        if (i + 1 == winners.size()) {
            std::printf("    {kAnyTokens, %s},\n", winners[i].name);
        } else {
            std::printf("    {%d, %s},\n", winners[i].tokens, winners[i].name);
        }
    }
    std::printf("};\n");
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
