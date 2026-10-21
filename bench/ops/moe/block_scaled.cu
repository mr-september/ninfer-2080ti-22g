#include "core/buffer.h"
#include "ninfer/ops/moe.h"
#include "ninfer_bench_common.h"
#include "timing.h"

#include <numeric>
#include <random>
#include <sstream>
#include <string>

using namespace ninfer;

namespace {
constexpr int H = 2560, I = 640, E = 512, TopK = 10;

struct Options {
    std::vector<int> tokens{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    std::string format       = "fp8";
    std::string distribution = "all", operation = "routed", cache = "both";
    int warmup = 5, repeat = 30;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int a = 1; a < argc; ++a) {
        const std::string key = argv[a];
        if (key == "--help") {
            std::puts(
                "--format fp8 --tokens M[,M...] --distribution independent|same|mixed|all "
                "--operation "
                "routed|routed-shared|complete|route --cache cold|warm|both --warmup N --repeat N");
            std::exit(0);
        }
        if (++a == argc) { throw std::invalid_argument("missing option value"); }
        const std::string value = argv[a];
        if (key == "--format") {
            o.format = value;
        } else if (key == "--tokens") {
            o.tokens.clear();
            std::stringstream input(value);
            std::string part;
            while (std::getline(input, part, ',')) { o.tokens.push_back(std::stoi(part)); }
        } else if (key == "--distribution") {
            o.distribution = value;
        } else if (key == "--operation") {
            o.operation = value;
        } else if (key == "--cache") {
            o.cache = value;
        } else if (key == "--warmup") {
            o.warmup = std::stoi(value);
        } else if (key == "--repeat") {
            o.repeat = std::stoi(value);
        } else {
            throw std::invalid_argument("unknown option: " + key);
        }
    }
    if (o.format != "fp8" || o.tokens.empty() ||
        std::any_of(o.tokens.begin(), o.tokens.end(), [](int x) { return x <= 0; }) ||
        o.warmup < 0 || o.repeat <= 0 ||
        (o.distribution != "all" && o.distribution != "independent" && o.distribution != "same" &&
         o.distribution != "mixed") ||
        (o.operation != "routed" && o.operation != "routed-shared" && o.operation != "complete" &&
         o.operation != "route") ||
        (o.cache != "cold" && o.cache != "warm" && o.cache != "both")) {
        throw std::invalid_argument("invalid benchmark options");
    }
    return o;
}

__global__ void fill_fp8(std::uint8_t* codes, std::size_t bytes, __nv_bfloat16* scales,
                         std::size_t groups, unsigned seed) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < bytes;
         i += gridDim.x * blockDim.x) {
        unsigned code = bench::fixture::bits(i, seed) & 255;
        if ((code & 127) == 127) { --code; }
        codes[i] = code;
    }
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < groups;
         i += gridDim.x * blockDim.x) {
        scales[i] = __float2bfloat16_rn((1 + (i * 7 + seed) % 13) * .0000083F);
    }
}

struct Bank {
    DeviceBuffer codes, scales;
    std::size_t code_bytes, scale_bytes;

    Bank(int rows, int columns, unsigned seed)
        : codes(bench::make_buffer(std::size_t(E) * rows * columns)),
          scales(bench::make_buffer(std::size_t(E) * (rows / 128 * (columns / 128) * 2))),
          code_bytes(codes.size() / E), scale_bytes(scales.size() / E) {
        fill_fp8<<<2048, 256>>>(static_cast<std::uint8_t*>(codes.data()), codes.size(),
                                static_cast<__nv_bfloat16*>(scales.data()), scales.size() / 2,
                                seed);
        CUDA_CHECK(cudaGetLastError());
    }

    ExpertMatrixDeviceView view() const {
        ExpertMatrixDeviceView v;
        v.codes        = static_cast<const std::byte*>(codes.data());
        v.scales       = static_cast<const std::byte*>(scales.data());
        v.code_stride  = code_bytes;
        v.scale_stride = scale_bytes;
        return v;
    }
};

Weight dense(const DeviceBuffer& b, int n, int k) {
    Weight w;
    w.device  = b.device();
    w.payload = w.qdata = b.data();
    w.payload_bytes     = b.size();
    w.qtype             = QType::BF16;
    w.layout            = QuantLayout::Contiguous;
    w.ndim              = 2;
    w.n = w.shape[0] = w.padded_shape[0] = n;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    return w;
}

struct Fixture {
    Bank gu, down;
    DeviceBuffer control     = bench::make_buffer(std::size_t(E + 1) * H * 2);
    DeviceBuffer shared_gu   = bench::make_bf16(2 * I * H, 991, -.012F, .012F);
    DeviceBuffer shared_down = bench::make_bf16(H * I, 1011, -.025F, .025F);
    bench::L2FlushBuffer flush{256ULL << 20};
    ops::MoeWeights weights;
    ops::MoeProfile profile;

    Fixture() : gu(2 * I, H, 811), down(H, I, 919) {
        std::vector<std::uint16_t> router(std::size_t(E + 1) * H);
        for (int e = 0; e <= E; ++e) { router[std::size_t(e) * H + e] = bench::f32_to_bf16(2); }
        control.copy_from_host(router.data(), control.size());
        weights         = {dense(control, E + 1, H),
                           {{gu.view(), down.view(), E, control.device()}, {}},
                           {{dense(shared_gu, 2 * I, H)}, {dense(shared_down, H, I)}}};
        const auto type   = QType::FP8_E4M3FN_BLOCK128X128_BF16;
        const auto layout = QuantLayout::BlockScaleM128K128;
        const auto policy = ops::LinearPolicy::AllowA8;
        profile           = {{{H, I, type, type, layout, layout, policy, policy}, E, TopK},
                             ops::swiglu_profile(weights.shared)};
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};

struct Workload {
    DeviceBuffer input, output, ids, alpha, beta, scratch;
    Tensor x, y, shared;
    ops::ExpertRoutes routes;
    MemoryArena arena;
    std::size_t workspace_bytes;
    int unique;
    std::vector<int> counts;

    Workload(Fixture& f, int m, const std::string& distribution, const std::string& operation)
        : input(bench::make_buffer(std::size_t(m) * H * 2)),
          output(bench::make_buffer(std::size_t(m) * H * 2)),
          ids(bench::make_buffer(std::size_t(m) * TopK * 4)),
          alpha(bench::make_buffer(std::size_t(m) * TopK * 4)), beta(bench::make_buffer(m * 4)),
          scratch(bench::make_buffer(capacity(f.profile, m, operation))),
          x(input.span(), DType::BF16, {H, m}), y(output.span(), DType::BF16, {H, m}),
          shared(beta.span(), DType::FP32, {m}),
          routes{Tensor(ids.span(), DType::I32, {TopK, m}),
                 Tensor(alpha.span(), DType::FP32, {TopK, m})},
          arena(scratch.span()), workspace_bytes(scratch.size()), counts(E) {
        std::mt19937 rng(1729);
        std::vector<int> pool(E), selected(m * TopK);
        std::vector<std::uint16_t> values(std::size_t(m) * H);
        std::vector<float> coefficients(m * TopK), shared_coefficients(m, .5F);
        for (int t = 0; t < m; ++t) {
            std::iota(pool.begin(), pool.end(), 0);
            if (distribution == "independent") { std::shuffle(pool.begin(), pool.end(), rng); }
            if (distribution == "mixed") { std::shuffle(pool.begin() + 6, pool.end(), rng); }
            for (int h = 0; h < H; ++h) {
                values[std::size_t(t) * H + h] = bench::f32_to_bf16(
                    h < E ? -.75F
                          : bench::fixture::uniform(std::size_t(t) * H + h, 151, -.7F, .7F));
            }
            double total = 0;
            for (int j = 0; j < TopK; ++j) { total += std::exp(2 * (.75F - j / 32.F)); }
            for (int j = 0; j < TopK; ++j) {
                selected[t * TopK + j] = pool[j];
                ++counts[pool[j]];
                const float value                    = .75F - j / 32.F;
                values[std::size_t(t) * H + pool[j]] = bench::f32_to_bf16(value);
                coefficients[t * TopK + j]           = std::exp(2 * value) / total;
            }
            values[std::size_t(t) * H + E] = 0;
        }
        unique = std::count_if(counts.begin(), counts.end(), [](int n) { return n != 0; });
        input.copy_from_host(values.data(), input.size());
        ids.copy_from_host(selected.data(), ids.size());
        alpha.copy_from_host(coefficients.data(), alpha.size());
        beta.copy_from_host(shared_coefficients.data(), beta.size());
    }

    static std::size_t capacity(const ops::MoeProfile& p, int m, const std::string& operation) {
        if (operation == "complete") {
            return ops::topk_softmax_swiglu_moe_workspace_capacity_bytes(p, ops::MoeEpilogue::Write,
                                                                         m, m);
        }
        if (operation == "route") {
            return ops::topk_softmax_route_workspace_capacity_bytes({H, E, TopK}, m, m);
        }
        if (operation == "routed-shared") {
            return ops::routed_swiglu_workspace_capacity_bytes(
                p.routed, p.shared, ops::ExpertSubsetKind::All, DType::BF16,
                ops::MoeEpilogue::Write, m, m);
        }
        return ops::routed_swiglu_workspace_capacity_bytes(
            p.routed, ops::ExpertSubsetKind::All, DType::BF16, ops::MoeEpilogue::Write, m, m);
    }

    void launch(Fixture& f, const std::string& operation, CudaExecutionView cuda) {
        if (operation == "complete") {
            ops::topk_softmax_swiglu_moe(x, f.profile, f.weights, ops::MoeEpilogue::Write, y, arena,
                                         cuda);
        } else if (operation == "route") {
            ops::topk_softmax_route(x, {H, E, TopK}, f.weights.control, routes, shared, arena,
                                    cuda);
        } else if (operation == "routed-shared") {
            ops::routed_swiglu(x, f.profile.routed, f.weights.routed, routes, {},
                               {f.weights.shared, shared}, ops::MoeEpilogue::Write, y, arena, cuda);
        } else {
            ops::routed_swiglu(x, f.profile.routed, f.weights.routed, routes, {},
                               ops::MoeEpilogue::Write, y, arena, cuda);
        }
    }
};
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        Fixture fixture;
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const auto execution = bench::device_context().execution_view(stream);
        std::puts(
            "format,operation,distribution,M,unique,cache,median_us,p95_us,min_us,workspace_MiB,"
            "useful_TFLOPs,unique_weight_GBps");
        for (const std::string distribution : {"independent", "same", "mixed"}) {
            if (options.distribution != "all" && options.distribution != distribution) { continue; }
            for (int m : options.tokens) {
                Workload state(fixture, m, distribution, options.operation);
                state.launch(fixture, options.operation, execution);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                bench::moe::BodyTimedGraph graph;
                graph.capture(execution,
                              [&](auto cuda) { state.launch(fixture, options.operation, cuda); });
                for (const std::string cache : {"cold", "warm"}) {
                    if (options.cache != "both" && options.cache != cache) { continue; }
                    std::vector<double> samples;
                    for (int r = -options.warmup; r < options.repeat; ++r) {
                        if (cache == "cold") { fixture.flush.evict(execution); }
                        const auto us = graph.launch_body_timed(execution);
                        if (r >= 0) { samples.push_back(us); }
                    }
                    const auto stats = bench::moe::summarize(samples);
                    double flops     = options.operation == "route" ? 2.0 * H * (E + 1) * m
                                                                    : 6.0 * H * I * TopK * m;
                    if (options.operation == "complete" || options.operation == "routed-shared") {
                        flops += 6.0 * H * I * m;
                    }
                    if (options.operation == "complete") { flops += 2.0 * H * (E + 1) * m; }
                    const double bytes =
                        options.operation == "route"
                            ? 2.0 * H * (E + 1)
                            : double(state.unique) *
                                  (fixture.gu.code_bytes + fixture.gu.scale_bytes +
                                   fixture.down.code_bytes + fixture.down.scale_bytes);
                    std::printf("%s,%s,%s,%d,%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                                options.format.c_str(), options.operation.c_str(),
                                distribution.c_str(), m, state.unique, cache.c_str(),
                                stats.median_us, stats.p95_us, stats.min_us,
                                state.workspace_bytes / 1048576.0, flops / stats.median_us / 1e6,
                                bytes / stats.median_us / 1e3);
                    std::fflush(stdout);
                }
            }
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
