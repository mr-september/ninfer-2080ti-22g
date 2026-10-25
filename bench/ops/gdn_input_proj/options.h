#pragma once

#include "op_measurement.cuh"
#include "ninfer/ops/gdn_input_proj.h"

namespace ninfer::bench::gdn_input {
struct Options {
    measurement::Options timing;
    std::string op = "snapshot", format = "q4q5", policy = "a16";
    int n = 16384, k = 5120, batch = 1;
    std::vector<int> tokens{1}, valid;
};

inline Options parse_options(int argc, char** argv) {
    Options o;
    bool explicit_tokens = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            std::cout << "GDN input Op benchmark\n"
                         "  --op snapshot|record --format q4q5|fp8\n"
                         "  --n N --k K --policy a16|allow-a8\n"
                         "  --tokens T[,T...] or --sweep START:END[:STEP]\n"
                         "  --batch B --valid-columns T[,T...]\n"
                         "Uses physical T and B. Defaults: Q4/Q5 "
                         "[16384,5120], T=1.\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--op")
            o.op = value;
        else if (flag == "--format")
            o.format = value;
        else if (flag == "--n")
            o.n = measurement::integer(value);
        else if (flag == "--k")
            o.k = measurement::integer(value);
        else if (flag == "--policy")
            o.policy = value;
        else if (flag == "--batch")
            o.batch = measurement::integer(value);
        else if (flag == "--tokens") {
            if (explicit_tokens) throw std::invalid_argument("specify one token extent or sweep");
            o.tokens        = measurement::integers(value);
            explicit_tokens = true;
        } else if (flag == "--sweep") {
            if (explicit_tokens) throw std::invalid_argument("specify one token extent or sweep");
            o.tokens        = measurement::sweep(value);
            explicit_tokens = true;
        } else if (flag == "--valid-columns")
            o.valid = measurement::integers(value);
        else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option: " + std::string(flag));
    }
    if (o.op != "snapshot" && o.op != "record") throw std::invalid_argument("unknown input Op");
    if (o.policy != "a16" && o.policy != "allow-a8")
        throw std::invalid_argument("unknown activation policy");
    if (o.format == "q4q5" && o.policy != "a16")
        throw std::invalid_argument("paired Q4/Q5 weights use A16");
    if (o.batch > 8) throw std::invalid_argument("invalid batch/valid for Op");
    if (!o.valid.empty() && (o.tokens.size() != 1 || o.valid.size() != std::size_t(o.batch) ||
                             *std::max_element(o.valid.begin(), o.valid.end()) > o.tokens.front()))
        throw std::invalid_argument("valid columns must match exact T/B");
    const bool supported =
        (o.format == "q4q5" || o.format == "fp8") && o.n == 16384 && o.k == 5120;
    if (!supported) throw std::invalid_argument("unsupported format/N/K");
    return o;
}

inline ops::LinearPolicy policy(const Options& o) {
    return o.policy == "a16" ? ops::LinearPolicy::A16Only : ops::LinearPolicy::AllowA8;
}
} // namespace ninfer::bench::gdn_input
