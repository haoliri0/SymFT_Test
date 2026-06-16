#include "rate_bench/rate_bench.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

namespace symft_rate_bench {

double seconds_between(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double>(stop - start).count();
}

std::uint64_t parse_u64(const char* raw, const char* name) {
    if (raw[0] == '-') {
        throw std::runtime_error(std::string(name) + " must be nonnegative");
    }
    errno = 0;
    char* end = nullptr;
    const auto value = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' || errno == ERANGE) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + raw);
    }
    return static_cast<std::uint64_t>(value);
}

int parse_positive_int(const char* raw, const char* name) {
    const std::uint64_t value = parse_u64(raw, name);
    if (value == 0 || value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + " must be in 1:Int32Max");
    }
    return static_cast<int>(value);
}

int parse_nonnegative_int(const char* raw, const char* name) {
    const std::uint64_t value = parse_u64(raw, name);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + " must fit in Int32Max");
    }
    return static_cast<int>(value);
}

bool parse_bool_flag(const char* raw, const char* name) {
    const std::string value(raw);
    if (value == "1" || value == "true" || value == "yes" || value == "on" || value == "postselect") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off" || value == "none") {
        return false;
    }
    throw std::runtime_error(std::string("invalid ") + name + ": " + raw);
}

int parse_block_shots(const char* raw) {
    if (std::string(raw) == "auto") {
        return 0;
    }
    return parse_positive_int(raw, "block_shots");
}

int parse_threads(const char* raw) {
    const std::string value(raw);
    if (value == "auto") {
        const unsigned int detected = std::thread::hardware_concurrency();
        return std::max(1, static_cast<int>(detected == 0 ? 1 : detected));
    }
    return parse_positive_int(raw, "threads");
}

SamplerMode parse_sampler_mode(const char* raw) {
    const std::string value(raw);
    if (value == "single" || value == "single-shot" || value == "single_shot") {
        return SamplerMode::Single;
    }
    if (value == "batch" || value == "batched") {
        return SamplerMode::Batch;
    }
    if (value == "both" || value == "all") {
        return SamplerMode::Both;
    }
    throw std::runtime_error(std::string("invalid sampler: ") + raw);
}

void print_usage(const char* argv0) {
    std::cerr
        << "usage: " << argv0 << " [circuit.stim] [shots] [batch_size] [repeats] [observable] [postselect_detectors]\n"
        << "       " << argv0 << " --circuit d5.stim --shots 1000000 --batch-size 64 --postselect-detectors\n"
        << "       " << argv0 << " --sampler both --circuit d5.stim --shots 1000000 --batch-size auto\n"
        << "\n"
        << "options:\n"
        << "  --circuit PATH, --file PATH        Stim circuit path\n"
        << "  --shots N                         Number of shots\n"
        << "  --sampler single|batch|both       Sampler to benchmark; default: batch\n"
        << "  --batch-size N|auto               Batch size; --block-shots is accepted as an alias\n"
        << "  --repeats N                       Number of repeats\n"
        << "  --observable N                    Observable index\n"
        << "  --threads N|auto                  Parallel worker threads over independent batches\n"
        << "  --postselect-detectors            Stop discarded single shots or compact discarded batch shots after fired detectors\n"
        << "  --no-postselect-detectors         Disable detector postselection abort\n";
}

std::string require_option_value(
    const std::string& option,
    const std::string& inline_value,
    int& idx,
    int argc,
    char** argv) {
    if (!inline_value.empty()) {
        return inline_value;
    }
    if (idx + 1 >= argc) {
        throw std::runtime_error(option + " requires a value");
    }
    return argv[++idx];
}

void apply_positional_option(Options& options, int position, const char* value) {
    switch (position) {
    case 0:
        options.path = value;
        break;
    case 1:
        options.shots = parse_u64(value, "shots");
        break;
    case 2:
        options.block_shots = parse_block_shots(value);
        break;
    case 3:
        options.repeats = parse_positive_int(value, "repeats");
        break;
    case 4:
        options.observable = parse_nonnegative_int(value, "observable");
        break;
    case 5:
        options.postselect_detectors = parse_bool_flag(value, "postselect_detectors");
        break;
    default:
        throw std::runtime_error("too many positional arguments");
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    int positional = 0;
    for (int idx = 1; idx < argc; ++idx) {
        std::string arg(argv[idx]);
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg.rfind("--", 0) != 0) {
            apply_positional_option(options, positional++, argv[idx]);
            continue;
        }

        std::string name = arg;
        std::string value;
        const std::size_t eq = arg.find('=');
        if (eq != std::string::npos) {
            name = arg.substr(0, eq);
            value = arg.substr(eq + 1);
        }

        if (name == "--circuit" || name == "--file") {
            options.path = require_option_value(name, value, idx, argc, argv);
        } else if (name == "--shots") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.shots = parse_u64(raw.c_str(), "shots");
        } else if (name == "--sampler") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.sampler = parse_sampler_mode(raw.c_str());
        } else if (name == "--batch-size" || name == "--block-shots") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.block_shots = parse_block_shots(raw.c_str());
        } else if (name == "--repeats") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.repeats = parse_positive_int(raw.c_str(), "repeats");
        } else if (name == "--observable") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.observable = parse_nonnegative_int(raw.c_str(), "observable");
        } else if (name == "--threads") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.threads = parse_threads(raw.c_str());
        } else if (name == "--postselect-detectors" || name == "--detector-postselection") {
            options.postselect_detectors = value.empty() ? true : parse_bool_flag(value.c_str(), "postselect_detectors");
        } else if (name == "--no-postselect-detectors") {
            options.postselect_detectors = false;
        } else {
            throw std::runtime_error("unknown option: " + name);
        }
    }
    return options;
}

} // namespace symft_rate_bench
