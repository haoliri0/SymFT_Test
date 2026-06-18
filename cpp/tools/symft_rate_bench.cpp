#include "frontend/stim.hpp"
#include "sampler/batch_sampler.hpp"
#include "sampler/exogenous.hpp"
#include "sampler/single_shot.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct RateCounts {
    std::uint64_t shots = 0;
    std::uint64_t discarded = 0;
    std::uint64_t accepted = 0;
    std::uint64_t logical_errors = 0;
};

struct WorkerResult {
    RateCounts counts;
    double presample_s = 0.0;
    double execute_s = 0.0;
    double accumulate_s = 0.0;
};

enum class SamplerMode {
    Single,
    Batch,
    Both,
};

struct BenchResult {
    std::string path;
    std::string sampler;
    std::uint64_t requested_shots = 0;
    RateCounts counts;
    int n = 0;
    int records = 0;
    int detectors = 0;
    int observable_includes = 0;
    int observable = 0;
    int max_k = 0;
    int batch_size = 0;
    int sample_chunk_shots = 0;
    int repeats = 1;
    int threads = 1;
    int requested_threads = 1;
    bool detector_postselection = false;
    int batch_mask_threshold_denominator = 0;
    std::string exogenous_mode;
    std::string rng_streams = "split_exogenous_branch";
    std::string phase_timing = "wall";
    double parse_s = 0.0;
    double plan_s = 0.0;
    double presample_s = 0.0;
    double execute_s = 0.0;
    double accumulate_s = 0.0;
    double sample_s = 0.0;
};

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

std::uint64_t ceil_div_u64(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) {
        throw std::runtime_error("division by zero in chunk sizing");
    }
    return numerator / denominator + (numerator % denominator != 0 ? 1 : 0);
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

int parse_batch_size(const char* raw) {
    if (std::string(raw) == "auto") {
        return 0;
    }
    return parse_positive_int(raw, "batch_size");
}

int parse_sample_chunk_shots(const char* raw) {
    const std::string value(raw);
    if (value == "auto" || value == "none" || value == "off") {
        return 0;
    }
    return parse_positive_int(raw, "sample_chunk_shots");
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

struct Options {
    std::string path = "d3.stim";
    std::uint64_t shots = 100000000ULL;
    int batch_size = 0;
    int sample_chunk_shots = 0;
    int repeats = 1;
    int observable = 0;
    int threads = 1;
    bool postselect_detectors = false;
    int batch_mask_threshold_denominator = 2;
    SamplerMode sampler = SamplerMode::Batch;
};

void print_usage(const char* argv0) {
    std::cerr
        << "usage: " << argv0 << " --circuit PATH --shots N [options]\n"
        << "       " << argv0 << " --sampler batch --circuit d5.stim --shots 1000000 --batch-size auto --postselect-detectors\n"
        << "       " << argv0 << " --sampler both --circuit d5.stim --shots 1000000 --batch-size auto\n"
        << "\n"
        << "common options:\n"
        << "  --circuit PATH, --file PATH        Stim circuit path\n"
        << "  --shots N                         Number of shots\n"
        << "  --sampler single|batch|both       Sampler to benchmark; default: batch\n"
        << "  --sample-chunk-shots N|auto       Presample requested shots in reusable chunks; default: 1024 for packed streaming\n"
        << "  --repeats N                       Number of repeats\n"
        << "  --observable N                    Observable index\n"
        << "  --postselect-detectors            Stop discarded single shots or compact discarded batch shots after fired detectors\n"
        << "  --no-postselect-detectors         Disable detector postselection abort\n"
        << "\n"
        << "batch-only options:\n"
        << "  --batch-size N|auto               Batch size; auto uses the postselection-tuned size when postselecting\n"
        << "  --threads N|auto                  Parallel worker threads over independent chunks\n"
        << "  --batch-mask-threshold-denominator N  Base batch postselection compaction threshold; expensive active-state ops compact more aggressively; default: 2\n";
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
        options.batch_size = parse_batch_size(value);
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
        } else if (name == "--batch-size") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.batch_size = parse_batch_size(raw.c_str());
        } else if (name == "--sample-chunk-shots") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.sample_chunk_shots = parse_sample_chunk_shots(raw.c_str());
        } else if (name == "--repeats") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.repeats = parse_positive_int(raw.c_str(), "repeats");
        } else if (name == "--observable") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.observable = parse_nonnegative_int(raw.c_str(), "observable");
        } else if (name == "--threads") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.threads = parse_threads(raw.c_str());
        } else if (name == "--postselect-detectors") {
            options.postselect_detectors = value.empty() ? true : parse_bool_flag(value.c_str(), "postselect_detectors");
        } else if (name == "--no-postselect-detectors") {
            options.postselect_detectors = false;
        } else if (name == "--batch-mask-threshold-denominator") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.batch_mask_threshold_denominator =
                parse_positive_int(raw.c_str(), "batch_mask_threshold_denominator");
        } else {
            throw std::runtime_error("unknown option: " + name);
        }
    }
    return options;
}

symft::BatchDetectorPostselectionOptions batch_postselection_options(
    const Options& options,
    const std::vector<std::vector<int>>& retained_record_uses) {
    symft::BatchDetectorPostselectionOptions out;
    out.mask_dead_shots_min_fraction_denominator = options.batch_mask_threshold_denominator;
    out.retained_record_uses = &retained_record_uses;
    return out;
}

int popcount64(std::uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(value);
#else
    int count = 0;
    while (value != 0) {
        value &= value - 1;
        ++count;
    }
    return count;
#endif
}

std::size_t batch_word_count(int shots) {
    return shots <= 0 ? 0 : static_cast<std::size_t>((shots + 63) >> 6);
}

std::uint64_t low_bits_mask(int nbits) {
    if (nbits <= 0) {
        return 0;
    }
    if (nbits >= 64) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return (std::uint64_t{1} << nbits) - 1;
}

std::uint64_t live_word_mask(int shots, std::size_t word) {
    return low_bits_mask(shots - static_cast<int>(word << 6));
}

std::size_t measurement_offset(std::size_t stride_words, int record, std::size_t word) {
    if (record <= 0) {
        throw std::runtime_error("record ids must be positive");
    }
    return static_cast<std::size_t>(record - 1) * stride_words + word;
}

void xor_records_into(
    std::vector<std::uint64_t>& out,
    const std::vector<std::uint64_t>& measurements,
    std::size_t stride_words,
    std::size_t nwords,
    const std::vector<int>& records) {
    std::fill(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(nwords), 0);
    for (int record : records) {
        for (std::size_t word = 0; word < nwords; ++word) {
            out[word] ^= measurements[measurement_offset(stride_words, record, word)];
        }
    }
}

void accumulate_block_counts(
    RateCounts& counts,
    const symft::BatchFactoredExecutorState& runtime,
    int block,
    const std::vector<std::vector<int>>& logical_records,
    std::vector<std::uint64_t>& discard_bits,
    std::vector<std::uint64_t>& logical_bits,
    std::vector<std::uint64_t>& scratch) {
    const std::size_t stride_words = runtime.batch_words;
    const std::size_t nwords = batch_word_count(block);
    std::fill(discard_bits.begin(), discard_bits.begin() + static_cast<std::ptrdiff_t>(nwords), 0);
    std::fill(logical_bits.begin(), logical_bits.begin() + static_cast<std::ptrdiff_t>(nwords), 0);

    for (int detector = 1; detector <= runtime.ndetectors; ++detector) {
        const std::size_t base = static_cast<std::size_t>(detector - 1) * stride_words;
        for (std::size_t word = 0; word < nwords; ++word) {
            discard_bits[word] |= runtime.detector_words[base + word];
        }
    }

    for (const auto& records : logical_records) {
        xor_records_into(scratch, runtime.measurement_words, stride_words, nwords, records);
        for (std::size_t word = 0; word < nwords; ++word) {
            logical_bits[word] ^= scratch[word];
        }
    }

    for (std::size_t word = 0; word < nwords; ++word) {
        const std::uint64_t live = live_word_mask(block, word);
        const std::uint64_t discarded_word = discard_bits[word] & live;
        const std::uint64_t accepted_word = (~discarded_word) & live;
        counts.discarded += static_cast<std::uint64_t>(popcount64(discarded_word));
        counts.accepted += static_cast<std::uint64_t>(popcount64(accepted_word));
        counts.logical_errors += static_cast<std::uint64_t>(popcount64(logical_bits[word] & accepted_word));
    }
}

void accumulate_logical_counts_for_survivors(
    RateCounts& counts,
    const symft::BatchFactoredExecutorState& runtime,
    const std::vector<std::vector<int>>& logical_records,
    std::vector<std::uint64_t>& logical_bits,
    std::vector<std::uint64_t>& scratch) {
    counts.accepted += static_cast<std::uint64_t>(runtime.active_shots);
    if (runtime.active_shots == 0) {
        return;
    }
    const std::size_t nwords = batch_word_count(runtime.active_shots);
    std::fill(logical_bits.begin(), logical_bits.begin() + static_cast<std::ptrdiff_t>(nwords), 0);
    for (const auto& records : logical_records) {
        xor_records_into(scratch, runtime.measurement_words, runtime.batch_words, nwords, records);
        for (std::size_t word = 0; word < nwords; ++word) {
            logical_bits[word] ^= scratch[word] & live_word_mask(runtime.active_shots, word);
        }
    }
    for (std::size_t word = 0; word < nwords; ++word) {
        counts.logical_errors += static_cast<std::uint64_t>(
            popcount64(logical_bits[word] & live_word_mask(runtime.active_shots, word)));
    }
}

std::uint64_t block_seed(std::uint64_t base, int repeat, std::uint64_t block_index) {
    return base ^
           (std::uint64_t{0x9e3779b97f4a7c15} * (block_index + 1)) ^
           (std::uint64_t{0xbf58476d1ce4e5b9} * static_cast<std::uint64_t>(repeat + 1));
}

std::vector<std::vector<int>> logical_records_for_observable(
    const std::vector<symft::StimObservableInclude>& observables,
    int observable) {
    std::vector<std::vector<int>> out;
    for (const auto& include : observables) {
        if (include.index == observable) {
            out.push_back(include.records);
        }
    }
    return out;
}

bool record_parity(const std::vector<std::uint64_t>& words, const std::vector<int>& records) {
    bool parity = false;
    for (int record : records) {
        if (record <= 0) {
            throw std::runtime_error("record ids must be positive");
        }
        parity ^= symft::packed_bit(words, record - 1);
    }
    return parity;
}

void accumulate_single_counts(
    RateCounts& counts,
    const std::vector<std::uint64_t>& measurement_words,
    const std::vector<std::uint64_t>& detector_words,
    int ndetectors,
    const std::vector<std::vector<int>>& logical_records) {
    bool discarded = false;
    for (int detector = 0; detector < ndetectors; ++detector) {
        discarded = discarded || symft::packed_bit(detector_words, detector);
    }
    if (discarded) {
        ++counts.discarded;
        return;
    }
    ++counts.accepted;
    bool logical = false;
    for (const auto& records : logical_records) {
        logical ^= record_parity(measurement_words, records);
    }
    if (logical) {
        ++counts.logical_errors;
    }
}

struct RateBenchSetup {
    symft::StimParseResult parsed;
    symft::FactoredInstructionProgram program;
    std::vector<std::vector<int>> logical_records;
    int batch_size = 0;
    int sample_chunk_shots = 0;
};

RateBenchSetup build_rate_bench_setup(
    const Options& options,
    double& parse_s,
    double& plan_s) {
    const auto parse_start = Clock::now();
    auto parsed = symft::parse_stim_file(options.path);
    const auto parse_stop = Clock::now();

    const auto plan_start = Clock::now();
    auto program = symft::plan_stim_factored_program(parsed);
    const auto plan_stop = Clock::now();

    RateBenchSetup setup;
    setup.batch_size = options.batch_size > 0
                           ? options.batch_size
                           : (options.postselect_detectors
                                  ? symft::default_postselected_batch_count(program.max_k)
                                  : symft::default_batch_count(program.max_k));
    setup.sample_chunk_shots =
        options.sample_chunk_shots > 0
            ? options.sample_chunk_shots
            : symft::default_single_shot_sample_chunk_shots();
    setup.logical_records = logical_records_for_observable(parsed.observables, options.observable);
    setup.parsed = std::move(parsed);
    setup.program = std::move(program);
    parse_s = seconds_between(parse_start, parse_stop);
    plan_s = seconds_between(plan_start, plan_stop);
    return setup;
}

BenchResult run_single_sampler(const Options& options) {
    BenchResult result;
    result.path = options.path;
    result.sampler = options.postselect_detectors ? "single_postselected" : "single";
    result.requested_shots = options.shots;
    result.repeats = options.repeats;
    result.observable = options.observable;
    result.threads = 1;
    result.requested_threads = options.threads;
    result.detector_postselection = options.postselect_detectors;
    result.exogenous_mode = "packed_presampled_expression_streaming";
    result.phase_timing = "wall";

    double parse_total = 0.0;
    double plan_total = 0.0;
    const auto setup = build_rate_bench_setup(options, parse_total, plan_total);
    result.n = setup.parsed.state.n;
    result.records = setup.program.nrecords;
    result.max_k = setup.program.max_k;
    result.detectors = static_cast<int>(setup.parsed.detectors.size());
    result.observable_includes = static_cast<int>(setup.parsed.observables.size());
    result.batch_size = setup.batch_size;
    result.sample_chunk_shots = setup.sample_chunk_shots;

    double presample_total = 0.0;
    double execute_total = 0.0;
    double accumulate_total = 0.0;
    double sample_wall_total = 0.0;

    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        const std::uint64_t nchunks = ceil_div_u64(
            options.shots,
            static_cast<std::uint64_t>(result.sample_chunk_shots));
        symft::FactoredExecutorState runtime(setup.program, block_seed(0x5eed1234ULL, repeat, 0));
        symft::PackedPresampledExogenous packed_samples;
        symft::PresampledExpressionPlan packed_expression_plan;
        symft::PresampledExpressionBlock packed_expression_block;
        symft::prepare_presampled_exogenous_packed(packed_samples, setup.program);
        symft::prepare_presampled_expression_plan(
            packed_expression_plan,
            setup.program,
            packed_samples);

        const auto sample_wall_start = Clock::now();
        RateCounts counts;
        for (std::uint64_t chunk_index = 0; chunk_index < nchunks; ++chunk_index) {
            const std::uint64_t offset = chunk_index * static_cast<std::uint64_t>(result.sample_chunk_shots);
            const int chunk = static_cast<int>(std::min<std::uint64_t>(
                static_cast<std::uint64_t>(result.sample_chunk_shots),
                options.shots - offset));

            const auto presample_start = Clock::now();
            symft::resample_prepared_exogenous_packed_in_place(
                packed_samples,
                setup.program,
                chunk,
                block_seed(0x7eed0000ULL, repeat, chunk_index));
            symft::evaluate_presampled_expression_block(
                packed_expression_block,
                packed_expression_plan,
                packed_samples);
            const auto presample_stop = Clock::now();

            runtime.rng_state = block_seed(0x5eed1234ULL, repeat, chunk_index);
            for (int shot = 0; shot < chunk; ++shot) {
                const auto execute_start = Clock::now();
                symft::reset_executor(runtime, setup.program);
                if (options.postselect_detectors) {
                    const bool survived = symft::execute_postselected_in_place(
                        runtime,
                        setup.program,
                        packed_expression_plan,
                        packed_expression_block,
                        shot);
                    const auto execute_stop = Clock::now();
                    if (!survived) {
                        ++counts.discarded;
                    } else {
                        accumulate_single_counts(
                            counts,
                            runtime.measurement_words,
                            runtime.detector_words,
                            setup.program.ndetectors,
                            setup.logical_records);
                    }
                    const auto accumulate_stop = Clock::now();
                    execute_total += seconds_between(execute_start, execute_stop);
                    accumulate_total += seconds_between(execute_stop, accumulate_stop);
                    continue;
                } else {
                    symft::execute_in_place(
                        runtime,
                        setup.program,
                        packed_expression_plan,
                        packed_expression_block,
                        shot);
                }
                const auto execute_stop = Clock::now();
                accumulate_single_counts(
                    counts,
                    runtime.measurement_words,
                    runtime.detector_words,
                    setup.program.ndetectors,
                    setup.logical_records);
                const auto accumulate_stop = Clock::now();
                execute_total += seconds_between(execute_start, execute_stop);
                accumulate_total += seconds_between(execute_stop, accumulate_stop);
            }
            counts.shots += static_cast<std::uint64_t>(chunk);
            presample_total += seconds_between(presample_start, presample_stop);
        }
        const auto sample_wall_stop = Clock::now();

        result.counts.shots += counts.shots;
        result.counts.discarded += counts.discarded;
        result.counts.accepted += counts.accepted;
        result.counts.logical_errors += counts.logical_errors;
        sample_wall_total += seconds_between(sample_wall_start, sample_wall_stop);
    }

    const double inv_repeats = 1.0 / static_cast<double>(options.repeats);
    result.parse_s = parse_total;
    result.plan_s = plan_total;
    result.presample_s = presample_total * inv_repeats;
    result.execute_s = execute_total * inv_repeats;
    result.accumulate_s = accumulate_total * inv_repeats;
    result.sample_s = sample_wall_total * inv_repeats;
    return result;
}

BenchResult run_batch_sampler(const Options& options) {
    BenchResult result;
    result.path = options.path;
    result.requested_shots = options.shots;
    result.repeats = options.repeats;
    result.observable = options.observable;
    result.requested_threads = options.threads;
    result.detector_postselection = options.postselect_detectors;
    result.batch_mask_threshold_denominator =
        options.postselect_detectors ? options.batch_mask_threshold_denominator : 0;
    result.exogenous_mode = options.postselect_detectors
                                 ? "packed_presampled_expression_streaming"
                                 : "packed_presampled_streaming";
    result.rng_streams = "split_exogenous_branch";

    double parse_total = 0.0;
    double plan_total = 0.0;
    const auto setup = build_rate_bench_setup(options, parse_total, plan_total);
    result.n = setup.parsed.state.n;
    result.records = setup.program.nrecords;
    result.max_k = setup.program.max_k;
    result.batch_size = setup.batch_size;
    result.sample_chunk_shots = setup.sample_chunk_shots;
    result.detectors = static_cast<int>(setup.parsed.detectors.size());
    result.observable_includes = static_cast<int>(setup.parsed.observables.size());

    double presample_total = 0.0;
    double execute_total = 0.0;
    double accumulate_total = 0.0;
    double sample_wall_total = 0.0;
    const std::uint64_t nchunks = ceil_div_u64(
        options.shots,
        static_cast<std::uint64_t>(result.sample_chunk_shots));
    const std::uint64_t blocks_per_chunk = ceil_div_u64(
        static_cast<std::uint64_t>(result.sample_chunk_shots),
        static_cast<std::uint64_t>(result.batch_size));
    const int active_threads = std::min<int>(
        options.threads,
        static_cast<int>(std::max<std::uint64_t>(1, nchunks)));
    const auto postselection_options = batch_postselection_options(options, setup.logical_records);

    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        struct BatchWorkerContext {
            WorkerResult worker_result;
            symft::BatchFactoredExecutorState runtime;
            std::vector<std::uint64_t> discard_bits;
            std::vector<std::uint64_t> logical_bits;
            std::vector<std::uint64_t> scratch;
            symft::BatchDetectorPostselectionScratch postselection_scratch;
            symft::PackedPresampledExogenous samples;
            symft::PresampledExpressionPlan expression_plan;
            symft::PresampledExpressionBlock expression_block;

            BatchWorkerContext(
                const symft::FactoredInstructionProgram& program,
                int batch_size,
                std::uint64_t seed)
                : runtime(program, batch_size, seed),
                  discard_bits(runtime.batch_words, 0),
                  logical_bits(runtime.batch_words, 0),
                  scratch(runtime.batch_words, 0) {}
        };

        std::vector<BatchWorkerContext> worker_contexts;
        worker_contexts.reserve(static_cast<std::size_t>(active_threads));
        for (int worker_id = 0; worker_id < active_threads; ++worker_id) {
            worker_contexts.emplace_back(
                setup.program,
                result.batch_size,
                block_seed(0x5eed1234ULL, repeat, static_cast<std::uint64_t>(worker_id)));
            auto& context = worker_contexts.back();
            if (options.postselect_detectors) {
                symft::prepare_batch_detector_postselection_scratch(
                    context.postselection_scratch,
                    context.runtime,
                    setup.program,
                    postselection_options);
            }
            symft::prepare_presampled_exogenous_packed(context.samples, setup.program);
            if (options.postselect_detectors) {
                symft::prepare_presampled_expression_plan(
                    context.expression_plan,
                    setup.program,
                    context.samples);
            }
        }

        const auto sample_wall_start = Clock::now();
        auto run_worker = [&](int worker_id) {
            WorkerResult local;
            auto& context = worker_contexts[static_cast<std::size_t>(worker_id)];
            auto& runtime = context.runtime;
            auto& discard_bits = context.discard_bits;
            auto& logical_bits = context.logical_bits;
            auto& scratch = context.scratch;
            auto& postselection_scratch = context.postselection_scratch;
            auto& samples = context.samples;
            auto& expression_plan = context.expression_plan;
            auto& expression_block = context.expression_block;
            for (std::uint64_t chunk_index = static_cast<std::uint64_t>(worker_id);
                 chunk_index < nchunks;
                 chunk_index += static_cast<std::uint64_t>(active_threads)) {
                const std::uint64_t chunk_offset =
                    chunk_index * static_cast<std::uint64_t>(result.sample_chunk_shots);
                const int chunk_shots = static_cast<int>(std::min<std::uint64_t>(
                    static_cast<std::uint64_t>(result.sample_chunk_shots),
                    options.shots - chunk_offset));

                const auto presample_start = Clock::now();
                symft::resample_prepared_exogenous_packed_in_place(
                    samples,
                    setup.program,
                    chunk_shots,
                    block_seed(0x7eed0000ULL, repeat, chunk_index));
                if (options.postselect_detectors) {
                    symft::evaluate_presampled_expression_block(
                        expression_block,
                        expression_plan,
                        samples);
                }
                const auto presample_stop = Clock::now();
                local.presample_s += seconds_between(presample_start, presample_stop);

                for (int chunk_local_offset = 0, local_block_index = 0;
                     chunk_local_offset < chunk_shots;
                     chunk_local_offset += result.batch_size, ++local_block_index) {
                    const int block = std::min(result.batch_size, chunk_shots - chunk_local_offset);
                    const std::uint64_t block_index =
                        chunk_index * blocks_per_chunk + static_cast<std::uint64_t>(local_block_index);
                    const auto execute_start = Clock::now();
                    symft::reset_batch_executor(
                        runtime,
                        setup.program,
                        block,
                        !options.postselect_detectors);
                    runtime.rng_state = block_seed(0x5eed1234ULL, repeat, block_index);
                    if (options.postselect_detectors) {
                        const auto postselection_result = symft::execute_batch_postselected_in_place(
                            runtime,
                            setup.program,
                            expression_plan,
                            expression_block,
                            chunk_local_offset,
                            postselection_scratch,
                            postselection_options);
                        local.counts.discarded +=
                            static_cast<std::uint64_t>(postselection_result.discarded);
                    } else {
                        symft::execute_batch_in_place(runtime, setup.program, samples, chunk_local_offset);
                    }
                    const auto execute_stop = Clock::now();
                    if (options.postselect_detectors) {
                        accumulate_logical_counts_for_survivors(
                            local.counts,
                            runtime,
                            setup.logical_records,
                            logical_bits,
                            scratch);
                    } else {
                        accumulate_block_counts(
                            local.counts,
                            runtime,
                            block,
                            setup.logical_records,
                            discard_bits,
                            logical_bits,
                            scratch);
                    }
                    const auto accumulate_stop = Clock::now();
                    local.counts.shots += static_cast<std::uint64_t>(block);
                    local.execute_s += seconds_between(execute_start, execute_stop);
                    local.accumulate_s += seconds_between(execute_stop, accumulate_stop);
                }
            }
            context.worker_result = local;
        };
        if (active_threads == 1) {
            run_worker(0);
        } else {
            std::vector<std::thread> workers;
            workers.reserve(static_cast<std::size_t>(active_threads));
            for (int worker_id = 0; worker_id < active_threads; ++worker_id) {
                workers.emplace_back(run_worker, worker_id);
            }
            for (auto& worker : workers) {
                worker.join();
            }
        }
        const auto sample_wall_stop = Clock::now();

        for (const auto& context : worker_contexts) {
            result.counts.shots += context.worker_result.counts.shots;
            result.counts.discarded += context.worker_result.counts.discarded;
            result.counts.accepted += context.worker_result.counts.accepted;
            result.counts.logical_errors += context.worker_result.counts.logical_errors;
            presample_total += context.worker_result.presample_s;
            execute_total += context.worker_result.execute_s;
            accumulate_total += context.worker_result.accumulate_s;
        }
        sample_wall_total += seconds_between(sample_wall_start, sample_wall_stop);
    }

    const double inv_repeats = 1.0 / static_cast<double>(options.repeats);
    result.sampler = options.postselect_detectors ? "batch_postselected" : "batch";
    result.threads = active_threads;
    result.phase_timing = active_threads > 1 ? "worker_sum" : "wall";
    result.parse_s = parse_total;
    result.plan_s = plan_total;
    result.presample_s = presample_total * inv_repeats;
    result.execute_s = execute_total * inv_repeats;
    result.accumulate_s = accumulate_total * inv_repeats;
    result.sample_s = sample_wall_total * inv_repeats;
    return result;
}

void print_result(const BenchResult& result) {
    const bool is_batch_sampler = result.sampler == "batch" || result.sampler == "batch_postselected";
    const double shots_per_s = result.sample_s > 0.0 ? static_cast<double>(result.requested_shots) / result.sample_s : 0.0;
    const double discard_rate = result.counts.shots == 0
                                    ? std::numeric_limits<double>::quiet_NaN()
                                    : static_cast<double>(result.counts.discarded) / static_cast<double>(result.counts.shots);
    const double logical_rate = result.counts.accepted == 0
                                    ? std::numeric_limits<double>::quiet_NaN()
                                    : static_cast<double>(result.counts.logical_errors) / static_cast<double>(result.counts.accepted);

    std::cout << "sampler " << result.sampler << "\n";
    std::cout << "file " << result.path << "\n";
    std::cout << "shots " << result.requested_shots << "\n";
    std::cout << "sampled_shots " << result.counts.shots << "\n";
    std::cout << "detector_postselection " << (result.detector_postselection ? "enabled" : "disabled") << "\n";
    if (is_batch_sampler) {
        std::cout << "batch_size " << result.batch_size << "\n";
    }
    std::cout << "sample_chunk_shots " << result.sample_chunk_shots << "\n";
    std::cout << "repeats " << result.repeats << "\n";
    if (is_batch_sampler) {
        std::cout << "threads " << result.threads << "\n";
        if (result.requested_threads != result.threads) {
            std::cout << "requested_threads " << result.requested_threads << "\n";
        }
    }
    if (result.sampler == "batch_postselected") {
        std::cout << "batch_mask_threshold_denominator "
                  << result.batch_mask_threshold_denominator << "\n";
    }
    std::cout << "sample_s_avg " << result.sample_s << "\n";
    std::cout << "sample_shots_per_s " << shots_per_s << "\n";
    std::cout << "presample_s_avg " << result.presample_s << "\n";
    std::cout << "execute_s_avg " << result.execute_s << "\n";
    std::cout << "accumulate_s_avg " << result.accumulate_s << "\n";
    std::cout << "discarded " << result.counts.discarded << "\n";
    std::cout << "accepted " << result.counts.accepted << "\n";
    std::cout << "logical_errors " << result.counts.logical_errors << "\n";
    std::cout << "discard_rate " << discard_rate << "\n";
    std::cout << "logical_error_rate " << logical_rate << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.sampler == SamplerMode::Single) {
            print_result(run_single_sampler(options));
        } else if (options.sampler == SamplerMode::Batch) {
            print_result(run_batch_sampler(options));
        } else {
            print_result(run_single_sampler(options));
            std::cout << "\n";
            print_result(run_batch_sampler(options));
        }
    } catch (const std::exception& ex) {
        std::cerr << "symft_rate_bench: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
