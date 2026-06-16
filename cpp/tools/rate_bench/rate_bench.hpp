#pragma once

#include "frontend/stim.hpp"
#include "sampler/batch_sampler.hpp"
#include "sampler/exogenous.hpp"
#include "sampler/single_shot.hpp"

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace symft_rate_bench {

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
    int block_shots = 0;
    int repeats = 1;
    int threads = 1;
    int requested_threads = 1;
    bool detector_postselection = false;
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

struct Options {
    std::string path = "d3.stim";
    std::uint64_t shots = 100000000ULL;
    int block_shots = 0;
    int repeats = 1;
    int observable = 0;
    int threads = 1;
    bool postselect_detectors = false;
    SamplerMode sampler = SamplerMode::Batch;
};

double seconds_between(Clock::time_point start, Clock::time_point stop);
Options parse_options(int argc, char** argv);
void print_usage(const char* argv0);
std::uint64_t block_seed(std::uint64_t base, int repeat, std::uint64_t block_index);
std::vector<std::vector<int>> logical_records_for_observable(
    const std::vector<symft::StimObservableInclude>& observables,
    int observable);
std::vector<std::vector<std::vector<int>>> detectors_by_max_record(
    const std::vector<symft::StimDetector>& detectors,
    int nrecords);
std::vector<int> instruction_records(const symft::FactoredInstructionProgram& program);
std::vector<int> condition_last_uses(const symft::FactoredInstructionProgram& program);
void accumulate_block_counts(
    RateCounts& counts,
    const symft::BatchFactoredExecutorState& runtime,
    int block,
    const std::vector<symft::StimDetector>& detectors,
    const std::vector<std::vector<int>>& logical_records,
    std::vector<std::uint64_t>& discard_bits,
    std::vector<std::uint64_t>& logical_bits,
    std::vector<std::uint64_t>& scratch);
void execute_postselected_block(
    RateCounts& counts,
    symft::BatchFactoredExecutorState& runtime,
    const symft::FactoredInstructionProgram& program,
    const symft::PresampledExogenous& samples,
    const std::vector<int>& instruction_records_by_index,
    const std::vector<int>& condition_last_use,
    const std::vector<std::vector<std::vector<int>>>& detectors_by_record,
    const std::vector<std::vector<int>>& logical_records,
    std::vector<std::uint64_t>& detector_bits,
    std::vector<std::uint64_t>& dead_bits,
    std::vector<std::uint64_t>& keep_bits,
    std::vector<std::uint64_t>& scratch,
    std::vector<std::uint64_t>& compact_scratch);
void accumulate_single_counts(
    RateCounts& counts,
    const std::vector<std::uint64_t>& measurement_words,
    const std::vector<symft::StimDetector>& detectors,
    const std::vector<std::vector<int>>& logical_records);
bool any_detector_fires(
    const std::vector<std::uint64_t>& measurement_words,
    const std::vector<std::vector<int>>& detectors);
BenchResult run_single_sampler(const Options& options);
BenchResult run_batch_sampler(const Options& options);
void print_result(const BenchResult& result);

} // namespace symft_rate_bench
