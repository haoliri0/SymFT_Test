#include "rate_bench/rate_bench.hpp"

#include <iostream>
#include <limits>

namespace symft_rate_bench {

void print_result(const BenchResult& result) {
    const double shots_per_s = result.sample_s > 0.0 ? static_cast<double>(result.requested_shots) / result.sample_s : 0.0;
    const double discard_rate = result.counts.shots == 0
                                    ? std::numeric_limits<double>::quiet_NaN()
                                    : static_cast<double>(result.counts.discarded) / static_cast<double>(result.counts.shots);
    const double logical_rate = result.counts.accepted == 0
                                    ? std::numeric_limits<double>::quiet_NaN()
                                    : static_cast<double>(result.counts.logical_errors) / static_cast<double>(result.counts.accepted);

    std::cout << "file " << result.path << "\n";
    std::cout << "qubits " << result.n << "\n";
    std::cout << "records " << result.records << "\n";
    std::cout << "detectors " << result.detectors << "\n";
    std::cout << "observable_includes " << result.observable_includes << "\n";
    std::cout << "observable " << result.observable << "\n";
    std::cout << "max_active_qubits " << result.max_k << "\n";
    std::cout << "simd_backend " << symft::active_simd_backend() << "\n";
    std::cout << "batch_backend " << symft::active_batch_backend() << "\n";
    std::cout << "shots " << result.requested_shots << "\n";
    std::cout << "sampled_shots " << result.counts.shots << "\n";
    std::cout << "batch_size " << result.block_shots << "\n";
    std::cout << "block_shots " << result.block_shots << "\n";
    std::cout << "repeats " << result.repeats << "\n";
    std::cout << "threads " << result.threads << "\n";
    std::cout << "requested_threads " << result.requested_threads << "\n";
    std::cout << "sampler " << result.sampler << "\n";
    std::cout << "detector_postselection " << (result.detector_postselection ? "enabled" : "disabled") << "\n";
    std::cout << "exogenous_mode " << result.exogenous_mode << "\n";
    std::cout << "rng_streams " << result.rng_streams << "\n";
    std::cout << "phase_timing " << result.phase_timing << "\n";
    std::cout << "parse_s_avg " << result.parse_s << "\n";
    std::cout << "plan_s_avg " << result.plan_s << "\n";
    std::cout << "presample_s_avg " << result.presample_s << "\n";
    std::cout << "execute_s_avg " << result.execute_s << "\n";
    std::cout << "accumulate_s_avg " << result.accumulate_s << "\n";
    std::cout << "sample_wall_s_avg " << result.sample_s << "\n";
    std::cout << "sample_s_avg " << result.sample_s << "\n";
    std::cout << "sample_shots_per_s " << shots_per_s << "\n";
    std::cout << "discarded " << result.counts.discarded << "\n";
    std::cout << "accepted " << result.counts.accepted << "\n";
    std::cout << "logical_errors " << result.counts.logical_errors << "\n";
    std::cout << "discard_rate " << discard_rate << "\n";
    std::cout << "logical_error_rate " << logical_rate << "\n";
}

} // namespace symft_rate_bench
