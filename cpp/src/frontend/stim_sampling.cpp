#include "frontend/stim.hpp"

#include "sampler/single_shot.hpp"

#include <limits>

namespace symft {

StimSampleSummary estimate_stim_logical_error_rate(const StimParseResult& parsed, int shots, std::uint64_t seed) {
    PendingFactoredState pending(parsed.state);
    const FactoredInstructionProgram program = plan_factored_updates(pending);
    const auto records = sample_measurements(program, shots, seed);
    StimSampleSummary summary;
    summary.shots = shots;
    for (const auto& shot_records : records) {
        bool discarded = false;
        for (const auto& detector : parsed.detectors) {
            bool parity = false;
            for (int record : detector.records) {
                parity ^= packed_bit(shot_records, record - 1);
            }
            discarded = discarded || parity;
        }
        if (discarded) {
            ++summary.discarded;
            continue;
        }
        ++summary.accepted;
        bool logical = false;
        for (const auto& observable : parsed.observables) {
            bool parity = false;
            for (int record : observable.records) {
                parity ^= packed_bit(shot_records, record - 1);
            }
            logical ^= parity;
        }
        if (logical) {
            ++summary.logical_errors;
        }
    }
    return summary;
}

double discard_rate(const StimSampleSummary& summary) {
    return summary.shots == 0 ? std::numeric_limits<double>::quiet_NaN()
                              : static_cast<double>(summary.discarded) / static_cast<double>(summary.shots);
}

double logical_error_rate(const StimSampleSummary& summary) {
    return summary.accepted == 0 ? std::numeric_limits<double>::quiet_NaN()
                                 : static_cast<double>(summary.logical_errors) / static_cast<double>(summary.accepted);
}

} // namespace symft
