#include "frontend/stim.hpp"

#include "core/internal.hpp"
#include "sampler/single_shot.hpp"

#include <limits>
#include <utility>
#include <vector>

namespace symft {
namespace {

using detail::fail;

int instruction_checkpoint_for_pending_prefix(
    const FactoredInstructionProgram& program,
    int pending_prefix) {
    if (pending_prefix < 0) {
        fail("detector pending-operation position is negative");
    }
    if (pending_prefix == 0 && program.pending_prefix_instruction_indices.empty()) {
        return 0;
    }
    if (pending_prefix >= static_cast<int>(program.pending_prefix_instruction_indices.size())) {
        fail("detector pending-operation position is out of range");
    }
    const int checkpoint = program.pending_prefix_instruction_indices[static_cast<std::size_t>(pending_prefix)];
    if (checkpoint < 0 || checkpoint > static_cast<int>(program.instructions.size())) {
        fail("detector instruction checkpoint is out of range");
    }
    return checkpoint;
}

SymbolicBool detector_expression(
    const StimDetector& detector,
    const std::vector<SymbolicBool>& measurement_records) {
    SymbolicBool out(false);
    for (int record : detector.records) {
        if (record <= 0 || record > static_cast<int>(measurement_records.size())) {
            fail("detector references an out-of-range measurement record");
        }
        out = xor_bool(out, measurement_records[static_cast<std::size_t>(record - 1)]);
    }
    return out;
}

FactoredInstructionProgram insert_stim_detector_events(
    FactoredInstructionProgram program,
    const std::vector<StimDetector>& detectors,
    const std::vector<SymbolicBool>& measurement_records) {
    if (detectors.empty()) {
        return program;
    }
    std::vector<std::vector<RecordDetector>> events(program.instructions.size() + 1);
    for (std::size_t idx = 0; idx < detectors.size(); ++idx) {
        const auto& detector = detectors[idx];
        RecordDetector instruction;
        instruction.detector = static_cast<int>(idx + 1);
        instruction.records = detector.records;
        instruction.outcome = detector_expression(detector, measurement_records);
        const int checkpoint = instruction_checkpoint_for_pending_prefix(program, detector.after_pending_operation);
        events[static_cast<std::size_t>(checkpoint)].push_back(std::move(instruction));
    }

    std::vector<FactoredInstruction> instructions;
    instructions.reserve(program.instructions.size() + detectors.size());
    for (auto& detector_instruction : events[0]) {
        instructions.push_back(std::move(detector_instruction));
    }
    for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
        instructions.push_back(std::move(program.instructions[idx]));
        for (auto& detector_instruction : events[idx + 1]) {
            instructions.push_back(std::move(detector_instruction));
        }
    }

    return FactoredInstructionProgram(
        program.n,
        program.initial_k,
        std::move(instructions),
        program.max_k,
        std::move(program.context));
}

} // namespace

FactoredInstructionProgram plan_stim_factored_program(const StimParseResult& parsed) {
    PendingFactoredState pending(parsed.state);
    auto program = plan_factored_updates(pending);
    return insert_stim_detector_events(
        std::move(program),
        parsed.detectors,
        parsed.measurement_records);
}

StimSampleSummary estimate_stim_logical_error_rate(const StimParseResult& parsed, int shots, std::uint64_t seed) {
    const auto program = plan_stim_factored_program(parsed);
    FactoredExecutorState runtime(program, seed);
    StimSampleSummary summary;
    summary.shots = shots;
    for (int shot = 0; shot < shots; ++shot) {
        reset_executor(runtime, program);
        execute_in_place(runtime, program);
        bool discarded = false;
        for (int detector = 0; detector < program.ndetectors; ++detector) {
            discarded = discarded || packed_bit(runtime.detector_words, detector);
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
                parity ^= packed_bit(runtime.measurement_words, record - 1);
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
