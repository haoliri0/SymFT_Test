#pragma once

#include "factored/factored.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace symft {

struct PresampledExogenous {
    int nshots = 0;
    int nsymbols = 0;
    std::size_t nwords = 0;
    std::uint64_t next_rng_state = 0;
    std::vector<std::uint64_t> exogenous_assigned_words;
    std::vector<std::uint64_t> value_words;
};

struct FactoredExecutorState {
    int n = 0;
    int k = 0;
    int ndormant = 0;
    int nsymbols = 0;
    int nrecords = 0;
    std::vector<double> active_re;
    std::vector<double> active_im;
    std::vector<double> active_scratch_re;
    std::vector<double> active_scratch_im;
    std::vector<std::uint64_t> value_words;
    std::vector<std::uint64_t> assigned_words;
    std::vector<std::uint64_t> measurement_words;
    std::uint64_t rng_state = 1;

    explicit FactoredExecutorState(const FactoredInstructionProgram& program, std::uint64_t seed = 1);
};

void reset_executor(FactoredExecutorState& runtime, const FactoredInstructionProgram& program);
PresampledExogenous presample_exogenous(const FactoredInstructionProgram& program, int shots, std::uint64_t seed = 1);
void execute_in_place(FactoredExecutorState& runtime, const FactoredInstructionProgram& program);
void execute_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExogenous& samples,
    int shot_index);
std::vector<std::uint64_t> execute(FactoredExecutorState& runtime, const FactoredInstructionProgram& program);
std::vector<std::uint64_t> sample_measurements(const FactoredInstructionProgram& program, std::uint64_t seed = 1);
std::vector<std::vector<std::uint64_t>> sample_measurements(const FactoredInstructionProgram& program, int shots, std::uint64_t seed = 1);
void assign_presampled_exogenous_in_place(
    FactoredExecutorState& runtime,
    const PresampledExogenous& samples,
    int shot_index);
void execute_instruction_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstruction& instruction);
} // namespace symft
