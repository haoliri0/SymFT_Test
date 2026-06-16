#pragma once

#include "factored/factored.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace symft {

struct PresampledExogenous;
struct BatchThreadPool;

// Active storage is a Julia-column-major equivalent SoA layout:
// active_re[basis * batches + shot] and active_im[basis * batches + shot].
// Thus shots are contiguous for each active basis column.
struct BatchFactoredExecutorState {
    int n = 0;
    int k = 0;
    int ndormant = 0;
    int batches = 0;
    int active_shots = 0;
    int nsymbols = 0;
    int nrecords = 0;
    int max_k = 0;
    std::size_t batch_words = 0;
    std::vector<double> active_re;
    std::vector<double> active_im;
    std::vector<double> scratch_re;
    std::vector<double> scratch_im;
    std::vector<std::uint64_t> value_words;
    std::vector<std::uint64_t> assigned_words;
    std::vector<std::uint64_t> measurement_words;
    std::vector<std::uint64_t> eval_scratch;
    std::vector<double> rotation_coefficients;
    std::vector<double> branch_prob_true;
    std::vector<double> branch_invnorms;
    std::uint64_t rng_state = 1;
    int threads = 1;
    std::shared_ptr<BatchThreadPool> thread_pool;

    explicit BatchFactoredExecutorState(
        const FactoredInstructionProgram& program,
        int batches = 0,
        std::uint64_t seed = 1,
        int threads = 1);
};

int default_batch_count(int max_k);
void set_batch_executor_threads(BatchFactoredExecutorState& runtime, int threads);
void reset_batch_executor(BatchFactoredExecutorState& runtime, const FactoredInstructionProgram& program, int shots);
void execute_batch_in_place(BatchFactoredExecutorState& runtime, const FactoredInstructionProgram& program);
void execute_batch_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExogenous& samples);
const char* active_batch_backend();
std::vector<std::vector<std::uint64_t>> sample_measurements_batch(
    const FactoredInstructionProgram& program,
    int shots,
    int batches = 0,
    std::uint64_t seed = 1,
    int threads = 1);
void assign_presampled_exogenous_batch_in_place(
    BatchFactoredExecutorState& runtime,
    const PresampledExogenous& samples);
void execute_batch_instruction_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstruction& instruction);

} // namespace symft
