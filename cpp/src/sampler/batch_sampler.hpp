#pragma once

#include "factored/factored.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace symft {

struct PresampledExogenous;
struct PackedPresampledExogenous;

struct BatchDetectorPostselectionPlan {
    std::vector<int> instruction_records_by_index;
    std::vector<std::vector<std::vector<int>>> detectors_by_record;
    std::vector<int> condition_last_use_by_index;
    std::vector<int> record_last_use_by_index;
};

struct BatchDetectorPostselectionScratch {
    std::vector<std::uint64_t> detector_bits;
    std::vector<std::uint64_t> dead_bits;
    std::vector<std::uint64_t> keep_bits;
    std::vector<std::uint64_t> scratch;
    std::vector<std::uint64_t> compact_scratch;
    std::vector<int> live_sources;
    std::vector<int> tail_fill_moves;
    int dead_count = 0;
};

struct BatchDetectorPostselectionResult {
    int discarded = 0;
    int accepted = 0;
};

struct BatchDetectorPostselectionOptions {
    int dense_over_dead_max_fraction_denominator = 4;
    int unordered_tail_fill_max_dead_fraction_denominator = 5;
};

// Active storage is a Julia-column-major equivalent SoA layout:
// active_re[basis * batches + shot] and active_im[basis * batches + shot].
// Here batches is the fixed shot pitch/capacity. active_shots is only the
// current dense live prefix length; it is never used as the active column pitch.
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

    explicit BatchFactoredExecutorState(
        const FactoredInstructionProgram& program,
        int batches = 0,
        std::uint64_t seed = 1);
};

int default_batch_count(int max_k);
void reset_batch_executor(BatchFactoredExecutorState& runtime, const FactoredInstructionProgram& program, int shots);
void execute_batch_in_place(BatchFactoredExecutorState& runtime, const FactoredInstructionProgram& program);
void execute_batch_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExogenous& samples);
void execute_batch_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PackedPresampledExogenous& samples,
    int first_sample_shot = 0);
void prepare_batch_detector_postselection_scratch(
    BatchDetectorPostselectionScratch& scratch,
    const BatchFactoredExecutorState& runtime);
BatchDetectorPostselectionResult execute_batch_postselected_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExogenous& samples,
    const BatchDetectorPostselectionPlan& postselection,
    BatchDetectorPostselectionScratch& scratch,
    BatchDetectorPostselectionOptions options = {});
BatchDetectorPostselectionResult execute_batch_postselected_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PackedPresampledExogenous& samples,
    int first_sample_shot,
    const BatchDetectorPostselectionPlan& postselection,
    BatchDetectorPostselectionScratch& scratch,
    BatchDetectorPostselectionOptions options = {});
const char* active_batch_backend();
std::vector<std::vector<std::uint64_t>> sample_measurements_batch(
    const FactoredInstructionProgram& program,
    int shots,
    int batches = 0,
    std::uint64_t seed = 1);
void assign_presampled_exogenous_batch_in_place(
    BatchFactoredExecutorState& runtime,
    const PresampledExogenous& samples);
void assign_presampled_exogenous_batch_in_place(
    BatchFactoredExecutorState& runtime,
    const PackedPresampledExogenous& samples,
    int first_sample_shot = 0);
void execute_batch_instruction_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstruction& instruction);

} // namespace symft
