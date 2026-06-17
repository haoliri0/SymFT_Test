#pragma once

#include "factored/factored.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace symft {

struct PresampledExogenous;
struct PackedPresampledExogenous;

struct BatchPostselectionBoundaryPlan {
    std::size_t instruction_index = 0;
    std::uint64_t future_page_basis_touches = 0;
    std::vector<int> transfer_conditions;
    std::vector<int> transfer_records;
};

struct BatchDetectorPostselectionPlan {
    std::vector<int> instruction_records_by_index;
    std::vector<std::vector<std::vector<int>>> detectors_by_record;
    std::vector<int> condition_last_use_by_index;
    std::vector<int> record_last_use_by_index;
    std::vector<BatchPostselectionBoundaryPlan> boundary_plans;
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

struct BatchPostselectionCoalescingStats {
    std::uint64_t checks = 0;
    std::uint64_t accepted = 0;
    std::uint64_t rejected_small_savings = 0;
    std::uint64_t rejected_cost = 0;
    std::uint64_t old_nonempty_pages = 0;
    std::uint64_t new_pages_if_coalesced = 0;
    std::uint64_t saved_pages = 0;
    std::uint64_t live_lanes = 0;
    std::uint64_t moved_lanes = 0;
    std::uint64_t actual_moved_lanes = 0;
    std::uint64_t keep_basis_columns = 0;
    std::uint64_t future_page_basis_touches = 0;
    std::uint64_t state_move_units = 0;
    std::uint64_t saved_page_units = 0;
    std::uint64_t estimated_move_bytes = 0;
    std::uint64_t full_pages_executed = 0;
    std::uint64_t partial_pages_executed = 0;
    std::uint64_t partial_page_live_lanes = 0;
    double segment_execution_s = 0.0;
    double page_local_compaction_s = 0.0;
    double cross_page_coalescing_s = 0.0;
};

struct BatchDetectorPostselectionOptions {
    int dense_over_dead_max_fraction_denominator = 4;
    int unordered_tail_fill_max_dead_fraction_denominator = 5;
    int cross_page_coalescing_benefit_factor = 8;
    BatchPostselectionCoalescingStats* coalescing_stats = nullptr;
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
void prepare_batch_detector_postselection_boundaries(
    BatchDetectorPostselectionPlan& postselection,
    const FactoredInstructionProgram& program);
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
BatchDetectorPostselectionResult execute_batch_postselected_page_pool_in_place(
    std::vector<BatchFactoredExecutorState>& pages,
    const FactoredInstructionProgram& program,
    const BatchDetectorPostselectionPlan& postselection,
    std::vector<BatchDetectorPostselectionScratch>& scratches,
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
