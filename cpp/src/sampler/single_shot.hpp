#pragma once

#include "factored/factored.hpp"

#include <cstdint>
#include <vector>

namespace symft {

struct PresampledExogenous;

struct DetectorPostselectionPlan {
    std::vector<int> instruction_records_by_index;
    std::vector<std::vector<std::vector<int>>> detectors_by_record;
    std::vector<std::vector<std::vector<std::uint64_t>>> detector_masks_by_record;
    std::size_t record_words = 0;
};

struct SingleShotPresampledExpression {
    bool constant = false;
    int block_expression_index = 0;
    SymbolicBoolEvaluationPlan residual_plan;
    std::vector<int> exogenous_word_indices;
    std::vector<std::uint64_t> exogenous_word_masks;
};

struct SingleShotPresampledExpressionPlan {
    std::vector<SingleShotPresampledExpression> instruction_expressions;
    std::vector<SingleShotPresampledExpression> block_expressions;
};

struct SingleShotPresampledExpressionBlock {
    int nshots = 0;
    std::size_t shot_words = 0;
    std::vector<std::uint64_t> expression_words;
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
void execute_in_place(FactoredExecutorState& runtime, const FactoredInstructionProgram& program);
void execute_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExogenous& samples,
    int shot_index);
void prepare_single_shot_presampled_expression_plan(
    SingleShotPresampledExpressionPlan& out,
    const FactoredInstructionProgram& program,
    const PresampledExogenous& samples);
void evaluate_single_shot_presampled_expression_block(
    SingleShotPresampledExpressionBlock& out,
    const SingleShotPresampledExpressionPlan& plan,
    const PresampledExogenous& samples);
void execute_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExogenous& samples,
    const SingleShotPresampledExpressionPlan& expression_plan,
    int shot_index);
void execute_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const SingleShotPresampledExpressionPlan& expression_plan,
    const SingleShotPresampledExpressionBlock& expression_block,
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
bool execute_postselected_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExogenous& samples,
    int shot_index,
    const DetectorPostselectionPlan& postselection);
bool execute_postselected_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExogenous& samples,
    const SingleShotPresampledExpressionPlan& expression_plan,
    int shot_index,
    const DetectorPostselectionPlan& postselection);
bool execute_postselected_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const SingleShotPresampledExpressionPlan& expression_plan,
    const SingleShotPresampledExpressionBlock& expression_block,
    int shot_index,
    const DetectorPostselectionPlan& postselection);
} // namespace symft
