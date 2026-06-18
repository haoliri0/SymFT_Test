#pragma once

#include "core/frames.hpp"
#include "sampler/active.hpp"

#include <memory>
#include <optional>
#include <variant>

namespace symft {

struct PendingPauliRotation {
    double theta = 0.0;
    SymbolicPauliString pauli;
};

struct PendingPauliMeasurement {
    SymbolicPauliString pauli;
    std::optional<int> record;
    std::optional<int> record_condition;
};

using PendingOperation = std::variant<PendingPauliRotation, PendingPauliMeasurement>;

bool operator==(const PendingPauliRotation& lhs, const PendingPauliRotation& rhs);
bool operator==(const PendingPauliMeasurement& lhs, const PendingPauliMeasurement& rhs);
bool operator==(const PendingOperation& lhs, const PendingOperation& rhs);

struct ApplyPrecomputedActivePauliRotation {
    PauliString pauli;
    ActivePauliAction action;
    PrecomputedActivePauliRotationKernel rotation_kernel;
    double theta = 0.0;
    SymbolicBool sign;
    SymbolicBoolEvaluationPlan sign_plan;
};

struct PromoteDormantRotation {
    double theta = 0.0;
    SymbolicBool sign;
    SymbolicBoolEvaluationPlan sign_plan;
};

struct RecordMeasurement {
    SymbolicBool outcome;
    std::optional<int> record;
    std::optional<int> record_condition;
    SymbolicBoolEvaluationPlan outcome_plan;
};

struct RecordDetector {
    SymbolicBool outcome;
    std::vector<int> records;
    int detector = 0;
    SymbolicBoolEvaluationPlan outcome_plan;
};

struct MeasureActiveLastZ {
    int branch = 0;
    SymbolicBool outcome;
    std::optional<int> record;
    std::optional<int> record_condition;
    SymbolicBoolEvaluationPlan outcome_plan;
};

struct MeasurePrecomputedActivePauli {
    PauliString pauli;
    PrecomputedActivePauliMeasurementKernel kernel;
    int branch = 0;
    SymbolicBool outcome;
    std::optional<int> record;
    std::optional<int> record_condition;
    SymbolicBoolEvaluationPlan outcome_plan;
};

struct IntroduceDormantMeasurementBranch {
    int branch = 0;
    SymbolicBool outcome;
    std::optional<int> record;
    std::optional<int> record_condition;
    SymbolicBoolEvaluationPlan outcome_plan;
};

using FactoredInstruction = std::variant<
    ApplyPrecomputedActivePauliRotation,
    PromoteDormantRotation,
    RecordMeasurement,
    RecordDetector,
    MeasureActiveLastZ,
    MeasurePrecomputedActivePauli,
    IntroduceDormantMeasurementBranch>;

bool operator==(const ApplyPrecomputedActivePauliRotation& lhs, const ApplyPrecomputedActivePauliRotation& rhs);
bool operator==(const PromoteDormantRotation& lhs, const PromoteDormantRotation& rhs);
bool operator==(const RecordMeasurement& lhs, const RecordMeasurement& rhs);
bool operator==(const RecordDetector& lhs, const RecordDetector& rhs);
bool operator==(const MeasureActiveLastZ& lhs, const MeasureActiveLastZ& rhs);
bool operator==(const MeasurePrecomputedActivePauli& lhs, const MeasurePrecomputedActivePauli& rhs);
bool operator==(const IntroduceDormantMeasurementBranch& lhs, const IntroduceDormantMeasurementBranch& rhs);
bool operator==(const FactoredInstruction& lhs, const FactoredInstruction& rhs);

struct BernoulliSampleGroup {
    double probability = 0.0;
    std::vector<int> conditions;
};

struct RareCategoricalSampleGroup {
    double event_probability = 0.0;
    int nbits = 0;
    std::vector<std::vector<int>> conditions;
    std::vector<std::vector<std::uint64_t>> assignments;
    std::vector<double> probabilities;
    std::vector<int> event_rows;
    std::vector<double> event_probabilities;
};

struct FrameFactoredState {
    int n = 0;
    int k = 0;
    CliffordFrame clifford;
    ActivePauliFrame active_frame;
    DormantState dormant;
    std::shared_ptr<SymbolicContext> context;
    std::vector<PendingOperation> pending_operations;

    explicit FrameFactoredState(int n = 0, int k = 0);
    FrameFactoredState(int n, int k, std::shared_ptr<SymbolicContext> context);
};

void left_H(FrameFactoredState& state, int q);
void left_S(FrameFactoredState& state, int q);
void left_SDG(FrameFactoredState& state, int q);
void left_X(FrameFactoredState& state, int q);
void left_Z(FrameFactoredState& state, int q);
void left_CX(FrameFactoredState& state, int control, int target);
void left_CZ(FrameFactoredState& state, int a, int b);
void left_SWAP(FrameFactoredState& state, int a, int b);
void apply_pauli(FrameFactoredState& state, const ConditionalPauliString& pauli);
void apply_pauli(FrameFactoredState& state, const PauliString& pauli, int condition);
void apply_pauli(FrameFactoredState& state, const PauliString& pauli, const SymbolicBool& condition);
PendingPauliRotation apply_pauli_rotation(FrameFactoredState& state, const PauliString& pauli, double theta);
PendingPauliMeasurement apply_pauli_measurement(FrameFactoredState& state, const PauliString& pauli);
PendingPauliMeasurement apply_pauli_measurement(
    FrameFactoredState& state,
    const PauliString& pauli,
    const SymbolicBool& sign,
    std::optional<int> record = std::nullopt,
    std::optional<int> record_condition = std::nullopt);

struct FactoredInstructionProgram;

struct PendingFactoredState {
    int n = 0;
    int initial_k = 0;
    int k = 0;
    int max_k = 0;
    DormantState dormant;
    std::shared_ptr<SymbolicContext> context;
    std::vector<PendingOperation> pending_operations;
    std::vector<FactoredInstruction> instructions;
    std::vector<int> pending_prefix_instruction_indices;
    int next_record = 1;

    PendingFactoredState() = default;
    explicit PendingFactoredState(const FrameFactoredState& state);
    PendingFactoredState(int n, int k = 0);
    PendingFactoredState(int n, int k, std::shared_ptr<SymbolicContext> context);
};

bool has_pending_operations(const PendingFactoredState& state);
std::optional<FactoredInstruction> process_next_pending_operation(PendingFactoredState& state);
std::optional<FactoredInstruction> process_pending_rotation(PendingFactoredState& state, const PendingPauliRotation& rotation);
std::optional<FactoredInstruction> process_pending_measurement(PendingFactoredState& state, const PendingPauliMeasurement& measurement);
std::vector<FactoredInstruction> process_pending_operations(PendingFactoredState& state);

struct FactoredInstructionProgram {
    int n = 0;
    int initial_k = 0;
    int max_k = 0;
    std::vector<FactoredInstruction> instructions;
    std::vector<int> pending_prefix_instruction_indices;
    SymbolicContext context;
    int nsymbols = 0;
    int nrecords = 0;
    int ndetectors = 0;
    std::vector<SymbolicCategoricalDistribution> sampled_categorical_distributions;
    std::vector<RareCategoricalSampleGroup> sampled_rare_categorical_groups;
    std::vector<int> sampled_bernoulli_conditions;
    std::vector<double> sampled_bernoulli_probabilities;
    std::vector<BernoulliSampleGroup> sampled_low_probability_bernoulli_groups;

    FactoredInstructionProgram() = default;
    FactoredInstructionProgram(
        int n,
        int initial_k,
        std::vector<FactoredInstruction> instructions,
        int max_k,
        SymbolicContext context = SymbolicContext(),
        std::vector<int> pending_prefix_instruction_indices = {});
};

FactoredInstructionProgram factored_instruction_program(const PendingFactoredState& state);
FactoredInstructionProgram plan_factored_updates(PendingFactoredState& state);

} // namespace symft
