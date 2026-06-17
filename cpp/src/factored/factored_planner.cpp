#include "factored/factored_internal.hpp"

#include <algorithm>
#include <optional>
#include <type_traits>
#include <utility>

namespace symft {
using namespace detail;

bool has_pending_operations(const PendingFactoredState& state) {
    return !state.pending_operations.empty();
}

namespace {

void set_planning_active_count(PendingFactoredState& state, int k) {
    const int ki = checked_nqubits(k);
    if (ki > state.n) {
        fail("active qubit count exceeds total qubit count");
    }
    state.k = ki;
    state.max_k = std::max(state.max_k, state.k);
    state.dormant = DormantState(state.n - state.k, state.context);
}

FactoredInstruction push_instruction(PendingFactoredState& state, FactoredInstruction instruction) {
    state.context->bump_next_condition(max_condition(instruction));
    state.instructions.push_back(instruction);
    return instruction;
}

std::optional<int> measurement_record(PendingFactoredState& state, const PendingPauliMeasurement& measurement) {
    if (!measurement.record) {
        if (measurement.record_condition) {
            return std::nullopt;
        }
        return state.next_record++;
    }
    state.next_record = std::max(state.next_record, *measurement.record + 1);
    return measurement.record;
}

PauliString forward_conjugate_H(const PauliString& pauli, int q) {
    CliffordFrame gate(pauli.nqubits);
    left_H(gate, q);
    return preimage(gate, pauli);
}

PauliString forward_conjugate_S(const PauliString& pauli, int q) {
    CliffordFrame gate(pauli.nqubits);
    left_SDG(gate, q);
    return preimage(gate, pauli);
}

PauliString forward_conjugate_CX(const PauliString& pauli, int c, int t) {
    CliffordFrame gate(pauli.nqubits);
    left_CX(gate, c, t);
    return preimage(gate, pauli);
}

PauliString forward_conjugate_CZ(const PauliString& pauli, int a, int b) {
    CliffordFrame gate(pauli.nqubits);
    left_CZ(gate, a, b);
    return preimage(gate, pauli);
}

PauliString forward_conjugate_SWAP(const PauliString& pauli, int a, int b) {
    CliffordFrame gate(pauli.nqubits);
    left_SWAP(gate, a, b);
    return preimage(gate, pauli);
}

struct BasisChange {
    enum Kind { H, S, CX, CZ, SWAP } kind;
    int a = 0;
    int b = 0;
};

PauliString transform_pauli_by_basis_changes(PauliString pauli, const std::vector<BasisChange>& changes) {
    for (const auto& change : changes) {
        switch (change.kind) {
        case BasisChange::H:
            pauli = forward_conjugate_H(pauli, change.a);
            break;
        case BasisChange::S:
            pauli = forward_conjugate_S(pauli, change.a);
            break;
        case BasisChange::CX:
            pauli = forward_conjugate_CX(pauli, change.a, change.b);
            break;
        case BasisChange::CZ:
            pauli = forward_conjugate_CZ(pauli, change.a, change.b);
            break;
        case BasisChange::SWAP:
            pauli = forward_conjugate_SWAP(pauli, change.a, change.b);
            break;
        }
    }
    return pauli;
}

PendingPauliRotation transform_operation_by_basis_changes(
    const PendingPauliRotation& operation,
    const std::vector<BasisChange>& changes) {
    return PendingPauliRotation{
        operation.theta,
        SymbolicPauliString(transform_pauli_by_basis_changes(operation.pauli.pauli, changes), operation.pauli.sign),
    };
}

PendingPauliMeasurement transform_operation_by_basis_changes(
    const PendingPauliMeasurement& operation,
    const std::vector<BasisChange>& changes) {
    return PendingPauliMeasurement{
        SymbolicPauliString(transform_pauli_by_basis_changes(operation.pauli.pauli, changes), operation.pauli.sign),
        operation.record,
        operation.record_condition,
    };
}

PendingOperation transform_operation_by_basis_changes(const PendingOperation& operation, const std::vector<BasisChange>& changes) {
    return std::visit([&](const auto& op) -> PendingOperation { return transform_operation_by_basis_changes(op, changes); }, operation);
}

void transform_pending_operations_by_basis_changes(PendingFactoredState& state, const std::vector<BasisChange>& changes) {
    for (auto& operation : state.pending_operations) {
        operation = transform_operation_by_basis_changes(operation, changes);
    }
}

PendingPauliRotation rewrite_current_and_pending_by_basis_change(
    PendingFactoredState& state,
    const PendingPauliRotation& current,
    BasisChange change) {
    const std::vector<BasisChange> changes{change};
    transform_pending_operations_by_basis_changes(state, changes);
    return transform_operation_by_basis_changes(current, changes);
}

PendingPauliMeasurement rewrite_current_and_pending_by_basis_change(
    PendingFactoredState& state,
    const PendingPauliMeasurement& current,
    BasisChange change) {
    const std::vector<BasisChange> changes{change};
    transform_pending_operations_by_basis_changes(state, changes);
    return transform_operation_by_basis_changes(current, changes);
}

PendingPauliRotation xor_operation_sign_if_anticommutes(
    const PendingPauliRotation& operation,
    const PauliString& pauli,
    const SymbolicBool& sign) {
    if (!pauli_anticommutes(pauli, operation.pauli.pauli)) {
        return operation;
    }
    return PendingPauliRotation{operation.theta, SymbolicPauliString(operation.pauli.pauli, xor_bool(operation.pauli.sign, sign))};
}

PendingPauliMeasurement xor_operation_sign_if_anticommutes(
    const PendingPauliMeasurement& operation,
    const PauliString& pauli,
    const SymbolicBool& sign) {
    if (!pauli_anticommutes(pauli, operation.pauli.pauli)) {
        return operation;
    }
    return PendingPauliMeasurement{
        SymbolicPauliString(operation.pauli.pauli, xor_bool(operation.pauli.sign, sign)),
        operation.record,
        operation.record_condition,
    };
}

void push_symbolic_pauli_through_pending_from(
    PendingFactoredState& state,
    int first_index_one_based,
    const PauliString& pauli,
    const SymbolicBool& sign) {
    state.context->bump_next_condition(sign);
    const std::size_t start = first_index_one_based <= 1 ? 0 : static_cast<std::size_t>(first_index_one_based - 1);
    for (std::size_t idx = start; idx < state.pending_operations.size(); ++idx) {
        state.pending_operations[idx] = std::visit(
            [&](const auto& op) -> PendingOperation { return xor_operation_sign_if_anticommutes(op, pauli, sign); },
            state.pending_operations[idx]);
    }
}

std::optional<int> first_dormant_x_qubit(const PendingFactoredState& state, const PauliString& pauli) {
    for (int q = state.k; q < state.n; ++q) {
        if (pauli.xbit(q)) {
            return q - state.k;
        }
    }
    return std::nullopt;
}

SymbolicBool rotation_sign_from_pauli(const SymbolicPauliString& pauli) {
    return xor_bool(pauli.sign, measurement_phase_sign(pauli.pauli));
}

ApplyPrecomputedActivePauliRotation active_rotation_instruction(
    const PauliString& active_body,
    double theta,
    const SymbolicBool& sign) {
    ActivePauliAction action(active_body);
    return ApplyPrecomputedActivePauliRotation{
        active_body,
        action,
        PrecomputedActivePauliRotationKernel(action, theta),
        theta,
        sign,
        SymbolicBoolEvaluationPlan(sign),
    };
}

std::optional<FactoredInstruction> process_diagonal_dormant_rotation(
    PendingFactoredState& state,
    const PendingPauliRotation& current) {
    const PauliString active_body = project_pauli_body(current.pauli.pauli, 0, state.k);
    const SymbolicBool sign = rotation_sign_from_pauli(current.pauli);
    if (!active_body.has_nonidentity_body()) {
        return std::nullopt;
    }
    if (!pauli_squares_to_identity(active_body)) {
        fail("active rotation Pauli must square to identity");
    }
    return push_instruction(state, active_rotation_instruction(active_body, current.theta, sign));
}

PendingPauliRotation eliminate_active_support_for_dormant_rotation(
    PendingFactoredState& state,
    PendingPauliRotation current,
    int picked_dormant,
    int target_active) {
    const int control = state.k + picked_dormant;
    if (current.pauli.pauli.xbit(target_active)) {
        current = rewrite_current_and_pending_by_basis_change(
            state,
            current,
            {BasisChange::CX, control, target_active});
    }
    if (current.pauli.pauli.zbit(target_active)) {
        current = rewrite_current_and_pending_by_basis_change(
            state,
            current,
            {BasisChange::CZ, control, target_active});
    }
    return current;
}

PendingPauliRotation eliminate_other_dormant_support_for_rotation(
    PendingFactoredState& state,
    PendingPauliRotation current,
    int picked_dormant,
    int target_dormant) {
    const int q = state.k + target_dormant;
    if (current.pauli.pauli.xbit(q) && current.pauli.pauli.zbit(q)) {
        current = rewrite_current_and_pending_by_basis_change(state, current, {BasisChange::S, q, 0});
    }
    const int control = state.k + picked_dormant;
    const int target = state.k + target_dormant;
    if (current.pauli.pauli.xbit(target)) {
        current = rewrite_current_and_pending_by_basis_change(state, current, {BasisChange::CX, control, target});
    }
    if (current.pauli.pauli.zbit(target)) {
        current = rewrite_current_and_pending_by_basis_change(state, current, {BasisChange::CZ, control, target});
    }
    return current;
}

PendingPauliRotation normalize_first_dormant_rotation_to_x(PendingFactoredState& state, PendingPauliRotation current) {
    const int q = state.k;
    if (!current.pauli.pauli.xbit(q)) {
        fail("dormant rotation reduction lost first-dormant X support");
    }
    if (current.pauli.pauli.zbit(q)) {
        current = rewrite_current_and_pending_by_basis_change(state, current, {BasisChange::S, q, 0});
    }
    if (!current.pauli.pauli.xbit(q) || current.pauli.pauli.zbit(q)) {
        fail("failed to normalize dormant rotation Pauli to first-dormant X");
    }
    return current;
}

std::optional<FactoredInstruction> process_nondiagonal_dormant_rotation(
    PendingFactoredState& state,
    PendingPauliRotation current,
    int picked_dormant) {
    for (int q = 0; q < state.k; ++q) {
        current = eliminate_active_support_for_dormant_rotation(state, current, picked_dormant, q);
    }
    for (int d = 0; d < state.n - state.k; ++d) {
        if (d == picked_dormant) {
            continue;
        }
        current = eliminate_other_dormant_support_for_rotation(state, current, picked_dormant, d);
    }
    if (picked_dormant != 0) {
        current = rewrite_current_and_pending_by_basis_change(
            state,
            current,
            {BasisChange::SWAP, state.k, state.k + picked_dormant});
    }
    current = normalize_first_dormant_rotation_to_x(state, current);
    const SymbolicBool sign = rotation_sign_from_pauli(current.pauli);
    PromoteDormantRotation instruction{current.theta, sign, SymbolicBoolEvaluationPlan(sign)};
    FactoredInstruction pushed = push_instruction(state, instruction);
    set_planning_active_count(state, state.k + 1);
    return pushed;
}

SymbolicBool measurement_base_outcome(const SymbolicPauliString& pauli) {
    return xor_bool(pauli.sign, measurement_phase_sign(pauli.pauli));
}

std::optional<FactoredInstruction> record_deterministic_measurement(
    PendingFactoredState& state,
    const PendingPauliMeasurement& measurement,
    const SymbolicBool& outcome) {
    return push_instruction(
        state,
        RecordMeasurement{
            outcome,
            measurement_record(state, measurement),
            measurement.record_condition,
            SymbolicBoolEvaluationPlan(outcome),
        });
}

PendingPauliMeasurement eliminate_active_support_for_dormant_measurement(
    PendingFactoredState& state,
    PendingPauliMeasurement current,
    int picked_dormant,
    int target_active) {
    const int control = state.k + picked_dormant;
    if (current.pauli.pauli.xbit(target_active)) {
        current = rewrite_current_and_pending_by_basis_change(
            state,
            current,
            {BasisChange::CX, control, target_active});
    }
    if (current.pauli.pauli.zbit(target_active)) {
        current = rewrite_current_and_pending_by_basis_change(
            state,
            current,
            {BasisChange::CZ, control, target_active});
    }
    return current;
}

PendingPauliMeasurement normalize_picked_dormant_measurement_to_z(
    PendingFactoredState& state,
    PendingPauliMeasurement current,
    int picked_dormant) {
    const int q = state.k + picked_dormant;
    const bool xb = current.pauli.pauli.xbit(q);
    const bool zb = current.pauli.pauli.zbit(q);
    if (xb && zb) {
        current = rewrite_current_and_pending_by_basis_change(state, current, {BasisChange::S, q, 0});
        return rewrite_current_and_pending_by_basis_change(state, current, {BasisChange::H, q, 0});
    }
    if (xb) {
        return rewrite_current_and_pending_by_basis_change(state, current, {BasisChange::H, q, 0});
    }
    if (zb) {
        return current;
    }
    fail("dormant measurement reduction lost the picked Pauli support");
}

std::optional<FactoredInstruction> measure_dormant_xy_pauli(
    PendingFactoredState& state,
    PendingPauliMeasurement current,
    int picked_dormant,
    bool queued_first) {
    for (int d = 0; d < state.n - state.k; ++d) {
        if (d == picked_dormant) {
            continue;
        }
        const int q = state.k + d;
        if (current.pauli.pauli.xbit(q)) {
            current = rewrite_current_and_pending_by_basis_change(
                state,
                current,
                {BasisChange::CX, state.k + picked_dormant, q});
        }
    }
    for (int q = 0; q < state.k; ++q) {
        current = eliminate_active_support_for_dormant_measurement(state, current, picked_dormant, q);
    }
    current = normalize_picked_dormant_measurement_to_z(state, current, picked_dormant);
    const SymbolicBool base_outcome = measurement_base_outcome(current.pauli);
    const int branch = state.context->fresh_condition();
    const SymbolicBool branch_bit = symbolic_bool(branch);
    push_symbolic_pauli_through_pending_from(
        state,
        queued_first ? 2 : 1,
        pauli_x(state.n, state.k + picked_dormant),
        branch_bit);
    return push_instruction(
        state,
        IntroduceDormantMeasurementBranch{
            branch,
            xor_bool(base_outcome, branch_bit),
            measurement_record(state, current),
            current.record_condition,
            SymbolicBoolEvaluationPlan(xor_bool(base_outcome, branch_bit)),
        });
}

PauliString transform_by_frame(const CliffordFrame& frame, const PauliString& pauli) {
    return coordinates_in_frame(frame, pauli);
}

PendingPauliRotation transform_operation_by_frame(const PendingPauliRotation& operation, const CliffordFrame& frame) {
    return PendingPauliRotation{
        operation.theta,
        SymbolicPauliString(transform_by_frame(frame, operation.pauli.pauli), operation.pauli.sign),
    };
}

PendingPauliMeasurement transform_operation_by_frame(const PendingPauliMeasurement& operation, const CliffordFrame& frame) {
    return PendingPauliMeasurement{
        SymbolicPauliString(transform_by_frame(frame, operation.pauli.pauli), operation.pauli.sign),
        operation.record,
        operation.record_condition,
    };
}

void transform_pending_operations_by_frame(PendingFactoredState& state, const CliffordFrame& frame) {
    for (auto& op : state.pending_operations) {
        op = std::visit([&](const auto& typed) -> PendingOperation { return transform_operation_by_frame(typed, frame); }, op);
    }
}

CliffordFrame active_measurement_coordinate_frame(
    const PendingFactoredState& state,
    const PauliString& active_body,
    const PrecomputedActivePauliMeasurementKernel& kernel) {
    CliffordFrame frame(state.n);
    const int k = state.k;
    const int pivot = kernel.pivot;
    const PauliString measured = embed_active_pauli(state.n, active_body);
    const PauliString fixed_x = kernel.is_diagonal ? pauli_x(state.n, pivot) : pauli_z(state.n, pivot);
    frame.copy_pauli_to_row(frame.xrow(k - 1), fixed_x);
    frame.copy_pauli_to_row(frame.zrow(k - 1), measured);
    int new_q = 0;
    for (int old_q = 0; old_q < k; ++old_q) {
        if (old_q == pivot) {
            continue;
        }
        PauliString zrow = pauli_z(state.n, old_q);
        PauliString xrow = pauli_x(state.n, old_q);
        if (kernel.is_diagonal) {
            if (active_body.zbit(old_q)) {
                xrow = xrow * pauli_x(state.n, pivot);
            }
        } else {
            if (active_body.xbit(old_q)) {
                zrow = zrow * pauli_z(state.n, pivot);
            }
            if (active_body.zbit(old_q)) {
                xrow = xrow * pauli_z(state.n, pivot);
            }
        }
        frame.copy_pauli_to_row(frame.xrow(new_q), xrow);
        frame.copy_pauli_to_row(frame.zrow(new_q), zrow);
        ++new_q;
    }
    if (new_q != k - 1) {
        fail("active measurement tableau dropped the wrong number of qubits");
    }
    return frame;
}

std::optional<FactoredInstruction> measure_active_pauli_branches(
    PendingFactoredState& state,
    const PendingPauliMeasurement& current,
    const PauliString& active_body,
    const SymbolicBool& base_outcome,
    bool queued_first) {
    if (!pauli_squares_to_identity(active_body)) {
        fail("active measurement Pauli must square to identity");
    }
    const PrecomputedActivePauliMeasurementKernel kernel(active_body);
    const CliffordFrame frame = active_measurement_coordinate_frame(state, active_body, kernel);
    transform_pending_operations_by_frame(state, frame);
    const int branch = state.context->fresh_condition();
    const SymbolicBool branch_bit = symbolic_bool(branch);
    push_symbolic_pauli_through_pending_from(
        state,
        queued_first ? 2 : 1,
        pauli_x(state.n, state.k - 1),
        branch_bit);
    const SymbolicBool outcome = xor_bool(base_outcome, branch_bit);
    MeasurePrecomputedActivePauli instruction{
        active_body,
        kernel,
        branch,
        outcome,
        measurement_record(state, current),
        current.record_condition,
        SymbolicBoolEvaluationPlan(outcome),
    };
    FactoredInstruction pushed = push_instruction(state, instruction);
    set_planning_active_count(state, state.k - 1);
    return pushed;
}

} // namespace

std::optional<FactoredInstruction> process_pending_rotation(PendingFactoredState& state, const PendingPauliRotation& rotation) {
    state.context->bump_next_condition(max_condition(rotation));
    PendingPauliRotation current = rotation;
    const auto picked = first_dormant_x_qubit(state, current.pauli.pauli);
    if (!picked) {
        return process_diagonal_dormant_rotation(state, current);
    }
    return process_nondiagonal_dormant_rotation(state, current, *picked);
}

std::optional<FactoredInstruction> process_pending_measurement(PendingFactoredState& state, const PendingPauliMeasurement& measurement) {
    state.context->bump_next_condition(max_condition(measurement));
    const bool queued_first = !state.pending_operations.empty() && state.pending_operations.front() == PendingOperation(measurement);
    PendingPauliMeasurement current = measurement;
    const PauliString active_body = project_pauli_body(current.pauli.pauli, 0, state.k);
    const auto picked = first_dormant_x_qubit(state, current.pauli.pauli);
    if (picked) {
        return measure_dormant_xy_pauli(state, current, *picked, queued_first);
    }
    const SymbolicBool base_outcome = measurement_base_outcome(current.pauli);
    if (!active_body.has_nonidentity_body()) {
        return record_deterministic_measurement(state, current, base_outcome);
    }
    return measure_active_pauli_branches(state, current, active_body, base_outcome, queued_first);
}

std::optional<FactoredInstruction> process_next_pending_operation(PendingFactoredState& state) {
    if (state.pending_operations.empty()) {
        return std::nullopt;
    }
    const PendingOperation operation = state.pending_operations.front();
    const std::size_t start = state.instructions.size();
    std::optional<FactoredInstruction> result = std::visit(
        [&](const auto& op) -> std::optional<FactoredInstruction> {
            if constexpr (std::is_same_v<std::decay_t<decltype(op)>, PendingPauliRotation>) {
                return process_pending_rotation(state, op);
            } else {
                return process_pending_measurement(state, op);
            }
        },
        operation);
    state.pending_operations.erase(state.pending_operations.begin());
    if (state.instructions.size() == start) {
        return result;
    }
    return state.instructions.back();
}

std::vector<FactoredInstruction> process_pending_operations(PendingFactoredState& state) {
    const std::size_t start = state.instructions.size();
    while (has_pending_operations(state)) {
        process_next_pending_operation(state);
    }
    return std::vector<FactoredInstruction>(state.instructions.begin() + static_cast<std::ptrdiff_t>(start), state.instructions.end());
}

namespace {

void refresh_instruction_plans(ApplyPrecomputedActivePauliRotation& instruction) {
    instruction.sign_plan = SymbolicBoolEvaluationPlan(instruction.sign);
}

void refresh_instruction_plans(PromoteDormantRotation& instruction) {
    instruction.sign_plan = SymbolicBoolEvaluationPlan(instruction.sign);
}

void refresh_instruction_plans(RecordMeasurement& instruction) {
    instruction.outcome_plan = SymbolicBoolEvaluationPlan(instruction.outcome);
}

void refresh_instruction_plans(MeasureActiveLastZ& instruction) {
    instruction.outcome_plan = SymbolicBoolEvaluationPlan(instruction.outcome);
}

void refresh_instruction_plans(MeasurePrecomputedActivePauli& instruction) {
    instruction.outcome_plan = SymbolicBoolEvaluationPlan(instruction.outcome);
}

void refresh_instruction_plans(IntroduceDormantMeasurementBranch& instruction) {
    instruction.outcome_plan = SymbolicBoolEvaluationPlan(instruction.outcome);
}

void refresh_instruction_plans(FactoredInstruction& instruction) {
    std::visit([](auto& inst) { refresh_instruction_plans(inst); }, instruction);
}

struct RareInfo {
    double event_probability = 0.0;
    std::vector<int> event_rows;
    std::vector<double> event_probabilities;
};

std::optional<RareInfo> rare_categorical_sample_info(const SymbolicCategoricalDistribution& distribution) {
    std::optional<std::size_t> false_row;
    for (std::size_t row = 0; row < distribution.assignments.size(); ++row) {
        bool any_true = false;
        for (int bit = 0; bit < distribution.nbits; ++bit) {
            any_true = any_true || packed_bit(distribution.assignments[row], bit);
        }
        if (!any_true) {
            false_row = row;
            break;
        }
    }
    if (!false_row) {
        return std::nullopt;
    }
    const double event_probability = 1.0 - distribution.probabilities[*false_row];
    if (!(event_probability < kLowProbabilitySampleThreshold)) {
        return std::nullopt;
    }
    RareInfo info;
    info.event_probability = event_probability;
    if (event_probability > 0.0) {
        const double inv_event = 1.0 / event_probability;
        for (std::size_t row = 0; row < distribution.assignments.size(); ++row) {
            if (row == *false_row || distribution.probabilities[row] <= 0.0) {
                continue;
            }
            info.event_rows.push_back(static_cast<int>(row));
            info.event_probabilities.push_back(distribution.probabilities[row] * inv_event);
        }
    }
    return info;
}

void push_bernoulli_sample_group(std::vector<BernoulliSampleGroup>& groups, double probability, int condition) {
    for (auto& group : groups) {
        if (group.probability == probability) {
            group.conditions.push_back(condition);
            return;
        }
    }
    groups.push_back(BernoulliSampleGroup{probability, {condition}});
}

void push_rare_categorical_sample_group(
    std::vector<RareCategoricalSampleGroup>& groups,
    const SymbolicCategoricalDistribution& distribution,
    const RareInfo& info) {
    for (auto& group : groups) {
        if (group.event_probability == info.event_probability &&
            group.nbits == distribution.nbits &&
            group.assignments == distribution.assignments &&
            group.probabilities == distribution.probabilities &&
            group.event_rows == info.event_rows &&
            group.event_probabilities == info.event_probabilities) {
            group.conditions.push_back(distribution.conditions);
            return;
        }
    }
    groups.push_back(RareCategoricalSampleGroup{
        info.event_probability,
        distribution.nbits,
        {distribution.conditions},
        distribution.assignments,
        distribution.probabilities,
        info.event_rows,
        info.event_probabilities,
    });
}

void build_categorical_sample_plan(
    const std::vector<SymbolicCategoricalDistribution>& distributions,
    std::vector<SymbolicCategoricalDistribution>& scalar,
    std::vector<RareCategoricalSampleGroup>& rare_groups) {
    for (const auto& distribution : distributions) {
        const auto info = rare_categorical_sample_info(distribution);
        if (info) {
            push_rare_categorical_sample_group(rare_groups, distribution, *info);
        } else {
            scalar.push_back(distribution);
        }
    }
}

} // namespace

FactoredInstructionProgram::FactoredInstructionProgram(
    int n_,
    int initial_k_,
    std::vector<FactoredInstruction> instructions_,
    int max_k_,
    SymbolicContext context_)
    : n(checked_nqubits(n_)),
      initial_k(checked_nqubits(initial_k_)),
      max_k(checked_nqubits(max_k_)),
      instructions(std::move(instructions_)),
      context(std::move(context_)) {
    if (initial_k > n || max_k > n || initial_k > max_k) {
        fail("invalid factored instruction program dimensions");
    }
    int record_count = 0;
    for (auto& instruction : instructions) {
        refresh_instruction_plans(instruction);
        context.bump_next_condition(max_condition(instruction));
        std::visit(
            [&](const auto& inst) {
                if constexpr (requires { inst.record; }) {
                    if (inst.record) {
                        record_count = std::max(record_count, *inst.record);
                    }
                }
            },
            instruction);
    }
    nsymbols = std::max(0, context.next_condition - 1);
    nrecords = record_count;
    build_categorical_sample_plan(
        context.categorical_distributions,
        sampled_categorical_distributions,
        sampled_rare_categorical_groups);
    for (const auto& [condition, probability] : context.bernoulli_probabilities) {
        if (probability < kLowProbabilitySampleThreshold) {
            push_bernoulli_sample_group(sampled_low_probability_bernoulli_groups, probability, condition);
        } else {
            sampled_bernoulli_conditions.push_back(condition);
            sampled_bernoulli_probabilities.push_back(probability);
        }
    }
}

FactoredInstructionProgram factored_instruction_program(const PendingFactoredState& state) {
    return FactoredInstructionProgram(
        state.n,
        state.initial_k,
        state.instructions,
        state.max_k,
        *state.context);
}

FactoredInstructionProgram plan_factored_updates(PendingFactoredState& state) {
    process_pending_operations(state);
    return factored_instruction_program(state);
}

} // namespace symft
