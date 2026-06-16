#include "../core/symft_internal.hpp"

#include <array>
#include <utility>

namespace symft {
using namespace detail;

namespace {

struct CircuitLoweringAccumulator {
    FrameFactoredState state;
    std::vector<SymbolicBool> records;
};

std::pair<int, int> reserve_measurement_record(std::vector<SymbolicBool>& records, SymbolicContext& context) {
    const int condition = context.fresh_condition();
    records.push_back(symbolic_bool(condition));
    return {static_cast<int>(records.size()), condition};
}

SymbolicBool measurement_sign_with_readout_error(SymbolicContext& context, bool inverted, double probability) {
    SymbolicBool sign(inverted);
    if (probability != 0.0) {
        sign = xor_bool(sign, context.fresh_bernoulli_bool(check_probability(probability)));
    }
    return sign;
}

PauliString pauli_on_axis(int n, char axis, int q) {
    if (axis == 'X') {
        return pauli_x(n, q);
    }
    if (axis == 'Y') {
        return pauli_y(n, q);
    }
    if (axis == 'Z') {
        return pauli_z(n, q);
    }
    fail("unsupported Pauli axis");
}

PauliString feedback_pauli(const FrameFactoredState& state, CircuitInstructionKind kind, int q) {
    if (kind == CircuitInstructionKind::FeedbackX) {
        return pauli_x(state.n, q);
    }
    if (kind == CircuitInstructionKind::FeedbackY) {
        return pauli_y(state.n, q);
    }
    if (kind == CircuitInstructionKind::FeedbackZ) {
        return pauli_z(state.n, q);
    }
    fail("unsupported feedback kind");
}

SymbolicBool record_symbol(const std::vector<SymbolicBool>& records, int record) {
    if (record <= 0 || record > static_cast<int>(records.size())) {
        fail("measurement record target out of range");
    }
    return records[static_cast<std::size_t>(record - 1)];
}

void require_even_targets(const CircuitInstruction& instruction, const char* message) {
    if ((instruction.qubits.size() & 1u) != 0u) {
        fail(message);
    }
}

void apply_depolarize1(FrameFactoredState& state, int q, double probability) {
    probability = check_probability(probability);
    const std::vector<std::vector<std::uint64_t>> assignments{
        packed_bits({false, false}),
        packed_bits({false, true}),
        packed_bits({true, false}),
        packed_bits({true, true}),
    };
    const auto bits = state.context->fresh_categorical_bools(
        2,
        assignments,
        {1.0 - probability, probability / 3.0, probability / 3.0, probability / 3.0});
    apply_pauli(state, pauli_x(state.n, q), bits[0]);
    apply_pauli(state, pauli_z(state.n, q), bits[1]);
}

void apply_depolarize2(FrameFactoredState& state, int q1, int q2, double probability) {
    probability = check_probability(probability);
    std::vector<std::vector<std::uint64_t>> assignments;
    assignments.push_back(packed_bits({false, false, false, false}));
    for (bool x1 : {false, true}) {
        for (bool z1 : {false, true}) {
            for (bool x2 : {false, true}) {
                for (bool z2 : {false, true}) {
                    if (x1 || z1 || x2 || z2) {
                        assignments.push_back(packed_bits({x1, z1, x2, z2}));
                    }
                }
            }
        }
    }
    std::vector<double> probabilities(16, probability / 15.0);
    probabilities[0] = 1.0 - probability;
    const auto bits = state.context->fresh_categorical_bools(4, assignments, probabilities);
    apply_pauli(state, pauli_x(state.n, q1), bits[0]);
    apply_pauli(state, pauli_z(state.n, q1), bits[1]);
    apply_pauli(state, pauli_x(state.n, q2), bits[2]);
    apply_pauli(state, pauli_z(state.n, q2), bits[3]);
}

void apply_record_feedback(
    FrameFactoredState& state,
    const std::vector<SymbolicBool>& records,
    const CircuitInstruction& instruction) {
    for (const auto& target : instruction.feedback_targets) {
        apply_pauli(
            state,
            feedback_pauli(state, instruction.kind, target.qubit),
            record_symbol(records, target.record));
    }
}

void apply_reset(FrameFactoredState& state, const PauliString& measurement_pauli, const PauliString& correction_pauli) {
    const int condition = state.context->fresh_condition();
    apply_pauli_measurement(state, measurement_pauli, SymbolicBool(false), std::nullopt, condition);
    apply_pauli(state, correction_pauli, symbolic_bool(condition));
}

void apply_measurement_reset(
    FrameFactoredState& state,
    std::vector<SymbolicBool>& records,
    const PauliString& measurement_pauli,
    const PauliString& correction_pauli,
    const CircuitMeasurementTarget& target,
    double probability) {
    const SymbolicBool sign = measurement_sign_with_readout_error(*state.context, target.inverted, probability);
    auto [record, record_condition] = reserve_measurement_record(records, *state.context);
    apply_pauli_measurement(state, measurement_pauli, sign, record, record_condition);
    apply_pauli(state, correction_pauli, xor_bool(symbolic_bool(record_condition), sign));
}

void apply_measurement(
    FrameFactoredState& state,
    std::vector<SymbolicBool>& records,
    char axis,
    const CircuitMeasurementTarget& target,
    double probability) {
    const SymbolicBool sign = measurement_sign_with_readout_error(*state.context, target.inverted, probability);
    auto [record, record_condition] = reserve_measurement_record(records, *state.context);
    apply_pauli_measurement(state, pauli_on_axis(state.n, axis, target.qubit), sign, record, record_condition);
}

void apply_instruction(CircuitLoweringAccumulator& acc, const CircuitInstruction& instruction) {
    FrameFactoredState& state = acc.state;
    switch (instruction.kind) {
    case CircuitInstructionKind::Tick:
        return;
    case CircuitInstructionKind::CX:
        require_even_targets(instruction, "CX requires paired targets");
        for (std::size_t i = 0; i < instruction.qubits.size(); i += 2) {
            left_CX(state, instruction.qubits[i], instruction.qubits[i + 1]);
        }
        return;
    case CircuitInstructionKind::CZ:
        require_even_targets(instruction, "CZ requires paired targets");
        for (std::size_t i = 0; i < instruction.qubits.size(); i += 2) {
            left_CZ(state, instruction.qubits[i], instruction.qubits[i + 1]);
        }
        return;
    case CircuitInstructionKind::CY:
        require_even_targets(instruction, "CY requires paired targets");
        for (std::size_t i = 0; i < instruction.qubits.size(); i += 2) {
            const int target = instruction.qubits[i + 1];
            left_SDG(state, target);
            left_CX(state, instruction.qubits[i], target);
            left_S(state, target);
        }
        return;
    case CircuitInstructionKind::SWAP:
        require_even_targets(instruction, "SWAP requires paired targets");
        for (std::size_t i = 0; i < instruction.qubits.size(); i += 2) {
            left_SWAP(state, instruction.qubits[i], instruction.qubits[i + 1]);
        }
        return;
    case CircuitInstructionKind::H:
        for (int q : instruction.qubits) {
            left_H(state, q);
        }
        return;
    case CircuitInstructionKind::S:
        for (int q : instruction.qubits) {
            left_S(state, q);
        }
        return;
    case CircuitInstructionKind::SDG:
        for (int q : instruction.qubits) {
            left_SDG(state, q);
        }
        return;
    case CircuitInstructionKind::X:
    case CircuitInstructionKind::Y:
    case CircuitInstructionKind::Z:
        for (int q : instruction.qubits) {
            if (instruction.kind == CircuitInstructionKind::X) {
                left_X(state, q);
            } else if (instruction.kind == CircuitInstructionKind::Z) {
                left_Z(state, q);
            } else {
                left_X(state, q);
                left_Z(state, q);
            }
        }
        return;
    case CircuitInstructionKind::T:
    case CircuitInstructionKind::TDag: {
        const double theta = instruction.kind == CircuitInstructionKind::T ? kPi / 8.0 : -kPi / 8.0;
        for (int q : instruction.qubits) {
            apply_pauli_rotation(state, pauli_z(state.n, q), theta);
        }
        return;
    }
    case CircuitInstructionKind::MZ:
    case CircuitInstructionKind::MX:
    case CircuitInstructionKind::MY: {
        const char axis = instruction.kind == CircuitInstructionKind::MX
                              ? 'X'
                              : (instruction.kind == CircuitInstructionKind::MY ? 'Y' : 'Z');
        for (const auto& target : instruction.measurement_targets) {
            apply_measurement(state, acc.records, axis, target, instruction.probability);
        }
        return;
    }
    case CircuitInstructionKind::MRZ:
    case CircuitInstructionKind::MRX:
    case CircuitInstructionKind::MRY:
        for (const auto& target : instruction.measurement_targets) {
            if (instruction.kind == CircuitInstructionKind::MRX) {
                apply_measurement_reset(
                    state,
                    acc.records,
                    pauli_x(state.n, target.qubit),
                    pauli_z(state.n, target.qubit),
                    target,
                    instruction.probability);
            } else if (instruction.kind == CircuitInstructionKind::MRY) {
                apply_measurement_reset(
                    state,
                    acc.records,
                    pauli_y(state.n, target.qubit),
                    pauli_x(state.n, target.qubit),
                    target,
                    instruction.probability);
            } else {
                apply_measurement_reset(
                    state,
                    acc.records,
                    pauli_z(state.n, target.qubit),
                    pauli_x(state.n, target.qubit),
                    target,
                    instruction.probability);
            }
        }
        return;
    case CircuitInstructionKind::RZ:
    case CircuitInstructionKind::RX:
    case CircuitInstructionKind::RY:
        for (int q : instruction.qubits) {
            if (instruction.kind == CircuitInstructionKind::RX) {
                apply_reset(state, pauli_x(state.n, q), pauli_z(state.n, q));
            } else if (instruction.kind == CircuitInstructionKind::RY) {
                apply_reset(state, pauli_y(state.n, q), pauli_x(state.n, q));
            } else {
                apply_reset(state, pauli_z(state.n, q), pauli_x(state.n, q));
            }
        }
        return;
    case CircuitInstructionKind::MPP:
        for (const auto& product : instruction.pauli_products) {
            const SymbolicBool sign =
                measurement_sign_with_readout_error(*state.context, product.inverted, instruction.probability);
            auto [record, record_condition] = reserve_measurement_record(acc.records, *state.context);
            apply_pauli_measurement(state, product.pauli, sign, record, record_condition);
        }
        return;
    case CircuitInstructionKind::XError:
    case CircuitInstructionKind::YError:
    case CircuitInstructionKind::ZError: {
        const double probability = check_probability(instruction.probability);
        const char axis = instruction.kind == CircuitInstructionKind::XError
                              ? 'X'
                              : (instruction.kind == CircuitInstructionKind::YError ? 'Y' : 'Z');
        for (int q : instruction.qubits) {
            apply_pauli(state, pauli_on_axis(state.n, axis, q), state.context->fresh_bernoulli_bool(probability));
        }
        return;
    }
    case CircuitInstructionKind::Depolarize1:
        for (int q : instruction.qubits) {
            apply_depolarize1(state, q, instruction.probability);
        }
        return;
    case CircuitInstructionKind::Depolarize2:
        require_even_targets(instruction, "DEPOLARIZE2 requires paired targets");
        for (std::size_t i = 0; i < instruction.qubits.size(); i += 2) {
            apply_depolarize2(state, instruction.qubits[i], instruction.qubits[i + 1], instruction.probability);
        }
        return;
    case CircuitInstructionKind::FeedbackX:
    case CircuitInstructionKind::FeedbackY:
    case CircuitInstructionKind::FeedbackZ:
        apply_record_feedback(state, acc.records, instruction);
        return;
    }
    fail("unsupported circuit operation");
}

} // namespace

CircuitLoweringResult lower_circuit_to_factored(const QuantumCircuit& circuit) {
    CircuitLoweringAccumulator acc{FrameFactoredState(circuit.nqubits, 0), {}};
    for (const auto& instruction : circuit.instructions) {
        apply_instruction(acc, instruction);
    }
    if (static_cast<int>(acc.records.size()) != circuit.nrecords) {
        fail("circuit measurement record count mismatch");
    }
    return CircuitLoweringResult{std::move(acc.state), std::move(acc.records)};
}

} // namespace symft
