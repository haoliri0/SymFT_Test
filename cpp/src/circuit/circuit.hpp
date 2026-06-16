#pragma once

#include "factored/factored.hpp"

#include <vector>

namespace symft {

enum class CircuitInstructionKind {
    Tick,
    H,
    S,
    SDG,
    X,
    Y,
    Z,
    CX,
    CY,
    CZ,
    SWAP,
    T,
    TDag,
    MZ,
    MX,
    MY,
    MRZ,
    MRX,
    MRY,
    RZ,
    RX,
    RY,
    MPP,
    XError,
    YError,
    ZError,
    Depolarize1,
    Depolarize2,
    FeedbackX,
    FeedbackY,
    FeedbackZ,
};

struct CircuitMeasurementTarget {
    int qubit = 0;
    bool inverted = false;
};

struct CircuitPauliProduct {
    PauliString pauli;
    bool inverted = false;
};

struct CircuitFeedbackTarget {
    int record = 0;
    int qubit = 0;
};

struct CircuitInstruction {
    CircuitInstructionKind kind = CircuitInstructionKind::Tick;
    double probability = 0.0;
    std::vector<int> qubits;
    std::vector<CircuitMeasurementTarget> measurement_targets;
    std::vector<CircuitPauliProduct> pauli_products;
    std::vector<CircuitFeedbackTarget> feedback_targets;
    int line = 0;
};

struct CircuitDetector {
    std::vector<int> records;
    std::vector<double> coords;
    int line = 0;
};

struct CircuitObservableInclude {
    int index = 0;
    std::vector<int> records;
    int line = 0;
};

struct QuantumCircuit {
    int nqubits = 0;
    int nrecords = 0;
    std::vector<CircuitInstruction> instructions;
    std::vector<CircuitDetector> detectors;
    std::vector<CircuitObservableInclude> observables;
};

struct CircuitLoweringResult {
    FrameFactoredState state;
    std::vector<SymbolicBool> measurement_records;
};

CircuitLoweringResult lower_circuit_to_factored(const QuantumCircuit& circuit);

} // namespace symft
