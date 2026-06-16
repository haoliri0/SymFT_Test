#pragma once

#include "symft/core/pauli.hpp"
#include "symft/core/symbolic.hpp"

#include <memory>

namespace symft {

struct ConditionalPauliString {
    PauliString pauli;
    int condition = 0;

    ConditionalPauliString() = default;
    ConditionalPauliString(PauliString pauli, int condition);
};

bool operator==(const ConditionalPauliString& lhs, const ConditionalPauliString& rhs);

struct SymbolicPauliString {
    PauliString pauli;
    SymbolicBool sign;

    SymbolicPauliString() = default;
    explicit SymbolicPauliString(PauliString pauli);
    SymbolicPauliString(PauliString pauli, SymbolicBool sign);
};

bool operator==(const SymbolicPauliString& lhs, const SymbolicPauliString& rhs);

struct ActivePauliFrame {
    int k = 0;
    std::vector<ConditionalPauliString> terms;
    std::shared_ptr<SymbolicContext> context;

    ActivePauliFrame() = default;
    explicit ActivePauliFrame(int k);
    ActivePauliFrame(int k, std::shared_ptr<SymbolicContext> context);

    ConditionalPauliString add_pauli(const PauliString& pauli);
    ConditionalPauliString add_pauli(const PauliString& pauli, int condition);
};

SymbolicPauliString conjugate_by(const ConditionalPauliString& cp, const PauliString& pauli);
SymbolicPauliString conjugate_by(const ConditionalPauliString& cp, const SymbolicPauliString& pauli);
SymbolicPauliString conjugate_by(const ActivePauliFrame& frame, const PauliString& pauli);
SymbolicPauliString conjugate_by(const ActivePauliFrame& frame, const SymbolicPauliString& pauli);

struct DormantState {
    int d = 0;
    std::vector<SymbolicBool> bits;
    std::shared_ptr<SymbolicContext> context;

    DormantState() = default;
    explicit DormantState(int d);
    DormantState(int d, std::shared_ptr<SymbolicContext> context);
    DormantState(std::vector<SymbolicBool> bits, std::shared_ptr<SymbolicContext> context);

    SymbolicBool dormant_bit(int q) const;
    void set_dormant_bit(int q, const SymbolicBool& value);
    SymbolicBool assign_dormant_symbol(int q);
};

struct CliffordFrame {
    int nqubits = 0;
    int nwords = 0;
    std::vector<PauliString> rows;

    CliffordFrame() = default;
    explicit CliffordFrame(int nqubits);

    int xrow(int q) const;
    int zrow(int q) const;
    void copy_pauli_to_row(int row, const PauliString& pauli);
};

bool operator==(const CliffordFrame& lhs, const CliffordFrame& rhs);
PauliString preimage(const CliffordFrame& frame, const PauliString& pauli);
PauliString coordinates_in_frame(const CliffordFrame& frame, const PauliString& pauli);

void left_H(CliffordFrame& frame, int q);
void left_S(CliffordFrame& frame, int q);
void left_SDG(CliffordFrame& frame, int q);
void left_X(CliffordFrame& frame, int q);
void left_Z(CliffordFrame& frame, int q);
void left_CX(CliffordFrame& frame, int control, int target);
void left_CZ(CliffordFrame& frame, int a, int b);
void left_SWAP(CliffordFrame& frame, int a, int b);
void right_H(CliffordFrame& frame, int q);
void right_S(CliffordFrame& frame, int q);
void right_SDG(CliffordFrame& frame, int q);
void right_X(CliffordFrame& frame, int q);
void right_Z(CliffordFrame& frame, int q);
void right_CX(CliffordFrame& frame, int control, int target);
void right_CZ(CliffordFrame& frame, int a, int b);
void right_SWAP(CliffordFrame& frame, int a, int b);

} // namespace symft
