#include "core/frames.hpp"

#include "core/internal.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace symft {
using namespace detail;

ConditionalPauliString::ConditionalPauliString(PauliString pauli_, int condition_)
    : pauli(std::move(pauli_)), condition(condition_) {
    if (condition <= 0) {
        fail("condition id must be positive");
    }
}

bool operator==(const ConditionalPauliString& lhs, const ConditionalPauliString& rhs) {
    return lhs.condition == rhs.condition && lhs.pauli == rhs.pauli;
}

SymbolicPauliString::SymbolicPauliString(PauliString pauli_) : pauli(std::move(pauli_)), sign(false) {}

SymbolicPauliString::SymbolicPauliString(PauliString pauli_, SymbolicBool sign_)
    : pauli(std::move(pauli_)), sign(std::move(sign_)) {}

bool operator==(const SymbolicPauliString& lhs, const SymbolicPauliString& rhs) {
    return lhs.pauli == rhs.pauli && lhs.sign == rhs.sign;
}

ActivePauliFrame::ActivePauliFrame(int k_) : ActivePauliFrame(k_, std::make_shared<SymbolicContext>()) {}

ActivePauliFrame::ActivePauliFrame(int k_, std::shared_ptr<SymbolicContext> context_)
    : k(checked_nqubits(k_)), context(std::move(context_)) {
    if (!context) {
        context = std::make_shared<SymbolicContext>();
    }
}

ConditionalPauliString ActivePauliFrame::add_pauli(const PauliString& pauli) {
    return add_pauli(pauli, context->fresh_condition());
}

ConditionalPauliString ActivePauliFrame::add_pauli(const PauliString& pauli, int condition) {
    if (pauli.nqubits != k) {
        fail("Pauli string dimension does not match active Pauli frame");
    }
    ConditionalPauliString cp(pauli, condition);
    terms.push_back(cp);
    context->bump_next_condition(condition);
    return cp;
}

SymbolicPauliString conjugate_by(const ConditionalPauliString& cp, const PauliString& pauli) {
    SymbolicPauliString out(pauli);
    if (pauli_anticommutes(cp.pauli, pauli)) {
        out.sign = xor_bool(out.sign, symbolic_bool(cp.condition));
    }
    return out;
}

SymbolicPauliString conjugate_by(const ConditionalPauliString& cp, const SymbolicPauliString& pauli) {
    SymbolicPauliString out = pauli;
    if (pauli_anticommutes(cp.pauli, pauli.pauli)) {
        out.sign = xor_bool(out.sign, symbolic_bool(cp.condition));
    }
    return out;
}

SymbolicPauliString conjugate_by(const ActivePauliFrame& frame, const PauliString& pauli) {
    if (pauli.nqubits != frame.k) {
        fail("Pauli string dimension does not match active Pauli frame");
    }
    SymbolicPauliString out(pauli);
    for (const auto& cp : frame.terms) {
        out = conjugate_by(cp, out);
    }
    return out;
}

SymbolicPauliString conjugate_by(const ActivePauliFrame& frame, const SymbolicPauliString& pauli) {
    if (pauli.pauli.nqubits != frame.k) {
        fail("Pauli string dimension does not match active Pauli frame");
    }
    SymbolicPauliString out = pauli;
    for (const auto& cp : frame.terms) {
        out = conjugate_by(cp, out);
    }
    return out;
}

DormantState::DormantState(int d_) : DormantState(d_, std::make_shared<SymbolicContext>()) {}

DormantState::DormantState(int d_, std::shared_ptr<SymbolicContext> context_)
    : d(checked_nqubits(d_)), bits(static_cast<std::size_t>(d), SymbolicBool(false)), context(std::move(context_)) {
    if (!context) {
        context = std::make_shared<SymbolicContext>();
    }
}

DormantState::DormantState(std::vector<SymbolicBool> bits_, std::shared_ptr<SymbolicContext> context_)
    : d(static_cast<int>(bits_.size())), bits(std::move(bits_)), context(std::move(context_)) {
    if (!context) {
        context = std::make_shared<SymbolicContext>();
    }
    for (const auto& bit : bits) {
        context->bump_next_condition(bit);
    }
}

SymbolicBool DormantState::dormant_bit(int q) const {
    check_qubit(d, q);
    return bits[static_cast<std::size_t>(q)];
}

void DormantState::set_dormant_bit(int q, const SymbolicBool& value) {
    check_qubit(d, q);
    bits[static_cast<std::size_t>(q)] = value;
    context->bump_next_condition(value);
}

SymbolicBool DormantState::assign_dormant_symbol(int q) {
    check_qubit(d, q);
    const SymbolicBool expr = symbolic_bool(context->fresh_condition());
    bits[static_cast<std::size_t>(q)] = expr;
    return expr;
}

CliffordFrame::CliffordFrame(int nqubits_) : nqubits(checked_nqubits(nqubits_)), nwords(nwords_for(nqubits)) {
    rows.assign(static_cast<std::size_t>(2 * nqubits), PauliString(nqubits));
    for (int q = 0; q < nqubits; ++q) {
        rows[static_cast<std::size_t>(xrow(q))].set_xbit(q);
        rows[static_cast<std::size_t>(zrow(q))].set_zbit(q);
    }
}

int CliffordFrame::xrow(int q) const {
    return check_qubit(nqubits, q);
}

int CliffordFrame::zrow(int q) const {
    return nqubits + check_qubit(nqubits, q);
}

void CliffordFrame::copy_pauli_to_row(int row, const PauliString& pauli) {
    if (pauli.nqubits != nqubits || row < 0 || row >= static_cast<int>(rows.size())) {
        fail("invalid Clifford frame row assignment");
    }
    rows[static_cast<std::size_t>(row)] = pauli;
}

bool operator==(const CliffordFrame& lhs, const CliffordFrame& rhs) {
    return lhs.nqubits == rhs.nqubits && lhs.rows == rhs.rows;
}

namespace {

void swap_rows(CliffordFrame& frame, int a, int b) {
    if (a != b) {
        std::swap(frame.rows[static_cast<std::size_t>(a)], frame.rows[static_cast<std::size_t>(b)]);
    }
}

void add_row_phase(CliffordFrame& frame, int row, int delta) {
    frame.rows[static_cast<std::size_t>(row)].phase_shift(delta);
}

void mul_rows(CliffordFrame& frame, int dst, int lhs, int rhs, int extra_phase = 0) {
    PauliString out = frame.rows[static_cast<std::size_t>(lhs)] * frame.rows[static_cast<std::size_t>(rhs)];
    out.phase_shift(extra_phase);
    frame.rows[static_cast<std::size_t>(dst)] = out;
}

PauliString axis_image_from_rows(
    const CliffordFrame& frame,
    const std::vector<PauliString>& rows,
    int q,
    char axis) {
    if (axis == '_') {
        return pauli_identity(frame.nqubits);
    }
    if (axis == 'X') {
        return rows[static_cast<std::size_t>(frame.xrow(q))];
    }
    if (axis == 'Z') {
        return rows[static_cast<std::size_t>(frame.zrow(q))];
    }
    if (axis == 'Y') {
        PauliString out =
            rows[static_cast<std::size_t>(frame.xrow(q))] * rows[static_cast<std::size_t>(frame.zrow(q))];
        out.phase_shift(1);
        return out;
    }
    fail("invalid Clifford image axis");
}

PauliString image_from_body(
    const CliffordFrame& frame,
    const std::vector<PauliString>& rows,
    const std::vector<int>& qubits,
    std::string body) {
    bool negative = false;
    if (!body.empty() && body[0] == '-') {
        negative = true;
        body = body.substr(1);
    }
    if (body.size() != qubits.size()) {
        fail("invalid Clifford image body");
    }
    PauliString out = pauli_identity(frame.nqubits);
    for (std::size_t i = 0; i < qubits.size(); ++i) {
        out = out * axis_image_from_rows(frame, rows, qubits[i], body[i]);
    }
    if (negative) {
        out.phase_shift(2);
    }
    return out;
}

void left_apply_single_qubit_images(CliffordFrame& frame, int q, const std::string& x_image, const std::string& z_image) {
    const int qi = check_qubit(frame.nqubits, q);
    const auto rows = frame.rows;
    const std::vector<int> qubits{qi};
    frame.rows[static_cast<std::size_t>(frame.xrow(qi))] = image_from_body(frame, rows, qubits, x_image);
    frame.rows[static_cast<std::size_t>(frame.zrow(qi))] = image_from_body(frame, rows, qubits, z_image);
}

void left_apply_two_qubit_images(
    CliffordFrame& frame,
    int a,
    int b,
    const std::string& xa_image,
    const std::string& za_image,
    const std::string& xb_image,
    const std::string& zb_image) {
    const int ai = check_qubit(frame.nqubits, a);
    const int bi = check_qubit(frame.nqubits, b);
    if (ai == bi) {
        fail("two-qubit Clifford gate requires distinct qubits");
    }
    const auto rows = frame.rows;
    const std::vector<int> qubits{ai, bi};
    frame.rows[static_cast<std::size_t>(frame.xrow(ai))] = image_from_body(frame, rows, qubits, xa_image);
    frame.rows[static_cast<std::size_t>(frame.zrow(ai))] = image_from_body(frame, rows, qubits, za_image);
    frame.rows[static_cast<std::size_t>(frame.xrow(bi))] = image_from_body(frame, rows, qubits, xb_image);
    frame.rows[static_cast<std::size_t>(frame.zrow(bi))] = image_from_body(frame, rows, qubits, zb_image);
}

void right_apply_clifford(CliffordFrame& frame, const CliffordFrame& gate) {
    if (frame.nqubits != gate.nqubits) {
        fail("Clifford frames act on different numbers of qubits");
    }
    for (auto& row : frame.rows) {
        row = preimage(gate, row);
    }
}

template <class Fn>
void right_apply_left_gate(CliffordFrame& frame, Fn&& fn) {
    CliffordFrame gate(frame.nqubits);
    fn(gate);
    right_apply_clifford(frame, gate);
}

} // namespace

PauliString preimage(const CliffordFrame& frame, const PauliString& pauli) {
    if (pauli.nqubits != frame.nqubits) {
        fail("Pauli string and Clifford frame have different numbers of qubits");
    }
    PauliString out(frame.nqubits);
    out.set_phase(pauli.phase_exponent());
    for (int q = 0; q < frame.nqubits; ++q) {
        if (pauli.xbit(q)) {
            out = out * frame.rows[static_cast<std::size_t>(frame.xrow(q))];
        }
        if (pauli.zbit(q)) {
            out = out * frame.rows[static_cast<std::size_t>(frame.zrow(q))];
        }
    }
    return out;
}

PauliString coordinates_in_frame(const CliffordFrame& frame, const PauliString& pauli) {
    if (pauli.nqubits != frame.nqubits) {
        fail("Pauli string and Clifford frame have different numbers of qubits");
    }
    PauliString out(frame.nqubits);
    for (int q = 0; q < frame.nqubits; ++q) {
        if (pauli_anticommutes(pauli, frame.rows[static_cast<std::size_t>(frame.zrow(q))])) {
            out.set_xbit(q);
        }
        if (pauli_anticommutes(pauli, frame.rows[static_cast<std::size_t>(frame.xrow(q))])) {
            out.set_zbit(q);
        }
    }
    const PauliString reconstructed = preimage(frame, out);
    if (!reconstructed.same_body(pauli)) {
        fail("frame rows do not span the Pauli body");
    }
    out.set_phase(pauli.phase_exponent() - reconstructed.phase_exponent());
    return out;
}

void left_H(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    swap_rows(frame, frame.xrow(qi), frame.zrow(qi));
}

void left_H_NXY(CliffordFrame& frame, int q) {
    left_apply_single_qubit_images(frame, q, "-Y", "-Z");
}

void left_H_NXZ(CliffordFrame& frame, int q) {
    left_apply_single_qubit_images(frame, q, "-Z", "-X");
}

void left_H_NYZ(CliffordFrame& frame, int q) {
    left_apply_single_qubit_images(frame, q, "-X", "-Y");
}

void left_H_XY(CliffordFrame& frame, int q) {
    left_apply_single_qubit_images(frame, q, "Y", "-Z");
}

void left_H_YZ(CliffordFrame& frame, int q) {
    left_apply_single_qubit_images(frame, q, "-X", "Y");
}

void left_C_NXYZ(CliffordFrame& frame, int q) {
    left_apply_single_qubit_images(frame, q, "-Y", "-X");
}

void left_C_NZYX(CliffordFrame& frame, int q) {
    left_apply_single_qubit_images(frame, q, "-Z", "-Y");
}

void left_C_XNYZ(CliffordFrame& frame, int q) {
    left_apply_single_qubit_images(frame, q, "-Y", "X");
}

void left_C_XYNZ(CliffordFrame& frame, int q) {
    left_apply_single_qubit_images(frame, q, "Y", "-X");
}

void left_C_XYZ(CliffordFrame& frame, int q) {
    left_apply_single_qubit_images(frame, q, "Y", "X");
}

void left_C_ZNYX(CliffordFrame& frame, int q) {
    left_apply_single_qubit_images(frame, q, "Z", "-Y");
}

void left_C_ZYNX(CliffordFrame& frame, int q) {
    left_apply_single_qubit_images(frame, q, "-Z", "Y");
}

void left_C_ZYX(CliffordFrame& frame, int q) {
    left_apply_single_qubit_images(frame, q, "Z", "Y");
}

void left_S(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    mul_rows(frame, frame.xrow(qi), frame.xrow(qi), frame.zrow(qi), 3);
}

void left_SDG(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    mul_rows(frame, frame.xrow(qi), frame.xrow(qi), frame.zrow(qi), 1);
}

void left_SQRT_X(CliffordFrame& frame, int q) {
    left_apply_single_qubit_images(frame, q, "X", "-Y");
}

void left_SQRT_X_DAG(CliffordFrame& frame, int q) {
    left_apply_single_qubit_images(frame, q, "X", "Y");
}

void left_SQRT_Y(CliffordFrame& frame, int q) {
    left_apply_single_qubit_images(frame, q, "-Z", "X");
}

void left_SQRT_Y_DAG(CliffordFrame& frame, int q) {
    left_apply_single_qubit_images(frame, q, "Z", "-X");
}

void left_X(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    add_row_phase(frame, frame.zrow(qi), 2);
}

void left_Y(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    add_row_phase(frame, frame.xrow(qi), 2);
    add_row_phase(frame, frame.zrow(qi), 2);
}

void left_Z(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    add_row_phase(frame, frame.xrow(qi), 2);
}

void left_CX(CliffordFrame& frame, int control, int target) {
    const int c = check_qubit(frame.nqubits, control);
    const int t = check_qubit(frame.nqubits, target);
    if (c == t) {
        fail("two-qubit Clifford gate requires distinct qubits");
    }
    mul_rows(frame, frame.xrow(c), frame.xrow(c), frame.xrow(t));
    mul_rows(frame, frame.zrow(t), frame.zrow(c), frame.zrow(t));
}

void left_CY(CliffordFrame& frame, int control, int target) {
    left_apply_two_qubit_images(frame, control, target, "XY", "Z_", "ZX", "ZZ");
}

void left_CZ(CliffordFrame& frame, int a, int b) {
    const int ai = check_qubit(frame.nqubits, a);
    const int bi = check_qubit(frame.nqubits, b);
    if (ai == bi) {
        fail("two-qubit Clifford gate requires distinct qubits");
    }
    mul_rows(frame, frame.xrow(ai), frame.xrow(ai), frame.zrow(bi));
    mul_rows(frame, frame.xrow(bi), frame.zrow(ai), frame.xrow(bi));
}

void left_SWAP(CliffordFrame& frame, int a, int b) {
    const int ai = check_qubit(frame.nqubits, a);
    const int bi = check_qubit(frame.nqubits, b);
    if (ai == bi) {
        fail("two-qubit Clifford gate requires distinct qubits");
    }
    swap_rows(frame, frame.xrow(ai), frame.xrow(bi));
    swap_rows(frame, frame.zrow(ai), frame.zrow(bi));
}

void left_CXSWAP(CliffordFrame& frame, int a, int b) {
    left_apply_two_qubit_images(frame, a, b, "XX", "_Z", "X_", "ZZ");
}

void left_CZSWAP(CliffordFrame& frame, int a, int b) {
    left_apply_two_qubit_images(frame, a, b, "ZX", "_Z", "XZ", "Z_");
}

void left_ISWAP(CliffordFrame& frame, int a, int b) {
    left_apply_two_qubit_images(frame, a, b, "ZY", "_Z", "YZ", "Z_");
}

void left_ISWAP_DAG(CliffordFrame& frame, int a, int b) {
    left_apply_two_qubit_images(frame, a, b, "-ZY", "_Z", "-YZ", "Z_");
}

void left_SQRT_XX(CliffordFrame& frame, int a, int b) {
    left_apply_two_qubit_images(frame, a, b, "X_", "-YX", "_X", "-XY");
}

void left_SQRT_XX_DAG(CliffordFrame& frame, int a, int b) {
    left_apply_two_qubit_images(frame, a, b, "X_", "YX", "_X", "XY");
}

void left_SQRT_YY(CliffordFrame& frame, int a, int b) {
    left_apply_two_qubit_images(frame, a, b, "-ZY", "XY", "-YZ", "YX");
}

void left_SQRT_YY_DAG(CliffordFrame& frame, int a, int b) {
    left_apply_two_qubit_images(frame, a, b, "ZY", "-XY", "YZ", "-YX");
}

void left_SQRT_ZZ(CliffordFrame& frame, int a, int b) {
    left_apply_two_qubit_images(frame, a, b, "YZ", "Z_", "ZY", "_Z");
}

void left_SQRT_ZZ_DAG(CliffordFrame& frame, int a, int b) {
    left_apply_two_qubit_images(frame, a, b, "-YZ", "Z_", "-ZY", "_Z");
}

void left_SWAPCX(CliffordFrame& frame, int a, int b) {
    left_apply_two_qubit_images(frame, a, b, "_X", "ZZ", "XX", "Z_");
}

void left_XCX(CliffordFrame& frame, int control, int target) {
    left_apply_two_qubit_images(frame, control, target, "X_", "ZX", "_X", "XZ");
}

void left_XCY(CliffordFrame& frame, int control, int target) {
    left_apply_two_qubit_images(frame, control, target, "X_", "ZY", "XX", "XZ");
}

void left_XCZ(CliffordFrame& frame, int control, int target) {
    left_apply_two_qubit_images(frame, control, target, "X_", "ZZ", "XX", "_Z");
}

void left_YCX(CliffordFrame& frame, int control, int target) {
    left_apply_two_qubit_images(frame, control, target, "XX", "ZX", "_X", "YZ");
}

void left_YCY(CliffordFrame& frame, int control, int target) {
    left_apply_two_qubit_images(frame, control, target, "XY", "ZY", "YX", "YZ");
}

void left_YCZ(CliffordFrame& frame, int control, int target) {
    left_apply_two_qubit_images(frame, control, target, "XZ", "ZZ", "YX", "_Z");
}

void right_H(CliffordFrame& frame, int q) {
    right_apply_left_gate(frame, [q](CliffordFrame& gate) { left_H(gate, q); });
}

void right_S(CliffordFrame& frame, int q) {
    right_apply_left_gate(frame, [q](CliffordFrame& gate) { left_S(gate, q); });
}

void right_SDG(CliffordFrame& frame, int q) {
    right_apply_left_gate(frame, [q](CliffordFrame& gate) { left_SDG(gate, q); });
}

void right_X(CliffordFrame& frame, int q) {
    right_apply_left_gate(frame, [q](CliffordFrame& gate) { left_X(gate, q); });
}

void right_Z(CliffordFrame& frame, int q) {
    right_apply_left_gate(frame, [q](CliffordFrame& gate) { left_Z(gate, q); });
}

void right_CX(CliffordFrame& frame, int control, int target) {
    right_apply_left_gate(frame, [control, target](CliffordFrame& gate) { left_CX(gate, control, target); });
}

void right_CZ(CliffordFrame& frame, int a, int b) {
    right_apply_left_gate(frame, [a, b](CliffordFrame& gate) { left_CZ(gate, a, b); });
}

void right_SWAP(CliffordFrame& frame, int a, int b) {
    right_apply_left_gate(frame, [a, b](CliffordFrame& gate) { left_SWAP(gate, a, b); });
}

} // namespace symft
