#include "symft_internal.hpp"

#include <algorithm>
#include <cctype>
#include <memory>
#include <ostream>
#include <sstream>
#include <utility>

namespace symft {
using namespace detail;

Error::Error(const std::string& message) : std::runtime_error(message) {}

int checked_nqubits(int nqubits) {
    if (nqubits < 0) {
        fail("number of qubits must be nonnegative");
    }
    return nqubits;
}

int nwords_for(int nqubits) {
    return (checked_nqubits(nqubits) + kWordBits - 1) / kWordBits;
}

std::size_t bit_word_count(int nbits) {
    if (nbits < 0) {
        fail("number of bits must be nonnegative");
    }
    return static_cast<std::size_t>((nbits + kWordBits - 1) / kWordBits);
}

bool packed_bit(const std::vector<std::uint64_t>& words, int bit_index) {
    if (bit_index < 0) {
        fail("bit index must be nonnegative");
    }
    const std::size_t word = static_cast<std::size_t>(bit_index / kWordBits);
    return word < words.size() && (words[word] & bit_mask(bit_index)) != 0;
}

void set_packed_bit(std::vector<std::uint64_t>& words, int bit_index, bool value) {
    if (bit_index < 0) {
        fail("bit index must be nonnegative");
    }
    const std::size_t word = static_cast<std::size_t>(bit_index / kWordBits);
    if (word >= words.size()) {
        words.resize(word + 1, 0);
    }
    if (value) {
        words[word] |= bit_mask(bit_index);
    } else {
        words[word] &= ~bit_mask(bit_index);
    }
}

std::vector<std::uint64_t> packed_bits(std::initializer_list<bool> bits) {
    std::vector<std::uint64_t> out(bit_word_count(static_cast<int>(bits.size())), 0);
    int bit_index = 0;
    for (bool bit : bits) {
        if (bit) {
            set_packed_bit(out, bit_index);
        }
        ++bit_index;
    }
    return out;
}

PauliString::PauliString(int nqubits_) : nqubits(checked_nqubits(nqubits_)) {
    const int nw = nwords_for(nqubits);
    x.assign(nw, 0);
    z.assign(nw, 0);
}

bool PauliString::xbit(int q) const {
    check_qubit(nqubits, q);
    return (x[word_index(q)] & bit_mask(q)) != 0;
}

bool PauliString::zbit(int q) const {
    check_qubit(nqubits, q);
    return (z[word_index(q)] & bit_mask(q)) != 0;
}

void PauliString::set_xbit(int q, bool value) {
    check_qubit(nqubits, q);
    auto& word = x[word_index(q)];
    if (value) {
        word |= bit_mask(q);
    } else {
        word &= ~bit_mask(q);
    }
}

void PauliString::set_zbit(int q, bool value) {
    check_qubit(nqubits, q);
    auto& word = z[word_index(q)];
    if (value) {
        word |= bit_mask(q);
    } else {
        word &= ~bit_mask(q);
    }
}

int PauliString::phase_exponent() const {
    return int(phase & 3u);
}

void PauliString::set_phase(int phase_exponent_) {
    phase = static_cast<std::uint8_t>(phase_exponent_ & 3);
}

void PauliString::phase_shift(int delta) {
    set_phase(phase_exponent() + delta);
}

bool PauliString::has_nonidentity_body() const {
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x[i] != 0 || z[i] != 0) {
            return true;
        }
    }
    return false;
}

bool PauliString::same_body(const PauliString& other) const {
    return nqubits == other.nqubits && x == other.x && z == other.z;
}

std::string PauliString::str() const {
    int ny = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        ny += popcount64(x[i] & z[i]);
    }
    const int coeff_phase = (phase_exponent() - ny) & 3;
    std::string out;
    if (coeff_phase == 1) {
        out = "i*";
    } else if (coeff_phase == 2) {
        out = "-";
    } else if (coeff_phase == 3) {
        out = "-i*";
    }
    if (nqubits == 0) {
        out += "I";
        return out;
    }
    for (int q = 0; q < nqubits; ++q) {
        const bool xb = xbit(q);
        const bool zb = zbit(q);
        out.push_back(xb ? (zb ? 'Y' : 'X') : (zb ? 'Z' : 'I'));
    }
    return out;
}

bool operator==(const PauliString& lhs, const PauliString& rhs) {
    return lhs.nqubits == rhs.nqubits && lhs.phase == rhs.phase && lhs.x == rhs.x && lhs.z == rhs.z;
}

bool operator!=(const PauliString& lhs, const PauliString& rhs) {
    return !(lhs == rhs);
}

PauliString operator*(const PauliString& lhs, const PauliString& rhs) {
    check_same_nqubits(lhs, rhs);
    PauliString out(lhs.nqubits);
    int carry = 0;
    for (std::size_t i = 0; i < lhs.x.size(); ++i) {
        carry += popcount64(lhs.z[i] & rhs.x[i]);
    }
    out.set_phase(lhs.phase_exponent() + rhs.phase_exponent() + 2 * (carry & 1));
    for (std::size_t i = 0; i < lhs.x.size(); ++i) {
        out.x[i] = lhs.x[i] ^ rhs.x[i];
        out.z[i] = lhs.z[i] ^ rhs.z[i];
    }
    return out;
}

std::ostream& operator<<(std::ostream& out, const PauliString& pauli) {
    out << pauli.str();
    return out;
}

PauliString pauli_identity(int nqubits) {
    return PauliString(nqubits);
}

PauliString pauli_x(int nqubits, int q) {
    PauliString out(nqubits);
    out.set_xbit(q);
    return out;
}

PauliString pauli_y(int nqubits, int q) {
    PauliString out(nqubits);
    out.set_xbit(q);
    out.set_zbit(q);
    out.set_phase(1);
    return out;
}

PauliString pauli_z(int nqubits, int q) {
    PauliString out(nqubits);
    out.set_zbit(q);
    return out;
}

PauliString pauli_string(const std::string& ops) {
    PauliString out(static_cast<int>(ops.size()));
    for (std::size_t i = 0; i < ops.size(); ++i) {
        const char ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ops[i])));
        const int q = static_cast<int>(i);
        if (ch == 'I' || ch == '_') {
            continue;
        }
        if (ch == 'X') {
            out.set_xbit(q);
        } else if (ch == 'Z') {
            out.set_zbit(q);
        } else if (ch == 'Y') {
            out.set_xbit(q);
            out.set_zbit(q);
            out.phase_shift(1);
        } else {
            fail("unsupported Pauli character");
        }
    }
    return out;
}

PauliString neg(PauliString pauli) {
    pauli.phase_shift(2);
    return pauli;
}

bool pauli_anticommutes(const PauliString& a, const PauliString& b) {
    check_same_nqubits(a, b);
    bool parity = false;
    for (std::size_t i = 0; i < a.x.size(); ++i) {
        parity ^= is_odd_popcount(a.x[i] & b.z[i]);
        parity ^= is_odd_popcount(a.z[i] & b.x[i]);
    }
    return parity;
}

int pauli_body_y_count(const PauliString& pauli) {
    int count = 0;
    for (std::size_t i = 0; i < pauli.x.size(); ++i) {
        count += popcount64(pauli.x[i] & pauli.z[i]);
    }
    return count;
}

bool pauli_squares_to_identity(const PauliString& pauli) {
    return ((pauli.phase_exponent() - pauli_body_y_count(pauli)) & 1) == 0;
}

bool measurement_phase_sign(const PauliString& pauli) {
    const int coeff_phase = (pauli.phase_exponent() - pauli_body_y_count(pauli)) & 3;
    if (coeff_phase == 0) {
        return false;
    }
    if (coeff_phase == 2) {
        return true;
    }
    fail("Pauli measurement requires a Hermitian Pauli with real coefficient");
}

SymbolicBool::SymbolicBool(bool constant_) : constant(constant_) {}

SymbolicBool::SymbolicBool(bool constant_, std::vector<int> conditions_)
    : constant(constant_), conditions(normalize_conditions(conditions_)) {}

int SymbolicBool::max_condition() const {
    return conditions.empty() ? 0 : conditions.back();
}

std::string SymbolicBool::str() const {
    std::ostringstream out;
    bool first = true;
    if (constant) {
        out << "1";
        first = false;
    }
    for (int condition : conditions) {
        if (!first) {
            out << " xor ";
        }
        out << "s" << condition;
        first = false;
    }
    if (first) {
        out << "0";
    }
    return out.str();
}

bool operator==(const SymbolicBool& lhs, const SymbolicBool& rhs) {
    return lhs.constant == rhs.constant && lhs.conditions == rhs.conditions;
}

bool operator!=(const SymbolicBool& lhs, const SymbolicBool& rhs) {
    return !(lhs == rhs);
}

SymbolicBool symbolic_bool(int condition) {
    return SymbolicBool(false, {condition});
}

SymbolicBool operator!(const SymbolicBool& expr) {
    return SymbolicBool(!expr.constant, expr.conditions);
}

SymbolicBool xor_bool(const SymbolicBool& lhs, const SymbolicBool& rhs) {
    std::vector<int> conditions = lhs.conditions;
    for (int condition : rhs.conditions) {
        toggle_condition(conditions, condition);
    }
    return SymbolicBool(lhs.constant != rhs.constant, conditions);
}

SymbolicBool xor_bool(const SymbolicBool& lhs, bool rhs) {
    return SymbolicBool(lhs.constant != rhs, lhs.conditions);
}

SymbolicBool xor_bool(bool lhs, const SymbolicBool& rhs) {
    return xor_bool(rhs, lhs);
}

std::ostream& operator<<(std::ostream& out, const SymbolicBool& expr) {
    out << expr.str();
    return out;
}

SymbolicBoolEvaluationPlan::SymbolicBoolEvaluationPlan(const SymbolicBool& expr)
    : constant(expr.constant), conditions(expr.conditions) {
    for (int condition : conditions) {
        const std::size_t word = symbol_word_index(condition);
        const std::uint64_t mask = symbol_bit_mask(condition);
        if (word_indices.empty() || word_indices.back() != static_cast<int>(word)) {
            word_indices.push_back(static_cast<int>(word));
            word_masks.push_back(mask);
        } else {
            word_masks.back() |= mask;
        }
    }
}

SymbolicContext::SymbolicContext(int next_condition_) : next_condition(next_condition_) {
    if (next_condition <= 0) {
        fail("next condition id must be positive");
    }
}

void SymbolicContext::bump_next_condition(int condition) {
    if (condition < 0) {
        fail("condition id must be nonnegative");
    }
    next_condition = std::max(next_condition, condition + 1);
}

void SymbolicContext::bump_next_condition(const SymbolicBool& expr) {
    bump_next_condition(expr.max_condition());
}

int SymbolicContext::fresh_condition() {
    return next_condition++;
}

int SymbolicContext::fresh_bernoulli_condition(double probability) {
    const double p = check_probability(probability);
    const int condition = fresh_condition();
    if (condition_to_categorical.find(condition) != condition_to_categorical.end()) {
        fail("condition already belongs to a categorical distribution");
    }
    bernoulli_probabilities[condition] = p;
    return condition;
}

SymbolicBool SymbolicContext::fresh_bernoulli_bool(double probability) {
    return symbolic_bool(fresh_bernoulli_condition(probability));
}

std::vector<int> SymbolicContext::fresh_categorical_conditions(
    int nbits,
    const std::vector<std::vector<std::uint64_t>>& assignments,
    const std::vector<double>& probabilities) {
    if (assignments.empty()) {
        fail("categorical symbolic distribution needs at least one assignment");
    }
    if (nbits <= 0 || assignments.size() != probabilities.size()) {
        fail("invalid categorical symbolic distribution");
    }
    const std::size_t nwords = bit_word_count(nbits);
    double total = 0.0;
    for (const auto& assignment : assignments) {
        if (assignment.size() != nwords) {
            fail("categorical assignment length mismatch");
        }
    }
    for (double probability : probabilities) {
        total += check_probability(probability);
    }
    if (std::abs(total - 1.0) > 1e-12) {
        fail("categorical symbolic distribution probabilities must sum to 1");
    }
    std::vector<int> conditions;
    conditions.reserve(static_cast<std::size_t>(nbits));
    for (int i = 0; i < nbits; ++i) {
        conditions.push_back(fresh_condition());
    }
    const std::size_t group = categorical_distributions.size();
    categorical_distributions.push_back({nbits, conditions, assignments, probabilities});
    for (int condition : conditions) {
        condition_to_categorical[condition] = group;
    }
    return conditions;
}

std::vector<SymbolicBool> SymbolicContext::fresh_categorical_bools(
    int nbits,
    const std::vector<std::vector<std::uint64_t>>& assignments,
    const std::vector<double>& probabilities) {
    const auto conditions = fresh_categorical_conditions(nbits, assignments, probabilities);
    std::vector<SymbolicBool> out;
    out.reserve(conditions.size());
    for (int condition : conditions) {
        out.push_back(symbolic_bool(condition));
    }
    return out;
}

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

void left_S(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    mul_rows(frame, frame.xrow(qi), frame.xrow(qi), frame.zrow(qi), 3);
}

void left_SDG(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    mul_rows(frame, frame.xrow(qi), frame.xrow(qi), frame.zrow(qi), 1);
}

void left_X(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
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
