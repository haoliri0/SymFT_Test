#include "symft/symft.hpp"

#include "symft/simd.hpp"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <type_traits>

namespace symft {
namespace {

constexpr int kWordBits = 64;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr std::size_t kNoSource = std::numeric_limits<std::size_t>::max();
constexpr double kLowProbabilitySampleThreshold = 0.02;
constexpr std::size_t kSimdPairRotationThreshold = 16;

[[noreturn]] void fail(const std::string& message) {
    throw Error(message);
}

int check_qubit(int nqubits, int q) {
    if (q < 0 || q >= nqubits) {
        fail("qubit index out of range");
    }
    return q;
}

void check_same_nqubits(const PauliString& a, const PauliString& b) {
    if (a.nqubits != b.nqubits) {
        fail("Pauli strings act on different numbers of qubits");
    }
}

std::uint64_t bit_mask(int q) {
    return std::uint64_t{1} << (q & 63);
}

int word_index(int q) {
    return q >> 6;
}

int popcount64(std::uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(value);
#else
    int count = 0;
    while (value != 0) {
        value &= value - 1;
        ++count;
    }
    return count;
#endif
}

std::uint64_t symbol_bit_mask(int condition) {
    if (condition <= 0) {
        fail("condition id must be positive");
    }
    return std::uint64_t{1} << ((condition - 1) & 63);
}

std::size_t symbol_word_index(int condition) {
    if (condition <= 0) {
        fail("condition id must be positive");
    }
    return static_cast<std::size_t>((condition - 1) >> 6);
}

std::size_t symbol_word_count(int nsymbols) {
    if (nsymbols <= 0) {
        return 0;
    }
    return static_cast<std::size_t>((nsymbols + 63) >> 6);
}

int trailing_zeros64(std::uint64_t value) {
    if (value == 0) {
        fail("trailing_zeros64 called with zero");
    }
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(value);
#else
    int count = 0;
    while ((value & 1u) == 0u) {
        value >>= 1;
        ++count;
    }
    return count;
#endif
}

bool is_odd_popcount(std::uint64_t value) {
    return (popcount64(value) & 1) != 0;
}

double check_probability(double probability) {
    if (!(probability >= 0.0 && probability <= 1.0)) {
        fail("probability must be between 0 and 1");
    }
    return probability;
}

void toggle_condition(std::vector<int>& conditions, int condition) {
    if (condition <= 0) {
        fail("condition id must be positive");
    }
    auto it = std::lower_bound(conditions.begin(), conditions.end(), condition);
    if (it != conditions.end() && *it == condition) {
        conditions.erase(it);
    } else {
        conditions.insert(it, condition);
    }
}

std::vector<int> normalize_conditions(const std::vector<int>& conditions) {
    std::vector<int> out;
    for (int condition : conditions) {
        toggle_condition(out, condition);
    }
    return out;
}

Complex phase_factor(int phase) {
    switch (phase & 3) {
    case 0:
        return {1.0, 0.0};
    case 1:
        return {0.0, 1.0};
    case 2:
        return {-1.0, 0.0};
    default:
        return {0.0, -1.0};
    }
}

std::size_t active_length(int k) {
    if (k < 0 || k >= 62) {
        fail("active qubit count is too large for machine basis indices");
    }
    return std::size_t{1} << k;
}

std::size_t insert_zero_bit(std::size_t packed, int bit) {
    const std::size_t low_mask = (std::size_t{1} << bit) - std::size_t{1};
    const std::size_t low = packed & low_mask;
    const std::size_t high = packed & ~low_mask;
    return low | (high << 1);
}

std::uint64_t active_mask_x(const PauliString& pauli) {
    return pauli.x.empty() ? 0 : pauli.x[0];
}

std::uint64_t active_mask_z(const PauliString& pauli) {
    return pauli.z.empty() ? 0 : pauli.z[0];
}

bool active_action_phase_odd(const ActivePauliAction& action, std::size_t basis) {
    return is_odd_popcount(static_cast<std::uint64_t>(basis) & action.zmask);
}

Complex active_action_phase(const ActivePauliAction& action, std::size_t basis) {
    return active_action_phase_odd(action, basis) ? action.odd_phase : action.even_phase;
}

bool is_near_zero(double value) {
    return std::abs(value) < 1e-14;
}

bool can_rotate_real_pair_flip(const ActivePauliAction& action) {
    if (action.zmask == 0 || !action.xz_overlap_odd) {
        return false;
    }
    return is_near_zero(action.even_phase.real()) && !is_near_zero(action.even_phase.imag());
}

inline void rotate_uniform_imag_pairs_scalar(
    Complex* alpha,
    const std::size_t* left_indices,
    const std::size_t* right_indices,
    std::size_t npairs,
    double c,
    double q) {
    auto* raw = reinterpret_cast<double*>(alpha);
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        double* a0 = raw + 2 * left_indices[idx];
        double* a1 = raw + 2 * right_indices[idx];
        const double r0 = a0[0];
        const double i0 = a0[1];
        const double r1 = a1[0];
        const double i1 = a1[1];
        a0[0] = c * r0 - q * i1;
        a0[1] = c * i0 + q * r1;
        a1[0] = c * r1 - q * i0;
        a1[1] = c * i1 + q * r0;
    }
}

inline void rotate_real_pair_flip_scalar(
    Complex* alpha,
    const std::size_t* left_indices,
    const std::size_t* right_indices,
    std::size_t npairs,
    double c,
    double base_coeff,
    std::uint64_t zmask) {
    auto* raw = reinterpret_cast<double*>(alpha);
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        const std::size_t i0 = left_indices[idx];
        const std::size_t i1 = right_indices[idx];
        const double q = is_odd_popcount(static_cast<std::uint64_t>(i0) & zmask) ? -base_coeff : base_coeff;
        double* a0 = raw + 2 * i0;
        double* a1 = raw + 2 * i1;
        const double r0 = a0[0];
        const double i0v = a0[1];
        const double r1 = a1[0];
        const double i1v = a1[1];
        a0[0] = c * r0 - q * r1;
        a0[1] = c * i0v - q * i1v;
        a1[0] = c * r1 + q * r0;
        a1[1] = c * i1v + q * i0v;
    }
}

PauliString project_pauli_body(const PauliString& pauli, int qstart, int qcount) {
    PauliString out(qcount);
    int y_count = 0;
    for (int q = 0; q < qcount; ++q) {
        const int src = qstart + q;
        const bool xb = pauli.xbit(src);
        const bool zb = pauli.zbit(src);
        if (xb) {
            out.set_xbit(q);
        }
        if (zb) {
            out.set_zbit(q);
        }
        if (xb && zb) {
            ++y_count;
        }
    }
    out.set_phase(y_count);
    return out;
}

PauliString embed_active_pauli(int n, const PauliString& active_body) {
    PauliString out(n);
    for (int q = 0; q < active_body.nqubits; ++q) {
        if (active_body.xbit(q)) {
            out.set_xbit(q);
        }
        if (active_body.zbit(q)) {
            out.set_zbit(q);
        }
    }
    out.set_phase(active_body.phase_exponent());
    return out;
}

bool optional_equal(const std::optional<int>& lhs, const std::optional<int>& rhs) {
    return lhs.has_value() == rhs.has_value() && (!lhs || *lhs == *rhs);
}

int max_condition(const SymbolicPauliString& pauli) {
    return pauli.sign.max_condition();
}

int max_condition(const PendingPauliRotation& rotation) {
    return max_condition(rotation.pauli);
}

int max_condition(const PendingPauliMeasurement& measurement) {
    int out = max_condition(measurement.pauli);
    if (measurement.record_condition) {
        out = std::max(out, *measurement.record_condition);
    }
    return out;
}

template <class... Ts>
struct Overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

int max_condition(const PendingOperation& operation) {
    return std::visit(Overloaded{
                          [](const PendingPauliRotation& op) { return max_condition(op); },
                          [](const PendingPauliMeasurement& op) { return max_condition(op); },
                      },
                      operation);
}

int max_condition(const ApplyPrecomputedActivePauliRotation& instruction) {
    return instruction.sign.max_condition();
}

int max_condition(const PromoteDormantRotation& instruction) {
    return instruction.sign.max_condition();
}

int max_condition(const RecordMeasurement& instruction) {
    int out = instruction.outcome.max_condition();
    if (instruction.record_condition) {
        out = std::max(out, *instruction.record_condition);
    }
    return out;
}

int max_condition(const MeasureActiveLastZ& instruction) {
    int out = std::max(instruction.branch, instruction.outcome.max_condition());
    if (instruction.record_condition) {
        out = std::max(out, *instruction.record_condition);
    }
    return out;
}

int max_condition(const MeasurePrecomputedActivePauli& instruction) {
    int out = std::max(instruction.branch, instruction.outcome.max_condition());
    if (instruction.record_condition) {
        out = std::max(out, *instruction.record_condition);
    }
    return out;
}

int max_condition(const IntroduceDormantMeasurementBranch& instruction) {
    int out = std::max(instruction.branch, instruction.outcome.max_condition());
    if (instruction.record_condition) {
        out = std::max(out, *instruction.record_condition);
    }
    return out;
}

int max_condition(const FactoredInstruction& instruction) {
    return std::visit([](const auto& inst) { return max_condition(inst); }, instruction);
}

} // namespace

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

ActiveState::ActiveState() : k(0), alpha{Complex(1.0, 0.0)} {}

ActiveState::ActiveState(int k_) : k(checked_nqubits(k_)), alpha(active_length(k), Complex(0.0, 0.0)) {
    alpha[0] = Complex(1.0, 0.0);
}

ActiveState::ActiveState(int k_, std::vector<Complex> alpha_) : k(checked_nqubits(k_)), alpha(std::move(alpha_)) {
    if (alpha.size() != active_length(k)) {
        fail("active vector length does not match active qubit count");
    }
}

std::size_t ActiveState::dim() const {
    return active_length(k);
}

void ActiveState::reserve_for_k(int max_k) {
    const std::size_t max_dim = active_length(max_k);
    if (alpha.size() < max_dim) {
        alpha.resize(max_dim, Complex(0.0, 0.0));
    }
}

ActivePauliAction::ActivePauliAction(const PauliString& pauli) : nqubits(pauli.nqubits) {
    if (nqubits >= 63) {
        fail("Pauli string has too many qubits for active-state basis indexing");
    }
    if (!pauli_squares_to_identity(pauli)) {
        fail("Pauli rotation requires P^2 == I");
    }
    xmask = active_mask_x(pauli);
    zmask = active_mask_z(pauli);
    even_phase = phase_factor(pauli.phase_exponent());
    odd_phase = -even_phase;
    xz_overlap_odd = is_odd_popcount(xmask & zmask);
}

PrecomputedActivePauliRotationKernel::PrecomputedActivePauliRotationKernel(const ActivePauliAction& action_, double theta_)
    : action(action_), is_diagonal(action.xmask == 0), theta(theta_), cos_theta(std::cos(theta_)) {
    const Complex minus_i_s(0.0, -std::sin(theta));
    const Complex plus_i_s(0.0, std::sin(theta));
    const std::size_t dim = active_length(action.nqubits);
    if (is_diagonal) {
        diagonal_minus_coefficients.resize(dim);
        diagonal_plus_coefficients.resize(dim);
        for (std::size_t basis = 0; basis < dim; ++basis) {
            const Complex phase = active_action_phase(action, basis);
            diagonal_minus_coefficients[basis] = minus_i_s * phase;
            diagonal_plus_coefficients[basis] = plus_i_s * phase;
        }
        return;
    }
    uniform_imag_pairs = action.zmask == 0;
    real_pair_flip = can_rotate_real_pair_flip(action);
    pair_bit = static_cast<unsigned>(trailing_zeros64(action.xmask));
    const std::size_t pair_selector = action.xmask & (~action.xmask + 1);
    for (std::size_t left = 0; left < dim; ++left) {
        if ((left & pair_selector) != 0) {
            continue;
        }
        const std::size_t right = left ^ action.xmask;
        pair_left_indices.push_back(left);
        pair_right_indices.push_back(right);
        if (real_pair_flip) {
            real_pair_flip_basis_phase_signs.push_back(
                is_odd_popcount(static_cast<std::uint64_t>(left) & action.zmask) ? -1.0 : 1.0);
        }
        const Complex left_phase = active_action_phase(action, left);
        const Complex right_phase = active_action_phase(action, right);
        pair_left_minus_coefficients.push_back(minus_i_s * left_phase);
        pair_right_minus_coefficients.push_back(minus_i_s * right_phase);
        pair_left_plus_coefficients.push_back(plus_i_s * left_phase);
        pair_right_plus_coefficients.push_back(plus_i_s * right_phase);
    }
}

namespace {

PrecomputedActivePauliMeasurementKernel precomputed_nondiagonal_measurement_kernel(const ActivePauliAction& action) {
    PrecomputedActivePauliMeasurementKernel out;
    out.action = action;
    out.is_diagonal = false;
    out.pivot = trailing_zeros64(action.xmask);
    const std::size_t dim = active_length(action.nqubits);
    const std::size_t out_dim = dim >> 1;
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    out.source0_false.reserve(out_dim);
    out.source1_false.reserve(out_dim);
    out.coeff0_false.reserve(out_dim);
    out.coeff1_false.reserve(out_dim);
    out.coeff1_false_real.reserve(out_dim);
    out.coeff1_false_imag.reserve(out_dim);
    out.source0_true.reserve(out_dim);
    out.source1_true.reserve(out_dim);
    out.coeff0_true.reserve(out_dim);
    out.coeff1_true.reserve(out_dim);
    for (std::size_t packed = 0; packed < out_dim; ++packed) {
        const std::size_t x0 = insert_zero_bit(packed, out.pivot);
        const std::size_t x1 = x0 ^ action.xmask;
        const Complex eta = active_action_phase(action, x0);
        const Complex coeff = (Complex(1.0, 0.0) / eta) * inv_sqrt2;
        out.source0_false.push_back(x0);
        out.source1_false.push_back(x1);
        out.coeff0_false.push_back(Complex(inv_sqrt2, 0.0));
        out.coeff1_false.push_back(coeff);
        out.coeff1_false_real.push_back(coeff.real());
        out.coeff1_false_imag.push_back(coeff.imag());
        out.source0_true.push_back(x0);
        out.source1_true.push_back(x1);
        out.coeff0_true.push_back(Complex(inv_sqrt2, 0.0));
        out.coeff1_true.push_back(-coeff);
    }
    return out;
}

PrecomputedActivePauliMeasurementKernel precomputed_diagonal_measurement_kernel(const ActivePauliAction& action) {
    PrecomputedActivePauliMeasurementKernel out;
    out.action = action;
    out.is_diagonal = true;
    out.pivot = trailing_zeros64(action.zmask);
    const std::size_t dim = active_length(action.nqubits);
    const std::size_t out_dim = dim >> 1;
    const bool negative_phase = std::abs(action.even_phase.real() + 1.0) < 1e-12 &&
                                std::abs(action.even_phase.imag()) < 1e-12;
    const bool positive_phase = std::abs(action.even_phase.real() - 1.0) < 1e-12 &&
                                std::abs(action.even_phase.imag()) < 1e-12;
    if (!negative_phase && !positive_phase) {
        fail("diagonal active measurement Pauli must have real eigenvalues");
    }
    const int phase_bit = negative_phase ? 1 : 0;
    const std::uint64_t z_without_pivot = action.zmask & ~(std::uint64_t{1} << out.pivot);
    out.source0_false.reserve(out_dim);
    out.source0_true.reserve(out_dim);
    out.coeff0_false.assign(out_dim, Complex(1.0, 0.0));
    out.coeff1_false.assign(out_dim, Complex(0.0, 0.0));
    out.coeff0_true.assign(out_dim, Complex(1.0, 0.0));
    out.coeff1_true.assign(out_dim, Complex(0.0, 0.0));
    out.source1_false.assign(out_dim, kNoSource);
    out.source1_true.assign(out_dim, kNoSource);
    for (std::size_t packed = 0; packed < out_dim; ++packed) {
        const std::size_t x_without_pivot = insert_zero_bit(packed, out.pivot);
        const int parity = popcount64(static_cast<std::uint64_t>(x_without_pivot) & z_without_pivot) & 1;
        const int false_pivot = phase_bit ^ parity;
        const int true_pivot = false_pivot ^ 1;
        out.source0_false.push_back(x_without_pivot | (std::size_t(false_pivot) << out.pivot));
        out.source0_true.push_back(x_without_pivot | (std::size_t(true_pivot) << out.pivot));
    }
    return out;
}

} // namespace

PrecomputedActivePauliMeasurementKernel::PrecomputedActivePauliMeasurementKernel(const PauliString& pauli)
    : PrecomputedActivePauliMeasurementKernel(ActivePauliAction(pauli)) {}

PrecomputedActivePauliMeasurementKernel::PrecomputedActivePauliMeasurementKernel(const ActivePauliAction& action_) {
    if (action_.nqubits <= 0) {
        fail("cannot build an active measurement kernel for k == 0");
    }
    if (action_.xmask != 0) {
        *this = precomputed_nondiagonal_measurement_kernel(action_);
    } else if (action_.zmask != 0) {
        *this = precomputed_diagonal_measurement_kernel(action_);
    } else {
        fail("cannot build an active measurement kernel for identity Pauli");
    }
}

void apply_pauli(std::vector<Complex>& out, const PauliString& pauli, const std::vector<Complex>& alpha) {
    if (pauli.nqubits >= 63) {
        fail("Pauli string has too many qubits for active-state basis indexing");
    }
    const std::size_t dim = active_length(pauli.nqubits);
    if (alpha.size() < dim || out.size() < dim || out.data() == alpha.data()) {
        fail("invalid active Pauli application buffers");
    }
    const std::uint64_t xmask = active_mask_x(pauli);
    const std::uint64_t zmask = active_mask_z(pauli);
    const Complex even = phase_factor(pauli.phase_exponent());
    const Complex odd = -even;
    for (std::size_t src = 0; src < dim; ++src) {
        const std::size_t dst = src ^ xmask;
        out[dst] = (is_odd_popcount(static_cast<std::uint64_t>(src) & zmask) ? odd : even) * alpha[src];
    }
}

void apply_pauli(ActiveState& state, const PauliString& pauli) {
    if (pauli.nqubits != state.k) {
        fail("Pauli string dimension does not match active state");
    }
    const ActivePauliAction action(pauli);
    const std::size_t dim = state.dim();
    if (action.xmask == 0) {
        for (std::size_t src = 0; src < dim; ++src) {
            state.alpha[src] *= active_action_phase(action, src);
        }
        return;
    }
    const std::size_t selector = action.xmask & (~action.xmask + 1);
    for (std::size_t base = 0; base < dim; ++base) {
        if ((base & selector) != 0) {
            continue;
        }
        const std::size_t paired = base ^ action.xmask;
        const Complex a0 = state.alpha[base];
        const Complex a1 = state.alpha[paired];
        const bool parity0 = active_action_phase_odd(action, base);
        const Complex phase0 = parity0 ? action.odd_phase : action.even_phase;
        const Complex phase1 = (parity0 != action.xz_overlap_odd) ? action.odd_phase : action.even_phase;
        state.alpha[base] = phase1 * a1;
        state.alpha[paired] = phase0 * a0;
    }
}

void rotate_pauli(ActiveState& state, const PauliString& pauli, double theta) {
    rotate_pauli(state, ActivePauliAction(pauli), theta);
}

void rotate_pauli(ActiveState& state, const ActivePauliAction& action, double theta) {
    if (action.nqubits != state.k) {
        fail("Pauli action dimension does not match active state");
    }
    const std::size_t dim = state.dim();
    const double c = std::cos(theta);
    const Complex minus_i_s(0.0, -std::sin(theta));
    if (action.xmask == 0) {
        for (std::size_t basis = 0; basis < dim; ++basis) {
            state.alpha[basis] *= Complex(c, 0.0) + minus_i_s * active_action_phase(action, basis);
        }
        return;
    }
    const std::size_t selector = action.xmask & (~action.xmask + 1);
    for (std::size_t base = 0; base < dim; ++base) {
        if ((base & selector) != 0) {
            continue;
        }
        const std::size_t paired = base ^ action.xmask;
        const Complex a0 = state.alpha[base];
        const Complex a1 = state.alpha[paired];
        const bool parity0 = active_action_phase_odd(action, base);
        const Complex phase0 = parity0 ? action.odd_phase : action.even_phase;
        const Complex phase1 = (parity0 != action.xz_overlap_odd) ? action.odd_phase : action.even_phase;
        state.alpha[base] = c * a0 + minus_i_s * phase1 * a1;
        state.alpha[paired] = c * a1 + minus_i_s * phase0 * a0;
    }
}

void rotate_pauli(ActiveState& state, const PrecomputedActivePauliRotationKernel& kernel, bool sign) {
    if (kernel.action.nqubits != state.k) {
        fail("rotation kernel dimension does not match active state");
    }
    const double c = kernel.cos_theta;
    if (kernel.is_diagonal) {
        const auto& coefficients = sign ? kernel.diagonal_plus_coefficients : kernel.diagonal_minus_coefficients;
        simd::dispatch_table().mul_assign(state.alpha.data(), coefficients.data(), c, coefficients.size());
        return;
    }
    if (kernel.uniform_imag_pairs) {
        const Complex coefficient = sign ? kernel.pair_left_plus_coefficients.front() : kernel.pair_left_minus_coefficients.front();
        const std::size_t npairs = kernel.pair_left_indices.size();
        if (npairs < kSimdPairRotationThreshold) {
            rotate_uniform_imag_pairs_scalar(
                state.alpha.data(),
                kernel.pair_left_indices.data(),
                kernel.pair_right_indices.data(),
                npairs,
                c,
                coefficient.imag());
        } else {
            simd::dispatch_table().rotate_uniform_imag_pairs(
                state.alpha.data(),
                kernel.pair_left_indices.data(),
                kernel.pair_right_indices.data(),
                npairs,
                c,
                coefficient.imag());
        }
        return;
    }
    if (kernel.real_pair_flip) {
        const Complex coefficient = sign ? kernel.pair_left_plus_coefficients.front() : kernel.pair_left_minus_coefficients.front();
        const std::size_t npairs = kernel.pair_left_indices.size();
        if (npairs < kSimdPairRotationThreshold) {
            rotate_real_pair_flip_scalar(
                state.alpha.data(),
                kernel.pair_left_indices.data(),
                kernel.pair_right_indices.data(),
                npairs,
                c,
                coefficient.real(),
                kernel.action.zmask);
        } else {
            simd::dispatch_table().rotate_real_pair_flip(
                state.alpha.data(),
                kernel.pair_left_indices.data(),
                kernel.pair_right_indices.data(),
                npairs,
                c,
                coefficient.real(),
                kernel.action.zmask);
        }
        return;
    }
    const auto& left_coeff = sign ? kernel.pair_left_plus_coefficients : kernel.pair_left_minus_coefficients;
    const auto& right_coeff = sign ? kernel.pair_right_plus_coefficients : kernel.pair_right_minus_coefficients;
    for (std::size_t idx = 0; idx < kernel.pair_left_indices.size(); ++idx) {
        const std::size_t i0 = kernel.pair_left_indices[idx];
        const std::size_t i1 = kernel.pair_right_indices[idx];
        const Complex a0 = state.alpha[i0];
        const Complex a1 = state.alpha[i1];
        state.alpha[i0] = c * a0 + right_coeff[idx] * a1;
        state.alpha[i1] = c * a1 + left_coeff[idx] * a0;
    }
}

std::string active_simd_backend() {
    return simd::dispatch_name();
}

bool operator==(const PendingPauliRotation& lhs, const PendingPauliRotation& rhs) {
    return lhs.theta == rhs.theta && lhs.pauli == rhs.pauli;
}

bool operator==(const PendingPauliMeasurement& lhs, const PendingPauliMeasurement& rhs) {
    return lhs.pauli == rhs.pauli && optional_equal(lhs.record, rhs.record) &&
           optional_equal(lhs.record_condition, rhs.record_condition);
}

bool operator==(const PendingOperation& lhs, const PendingOperation& rhs) {
    return std::visit(
        [](const auto& a, const auto& b) -> bool {
            using A = std::decay_t<decltype(a)>;
            using B = std::decay_t<decltype(b)>;
            if constexpr (std::is_same_v<A, B>) {
                return a == b;
            } else {
                return false;
            }
        },
        lhs,
        rhs);
}

bool operator==(const ApplyPrecomputedActivePauliRotation& lhs, const ApplyPrecomputedActivePauliRotation& rhs) {
    return lhs.pauli == rhs.pauli && lhs.theta == rhs.theta && lhs.sign == rhs.sign;
}

bool operator==(const PromoteDormantRotation& lhs, const PromoteDormantRotation& rhs) {
    return lhs.theta == rhs.theta && lhs.sign == rhs.sign;
}

bool operator==(const RecordMeasurement& lhs, const RecordMeasurement& rhs) {
    return lhs.outcome == rhs.outcome && optional_equal(lhs.record, rhs.record) &&
           optional_equal(lhs.record_condition, rhs.record_condition);
}

bool operator==(const MeasureActiveLastZ& lhs, const MeasureActiveLastZ& rhs) {
    return lhs.branch == rhs.branch && lhs.outcome == rhs.outcome && optional_equal(lhs.record, rhs.record) &&
           optional_equal(lhs.record_condition, rhs.record_condition);
}

bool operator==(const MeasurePrecomputedActivePauli& lhs, const MeasurePrecomputedActivePauli& rhs) {
    return lhs.pauli == rhs.pauli && lhs.branch == rhs.branch && lhs.outcome == rhs.outcome &&
           optional_equal(lhs.record, rhs.record) && optional_equal(lhs.record_condition, rhs.record_condition);
}

bool operator==(const IntroduceDormantMeasurementBranch& lhs, const IntroduceDormantMeasurementBranch& rhs) {
    return lhs.branch == rhs.branch && lhs.outcome == rhs.outcome && optional_equal(lhs.record, rhs.record) &&
           optional_equal(lhs.record_condition, rhs.record_condition);
}

bool operator==(const FactoredInstruction& lhs, const FactoredInstruction& rhs) {
    return std::visit(
        [](const auto& a, const auto& b) -> bool {
            using A = std::decay_t<decltype(a)>;
            using B = std::decay_t<decltype(b)>;
            if constexpr (std::is_same_v<A, B>) {
                return a == b;
            } else {
                return false;
            }
        },
        lhs,
        rhs);
}

FrameFactoredState::FrameFactoredState(int n_, int k_)
    : FrameFactoredState(n_, k_, std::make_shared<SymbolicContext>()) {}

FrameFactoredState::FrameFactoredState(int n_, int k_, std::shared_ptr<SymbolicContext> context_)
    : n(checked_nqubits(n_)),
      k(checked_nqubits(k_)),
      clifford(n),
      active(k),
      context(std::move(context_)) {
    if (k > n) {
        fail("active qubit count exceeds total qubit count");
    }
    if (!context) {
        context = std::make_shared<SymbolicContext>();
    }
    active_frame = ActivePauliFrame(n, context);
    dormant = DormantState(n - k, context);
}

void left_H(FrameFactoredState& state, int q) {
    left_H(state.clifford, q);
}

void left_S(FrameFactoredState& state, int q) {
    left_S(state.clifford, q);
}

void left_SDG(FrameFactoredState& state, int q) {
    left_SDG(state.clifford, q);
}

void left_X(FrameFactoredState& state, int q) {
    left_X(state.clifford, q);
}

void left_Z(FrameFactoredState& state, int q) {
    left_Z(state.clifford, q);
}

void left_CX(FrameFactoredState& state, int control, int target) {
    left_CX(state.clifford, control, target);
}

void left_CZ(FrameFactoredState& state, int a, int b) {
    left_CZ(state.clifford, a, b);
}

void left_SWAP(FrameFactoredState& state, int a, int b) {
    left_SWAP(state.clifford, a, b);
}

namespace {

SymbolicPauliString prepare_pending_pauli(FrameFactoredState& state, const PauliString& pauli) {
    if (pauli.nqubits != state.n) {
        fail("Pauli string dimension does not match frame-factored state");
    }
    const PauliString pre = preimage(state.clifford, pauli);
    return conjugate_by(state.active_frame, pre);
}

} // namespace

void apply_pauli(FrameFactoredState& state, const ConditionalPauliString& pauli) {
    if (pauli.pauli.nqubits != state.n) {
        fail("Pauli string dimension does not match frame-factored state");
    }
    state.context->bump_next_condition(pauli.condition);
    const PauliString pre = preimage(state.clifford, pauli.pauli);
    if (pre.has_nonidentity_body()) {
        state.active_frame.add_pauli(pre, pauli.condition);
    }
}

void apply_pauli(FrameFactoredState& state, const PauliString& pauli, int condition) {
    apply_pauli(state, ConditionalPauliString(pauli, condition));
}

void apply_pauli(FrameFactoredState& state, const PauliString& pauli, const SymbolicBool& condition) {
    if (condition.constant) {
        apply_pauli(state, pauli, state.context->fresh_bernoulli_condition(1.0));
    }
    for (int condition_id : condition.conditions) {
        apply_pauli(state, pauli, condition_id);
    }
}

PendingPauliRotation apply_pauli_rotation(FrameFactoredState& state, const PauliString& pauli, double theta) {
    PendingPauliRotation rotation{theta, prepare_pending_pauli(state, pauli)};
    state.pending_operations.push_back(rotation);
    return rotation;
}

PendingPauliMeasurement apply_pauli_measurement(FrameFactoredState& state, const PauliString& pauli) {
    PendingPauliMeasurement measurement{prepare_pending_pauli(state, pauli), std::nullopt, std::nullopt};
    state.pending_operations.push_back(measurement);
    return measurement;
}

PendingPauliMeasurement apply_pauli_measurement(
    FrameFactoredState& state,
    const PauliString& pauli,
    const SymbolicBool& sign,
    std::optional<int> record,
    std::optional<int> record_condition) {
    SymbolicPauliString prepared = prepare_pending_pauli(state, pauli);
    PendingPauliMeasurement measurement{
        SymbolicPauliString(prepared.pauli, xor_bool(prepared.sign, sign)),
        record,
        record_condition,
    };
    state.pending_operations.push_back(measurement);
    return measurement;
}

PendingFactoredState::PendingFactoredState(int n_, int k_)
    : PendingFactoredState(n_, k_, std::make_shared<SymbolicContext>()) {}

PendingFactoredState::PendingFactoredState(int n_, int k_, std::shared_ptr<SymbolicContext> context_)
    : n(checked_nqubits(n_)),
      k(checked_nqubits(k_)),
      max_k(k),
      active(k),
      context(std::move(context_)) {
    if (k > n) {
        fail("active qubit count exceeds total qubit count");
    }
    if (!context) {
        context = std::make_shared<SymbolicContext>();
    }
    dormant = DormantState(n - k, context);
}

PendingFactoredState::PendingFactoredState(const FrameFactoredState& state)
    : n(state.n),
      k(state.k),
      max_k(state.k),
      active(state.active),
      dormant(state.dormant.bits, state.context),
      context(state.context),
      pending_operations(state.pending_operations) {
    for (const auto& op : pending_operations) {
        context->bump_next_condition(max_condition(op));
        if (auto measurement = std::get_if<PendingPauliMeasurement>(&op); measurement && measurement->record) {
            next_record = std::max(next_record, *measurement->record + 1);
        }
    }
}

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

PendingPauliRotation prepare_active_qubit_for_dormant_rotation(
    PendingFactoredState& state,
    const PendingPauliRotation& current,
    int q) {
    const bool xb = current.pauli.pauli.xbit(q);
    const bool zb = current.pauli.pauli.zbit(q);
    if (xb && zb) {
        return rewrite_current_and_pending_by_basis_change(state, current, {BasisChange::S, q, 0});
    }
    if (zb) {
        return rewrite_current_and_pending_by_basis_change(state, current, {BasisChange::H, q, 0});
    }
    return current;
}

PendingPauliRotation eliminate_active_x_for_dormant_rotation(
    PendingFactoredState& state,
    const PendingPauliRotation& current,
    int picked_dormant,
    int target_active) {
    return rewrite_current_and_pending_by_basis_change(
        state,
        current,
        {BasisChange::CX, state.k + picked_dormant, target_active});
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
        current = prepare_active_qubit_for_dormant_rotation(state, current, q);
        if (current.pauli.pauli.xbit(q)) {
            current = eliminate_active_x_for_dormant_rotation(state, current, picked_dormant, q);
        }
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

PendingPauliMeasurement prepare_active_qubit_for_dormant_measurement(
    PendingFactoredState& state,
    const PendingPauliMeasurement& current,
    int q) {
    const bool xb = current.pauli.pauli.xbit(q);
    const bool zb = current.pauli.pauli.zbit(q);
    if (xb && zb) {
        return rewrite_current_and_pending_by_basis_change(state, current, {BasisChange::S, q, 0});
    }
    if (zb) {
        return rewrite_current_and_pending_by_basis_change(state, current, {BasisChange::H, q, 0});
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
        current = prepare_active_qubit_for_dormant_measurement(state, current, q);
        if (current.pauli.pauli.xbit(q)) {
            current = rewrite_current_and_pending_by_basis_change(
                state,
                current,
                {BasisChange::CX, state.k + picked_dormant, q});
        }
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
    ActiveState initial_active_,
    std::vector<FactoredInstruction> instructions_,
    int max_k_,
    SymbolicContext context_)
    : n(checked_nqubits(n_)),
      initial_k(checked_nqubits(initial_k_)),
      max_k(checked_nqubits(max_k_)),
      initial_active(std::move(initial_active_)),
      instructions(std::move(instructions_)),
      context(std::move(context_)) {
    if (initial_k > n || max_k > n || initial_k > max_k || initial_active.k != initial_k) {
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
    return FactoredInstructionProgram(state.n, state.active.k, state.active, state.instructions, state.max_k, *state.context);
}

FactoredInstructionProgram plan_factored_updates(PendingFactoredState& state) {
    process_pending_operations(state);
    return factored_instruction_program(state);
}

namespace {

std::uint64_t next_random_u64(std::uint64_t& state) {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

double rand_float(std::uint64_t& state) {
    return static_cast<double>(next_random_u64(state) >> 11) * 0x1.0p-53;
}

bool sample_bernoulli(std::uint64_t& rng_state, double probability) {
    const double p = check_probability(probability);
    if (p <= 0.0) {
        return false;
    }
    if (p >= 1.0) {
        return true;
    }
    return rand_float(rng_state) < p;
}

int sample_categorical_row(std::uint64_t& rng_state, const std::vector<double>& probabilities) {
    const double r = rand_float(rng_state);
    double cumulative = 0.0;
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        cumulative += probabilities[i];
        if (r <= cumulative) {
            return static_cast<int>(i);
        }
    }
    return static_cast<int>(probabilities.size() - 1);
}

void check_symbol_slot(const FactoredExecutorState& runtime, int condition) {
    if (condition <= 0 || condition > runtime.nsymbols) {
        fail("symbolic condition exceeds executor symbol table");
    }
}

bool is_assigned(const FactoredExecutorState& runtime, int condition) {
    check_symbol_slot(runtime, condition);
    const std::size_t word = symbol_word_index(condition);
    return word < runtime.assigned_words.size() && (runtime.assigned_words[word] & symbol_bit_mask(condition)) != 0;
}

void set_symbol_assignment_unchecked(FactoredExecutorState& runtime, int condition, bool value) {
    const std::size_t word = symbol_word_index(condition);
    const std::uint64_t mask = symbol_bit_mask(condition);
    runtime.assigned_words[word] |= mask;
    if (value) {
        runtime.value_words[word] |= mask;
    } else {
        runtime.value_words[word] &= ~mask;
    }
}

void set_assigned_symbol_true_unchecked(FactoredExecutorState& runtime, int condition) {
    const std::size_t word = symbol_word_index(condition);
    const std::uint64_t mask = symbol_bit_mask(condition);
    runtime.value_words[word] |= mask;
}

void assign_symbol(FactoredExecutorState& runtime, std::optional<int> condition, bool value) {
    if (!condition) {
        return;
    }
    check_symbol_slot(runtime, *condition);
    const std::size_t word = symbol_word_index(*condition);
    const std::uint64_t mask = symbol_bit_mask(*condition);
    if ((runtime.assigned_words[word] & mask) != 0) {
        if (((runtime.value_words[word] & mask) != 0) != value) {
            fail("symbolic condition was assigned inconsistent concrete values");
        }
        return;
    }
    set_symbol_assignment_unchecked(runtime, *condition, value);
}

void assign_symbol(FactoredExecutorState& runtime, int condition, bool value) {
    assign_symbol(runtime, std::optional<int>(condition), value);
}

bool eval_symbolic_bool_packed(const SymbolicBoolEvaluationPlan& plan, const FactoredExecutorState& runtime) {
    const std::size_t max_word = static_cast<std::size_t>(plan.word_indices.back());
    if (max_word >= runtime.assigned_words.size()) {
        fail("symbolic condition expression has no concrete value");
    }
    const auto* word_indices = plan.word_indices.data();
    const auto* word_masks = plan.word_masks.data();
    const auto* assigned_words = runtime.assigned_words.data();
    const auto* value_words = runtime.value_words.data();
    std::uint64_t parity_bits = 0;
    std::uint64_t missing = 0;
    for (std::size_t i = 0; i < plan.word_indices.size(); ++i) {
        const std::size_t word = static_cast<std::size_t>(word_indices[i]);
        const std::uint64_t mask = word_masks[i];
        missing |= mask & ~assigned_words[word];
        parity_bits ^= value_words[word] & mask;
    }
    if (missing != 0) {
        fail("symbolic condition expression has no concrete value");
    }
    return plan.constant != is_odd_popcount(parity_bits);
}

bool eval_symbolic_bool(const SymbolicBoolEvaluationPlan& plan, const FactoredExecutorState& runtime) {
    if (plan.word_indices.empty()) {
        return plan.constant;
    }
    return eval_symbolic_bool_packed(plan, runtime);
}

void write_measurement_record(
    FactoredExecutorState& runtime,
    std::optional<int> record,
    bool outcome,
    std::optional<int> record_condition) {
    if (record) {
        if (*record <= 0) {
            fail("measurement record id must be positive");
        }
        if (*record > runtime.nrecords) {
            runtime.nrecords = *record;
        }
        const std::size_t nwords = symbol_word_count(runtime.nrecords);
        if (runtime.measurement_words.size() < nwords) {
            runtime.measurement_words.resize(nwords, 0);
        }
        const std::size_t word = symbol_word_index(*record);
        const std::uint64_t mask = symbol_bit_mask(*record);
        if (outcome) {
            runtime.measurement_words[word] |= mask;
        } else {
            runtime.measurement_words[word] &= ~mask;
        }
    }
    assign_symbol(runtime, record_condition, outcome);
}

void sample_categorical_distribution(
    FactoredExecutorState& runtime,
    const std::vector<int>& conditions,
    int nbits,
    const std::vector<std::vector<std::uint64_t>>& assignments,
    const std::vector<double>& probabilities) {
    if (static_cast<int>(conditions.size()) != nbits) {
        fail("categorical condition count does not match assignment bit count");
    }
    bool any_assigned = false;
    bool all_assigned = true;
    for (int condition : conditions) {
        const bool assigned = is_assigned(runtime, condition);
        any_assigned = any_assigned || assigned;
        all_assigned = all_assigned && assigned;
    }
    if (all_assigned) {
        return;
    }
    if (any_assigned) {
        fail("categorical symbolic distribution was only partially preassigned");
    }
    const int row = sample_categorical_row(runtime.rng_state, probabilities);
    for (std::size_t i = 0; i < conditions.size(); ++i) {
        assign_symbol(runtime, conditions[i], packed_bit(assignments[static_cast<std::size_t>(row)], static_cast<int>(i)));
    }
}

double sample_geometric_gap(std::uint64_t& rng_state, double probability) {
    if (!(probability > 0.0 && probability < 1.0)) {
        fail("geometric gap probability must be in (0, 1)");
    }
    const double u = std::max(rand_float(rng_state), std::numeric_limits<double>::min());
    const double gap = std::floor(std::log(u) / std::log1p(-probability));
    if (!std::isfinite(gap) || gap >= static_cast<double>(std::numeric_limits<int>::max())) {
        return static_cast<double>(std::numeric_limits<int>::max());
    }
    return gap;
}

bool any_assigned(const FactoredExecutorState& runtime, const std::vector<int>& conditions) {
    for (int condition : conditions) {
        if (is_assigned(runtime, condition)) {
            return true;
        }
    }
    return false;
}

bool any_categorical_group_assigned(
    const FactoredExecutorState& runtime,
    const std::vector<std::vector<int>>& condition_sets) {
    for (const auto& conditions : condition_sets) {
        if (any_assigned(runtime, conditions)) {
            return true;
        }
    }
    return false;
}

void assign_conditions_false(FactoredExecutorState& runtime, const std::vector<int>& conditions) {
    for (int condition : conditions) {
        check_symbol_slot(runtime, condition);
        set_symbol_assignment_unchecked(runtime, condition, false);
    }
}

void assign_categorical_group_false(
    FactoredExecutorState& runtime,
    const std::vector<std::vector<int>>& condition_sets) {
    for (const auto& conditions : condition_sets) {
        assign_conditions_false(runtime, conditions);
    }
}

void sample_rare_categorical_group(FactoredExecutorState& runtime, const RareCategoricalSampleGroup& group) {
    if (any_categorical_group_assigned(runtime, group.conditions)) {
        for (const auto& conditions : group.conditions) {
            sample_categorical_distribution(runtime, conditions, group.nbits, group.assignments, group.probabilities);
        }
        return;
    }

    assign_categorical_group_false(runtime, group.conditions);
    if (group.event_probability <= 0.0) {
        return;
    }
    int idx = 0;
    const int nsets = static_cast<int>(group.conditions.size());
    while (true) {
        const int gap = static_cast<int>(sample_geometric_gap(runtime.rng_state, group.event_probability));
        if (gap >= nsets - idx) {
            return;
        }
        idx += gap;
        const int row = group.event_rows[static_cast<std::size_t>(sample_categorical_row(runtime.rng_state, group.event_probabilities))];
        const auto& conditions = group.conditions[static_cast<std::size_t>(idx)];
        const auto& assignment = group.assignments[static_cast<std::size_t>(row)];
        for (std::size_t bit_idx = 0; bit_idx < conditions.size(); ++bit_idx) {
            if (packed_bit(assignment, static_cast<int>(bit_idx))) {
                set_assigned_symbol_true_unchecked(runtime, conditions[bit_idx]);
            }
        }
        ++idx;
    }
}

void sample_low_probability_bernoulli_group(FactoredExecutorState& runtime, const BernoulliSampleGroup& group) {
    if (any_assigned(runtime, group.conditions)) {
        for (int condition : group.conditions) {
            if (!is_assigned(runtime, condition)) {
                assign_symbol(runtime, condition, sample_bernoulli(runtime.rng_state, group.probability));
            }
        }
        return;
    }

    assign_conditions_false(runtime, group.conditions);
    if (group.probability <= 0.0) {
        return;
    }
    int idx = 0;
    const int nconditions = static_cast<int>(group.conditions.size());
    while (true) {
        const int gap = static_cast<int>(sample_geometric_gap(runtime.rng_state, group.probability));
        if (gap >= nconditions - idx) {
            return;
        }
        idx += gap;
        set_assigned_symbol_true_unchecked(runtime, group.conditions[static_cast<std::size_t>(idx)]);
        ++idx;
    }
}

void sample_exogenous_symbols(FactoredExecutorState& runtime, const FactoredInstructionProgram& program) {
    for (const auto& distribution : program.sampled_categorical_distributions) {
        sample_categorical_distribution(runtime, distribution.conditions, distribution.nbits, distribution.assignments, distribution.probabilities);
    }
    for (const auto& group : program.sampled_rare_categorical_groups) {
        sample_rare_categorical_group(runtime, group);
    }
    for (std::size_t i = 0; i < program.sampled_bernoulli_conditions.size(); ++i) {
        const int condition = program.sampled_bernoulli_conditions[i];
        if (!is_assigned(runtime, condition)) {
            assign_symbol(runtime, condition, sample_bernoulli(runtime.rng_state, program.sampled_bernoulli_probabilities[i]));
        }
    }
    for (const auto& group : program.sampled_low_probability_bernoulli_groups) {
        sample_low_probability_bernoulli_group(runtime, group);
    }
}

std::size_t presampled_word_index(std::size_t nwords, int shot_index, std::size_t word) {
    return static_cast<std::size_t>(shot_index) * nwords + word;
}

void xor_presampled_condition(
    std::vector<std::uint64_t>& value_words,
    std::size_t nwords,
    int shot_index,
    int condition) {
    const std::size_t word = symbol_word_index(condition);
    value_words[presampled_word_index(nwords, shot_index, word)] ^= symbol_bit_mask(condition);
}

void mark_exogenous_conditions(std::vector<std::uint64_t>& words, const std::vector<int>& conditions) {
    for (int condition : conditions) {
        const std::size_t word = symbol_word_index(condition);
        if (word >= words.size()) {
            fail("exogenous condition exceeds program symbol table");
        }
        words[word] |= symbol_bit_mask(condition);
    }
}

std::vector<std::uint64_t> exogenous_assigned_words(const FactoredInstructionProgram& program) {
    std::vector<std::uint64_t> words(symbol_word_count(program.nsymbols), 0);
    for (const auto& distribution : program.sampled_categorical_distributions) {
        mark_exogenous_conditions(words, distribution.conditions);
    }
    for (const auto& group : program.sampled_rare_categorical_groups) {
        for (const auto& conditions : group.conditions) {
            mark_exogenous_conditions(words, conditions);
        }
    }
    mark_exogenous_conditions(words, program.sampled_bernoulli_conditions);
    for (const auto& group : program.sampled_low_probability_bernoulli_groups) {
        mark_exogenous_conditions(words, group.conditions);
    }
    return words;
}

void presample_categorical_distribution(
    std::vector<std::uint64_t>& value_words,
    std::size_t nwords,
    std::uint64_t& rng_state,
    const SymbolicCategoricalDistribution& distribution,
    int shots) {
    for (int shot = 0; shot < shots; ++shot) {
        const int row = sample_categorical_row(rng_state, distribution.probabilities);
        const auto& assignment = distribution.assignments[static_cast<std::size_t>(row)];
        for (std::size_t bit_idx = 0; bit_idx < distribution.conditions.size(); ++bit_idx) {
            if (packed_bit(assignment, static_cast<int>(bit_idx))) {
                xor_presampled_condition(value_words, nwords, shot, distribution.conditions[bit_idx]);
            }
        }
    }
}

void presample_rare_categorical_group(
    std::vector<std::uint64_t>& value_words,
    std::size_t nwords,
    std::uint64_t& rng_state,
    const RareCategoricalSampleGroup& group,
    int shots) {
    const int nsets = static_cast<int>(group.conditions.size());
    const std::int64_t total_draws = static_cast<std::int64_t>(shots) * static_cast<std::int64_t>(nsets);
    if (group.event_probability <= 0.0 || nsets == 0) {
        return;
    }
    std::int64_t draw = 0;
    while (true) {
        const auto gap = static_cast<std::int64_t>(sample_geometric_gap(rng_state, group.event_probability));
        if (gap >= total_draws - draw) {
            return;
        }
        draw += gap;
        const int shot = static_cast<int>(draw / nsets);
        const int set_idx = static_cast<int>(draw % nsets);
        const int row = group.event_rows[static_cast<std::size_t>(sample_categorical_row(rng_state, group.event_probabilities))];
        const auto& conditions = group.conditions[static_cast<std::size_t>(set_idx)];
        const auto& assignment = group.assignments[static_cast<std::size_t>(row)];
        for (std::size_t bit_idx = 0; bit_idx < conditions.size(); ++bit_idx) {
            if (packed_bit(assignment, static_cast<int>(bit_idx))) {
                xor_presampled_condition(value_words, nwords, shot, conditions[bit_idx]);
            }
        }
        ++draw;
    }
}

void presample_bernoulli_condition(
    std::vector<std::uint64_t>& value_words,
    std::size_t nwords,
    std::uint64_t& rng_state,
    int condition,
    double probability,
    int shots) {
    const double p = check_probability(probability);
    if (p <= 0.0) {
        return;
    }
    if (p >= 1.0) {
        for (int shot = 0; shot < shots; ++shot) {
            xor_presampled_condition(value_words, nwords, shot, condition);
        }
        return;
    }
    if (p < kLowProbabilitySampleThreshold) {
        int draw = 0;
        while (true) {
            const int gap = static_cast<int>(sample_geometric_gap(rng_state, p));
            if (gap >= shots - draw) {
                return;
            }
            draw += gap;
            xor_presampled_condition(value_words, nwords, draw, condition);
            ++draw;
        }
    }
    for (int shot = 0; shot < shots; ++shot) {
        if (sample_bernoulli(rng_state, p)) {
            xor_presampled_condition(value_words, nwords, shot, condition);
        }
    }
}

void presample_low_probability_bernoulli_group(
    std::vector<std::uint64_t>& value_words,
    std::size_t nwords,
    std::uint64_t& rng_state,
    const BernoulliSampleGroup& group,
    int shots) {
    const int nconditions = static_cast<int>(group.conditions.size());
    const std::int64_t total_draws = static_cast<std::int64_t>(shots) * static_cast<std::int64_t>(nconditions);
    if (group.probability <= 0.0 || nconditions == 0) {
        return;
    }
    std::int64_t draw = 0;
    while (true) {
        const auto gap = static_cast<std::int64_t>(sample_geometric_gap(rng_state, group.probability));
        if (gap >= total_draws - draw) {
            return;
        }
        draw += gap;
        const int shot = static_cast<int>(draw / nconditions);
        const int condition_idx = static_cast<int>(draw % nconditions);
        xor_presampled_condition(value_words, nwords, shot, group.conditions[static_cast<std::size_t>(condition_idx)]);
        ++draw;
    }
}

void assign_presampled_exogenous(FactoredExecutorState& runtime, const PresampledExogenous& samples, int shot_index) {
    if (shot_index < 0 || shot_index >= samples.nshots) {
        fail("presampled shot index is out of range");
    }
    if (runtime.nsymbols != samples.nsymbols || runtime.value_words.size() < samples.nwords ||
        runtime.assigned_words.size() < samples.nwords ||
        samples.exogenous_assigned_words.size() != samples.nwords ||
        samples.value_words.size() != static_cast<std::size_t>(samples.nshots) * samples.nwords) {
        fail("presampled exogenous table does not match executor symbol table");
    }
    const std::size_t base = static_cast<std::size_t>(shot_index) * samples.nwords;
    for (std::size_t word = 0; word < samples.nwords; ++word) {
        const std::uint64_t exogenous = samples.exogenous_assigned_words[word];
        const std::uint64_t row = samples.value_words[base + word] & exogenous;
        const std::uint64_t overlap = runtime.assigned_words[word] & exogenous;
        if ((overlap & (runtime.value_words[word] ^ row)) != 0) {
            fail("presampled exogenous condition conflicts with an existing assignment");
        }
        runtime.assigned_words[word] |= exogenous;
        runtime.value_words[word] |= row;
    }
}

void promote_first_dormant_rotation(FactoredExecutorState& runtime, double theta) {
    if (runtime.ndormant <= 0) {
        fail("cannot promote a dormant qubit when none remain");
    }
    const std::size_t dim = runtime.active.dim();
    const std::size_t promoted_dim = 2 * dim;
    if (runtime.active.alpha.size() < promoted_dim) {
        runtime.active.alpha.resize(promoted_dim, Complex(0.0, 0.0));
    }
    if (runtime.active_scratch.size() < promoted_dim) {
        runtime.active_scratch.resize(promoted_dim, Complex(0.0, 0.0));
    }
    const double c = std::cos(theta);
    const Complex minus_i_s(0.0, -std::sin(theta));
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const Complex amp = runtime.active.alpha[basis];
        runtime.active.alpha[basis] = c * amp;
        runtime.active.alpha[dim + basis] = minus_i_s * amp;
    }
    ++runtime.k;
    --runtime.ndormant;
    runtime.active.k = runtime.k;
}

double active_last_z_probability_one(const ActiveState& active) {
    if (active.k <= 0) {
        fail("cannot measure last active qubit when k == 0");
    }
    const std::size_t mask = std::size_t{1} << (active.k - 1);
    const std::size_t dim = active.dim();
    double probability = 0.0;
    for (std::size_t basis = 0; basis < dim; ++basis) {
        if ((basis & mask) != 0) {
            probability += std::norm(active.alpha[basis]);
        }
    }
    return std::clamp(probability, 0.0, 1.0);
}

void project_active_last_z(ActiveState& active, bool branch, double prob1) {
    const int new_k = active.k - 1;
    const std::size_t dim = active_length(new_k);
    const double probability = branch ? prob1 : 1.0 - prob1;
    if (probability <= 0.0) {
        fail("sampled an impossible active measurement branch");
    }
    const double invnorm = 1.0 / std::sqrt(probability);
    const std::size_t branch_offset = branch ? dim : 0;
    for (std::size_t basis = 0; basis < dim; ++basis) {
        active.alpha[basis] = active.alpha[branch_offset + basis] * invnorm;
    }
    active.k = new_k;
}

double active_diagonal_measurement_branch_probability(
    const ActiveState& active,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch) {
    const auto& sources = branch ? kernel.source0_true : kernel.source0_false;
    double probability = simd::dispatch_table().norm_sum(active.alpha.data(), sources.data(), sources.size());
    return std::clamp(probability, 0.0, 1.0);
}

double active_measurement_branch_probability(
    const ActiveState& active,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch) {
    const auto& sources0 = branch ? kernel.source0_true : kernel.source0_false;
    const auto& sources1 = branch ? kernel.source1_true : kernel.source1_false;
    const auto& coeffs0 = branch ? kernel.coeff0_true : kernel.coeff0_false;
    const auto& coeffs1 = branch ? kernel.coeff1_true : kernel.coeff1_false;
    double probability = 0.0;
    for (std::size_t idx = 0; idx < sources0.size(); ++idx) {
        Complex amp = coeffs0[idx] * active.alpha[sources0[idx]];
        if (sources1[idx] != kNoSource) {
            amp += coeffs1[idx] * active.alpha[sources1[idx]];
        }
        probability += std::norm(amp);
    }
    return std::clamp(probability, 0.0, 1.0);
}

void project_diagonal_active_pauli_measurement(
    ActiveState& active,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch,
    double probability) {
    if (probability <= 0.0) {
        fail("sampled an impossible active measurement branch");
    }
    const auto& sources = branch ? kernel.source0_true : kernel.source0_false;
    const double invnorm = 1.0 / std::sqrt(probability);
    for (std::size_t idx = 0; idx < sources.size(); ++idx) {
        active.alpha[idx] = active.alpha[sources[idx]] * invnorm;
    }
    --active.k;
}

void project_active_pauli_measurement(
    ActiveState& active,
    std::vector<Complex>& scratch,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch,
    double probability) {
    if (probability <= 0.0) {
        fail("sampled an impossible active measurement branch");
    }
    const auto& sources0 = branch ? kernel.source0_true : kernel.source0_false;
    const auto& sources1 = branch ? kernel.source1_true : kernel.source1_false;
    const auto& coeffs0 = branch ? kernel.coeff0_true : kernel.coeff0_false;
    const auto& coeffs1 = branch ? kernel.coeff1_true : kernel.coeff1_false;
    if (scratch.size() < sources0.size()) {
        scratch.resize(sources0.size());
    }
    const double invnorm = 1.0 / std::sqrt(probability);
    for (std::size_t idx = 0; idx < sources0.size(); ++idx) {
        Complex amp = coeffs0[idx] * active.alpha[sources0[idx]];
        if (sources1[idx] != kNoSource) {
            amp += coeffs1[idx] * active.alpha[sources1[idx]];
        }
        scratch[idx] = amp * invnorm;
    }
    std::copy(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(sources0.size()), active.alpha.begin());
    --active.k;
}

void execute_instruction(FactoredExecutorState& runtime, const ApplyPrecomputedActivePauliRotation& instruction) {
    const bool sign = eval_symbolic_bool(instruction.sign_plan, runtime);
    rotate_pauli(runtime.active, instruction.rotation_kernel, sign);
}

void execute_instruction(FactoredExecutorState& runtime, const PromoteDormantRotation& instruction) {
    const bool sign = eval_symbolic_bool(instruction.sign_plan, runtime);
    promote_first_dormant_rotation(runtime, sign ? -instruction.theta : instruction.theta);
}

void execute_instruction(FactoredExecutorState& runtime, const RecordMeasurement& instruction) {
    const bool outcome = eval_symbolic_bool(instruction.outcome_plan, runtime);
    write_measurement_record(runtime, instruction.record, outcome, instruction.record_condition);
}

void execute_instruction(FactoredExecutorState& runtime, const MeasureActiveLastZ& instruction) {
    const double prob1 = active_last_z_probability_one(runtime.active);
    const bool branch = sample_bernoulli(runtime.rng_state, prob1);
    assign_symbol(runtime, instruction.branch, branch);
    project_active_last_z(runtime.active, branch, prob1);
    runtime.k = runtime.active.k;
    ++runtime.ndormant;
    const bool outcome = eval_symbolic_bool(instruction.outcome_plan, runtime);
    write_measurement_record(runtime, instruction.record, outcome, instruction.record_condition);
}

void execute_instruction(FactoredExecutorState& runtime, const MeasurePrecomputedActivePauli& instruction) {
    if (runtime.k <= 0) {
        fail("cannot measure an active Pauli when k == 0");
    }
    double prob_true = instruction.kernel.is_diagonal
                           ? active_diagonal_measurement_branch_probability(runtime.active, instruction.kernel, true)
                           : active_measurement_branch_probability(runtime.active, instruction.kernel, true);
    prob_true = std::clamp(prob_true, 0.0, 1.0);
    const double prob_false = 1.0 - prob_true;
    const bool branch = sample_bernoulli(runtime.rng_state, prob_true);
    const double probability = branch ? prob_true : prob_false;
    assign_symbol(runtime, instruction.branch, branch);
    if (instruction.kernel.is_diagonal) {
        project_diagonal_active_pauli_measurement(runtime.active, instruction.kernel, branch, probability);
    } else {
        project_active_pauli_measurement(runtime.active, runtime.active_scratch, instruction.kernel, branch, probability);
    }
    runtime.k = runtime.active.k;
    ++runtime.ndormant;
    const bool outcome = eval_symbolic_bool(instruction.outcome_plan, runtime);
    write_measurement_record(runtime, instruction.record, outcome, instruction.record_condition);
}

void execute_instruction(FactoredExecutorState& runtime, const IntroduceDormantMeasurementBranch& instruction) {
    const bool branch = sample_bernoulli(runtime.rng_state, 0.5);
    assign_symbol(runtime, instruction.branch, branch);
    const bool outcome = eval_symbolic_bool(instruction.outcome_plan, runtime);
    write_measurement_record(runtime, instruction.record, outcome, instruction.record_condition);
}

} // namespace

FactoredExecutorState::FactoredExecutorState(const FactoredInstructionProgram& program, std::uint64_t seed)
    : n(program.n),
      k(program.initial_k),
      ndormant(program.n - program.initial_k),
      nsymbols(program.nsymbols),
      nrecords(program.nrecords),
      active(program.initial_active),
      active_scratch(active_length(program.max_k), Complex(0.0, 0.0)),
      value_words(symbol_word_count(program.nsymbols), 0),
      assigned_words(symbol_word_count(program.nsymbols), 0),
      measurement_words(symbol_word_count(program.nrecords), 0),
      rng_state(seed) {
    active.reserve_for_k(program.max_k);
}

void reset_executor(FactoredExecutorState& runtime, const FactoredInstructionProgram& program) {
    runtime.n = program.n;
    runtime.k = program.initial_k;
    runtime.ndormant = program.n - program.initial_k;
    runtime.nsymbols = program.nsymbols;
    runtime.nrecords = program.nrecords;
    runtime.active.k = program.initial_active.k;
    runtime.active.reserve_for_k(program.max_k);
    const std::size_t dim = program.initial_active.dim();
    std::copy(program.initial_active.alpha.begin(), program.initial_active.alpha.begin() + static_cast<std::ptrdiff_t>(dim), runtime.active.alpha.begin());
    const std::size_t max_dim = active_length(program.max_k);
    if (runtime.active_scratch.size() < max_dim) {
        runtime.active_scratch.resize(max_dim);
    }
    const std::size_t nwords = symbol_word_count(program.nsymbols);
    if (runtime.value_words.size() != nwords) {
        runtime.value_words.resize(nwords);
    }
    std::fill(runtime.value_words.begin(), runtime.value_words.end(), 0);
    if (runtime.assigned_words.size() != nwords) {
        runtime.assigned_words.resize(nwords);
    }
    std::fill(runtime.assigned_words.begin(), runtime.assigned_words.end(), 0);
    const std::size_t record_words = symbol_word_count(program.nrecords);
    if (runtime.measurement_words.size() != record_words) {
        runtime.measurement_words.resize(record_words);
    }
    std::fill(runtime.measurement_words.begin(), runtime.measurement_words.end(), 0);
}

PresampledExogenous presample_exogenous(const FactoredInstructionProgram& program, int shots, std::uint64_t seed) {
    if (shots < 0) {
        fail("presampled shot count must be nonnegative");
    }
    PresampledExogenous samples;
    samples.nshots = shots;
    samples.nsymbols = program.nsymbols;
    samples.nwords = symbol_word_count(program.nsymbols);
    samples.exogenous_assigned_words = exogenous_assigned_words(program);
    samples.value_words.assign(static_cast<std::size_t>(shots) * samples.nwords, 0);

    std::uint64_t rng_state = seed;
    for (const auto& distribution : program.sampled_categorical_distributions) {
        presample_categorical_distribution(samples.value_words, samples.nwords, rng_state, distribution, shots);
    }
    for (const auto& group : program.sampled_rare_categorical_groups) {
        presample_rare_categorical_group(samples.value_words, samples.nwords, rng_state, group, shots);
    }
    for (std::size_t i = 0; i < program.sampled_bernoulli_conditions.size(); ++i) {
        presample_bernoulli_condition(
            samples.value_words,
            samples.nwords,
            rng_state,
            program.sampled_bernoulli_conditions[i],
            program.sampled_bernoulli_probabilities[i],
            shots);
    }
    for (const auto& group : program.sampled_low_probability_bernoulli_groups) {
        presample_low_probability_bernoulli_group(samples.value_words, samples.nwords, rng_state, group, shots);
    }
    samples.next_rng_state = rng_state;
    return samples;
}

void execute_in_place(FactoredExecutorState& runtime, const FactoredInstructionProgram& program) {
    if (runtime.n != program.n || runtime.k != runtime.active.k || runtime.k + runtime.ndormant != runtime.n) {
        fail("executor state does not match program");
    }
    sample_exogenous_symbols(runtime, program);
    for (const auto& instruction : program.instructions) {
        std::visit([&](const auto& inst) { execute_instruction(runtime, inst); }, instruction);
    }
}

void execute_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExogenous& samples,
    int shot_index) {
    if (runtime.n != program.n || runtime.k != runtime.active.k || runtime.k + runtime.ndormant != runtime.n) {
        fail("executor state does not match program");
    }
    if (program.nsymbols != samples.nsymbols) {
        fail("presampled exogenous table does not match program");
    }
    assign_presampled_exogenous(runtime, samples, shot_index);
    for (const auto& instruction : program.instructions) {
        std::visit([&](const auto& inst) { execute_instruction(runtime, inst); }, instruction);
    }
}

std::vector<std::uint64_t> execute(FactoredExecutorState& runtime, const FactoredInstructionProgram& program) {
    execute_in_place(runtime, program);
    return runtime.measurement_words;
}

std::vector<std::uint64_t> sample_measurements(const FactoredInstructionProgram& program, std::uint64_t seed) {
    FactoredExecutorState runtime(program, seed);
    return execute(runtime, program);
}

std::vector<std::vector<std::uint64_t>> sample_measurements(const FactoredInstructionProgram& program, int shots, std::uint64_t seed) {
    if (shots < 0) {
        fail("shot count must be nonnegative");
    }
    const auto samples = presample_exogenous(program, shots, seed);
    FactoredExecutorState runtime(program, samples.next_rng_state);
    std::vector<std::vector<std::uint64_t>> out;
    out.reserve(static_cast<std::size_t>(shots));
    for (int shot = 0; shot < shots; ++shot) {
        reset_executor(runtime, program);
        execute_in_place(runtime, program, samples, shot);
        out.push_back(runtime.measurement_words);
    }
    return out;
}

namespace {

std::string trim(const std::string& input) {
    std::size_t first = 0;
    while (first < input.size() && std::isspace(static_cast<unsigned char>(input[first]))) {
        ++first;
    }
    std::size_t last = input.size();
    while (last > first && std::isspace(static_cast<unsigned char>(input[last - 1]))) {
        --last;
    }
    return input.substr(first, last - first);
}

std::vector<std::string> split_ws(const std::string& input) {
    std::istringstream stream(input);
    std::vector<std::string> out;
    std::string item;
    while (stream >> item) {
        out.push_back(item);
    }
    return out;
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string upper(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return value;
}

bool all_digits(const std::string& value) {
    return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char ch) { return std::isdigit(ch); });
}

std::string strip_stim_comment(const std::string& line) {
    bool in_brackets = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '[') {
            in_brackets = true;
        } else if (ch == ']') {
            in_brackets = false;
        } else if (ch == '#' && !in_brackets) {
            return trim(line.substr(0, i));
        }
    }
    return trim(line);
}

struct StimInstruction {
    std::string op;
    std::optional<std::string> tag;
    std::vector<double> parens;
    std::vector<std::string> targets;
    int line = 0;
};

struct StimRepeatBlock;
using StimNode = std::variant<StimInstruction, std::shared_ptr<StimRepeatBlock>>;

struct StimRepeatBlock {
    int count = 0;
    std::vector<StimNode> body;
    int line = 0;
};

StimInstruction parse_instruction(const std::string& body, int line) {
    std::size_t idx = 0;
    while (idx < body.size() && (std::isalpha(static_cast<unsigned char>(body[idx])) ||
                                 (idx > 0 && (std::isdigit(static_cast<unsigned char>(body[idx])) || body[idx] == '_')))) {
        ++idx;
    }
    if (idx == 0) {
        fail("invalid Stim instruction");
    }
    StimInstruction inst;
    inst.op = upper(body.substr(0, idx));
    inst.line = line;
    if (idx < body.size() && body[idx] == '[') {
        const std::size_t close = body.find(']', idx + 1);
        if (close == std::string::npos) {
            fail("unterminated Stim tag");
        }
        inst.tag = body.substr(idx + 1, close - idx - 1);
        idx = close + 1;
    }
    if (idx < body.size() && body[idx] == '(') {
        const std::size_t close = body.find(')', idx + 1);
        if (close == std::string::npos) {
            fail("unterminated Stim argument list");
        }
        const std::string args = trim(body.substr(idx + 1, close - idx - 1));
        if (!args.empty()) {
            std::size_t start = 0;
            while (start <= args.size()) {
                const std::size_t comma = args.find(',', start);
                const std::string piece = trim(args.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
                inst.parens.push_back(std::stod(piece));
                if (comma == std::string::npos) {
                    break;
                }
                start = comma + 1;
            }
        }
        idx = close + 1;
    }
    if (idx < body.size()) {
        const std::string rest = trim(body.substr(idx));
        inst.targets = split_ws(rest);
    }
    return inst;
}

std::vector<StimNode> parse_nodes(const std::vector<std::string>& lines, std::size_t& pos, bool in_block) {
    std::vector<StimNode> nodes;
    while (pos < lines.size()) {
        std::string body = strip_stim_comment(lines[pos]);
        const int line = static_cast<int>(pos + 1);
        ++pos;
        if (body.empty()) {
            continue;
        }
        if (body == "}") {
            if (!in_block) {
                fail("unmatched Stim block terminator");
            }
            return nodes;
        }
        bool block_start = false;
        if (ends_with(body, "{")) {
            block_start = true;
            body = trim(body.substr(0, body.size() - 1));
        }
        StimInstruction inst = parse_instruction(body, line);
        if (block_start) {
            if (inst.op != "REPEAT" || inst.targets.size() != 1 || !all_digits(inst.targets[0])) {
                fail("invalid Stim REPEAT block");
            }
            auto block = std::make_shared<StimRepeatBlock>();
            block->count = std::stoi(inst.targets[0]);
            block->line = line;
            block->body = parse_nodes(lines, pos, true);
            nodes.push_back(block);
        } else {
            if (inst.op == "REPEAT") {
                fail("REPEAT must start a block");
            }
            nodes.push_back(inst);
        }
    }
    if (in_block) {
        fail("unterminated Stim block");
    }
    return nodes;
}

std::pair<int, bool> stim_qubit_target(const std::string& target, int line) {
    bool inverted = false;
    std::string body = target;
    if (starts_with(body, "!")) {
        inverted = true;
        body = body.substr(1);
    }
    if (!all_digits(body)) {
        (void)line;
        fail("invalid qubit target");
    }
    return {std::stoi(body), inverted};
}

std::vector<int> stim_qubit_targets(const StimInstruction& inst) {
    std::vector<int> out;
    for (const auto& target : inst.targets) {
        auto [q, inverted] = stim_qubit_target(target, inst.line);
        if (inverted) {
            fail("operation does not accept inverted targets");
        }
        out.push_back(q);
    }
    return out;
}

std::vector<std::pair<int, bool>> stim_measurement_targets(const StimInstruction& inst) {
    std::vector<std::pair<int, bool>> out;
    for (const auto& target : inst.targets) {
        out.push_back(stim_qubit_target(target, inst.line));
    }
    return out;
}

int stim_record_index(const std::string& target, int nrecords) {
    if (!starts_with(target, "rec[") || !ends_with(target, "]")) {
        fail("invalid measurement record target");
    }
    const int offset = std::stoi(target.substr(4, target.size() - 5));
    const int idx = nrecords + offset + 1;
    if (idx <= 0 || idx > nrecords) {
        fail("measurement record target out of range");
    }
    return idx;
}

std::vector<int> stim_record_indices(const StimInstruction& inst, const std::vector<SymbolicBool>& records) {
    std::vector<int> out;
    for (const auto& target : inst.targets) {
        out.push_back(stim_record_index(target, static_cast<int>(records.size())));
    }
    return out;
}

double stim_paren_probability(const StimInstruction& inst, double default_probability = 0.0) {
    if (inst.parens.empty()) {
        return default_probability;
    }
    if (inst.parens.size() != 1) {
        fail("operation expects at most one probability argument");
    }
    return check_probability(inst.parens[0]);
}

double stim_required_probability(const StimInstruction& inst) {
    if (inst.parens.size() != 1) {
        fail("operation expects one probability argument");
    }
    return check_probability(inst.parens[0]);
}

int target_max_qubit(const std::string& target) {
    if (target == "*" || starts_with(target, "rec[") || starts_with(target, "sweep[")) {
        return -1;
    }
    std::string body = target;
    if (starts_with(body, "!")) {
        body = body.substr(1);
    }
    int max_q = -1;
    std::size_t start = 0;
    while (start <= body.size()) {
        const std::size_t star = body.find('*', start);
        std::string factor = body.substr(start, star == std::string::npos ? std::string::npos : star - start);
        if (starts_with(factor, "!")) {
            factor = factor.substr(1);
        }
        if (!factor.empty() && (factor[0] == 'X' || factor[0] == 'Y' || factor[0] == 'Z')) {
            factor = factor.substr(1);
        }
        if (!factor.empty() && all_digits(factor)) {
            max_q = std::max(max_q, std::stoi(factor));
        }
        if (star == std::string::npos) {
            break;
        }
        start = star + 1;
    }
    return max_q;
}

int all_qubits(const std::vector<StimNode>& nodes) {
    int max_q = -1;
    for (const auto& node : nodes) {
        if (const auto* block = std::get_if<std::shared_ptr<StimRepeatBlock>>(&node)) {
            max_q = std::max(max_q, all_qubits((*block)->body) - 1);
            continue;
        }
        const auto& inst = std::get<StimInstruction>(node);
        if (inst.op == "DETECTOR" || inst.op == "OBSERVABLE_INCLUDE" || inst.op == "TICK" || inst.op == "SHIFT_COORDS") {
            continue;
        }
        for (const auto& target : inst.targets) {
            max_q = std::max(max_q, target_max_qubit(target));
        }
    }
    return max_q + 1;
}

PauliString pauli_on_target(int n, char axis, int q) {
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

std::pair<PauliString, bool> stim_mpp_pauli(int n, const std::string& target) {
    PauliString out = pauli_identity(n);
    bool inverted = false;
    std::size_t start = 0;
    while (start <= target.size()) {
        const std::size_t star = target.find('*', start);
        std::string factor = target.substr(start, star == std::string::npos ? std::string::npos : star - start);
        if (starts_with(factor, "!")) {
            inverted = !inverted;
            factor = factor.substr(1);
        }
        if (factor.size() < 2 || !(factor[0] == 'X' || factor[0] == 'Y' || factor[0] == 'Z')) {
            fail("invalid MPP factor");
        }
        const int q = std::stoi(factor.substr(1));
        out = out * pauli_on_target(n, factor[0], q);
        if (star == std::string::npos) {
            break;
        }
        start = star + 1;
    }
    return {out, inverted};
}

std::vector<std::string> stim_mpp_targets(const StimInstruction& inst) {
    std::vector<std::string> groups;
    std::string current;
    for (const auto& target : inst.targets) {
        if (target == "*") {
            if (current.empty() || ends_with(current, "*")) {
                fail("misplaced MPP combiner");
            }
            current += "*";
        } else if (current.empty()) {
            current = target;
        } else if (ends_with(current, "*")) {
            current += target;
        } else {
            groups.push_back(current);
            current = target;
        }
    }
    if (!current.empty()) {
        if (ends_with(current, "*")) {
            fail("dangling MPP combiner");
        }
        groups.push_back(current);
    }
    return groups;
}

std::pair<int, int> reserve_measurement_record(std::vector<SymbolicBool>& records, SymbolicContext& context) {
    const int condition = context.fresh_condition();
    records.push_back(symbolic_bool(condition));
    return {static_cast<int>(records.size()), condition};
}

SymbolicBool measurement_sign_with_readout_error(SymbolicContext& context, bool inverted, double probability) {
    SymbolicBool sign(inverted);
    if (probability != 0.0) {
        sign = xor_bool(sign, context.fresh_bernoulli_bool(probability));
    }
    return sign;
}

std::vector<double> coords_with_shift(const std::vector<double>& coords, const std::vector<double>& shift) {
    std::vector<double> out(std::max(coords.size(), shift.size()), 0.0);
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (i < coords.size()) {
            out[i] += coords[i];
        }
        if (i < shift.size()) {
            out[i] += shift[i];
        }
    }
    return out;
}

struct StimAccumulator {
    FrameFactoredState state;
    std::vector<SymbolicBool> records;
    std::vector<StimDetector> detectors;
    std::vector<StimObservableInclude> observables;
    std::vector<double> coord_shift;
};

void apply_depolarize1(FrameFactoredState& state, int q, double probability) {
    const std::vector<std::vector<std::uint64_t>> assignments{
        packed_bits({false, false}),
        packed_bits({false, true}),
        packed_bits({true, false}),
        packed_bits({true, true}),
    };
    const auto bits = state.context->fresh_categorical_bools(2, assignments, {1.0 - probability, probability / 3.0, probability / 3.0, probability / 3.0});
    apply_pauli(state, pauli_x(state.n, q), bits[0]);
    apply_pauli(state, pauli_z(state.n, q), bits[1]);
}

void apply_depolarize2(FrameFactoredState& state, int q1, int q2, double probability) {
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

void apply_record_feedback(FrameFactoredState& state, const std::vector<SymbolicBool>& records, const StimInstruction& inst) {
    if ((inst.targets.size() & 1u) != 0u) {
        fail("record-controlled feedback requires paired targets");
    }
    for (std::size_t i = 0; i < inst.targets.size(); i += 2) {
        const int record = stim_record_index(inst.targets[i], static_cast<int>(records.size()));
        auto [q, inverted] = stim_qubit_target(inst.targets[i + 1], inst.line);
        if (inverted) {
            fail("record-controlled feedback does not accept inverted qubit targets");
        }
        const PauliString pauli = (inst.op == "CX" || inst.op == "CNOT") ? pauli_x(state.n, q) : pauli_z(state.n, q);
        apply_pauli(state, pauli, records[static_cast<std::size_t>(record - 1)]);
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
    bool inverted,
    double probability) {
    const SymbolicBool sign = measurement_sign_with_readout_error(*state.context, inverted, probability);
    auto [record, record_condition] = reserve_measurement_record(records, *state.context);
    apply_pauli_measurement(state, measurement_pauli, sign, record, record_condition);
    apply_pauli(state, correction_pauli, xor_bool(symbolic_bool(record_condition), sign));
}

void apply_instruction(StimAccumulator& acc, const StimInstruction& inst) {
    FrameFactoredState& state = acc.state;
    const std::string& op = inst.op;
    if (op == "QUBIT_COORDS" || op == "TICK") {
        return;
    }
    if (op == "SHIFT_COORDS") {
        if (acc.coord_shift.size() < inst.parens.size()) {
            acc.coord_shift.resize(inst.parens.size(), 0.0);
        }
        for (std::size_t i = 0; i < inst.parens.size(); ++i) {
            acc.coord_shift[i] += inst.parens[i];
        }
        return;
    }
    if (op == "DETECTOR") {
        acc.detectors.push_back({stim_record_indices(inst, acc.records), coords_with_shift(inst.parens, acc.coord_shift), inst.line});
        return;
    }
    if (op == "OBSERVABLE_INCLUDE") {
        if (inst.parens.size() != 1 || inst.parens[0] < 0.0) {
            fail("OBSERVABLE_INCLUDE expects one nonnegative observable index");
        }
        acc.observables.push_back({static_cast<int>(std::llround(inst.parens[0])), stim_record_indices(inst, acc.records), inst.line});
        return;
    }
    if ((op == "CX" || op == "CNOT" || op == "CZ") && !inst.targets.empty() && starts_with(inst.targets[0], "rec[")) {
        apply_record_feedback(state, acc.records, inst);
        return;
    }
    if (op == "CX" || op == "CNOT") {
        const auto qs = stim_qubit_targets(inst);
        if ((qs.size() & 1u) != 0u) {
            fail("CX requires paired targets");
        }
        for (std::size_t i = 0; i < qs.size(); i += 2) {
            left_CX(state, qs[i], qs[i + 1]);
        }
    } else if (op == "CZ") {
        const auto qs = stim_qubit_targets(inst);
        if ((qs.size() & 1u) != 0u) {
            fail("CZ requires paired targets");
        }
        for (std::size_t i = 0; i < qs.size(); i += 2) {
            left_CZ(state, qs[i], qs[i + 1]);
        }
    } else if (op == "SWAP") {
        const auto qs = stim_qubit_targets(inst);
        if ((qs.size() & 1u) != 0u) {
            fail("SWAP requires paired targets");
        }
        for (std::size_t i = 0; i < qs.size(); i += 2) {
            left_SWAP(state, qs[i], qs[i + 1]);
        }
    } else if (op == "H") {
        for (int q : stim_qubit_targets(inst)) {
            left_H(state, q);
        }
    } else if (op == "S" || op == "SQRT_Z") {
        for (int q : stim_qubit_targets(inst)) {
            left_S(state, q);
        }
    } else if (op == "S_DAG" || op == "SQRT_Z_DAG") {
        for (int q : stim_qubit_targets(inst)) {
            left_SDG(state, q);
        }
    } else if (op == "X" || op == "Y" || op == "Z") {
        for (int q : stim_qubit_targets(inst)) {
            if (op == "X") {
                left_X(state, q);
            } else if (op == "Z") {
                left_Z(state, q);
            } else {
                left_X(state, q);
                left_Z(state, q);
            }
        }
    } else if (op == "T" || op == "T_DAG") {
        const double theta = (op == "T") ? kPi / 8.0 : -kPi / 8.0;
        for (int q : stim_qubit_targets(inst)) {
            apply_pauli_rotation(state, pauli_z(state.n, q), theta);
        }
    } else if (op == "M" || op == "MZ" || op == "MX" || op == "MY") {
        const double probability = stim_paren_probability(inst);
        for (auto [q, inverted] : stim_measurement_targets(inst)) {
            const SymbolicBool sign = measurement_sign_with_readout_error(*state.context, inverted, probability);
            auto [record, record_condition] = reserve_measurement_record(acc.records, *state.context);
            const PauliString pauli = op == "MX" ? pauli_x(state.n, q) : (op == "MY" ? pauli_y(state.n, q) : pauli_z(state.n, q));
            apply_pauli_measurement(state, pauli, sign, record, record_condition);
        }
    } else if (op == "MR" || op == "MRZ" || op == "MRX" || op == "MRY") {
        const double probability = stim_paren_probability(inst);
        for (auto [q, inverted] : stim_measurement_targets(inst)) {
            if (op == "MRX") {
                apply_measurement_reset(state, acc.records, pauli_x(state.n, q), pauli_z(state.n, q), inverted, probability);
            } else if (op == "MRY") {
                apply_measurement_reset(state, acc.records, pauli_y(state.n, q), pauli_x(state.n, q), inverted, probability);
            } else {
                apply_measurement_reset(state, acc.records, pauli_z(state.n, q), pauli_x(state.n, q), inverted, probability);
            }
        }
    } else if (op == "R" || op == "RZ" || op == "RX" || op == "RY") {
        for (int q : stim_qubit_targets(inst)) {
            if (op == "RX") {
                apply_reset(state, pauli_x(state.n, q), pauli_z(state.n, q));
            } else if (op == "RY") {
                apply_reset(state, pauli_y(state.n, q), pauli_x(state.n, q));
            } else {
                apply_reset(state, pauli_z(state.n, q), pauli_x(state.n, q));
            }
        }
    } else if (op == "MPP") {
        const double probability = stim_paren_probability(inst);
        for (const auto& target : stim_mpp_targets(inst)) {
            auto [pauli, inverted] = stim_mpp_pauli(state.n, target);
            const SymbolicBool sign = measurement_sign_with_readout_error(*state.context, inverted, probability);
            auto [record, record_condition] = reserve_measurement_record(acc.records, *state.context);
            apply_pauli_measurement(state, pauli, sign, record, record_condition);
        }
    } else if (op == "X_ERROR" || op == "Y_ERROR" || op == "Z_ERROR") {
        const double probability = stim_required_probability(inst);
        for (int q : stim_qubit_targets(inst)) {
            const PauliString pauli = op == "X_ERROR" ? pauli_x(state.n, q) : (op == "Y_ERROR" ? pauli_y(state.n, q) : pauli_z(state.n, q));
            apply_pauli(state, pauli, state.context->fresh_bernoulli_bool(probability));
        }
    } else if (op == "DEPOLARIZE1") {
        const double probability = stim_required_probability(inst);
        for (int q : stim_qubit_targets(inst)) {
            apply_depolarize1(state, q, probability);
        }
    } else if (op == "DEPOLARIZE2") {
        const double probability = stim_required_probability(inst);
        const auto qs = stim_qubit_targets(inst);
        if ((qs.size() & 1u) != 0u) {
            fail("DEPOLARIZE2 requires paired targets");
        }
        for (std::size_t i = 0; i < qs.size(); i += 2) {
            apply_depolarize2(state, qs[i], qs[i + 1], probability);
        }
    } else {
        fail("unsupported Stim operation: " + op);
    }
}

void apply_nodes(StimAccumulator& acc, const std::vector<StimNode>& nodes) {
    for (const auto& node : nodes) {
        if (const auto* block = std::get_if<std::shared_ptr<StimRepeatBlock>>(&node)) {
            for (int i = 0; i < (*block)->count; ++i) {
                apply_nodes(acc, (*block)->body);
            }
        } else {
            apply_instruction(acc, std::get<StimInstruction>(node));
        }
    }
}

} // namespace

StimParseResult parse_stim_lines(const std::vector<std::string>& lines) {
    std::size_t pos = 0;
    const auto nodes = parse_nodes(lines, pos, false);
    const int n = all_qubits(nodes);
    StimAccumulator acc{FrameFactoredState(n, 0), {}, {}, {}, {}};
    apply_nodes(acc, nodes);
    return StimParseResult{acc.state, acc.records, acc.detectors, acc.observables};
}

StimParseResult parse_stim_text(const std::string& text) {
    std::istringstream stream(text);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(stream, line)) {
        lines.push_back(line);
    }
    return parse_stim_lines(lines);
}

StimParseResult parse_stim_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        fail("failed to open Stim file: " + path);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        lines.push_back(line);
    }
    return parse_stim_lines(lines);
}

StimSampleSummary estimate_stim_logical_error_rate(const StimParseResult& parsed, int shots, std::uint64_t seed) {
    PendingFactoredState pending(parsed.state);
    const FactoredInstructionProgram program = plan_factored_updates(pending);
    const auto records = sample_measurements(program, shots, seed);
    StimSampleSummary summary;
    summary.shots = shots;
    for (const auto& shot_records : records) {
        bool discarded = false;
        for (const auto& detector : parsed.detectors) {
            bool parity = false;
            for (int record : detector.records) {
                parity ^= packed_bit(shot_records, record - 1);
            }
            discarded = discarded || parity;
        }
        if (discarded) {
            ++summary.discarded;
            continue;
        }
        ++summary.accepted;
        bool logical = false;
        for (const auto& observable : parsed.observables) {
            bool parity = false;
            for (int record : observable.records) {
                parity ^= packed_bit(shot_records, record - 1);
            }
            logical ^= parity;
        }
        if (logical) {
            ++summary.logical_errors;
        }
    }
    return summary;
}

double discard_rate(const StimSampleSummary& summary) {
    return summary.shots == 0 ? std::numeric_limits<double>::quiet_NaN()
                              : static_cast<double>(summary.discarded) / static_cast<double>(summary.shots);
}

double logical_error_rate(const StimSampleSummary& summary) {
    return summary.accepted == 0 ? std::numeric_limits<double>::quiet_NaN()
                                 : static_cast<double>(summary.logical_errors) / static_cast<double>(summary.accepted);
}

} // namespace symft
