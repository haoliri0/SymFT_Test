#pragma once

#include "symft/batch_sampler.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace symft {

using Complex = std::complex<double>;

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& message);
};

int checked_nqubits(int nqubits);
int nwords_for(int nqubits);
std::size_t bit_word_count(int nbits);
bool packed_bit(const std::vector<std::uint64_t>& words, int bit_index);
void set_packed_bit(std::vector<std::uint64_t>& words, int bit_index, bool value = true);
std::vector<std::uint64_t> packed_bits(std::initializer_list<bool> bits);

struct PauliString {
    int nqubits = 0;
    std::vector<std::uint64_t> x;
    std::vector<std::uint64_t> z;
    std::uint8_t phase = 0;

    PauliString() = default;
    explicit PauliString(int nqubits);

    bool xbit(int q) const;
    bool zbit(int q) const;
    void set_xbit(int q, bool value = true);
    void set_zbit(int q, bool value = true);
    int phase_exponent() const;
    void set_phase(int phase_exponent);
    void phase_shift(int delta);
    bool has_nonidentity_body() const;
    bool same_body(const PauliString& other) const;
    std::string str() const;
};

bool operator==(const PauliString& lhs, const PauliString& rhs);
bool operator!=(const PauliString& lhs, const PauliString& rhs);
PauliString operator*(const PauliString& lhs, const PauliString& rhs);
std::ostream& operator<<(std::ostream& out, const PauliString& pauli);

PauliString pauli_identity(int nqubits);
PauliString pauli_x(int nqubits, int q);
PauliString pauli_y(int nqubits, int q);
PauliString pauli_z(int nqubits, int q);
PauliString pauli_string(const std::string& ops);
PauliString neg(PauliString pauli);
bool pauli_anticommutes(const PauliString& a, const PauliString& b);
bool pauli_squares_to_identity(const PauliString& pauli);
int pauli_body_y_count(const PauliString& pauli);
bool measurement_phase_sign(const PauliString& pauli);

struct SymbolicBool {
    bool constant = false;
    std::vector<int> conditions;

    SymbolicBool() = default;
    explicit SymbolicBool(bool constant);
    SymbolicBool(bool constant, std::vector<int> conditions);

    int max_condition() const;
    std::string str() const;
};

bool operator==(const SymbolicBool& lhs, const SymbolicBool& rhs);
bool operator!=(const SymbolicBool& lhs, const SymbolicBool& rhs);
SymbolicBool symbolic_bool(int condition);
SymbolicBool operator!(const SymbolicBool& expr);
SymbolicBool xor_bool(const SymbolicBool& lhs, const SymbolicBool& rhs);
SymbolicBool xor_bool(const SymbolicBool& lhs, bool rhs);
SymbolicBool xor_bool(bool lhs, const SymbolicBool& rhs);
std::ostream& operator<<(std::ostream& out, const SymbolicBool& expr);

struct SymbolicBoolEvaluationPlan {
    bool constant = false;
    std::vector<int> conditions;
    std::vector<int> word_indices;
    std::vector<std::uint64_t> word_masks;

    SymbolicBoolEvaluationPlan() = default;
    explicit SymbolicBoolEvaluationPlan(const SymbolicBool& expr);
};

struct SymbolicCategoricalDistribution {
    int nbits = 0;
    std::vector<int> conditions;
    std::vector<std::vector<std::uint64_t>> assignments;
    std::vector<double> probabilities;
};

struct SymbolicContext {
    int next_condition = 1;
    std::map<int, double> bernoulli_probabilities;
    std::vector<SymbolicCategoricalDistribution> categorical_distributions;
    std::unordered_map<int, std::size_t> condition_to_categorical;

    SymbolicContext() = default;
    explicit SymbolicContext(int next_condition);

    void bump_next_condition(int condition);
    void bump_next_condition(const SymbolicBool& expr);
    int fresh_condition();
    int fresh_bernoulli_condition(double probability);
    SymbolicBool fresh_bernoulli_bool(double probability);
    std::vector<int> fresh_categorical_conditions(
        int nbits,
        const std::vector<std::vector<std::uint64_t>>& assignments,
        const std::vector<double>& probabilities);
    std::vector<SymbolicBool> fresh_categorical_bools(
        int nbits,
        const std::vector<std::vector<std::uint64_t>>& assignments,
        const std::vector<double>& probabilities);
};

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

struct ActiveState {
    int k = 0;
    std::vector<Complex> alpha;

    ActiveState();
    explicit ActiveState(int k);
    ActiveState(int k, std::vector<Complex> alpha);

    std::size_t dim() const;
    void reserve_for_k(int max_k);
};

struct ActivePauliAction {
    int nqubits = 0;
    std::uint64_t xmask = 0;
    std::uint64_t zmask = 0;
    Complex even_phase = Complex(1.0, 0.0);
    Complex odd_phase = Complex(-1.0, 0.0);
    bool xz_overlap_odd = false;

    ActivePauliAction() = default;
    explicit ActivePauliAction(const PauliString& pauli);
};

struct PrecomputedActivePauliRotationKernel {
    ActivePauliAction action;
    bool is_diagonal = true;
    bool uniform_imag_pairs = false;
    bool real_pair_flip = false;
    unsigned pair_bit = 0;
    double theta = 0.0;
    double cos_theta = 1.0;
    std::vector<Complex> diagonal_minus_coefficients;
    std::vector<Complex> diagonal_plus_coefficients;
    std::vector<std::size_t> pair_left_indices;
    std::vector<std::size_t> pair_right_indices;
    std::vector<double> real_pair_flip_basis_phase_signs;
    std::vector<Complex> pair_left_minus_coefficients;
    std::vector<Complex> pair_right_minus_coefficients;
    std::vector<Complex> pair_left_plus_coefficients;
    std::vector<Complex> pair_right_plus_coefficients;

    PrecomputedActivePauliRotationKernel() = default;
    PrecomputedActivePauliRotationKernel(const ActivePauliAction& action, double theta);
};

struct PrecomputedActivePauliMeasurementKernel {
    ActivePauliAction action;
    int pivot = 0;
    bool is_diagonal = true;
    std::vector<std::size_t> source0_false;
    std::vector<std::size_t> source1_false;
    std::vector<Complex> coeff0_false;
    std::vector<Complex> coeff1_false;
    std::vector<double> coeff1_false_real;
    std::vector<double> coeff1_false_imag;
    std::vector<std::size_t> source0_true;
    std::vector<std::size_t> source1_true;
    std::vector<Complex> coeff0_true;
    std::vector<Complex> coeff1_true;

    PrecomputedActivePauliMeasurementKernel() = default;
    explicit PrecomputedActivePauliMeasurementKernel(const PauliString& pauli);
    explicit PrecomputedActivePauliMeasurementKernel(const ActivePauliAction& action);
};

void apply_pauli(std::vector<Complex>& out, const PauliString& pauli, const std::vector<Complex>& alpha);
void apply_pauli(ActiveState& state, const PauliString& pauli);
void rotate_pauli(ActiveState& state, const PauliString& pauli, double theta);
void rotate_pauli(ActiveState& state, const ActivePauliAction& action, double theta);
void rotate_pauli(ActiveState& state, const PrecomputedActivePauliRotationKernel& kernel, bool sign);

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
    MeasureActiveLastZ,
    MeasurePrecomputedActivePauli,
    IntroduceDormantMeasurementBranch>;

bool operator==(const ApplyPrecomputedActivePauliRotation& lhs, const ApplyPrecomputedActivePauliRotation& rhs);
bool operator==(const PromoteDormantRotation& lhs, const PromoteDormantRotation& rhs);
bool operator==(const RecordMeasurement& lhs, const RecordMeasurement& rhs);
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
    ActiveState active;
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
    int k = 0;
    int max_k = 0;
    ActiveState active;
    DormantState dormant;
    std::shared_ptr<SymbolicContext> context;
    std::vector<PendingOperation> pending_operations;
    std::vector<FactoredInstruction> instructions;
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
    ActiveState initial_active;
    std::vector<FactoredInstruction> instructions;
    SymbolicContext context;
    int nsymbols = 0;
    int nrecords = 0;
    std::vector<SymbolicCategoricalDistribution> sampled_categorical_distributions;
    std::vector<RareCategoricalSampleGroup> sampled_rare_categorical_groups;
    std::vector<int> sampled_bernoulli_conditions;
    std::vector<double> sampled_bernoulli_probabilities;
    std::vector<BernoulliSampleGroup> sampled_low_probability_bernoulli_groups;

    FactoredInstructionProgram() = default;
    FactoredInstructionProgram(
        int n,
        int initial_k,
        ActiveState initial_active,
        std::vector<FactoredInstruction> instructions,
        int max_k,
        SymbolicContext context = SymbolicContext());
};

FactoredInstructionProgram factored_instruction_program(const PendingFactoredState& state);
FactoredInstructionProgram plan_factored_updates(PendingFactoredState& state);

struct PresampledExogenous {
    int nshots = 0;
    int nsymbols = 0;
    std::size_t nwords = 0;
    std::uint64_t next_rng_state = 0;
    std::vector<std::uint64_t> exogenous_assigned_words;
    std::vector<std::uint64_t> value_words;
};

struct FactoredExecutorState {
    int n = 0;
    int k = 0;
    int ndormant = 0;
    int nsymbols = 0;
    int nrecords = 0;
    ActiveState active;
    std::vector<Complex> active_scratch;
    std::vector<std::uint64_t> value_words;
    std::vector<std::uint64_t> assigned_words;
    std::vector<std::uint64_t> measurement_words;
    std::uint64_t rng_state = 1;

    explicit FactoredExecutorState(const FactoredInstructionProgram& program, std::uint64_t seed = 1);
};

void reset_executor(FactoredExecutorState& runtime, const FactoredInstructionProgram& program);
PresampledExogenous presample_exogenous(const FactoredInstructionProgram& program, int shots, std::uint64_t seed = 1);
void execute_in_place(FactoredExecutorState& runtime, const FactoredInstructionProgram& program);
void execute_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExogenous& samples,
    int shot_index);
std::vector<std::uint64_t> execute(FactoredExecutorState& runtime, const FactoredInstructionProgram& program);
std::vector<std::uint64_t> sample_measurements(const FactoredInstructionProgram& program, std::uint64_t seed = 1);
std::vector<std::vector<std::uint64_t>> sample_measurements(const FactoredInstructionProgram& program, int shots, std::uint64_t seed = 1);

struct StimDetector {
    std::vector<int> records;
    std::vector<double> coords;
    int line = 0;
};

struct StimObservableInclude {
    int index = 0;
    std::vector<int> records;
    int line = 0;
};

struct StimParseResult {
    FrameFactoredState state;
    std::vector<SymbolicBool> measurement_records;
    std::vector<StimDetector> detectors;
    std::vector<StimObservableInclude> observables;
};

StimParseResult parse_stim_lines(const std::vector<std::string>& lines);
StimParseResult parse_stim_text(const std::string& text);
StimParseResult parse_stim_file(const std::string& path);

struct StimSampleSummary {
    int shots = 0;
    int discarded = 0;
    int accepted = 0;
    int logical_errors = 0;
};

StimSampleSummary estimate_stim_logical_error_rate(const StimParseResult& parsed, int shots, std::uint64_t seed = 1);
double discard_rate(const StimSampleSummary& summary);
double logical_error_rate(const StimSampleSummary& summary);

std::string active_simd_backend();

} // namespace symft
