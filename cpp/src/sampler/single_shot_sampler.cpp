#include "active_kernels.hpp"

#include "sampler/exogenous.hpp"
#include "sampler/random.hpp"
#include "sampler/single_shot.hpp"
#include "simd/simd.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace symft {
using namespace detail;

namespace {

constexpr int kDefaultSingleShotSampleChunkShots = 2048;
constexpr std::uint64_t kSingleShotBranchSeedXor = 0x5eed1234ULL;

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

bool eval_symbolic_bool_packed_unchecked(const SymbolicBoolEvaluationPlan& plan, const FactoredExecutorState& runtime) {
    const auto* word_indices = plan.word_indices.data();
    const auto* word_masks = plan.word_masks.data();
    const auto* value_words = runtime.value_words.data();
    std::uint64_t parity_bits = 0;
    for (std::size_t i = 0; i < plan.word_indices.size(); ++i) {
        const std::size_t word = static_cast<std::size_t>(word_indices[i]);
        parity_bits ^= value_words[word] & word_masks[i];
    }
    return plan.constant != is_odd_popcount(parity_bits);
}

bool eval_symbolic_bool_scalar(const SymbolicBoolEvaluationPlan& plan, const FactoredExecutorState& runtime) {
    bool out = plan.constant;
    for (int condition : plan.conditions) {
        check_symbol_slot(runtime, condition);
        const std::size_t word = symbol_word_index(condition);
        const std::uint64_t mask = symbol_bit_mask(condition);
        if ((runtime.assigned_words[word] & mask) == 0) {
            fail("symbolic condition expression has no concrete value");
        }
        out = out != ((runtime.value_words[word] & mask) != 0);
    }
    return out;
}

[[maybe_unused]] bool eval_symbolic_bool(const SymbolicBoolEvaluationPlan& plan, const FactoredExecutorState& runtime) {
    if (plan.word_indices.empty()) {
        return eval_symbolic_bool_scalar(plan, runtime);
    }
    return eval_symbolic_bool_packed(plan, runtime);
}

bool eval_symbolic_bool_unchecked(const SymbolicBoolEvaluationPlan& plan, const FactoredExecutorState& runtime) {
    // Hot planned execution relies on the planner's assignment-before-use invariant.
    // Keep eval_symbolic_bool above as the checked counterpart for validation/debugging.
    if (plan.word_indices.empty()) {
        return plan.constant;
    }
    return eval_symbolic_bool_packed_unchecked(plan, runtime);
}

bool record_parity_from_measurements(
    const std::vector<std::uint64_t>& measurement_words,
    int nrecords,
    const std::vector<int>& records) {
    bool parity = false;
    for (int record : records) {
        if (record <= 0 || record > nrecords) {
            fail("detector references an out-of-range measurement record");
        }
        parity = parity != packed_bit(measurement_words, record - 1);
    }
    return parity;
}

bool detector_outcome_from_runtime(
    const FactoredExecutorState& runtime,
    const RecordDetector& instruction) {
    if (!instruction.records.empty()) {
        return record_parity_from_measurements(
            runtime.measurement_words,
            runtime.nrecords,
            instruction.records);
    }
    return eval_symbolic_bool_unchecked(instruction.outcome_plan, runtime);
}

int normalize_single_shot_sample_chunk_shots(int sample_chunk_shots) {
    if (sample_chunk_shots < 0) {
        fail("single-shot sample chunk count must be nonnegative");
    }
    return sample_chunk_shots > 0 ? sample_chunk_shots : kDefaultSingleShotSampleChunkShots;
}

struct SingleShotExpressionEvaluator {
    const PresampledExpressionPlan& expression_plan;
    const PresampledExpressionBlock& expression_block;
    int shot_index = 0;

    bool eval(std::size_t instruction_index, const FactoredExecutorState& runtime) const {
        if (instruction_index >= expression_plan.instruction_expressions.size()) {
            fail("single-shot presampled expression plan does not match program");
        }
        const auto& expression = expression_plan.instruction_expressions[instruction_index];
        if (shot_index < 0 || shot_index >= expression_block.nshots) {
            fail("single-shot presampled expression shot index is out of range");
        }
        if (expression.block_expression_index < 0 ||
            expression.block_expression_index >= static_cast<int>(expression_plan.block_expressions.size())) {
            fail("single-shot presampled expression references an out-of-range block expression");
        }
        bool out = presampled_expression_block_bit(
            expression_block,
            expression.block_expression_index,
            shot_index);
        if (!expression.residual_plan.conditions.empty()) {
            out = out != eval_symbolic_bool_unchecked(expression.residual_plan, runtime);
        }
        return out;
    }
};

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

void write_detector_record(FactoredExecutorState& runtime, int detector, bool outcome) {
    if (detector <= 0) {
        fail("detector id must be positive");
    }
    if (detector > runtime.ndetectors) {
        runtime.ndetectors = detector;
    }
    const std::size_t nwords = symbol_word_count(runtime.ndetectors);
    if (runtime.detector_words.size() < nwords) {
        runtime.detector_words.resize(nwords, 0);
    }
    const std::size_t word = symbol_word_index(detector);
    const std::uint64_t mask = symbol_bit_mask(detector);
    if (outcome) {
        runtime.detector_words[word] |= mask;
    } else {
        runtime.detector_words[word] &= ~mask;
    }
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

std::size_t runtime_active_dim(const FactoredExecutorState& runtime) {
    return active_length(runtime.k);
}

void ensure_runtime_active_capacity(FactoredExecutorState& runtime, int max_k) {
    const std::size_t max_dim = active_length(max_k);
    if (runtime.active_re.size() < max_dim) {
        runtime.active_re.resize(max_dim, 0.0);
    }
    if (runtime.active_im.size() < max_dim) {
        runtime.active_im.resize(max_dim, 0.0);
    }
    if (runtime.active_scratch_re.size() < max_dim) {
        runtime.active_scratch_re.resize(max_dim, 0.0);
    }
    if (runtime.active_scratch_im.size() < max_dim) {
        runtime.active_scratch_im.resize(max_dim, 0.0);
    }
}

void promote_first_dormant_rotation(FactoredExecutorState& runtime, double kernel_angle) {
    if (runtime.ndormant <= 0) {
        fail("cannot promote a dormant qubit when none remain");
    }
    const std::size_t dim = runtime_active_dim(runtime);
    const std::size_t promoted_dim = 2 * dim;
    if (runtime.active_re.size() < promoted_dim) {
        runtime.active_re.resize(promoted_dim, 0.0);
    }
    if (runtime.active_im.size() < promoted_dim) {
        runtime.active_im.resize(promoted_dim, 0.0);
    }
    if (runtime.active_scratch_re.size() < promoted_dim) {
        runtime.active_scratch_re.resize(promoted_dim, 0.0);
    }
    if (runtime.active_scratch_im.size() < promoted_dim) {
        runtime.active_scratch_im.resize(promoted_dim, 0.0);
    }
    const double c = std::cos(kernel_angle);
    const double s = std::sin(kernel_angle);
    SYMFT_SINGLE_SIMD_LOOP
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const double r = runtime.active_re[basis];
        const double i = runtime.active_im[basis];
        runtime.active_re[basis] = c * r;
        runtime.active_im[basis] = c * i;
        runtime.active_re[dim + basis] = s * i;
        runtime.active_im[dim + basis] = -s * r;
    }
    ++runtime.k;
    --runtime.ndormant;
}

void rotate_pauli(FactoredExecutorState& runtime, const PrecomputedActivePauliRotationKernel& kernel, bool sign) {
    if (kernel.action.nqubits != runtime.k) {
        fail("rotation kernel dimension does not match active state");
    }
    const double c = kernel.cos_kernel_angle;
    double* active_re = runtime.active_re.data();
    double* active_im = runtime.active_im.data();
    const std::size_t dim = runtime_active_dim(runtime);
    if (kernel.is_diagonal) {
        SYMFT_SINGLE_SIMD_LOOP
        for (std::size_t basis = 0; basis < dim; ++basis) {
            const Complex coefficient = compact_rotation_coefficient(kernel, basis, sign);
            const double fr = c + coefficient.real();
            const double fi = coefficient.imag();
            const double r = active_re[basis];
            const double i = active_im[basis];
            active_re[basis] = fr * r - fi * i;
            active_im[basis] = fr * i + fi * r;
        }
        return;
    }
    const std::size_t npairs = kernel.pair_count;
    if (kernel.uniform_imag_pairs) {
        const Complex coefficient = compact_rotation_coefficient(kernel, 0, sign);
        if (npairs < kSimdPairRotationThreshold) {
            rotate_uniform_imag_pairs_soa_inline(active_re, active_im, dim, kernel.action.xmask, kernel.pair_bit, c, coefficient.imag());
        } else {
            simd::dispatch_table().rotate_uniform_imag_pairs_soa(
                active_re,
                active_im,
                dim,
                kernel.action.xmask,
                kernel.pair_bit,
                c,
                coefficient.imag());
        }
        return;
    }
    SYMFT_SINGLE_SIMD_LOOP
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        const std::size_t left = insert_zero_bit(idx, static_cast<int>(kernel.pair_bit));
        const std::size_t right = left ^ static_cast<std::size_t>(kernel.action.xmask);
        const bool left_odd = active_action_phase_odd(kernel.action, left);
        const double left_direction = sign != left_odd ? -1.0 : 1.0;
        const double right_direction = kernel.action.xz_overlap_odd ? -left_direction : left_direction;
        const double left_re = left_direction * kernel.minus_even_coefficient.real();
        const double left_im = left_direction * kernel.minus_even_coefficient.imag();
        const double right_re = right_direction * kernel.minus_even_coefficient.real();
        const double right_im = right_direction * kernel.minus_even_coefficient.imag();
        const double r0 = active_re[left];
        const double i0 = active_im[left];
        const double r1 = active_re[right];
        const double i1 = active_im[right];
        active_re[left] = c * r0 + right_re * r1 - right_im * i1;
        active_im[left] = c * i0 + right_re * i1 + right_im * r1;
        active_re[right] = c * r1 + left_re * r0 - left_im * i0;
        active_im[right] = c * i1 + left_re * i0 + left_im * r0;
    }
}

double active_diagonal_measurement_branch_probability(
    const FactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch) {
    double probability = 0.0;
    SYMFT_SINGLE_SIMD_LOOP
    for (std::size_t idx = 0; idx < kernel.out_dim; ++idx) {
        const std::size_t source = compact_diagonal_measurement_source(kernel, idx, branch);
        const double r = runtime.active_re[source];
        const double i = runtime.active_im[source];
        probability += r * r + i * i;
    }
    return std::clamp(probability, 0.0, 1.0);
}

double active_measurement_branch_probability(
    const FactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch) {
    const double probability = simd::dispatch_table().measure_nondiagonal_probability_soa(
        runtime.active_re.data(),
        runtime.active_im.data(),
        kernel.out_dim << 1,
        kernel.action.xmask,
        kernel.action.zmask,
        static_cast<unsigned>(kernel.pivot),
        kernel.nondiagonal_coefficient1_even,
        branch);
    return std::clamp(probability, 0.0, 1.0);
}

void project_diagonal_active_pauli_measurement(
    FactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch,
    double probability) {
    if (probability <= 0.0) {
        fail("sampled an impossible active measurement branch");
    }
    const double invnorm = 1.0 / std::sqrt(probability);
    SYMFT_SINGLE_SIMD_LOOP
    for (std::size_t idx = 0; idx < kernel.out_dim; ++idx) {
        const std::size_t source = compact_diagonal_measurement_source(kernel, idx, branch);
        runtime.active_re[idx] = runtime.active_re[source] * invnorm;
        runtime.active_im[idx] = runtime.active_im[source] * invnorm;
    }
    --runtime.k;
}

void project_active_pauli_measurement(
    FactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch,
    double probability) {
    if (probability <= 0.0) {
        fail("sampled an impossible active measurement branch");
    }
    const std::size_t out_dim = kernel.out_dim;
    const double invnorm = 1.0 / std::sqrt(probability);
    if (runtime.active_scratch_re.size() < out_dim) {
        runtime.active_scratch_re.resize(out_dim, 0.0);
    }
    if (runtime.active_scratch_im.size() < out_dim) {
        runtime.active_scratch_im.resize(out_dim, 0.0);
    }
    simd::dispatch_table().project_nondiagonal_soa(
        runtime.active_re.data(),
        runtime.active_im.data(),
        runtime.active_scratch_re.data(),
        runtime.active_scratch_im.data(),
        out_dim << 1,
        kernel.action.xmask,
        kernel.action.zmask,
        static_cast<unsigned>(kernel.pivot),
        kernel.nondiagonal_coefficient1_even,
        branch,
        invnorm);
    std::copy_n(runtime.active_scratch_re.data(), out_dim, runtime.active_re.data());
    std::copy_n(runtime.active_scratch_im.data(), out_dim, runtime.active_im.data());
    --runtime.k;
}

void execute_instruction(FactoredExecutorState& runtime, const ApplyPrecomputedActivePauliRotation& instruction) {
    const bool sign = eval_symbolic_bool_unchecked(instruction.sign_plan, runtime);
    rotate_pauli(runtime, instruction.rotation_kernel, sign);
}

void execute_instruction(FactoredExecutorState& runtime, const PromoteDormantRotation& instruction) {
    const bool sign = eval_symbolic_bool_unchecked(instruction.sign_plan, runtime);
    promote_first_dormant_rotation(runtime, sign ? -instruction.kernel_angle : instruction.kernel_angle);
}

void execute_instruction(FactoredExecutorState& runtime, const RecordMeasurement& instruction) {
    const bool outcome = eval_symbolic_bool_unchecked(instruction.outcome_plan, runtime);
    write_measurement_record(runtime, instruction.record, outcome, instruction.record_condition);
}

void execute_instruction(FactoredExecutorState& runtime, const RecordDetector& instruction) {
    const bool outcome = detector_outcome_from_runtime(runtime, instruction);
    write_detector_record(runtime, instruction.detector, outcome);
}

void execute_instruction(FactoredExecutorState& runtime, const MeasurePrecomputedActivePauli& instruction) {
    if (runtime.k <= 0) {
        fail("cannot measure an active Pauli when k == 0");
    }
    double prob_true = instruction.kernel.is_diagonal
                           ? active_diagonal_measurement_branch_probability(runtime, instruction.kernel, true)
                           : active_measurement_branch_probability(runtime, instruction.kernel, true);
    prob_true = std::clamp(prob_true, 0.0, 1.0);
    const double prob_false = 1.0 - prob_true;
    const bool branch = sample_bernoulli(runtime.rng_state, prob_true);
    const double probability = branch ? prob_true : prob_false;
    assign_symbol(runtime, instruction.branch, branch);
    if (instruction.kernel.is_diagonal) {
        project_diagonal_active_pauli_measurement(runtime, instruction.kernel, branch, probability);
    } else {
        project_active_pauli_measurement(runtime, instruction.kernel, branch, probability);
    }
    ++runtime.ndormant;
    const bool outcome = eval_symbolic_bool_unchecked(instruction.outcome_plan, runtime);
    write_measurement_record(runtime, instruction.record, outcome, instruction.record_condition);
}

void execute_instruction(FactoredExecutorState& runtime, const IntroduceDormantMeasurementBranch& instruction) {
    const bool branch = sample_bernoulli(runtime.rng_state, 0.5);
    assign_symbol(runtime, instruction.branch, branch);
    const bool outcome = eval_symbolic_bool_unchecked(instruction.outcome_plan, runtime);
    write_measurement_record(runtime, instruction.record, outcome, instruction.record_condition);
}

void execute_instruction_presampled(
    FactoredExecutorState& runtime,
    const ApplyPrecomputedActivePauliRotation& instruction,
    const SingleShotExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    const bool sign = evaluator.eval(instruction_index, runtime);
    rotate_pauli(runtime, instruction.rotation_kernel, sign);
}

void execute_instruction_presampled(
    FactoredExecutorState& runtime,
    const PromoteDormantRotation& instruction,
    const SingleShotExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    const bool sign = evaluator.eval(instruction_index, runtime);
    promote_first_dormant_rotation(runtime, sign ? -instruction.kernel_angle : instruction.kernel_angle);
}

void execute_instruction_presampled(
    FactoredExecutorState& runtime,
    const RecordMeasurement& instruction,
    const SingleShotExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    const bool outcome = evaluator.eval(instruction_index, runtime);
    write_measurement_record(runtime, instruction.record, outcome, instruction.record_condition);
}

void execute_instruction_presampled(
    FactoredExecutorState& runtime,
    const RecordDetector& instruction,
    const SingleShotExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    const bool outcome = !instruction.records.empty() || instruction.outcome.conditions.empty()
                             ? detector_outcome_from_runtime(runtime, instruction)
                             : evaluator.eval(instruction_index, runtime);
    write_detector_record(runtime, instruction.detector, outcome);
}

void execute_instruction_presampled(
    FactoredExecutorState& runtime,
    const MeasurePrecomputedActivePauli& instruction,
    const SingleShotExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    if (runtime.k <= 0) {
        fail("cannot measure an active Pauli when k == 0");
    }
    double prob_true = instruction.kernel.is_diagonal
                           ? active_diagonal_measurement_branch_probability(runtime, instruction.kernel, true)
                           : active_measurement_branch_probability(runtime, instruction.kernel, true);
    prob_true = std::clamp(prob_true, 0.0, 1.0);
    const double prob_false = 1.0 - prob_true;
    const bool branch = sample_bernoulli(runtime.rng_state, prob_true);
    const double probability = branch ? prob_true : prob_false;
    assign_symbol(runtime, instruction.branch, branch);
    if (instruction.kernel.is_diagonal) {
        project_diagonal_active_pauli_measurement(runtime, instruction.kernel, branch, probability);
    } else {
        project_active_pauli_measurement(runtime, instruction.kernel, branch, probability);
    }
    ++runtime.ndormant;
    const bool outcome = evaluator.eval(instruction_index, runtime);
    write_measurement_record(runtime, instruction.record, outcome, instruction.record_condition);
}

void execute_instruction_presampled(
    FactoredExecutorState& runtime,
    const IntroduceDormantMeasurementBranch& instruction,
    const SingleShotExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    const bool branch = sample_bernoulli(runtime.rng_state, 0.5);
    assign_symbol(runtime, instruction.branch, branch);
    const bool outcome = evaluator.eval(instruction_index, runtime);
    write_measurement_record(runtime, instruction.record, outcome, instruction.record_condition);
}

void execute_instruction_presampled(
    FactoredExecutorState& runtime,
    const FactoredInstruction& instruction,
    const SingleShotExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    std::visit(
        [&](const auto& inst) {
            execute_instruction_presampled(
                runtime,
                inst,
                evaluator,
                instruction_index);
        },
        instruction);
}

template <typename Instruction>
bool execute_instruction_postselected(FactoredExecutorState& runtime, const Instruction& instruction) {
    execute_instruction(runtime, instruction);
    return true;
}

bool execute_instruction_postselected(FactoredExecutorState& runtime, const RecordDetector& instruction) {
    return !detector_outcome_from_runtime(runtime, instruction);
}

template <typename Instruction>
bool execute_instruction_postselected(
    FactoredExecutorState& runtime,
    const Instruction& instruction,
    const SingleShotExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    execute_instruction_presampled(runtime, instruction, evaluator, instruction_index);
    return true;
}

bool execute_instruction_postselected(
    FactoredExecutorState& runtime,
    const RecordDetector& instruction,
    const SingleShotExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    const bool outcome = !instruction.records.empty() || instruction.outcome.conditions.empty()
                             ? detector_outcome_from_runtime(runtime, instruction)
                             : evaluator.eval(instruction_index, runtime);
    return !outcome;
}

} // namespace

FactoredExecutorState::FactoredExecutorState(const FactoredInstructionProgram& program, std::uint64_t seed)
    : n(program.n),
      k(program.initial_k),
      ndormant(program.n - program.initial_k),
      nsymbols(program.nsymbols),
      nrecords(program.nrecords),
      ndetectors(program.ndetectors),
      active_re(active_length(program.max_k), 0.0),
      active_im(active_length(program.max_k), 0.0),
      active_scratch_re(active_length(program.max_k), 0.0),
      active_scratch_im(active_length(program.max_k), 0.0),
      value_words(symbol_word_count(program.nsymbols), 0),
      assigned_words(symbol_word_count(program.nsymbols), 0),
      measurement_words(symbol_word_count(program.nrecords), 0),
      detector_words(symbol_word_count(program.ndetectors), 0),
      rng_state(seed) {
    const std::size_t dim = active_length(program.initial_k);
    std::fill_n(active_re.data(), dim, 0.0);
    std::fill_n(active_im.data(), dim, 0.0);
    active_re[0] = 1.0;
}

void reset_executor(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    bool clear_detector_records) {
    runtime.n = program.n;
    runtime.k = program.initial_k;
    runtime.ndormant = program.n - program.initial_k;
    runtime.nsymbols = program.nsymbols;
    runtime.nrecords = program.nrecords;
    runtime.ndetectors = program.ndetectors;
    ensure_runtime_active_capacity(runtime, program.max_k);
    const std::size_t dim = active_length(program.initial_k);
    std::fill_n(runtime.active_re.data(), dim, 0.0);
    std::fill_n(runtime.active_im.data(), dim, 0.0);
    runtime.active_re[0] = 1.0;
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
    const std::size_t detector_words = symbol_word_count(program.ndetectors);
    if (runtime.detector_words.size() != detector_words) {
        runtime.detector_words.resize(detector_words);
    }
    if (clear_detector_records) {
        std::fill(runtime.detector_words.begin(), runtime.detector_words.end(), 0);
    }
}

void execute_in_place(FactoredExecutorState& runtime, const FactoredInstructionProgram& program) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
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
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
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

void execute_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExpressionPlan& expression_plan,
    const PresampledExpressionBlock& expression_block,
    int shot_index) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
        fail("executor state does not match program");
    }
    if (expression_plan.instruction_expressions.size() != program.instructions.size()) {
        fail("single-shot presampled expression plan does not match program");
    }
    if (shot_index < 0 || shot_index >= expression_block.nshots) {
        fail("single-shot presampled expression shot index is out of range");
    }
    SingleShotExpressionEvaluator evaluator{expression_plan, expression_block, shot_index};
    for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
        execute_instruction_presampled(
            runtime,
            program.instructions[idx],
            evaluator,
            idx);
    }
}

void assign_presampled_exogenous_in_place(
    FactoredExecutorState& runtime,
    const PresampledExogenous& samples,
    int shot_index) {
    assign_presampled_exogenous(runtime, samples, shot_index);
}

void execute_instruction_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstruction& instruction) {
    std::visit([&](const auto& inst) { execute_instruction(runtime, inst); }, instruction);
}

bool execute_postselected_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExogenous& samples,
    int shot_index) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
        fail("executor state does not match program");
    }
    assign_presampled_exogenous(runtime, samples, shot_index);
    for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
        const bool survived = std::visit(
            [&](const auto& inst) {
                return execute_instruction_postselected(runtime, inst);
            },
            program.instructions[idx]);
        if (!survived) {
            return false;
        }
    }
    return true;
}

bool execute_postselected_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExpressionPlan& expression_plan,
    const PresampledExpressionBlock& expression_block,
    int shot_index) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
        fail("executor state does not match program");
    }
    if (expression_plan.instruction_expressions.size() != program.instructions.size()) {
        fail("single-shot presampled expression plan does not match program");
    }
    if (shot_index < 0 || shot_index >= expression_block.nshots) {
        fail("single-shot presampled expression shot index is out of range");
    }
    SingleShotExpressionEvaluator evaluator{expression_plan, expression_block, shot_index};
    for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
        const bool survived = std::visit(
            [&](const auto& inst) {
                return execute_instruction_postselected(runtime, inst, evaluator, idx);
            },
            program.instructions[idx]);
        if (!survived) {
            return false;
        }
    }
    return true;
}

std::vector<std::uint64_t> execute(FactoredExecutorState& runtime, const FactoredInstructionProgram& program) {
    execute_in_place(runtime, program);
    return runtime.measurement_words;
}

std::vector<std::uint64_t> sample_measurements(const FactoredInstructionProgram& program, std::uint64_t seed) {
    FactoredExecutorState runtime(program, seed);
    return execute(runtime, program);
}

int default_single_shot_sample_chunk_shots() {
    return kDefaultSingleShotSampleChunkShots;
}

std::vector<std::vector<std::uint64_t>> sample_measurements(
    const FactoredInstructionProgram& program,
    int shots,
    std::uint64_t seed,
    int sample_chunk_shots) {
    if (shots < 0) {
        fail("shot count must be nonnegative");
    }

    const int chunk_shots = normalize_single_shot_sample_chunk_shots(sample_chunk_shots);
    PackedPresampledExogenous packed_samples;
    prepare_presampled_exogenous_packed(packed_samples, program);
    PresampledExpressionPlan expression_plan;
    prepare_presampled_expression_plan(expression_plan, program, packed_samples);
    PresampledExpressionBlock expression_block;
    FactoredExecutorState runtime(program, seed ^ kSingleShotBranchSeedXor);

    std::vector<std::vector<std::uint64_t>> out;
    out.reserve(static_cast<std::size_t>(shots));

    std::uint64_t exogenous_rng_state = seed;
    for (int offset = 0; offset < shots; offset += chunk_shots) {
        const int chunk = std::min(chunk_shots, shots - offset);
        resample_prepared_exogenous_packed_in_place(
            packed_samples,
            program,
            chunk,
            exogenous_rng_state);
        exogenous_rng_state = packed_samples.next_rng_state;
        evaluate_presampled_expression_block(
            expression_block,
            expression_plan,
            packed_samples);
        for (int shot = 0; shot < chunk; ++shot) {
            reset_executor(runtime, program);
            execute_in_place(runtime, program, expression_plan, expression_block, shot);
            out.push_back(runtime.measurement_words);
        }
    }
    return out;
}

std::vector<std::vector<std::uint64_t>> sample_measurements(
    const FactoredInstructionProgram& program,
    int shots,
    std::uint64_t seed) {
    return sample_measurements(program, shots, seed, 0);
}

} // namespace symft
