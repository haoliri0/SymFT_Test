#include "active_kernels.hpp"

#include "symft/simd.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace symft {
using namespace detail;

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

void promote_first_dormant_rotation(FactoredExecutorState& runtime, double theta) {
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
    const double c = std::cos(theta);
    const double s = std::sin(theta);
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

void apply_active_basis_change(FactoredExecutorState& runtime, char kind, int q) {
    if (q < 0 || q >= runtime.k) {
        fail("active basis-change qubit is out of range");
    }
    const std::size_t dim = runtime_active_dim(runtime);
    const std::size_t mask = std::size_t{1} << q;
    if (kind == 'H') {
        const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
        const std::size_t step = mask << 1;
        for (std::size_t block = 0; block < dim; block += step) {
            SYMFT_SINGLE_SIMD_LOOP
            for (std::size_t offset = 0; offset < mask; ++offset) {
                const std::size_t base = block + offset;
                const std::size_t paired = base | mask;
                const double r0 = runtime.active_re[base];
                const double i0 = runtime.active_im[base];
                const double r1 = runtime.active_re[paired];
                const double i1 = runtime.active_im[paired];
                runtime.active_re[base] = (r0 + r1) * inv_sqrt2;
                runtime.active_im[base] = (i0 + i1) * inv_sqrt2;
                runtime.active_re[paired] = (r0 - r1) * inv_sqrt2;
                runtime.active_im[paired] = (i0 - i1) * inv_sqrt2;
            }
        }
    } else if (kind == 'S') {
        const std::size_t step = mask << 1;
        for (std::size_t block = mask; block < dim; block += step) {
            SYMFT_SINGLE_SIMD_LOOP
            for (std::size_t offset = 0; offset < mask; ++offset) {
                const std::size_t basis = block + offset;
                const double r = runtime.active_re[basis];
                const double i = runtime.active_im[basis];
                runtime.active_re[basis] = -i;
                runtime.active_im[basis] = r;
            }
        }
    } else {
        fail("unsupported active basis change");
    }
}

void rotate_pauli(FactoredExecutorState& runtime, const PrecomputedActivePauliRotationKernel& kernel, bool sign) {
    if (kernel.action.nqubits != runtime.k) {
        fail("rotation kernel dimension does not match active state");
    }
    const double c = kernel.cos_theta;
    auto& simd_table = simd::dispatch_table();
    double* active_re = runtime.active_re.data();
    double* active_im = runtime.active_im.data();
    if (kernel.is_diagonal) {
        const auto& coefficients = sign ? kernel.diagonal_plus_coefficients : kernel.diagonal_minus_coefficients;
        if (coefficients.size() < kSimdPairRotationThreshold) {
            mul_assign_soa_inline(active_re, active_im, coefficients.data(), c, coefficients.size());
        } else {
            simd_table.mul_assign_soa(active_re, active_im, coefficients.data(), c, coefficients.size());
        }
        return;
    }
    const std::size_t dim = runtime_active_dim(runtime);
    const std::size_t npairs = kernel.pair_left_indices.size();
    if (kernel.uniform_imag_pairs) {
        const Complex coefficient = sign ? kernel.pair_left_plus_coefficients.front() : kernel.pair_left_minus_coefficients.front();
        if (npairs < kSimdPairRotationThreshold) {
            rotate_uniform_imag_pairs_soa_inline(active_re, active_im, dim, kernel.action.xmask, kernel.pair_bit, c, coefficient.imag());
        } else {
            simd_table.rotate_uniform_imag_pairs_soa(
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
    if (kernel.real_pair_flip) {
        const Complex coefficient = sign ? kernel.pair_left_plus_coefficients.front() : kernel.pair_left_minus_coefficients.front();
        if (npairs < kSimdPairRotationThreshold) {
            rotate_real_pair_flip_soa_inline(
                active_re,
                active_im,
                dim,
                kernel.action.xmask,
                kernel.pair_bit,
                kernel.real_pair_flip_basis_phase_signs.data(),
                c,
                coefficient.real());
        } else {
            simd_table.rotate_real_pair_flip_soa(
                active_re,
                active_im,
                dim,
                kernel.action.xmask,
                kernel.pair_bit,
                kernel.real_pair_flip_basis_phase_signs.data(),
                c,
                coefficient.real());
        }
        return;
    }
    const auto& left_coeff = sign ? kernel.pair_left_plus_coefficients : kernel.pair_left_minus_coefficients;
    const auto& right_coeff = sign ? kernel.pair_right_plus_coefficients : kernel.pair_right_minus_coefficients;
    if (npairs < kSimdPairRotationThreshold) {
        rotate_general_pairs_soa_inline(
            active_re, active_im, dim, kernel.action.xmask, kernel.pair_bit, left_coeff.data(), right_coeff.data(), c);
    } else {
        simd_table.rotate_general_pairs_soa(
            active_re,
            active_im,
            dim,
            kernel.action.xmask,
            kernel.pair_bit,
            left_coeff.data(),
            right_coeff.data(),
            c);
    }
}

double active_last_z_probability_one(const FactoredExecutorState& runtime) {
    if (runtime.k <= 0) {
        fail("cannot measure last active qubit when k == 0");
    }
    const std::size_t mask = std::size_t{1} << (runtime.k - 1);
    const std::size_t dim = runtime_active_dim(runtime);
    double probability = 0.0;
    SYMFT_SINGLE_SIMD_LOOP
    for (std::size_t basis = mask; basis < dim; ++basis) {
        const double r = runtime.active_re[basis];
        const double i = runtime.active_im[basis];
        probability += r * r + i * i;
    }
    return std::clamp(probability, 0.0, 1.0);
}

void project_active_last_z(FactoredExecutorState& runtime, bool branch, double prob1) {
    const int new_k = runtime.k - 1;
    const std::size_t dim = active_length(new_k);
    const double probability = branch ? prob1 : 1.0 - prob1;
    if (probability <= 0.0) {
        fail("sampled an impossible active measurement branch");
    }
    const double invnorm = 1.0 / std::sqrt(probability);
    const std::size_t branch_offset = branch ? dim : 0;
    SYMFT_SINGLE_SIMD_LOOP
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const std::size_t source = branch_offset + basis;
        runtime.active_re[basis] = runtime.active_re[source] * invnorm;
        runtime.active_im[basis] = runtime.active_im[source] * invnorm;
    }
    runtime.k = new_k;
}

double active_diagonal_measurement_branch_probability(
    const FactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch) {
    const auto& sources = branch ? kernel.source0_true : kernel.source0_false;
    const double probability = sources.size() < kSimdPairRotationThreshold
                                   ? norm_sum_soa_inline(
                                         runtime.active_re.data(), runtime.active_im.data(), sources.data(), sources.size())
                                   : simd::dispatch_table().norm_sum_soa(
                                         runtime.active_re.data(), runtime.active_im.data(), sources.data(), sources.size());
    return std::clamp(probability, 0.0, 1.0);
}

double active_measurement_branch_probability(
    const FactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch) {
    const auto& sources0 = branch ? kernel.source0_true : kernel.source0_false;
    const auto& sources1 = branch ? kernel.source1_true : kernel.source1_false;
    const auto& coeffs0 = branch ? kernel.coeff0_true : kernel.coeff0_false;
    const auto& coeffs1 = branch ? kernel.coeff1_true : kernel.coeff1_false;
    const auto* coeff0 = reinterpret_cast<const double*>(coeffs0.data());
    const auto* coeff1 = reinterpret_cast<const double*>(coeffs1.data());
    double probability = 0.0;
    SYMFT_SINGLE_SIMD_LOOP
    for (std::size_t idx = 0; idx < sources0.size(); ++idx) {
        const std::size_t source0 = sources0[idx];
        const double c0r = coeff0[2 * idx];
        const double c0i = coeff0[2 * idx + 1];
        double ar = c0r * runtime.active_re[source0] - c0i * runtime.active_im[source0];
        double ai = c0r * runtime.active_im[source0] + c0i * runtime.active_re[source0];
        const std::size_t source1 = sources1[idx];
        if (source1 != kNoSource) {
            const double c1r = coeff1[2 * idx];
            const double c1i = coeff1[2 * idx + 1];
            ar += c1r * runtime.active_re[source1] - c1i * runtime.active_im[source1];
            ai += c1r * runtime.active_im[source1] + c1i * runtime.active_re[source1];
        }
        probability += ar * ar + ai * ai;
    }
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
    const auto& sources = branch ? kernel.source0_true : kernel.source0_false;
    const double invnorm = 1.0 / std::sqrt(probability);
    SYMFT_SINGLE_SIMD_LOOP
    for (std::size_t idx = 0; idx < sources.size(); ++idx) {
        const std::size_t source = sources[idx];
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
    const auto& sources0 = branch ? kernel.source0_true : kernel.source0_false;
    const auto& sources1 = branch ? kernel.source1_true : kernel.source1_false;
    const auto& coeffs0 = branch ? kernel.coeff0_true : kernel.coeff0_false;
    const auto& coeffs1 = branch ? kernel.coeff1_true : kernel.coeff1_false;
    if (runtime.active_scratch_re.size() < sources0.size()) {
        runtime.active_scratch_re.resize(sources0.size(), 0.0);
    }
    if (runtime.active_scratch_im.size() < sources0.size()) {
        runtime.active_scratch_im.resize(sources0.size(), 0.0);
    }
    const double invnorm = 1.0 / std::sqrt(probability);
    const auto* coeff0 = reinterpret_cast<const double*>(coeffs0.data());
    const auto* coeff1 = reinterpret_cast<const double*>(coeffs1.data());
    SYMFT_SINGLE_SIMD_LOOP
    for (std::size_t idx = 0; idx < sources0.size(); ++idx) {
        const std::size_t source0 = sources0[idx];
        const double c0r = coeff0[2 * idx];
        const double c0i = coeff0[2 * idx + 1];
        double ar = c0r * runtime.active_re[source0] - c0i * runtime.active_im[source0];
        double ai = c0r * runtime.active_im[source0] + c0i * runtime.active_re[source0];
        const std::size_t source1 = sources1[idx];
        if (source1 != kNoSource) {
            const double c1r = coeff1[2 * idx];
            const double c1i = coeff1[2 * idx + 1];
            ar += c1r * runtime.active_re[source1] - c1i * runtime.active_im[source1];
            ai += c1r * runtime.active_im[source1] + c1i * runtime.active_re[source1];
        }
        runtime.active_scratch_re[idx] = ar * invnorm;
        runtime.active_scratch_im[idx] = ai * invnorm;
    }
    std::copy_n(runtime.active_scratch_re.data(), sources0.size(), runtime.active_re.data());
    std::copy_n(runtime.active_scratch_im.data(), sources0.size(), runtime.active_im.data());
    --runtime.k;
}

void execute_instruction(FactoredExecutorState& runtime, const ApplyPrecomputedActivePauliRotation& instruction) {
    const bool sign = eval_symbolic_bool_unchecked(instruction.sign_plan, runtime);
    rotate_pauli(runtime, instruction.rotation_kernel, sign);
}

void execute_instruction(FactoredExecutorState& runtime, const ApplyActiveBasisChange& instruction) {
    apply_active_basis_change(runtime, instruction.kind, instruction.qubit);
}

void execute_instruction(FactoredExecutorState& runtime, const PromoteDormantRotation& instruction) {
    const bool sign = eval_symbolic_bool_unchecked(instruction.sign_plan, runtime);
    promote_first_dormant_rotation(runtime, sign ? -instruction.theta : instruction.theta);
}

void execute_instruction(FactoredExecutorState& runtime, const RecordMeasurement& instruction) {
    const bool outcome = eval_symbolic_bool_unchecked(instruction.outcome_plan, runtime);
    write_measurement_record(runtime, instruction.record, outcome, instruction.record_condition);
}

void execute_instruction(FactoredExecutorState& runtime, const MeasureActiveLastZ& instruction) {
    const double prob1 = active_last_z_probability_one(runtime);
    const bool branch = sample_bernoulli(runtime.rng_state, prob1);
    assign_symbol(runtime, instruction.branch, branch);
    project_active_last_z(runtime, branch, prob1);
    ++runtime.ndormant;
    const bool outcome = eval_symbolic_bool_unchecked(instruction.outcome_plan, runtime);
    write_measurement_record(runtime, instruction.record, outcome, instruction.record_condition);
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

} // namespace

FactoredExecutorState::FactoredExecutorState(const FactoredInstructionProgram& program, std::uint64_t seed)
    : n(program.n),
      k(program.initial_k),
      ndormant(program.n - program.initial_k),
      nsymbols(program.nsymbols),
      nrecords(program.nrecords),
      active_re(active_length(program.max_k), 0.0),
      active_im(active_length(program.max_k), 0.0),
      active_scratch_re(active_length(program.max_k), 0.0),
      active_scratch_im(active_length(program.max_k), 0.0),
      value_words(symbol_word_count(program.nsymbols), 0),
      assigned_words(symbol_word_count(program.nsymbols), 0),
      measurement_words(symbol_word_count(program.nrecords), 0),
      rng_state(seed) {
    const std::size_t dim = active_length(program.initial_k);
    std::fill_n(active_re.data(), dim, 0.0);
    std::fill_n(active_im.data(), dim, 0.0);
    active_re[0] = 1.0;
}

void reset_executor(FactoredExecutorState& runtime, const FactoredInstructionProgram& program) {
    runtime.n = program.n;
    runtime.k = program.initial_k;
    runtime.ndormant = program.n - program.initial_k;
    runtime.nsymbols = program.nsymbols;
    runtime.nrecords = program.nrecords;
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

} // namespace symft
