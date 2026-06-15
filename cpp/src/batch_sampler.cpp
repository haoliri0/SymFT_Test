#include "symft/symft.hpp"
#include "symft/batch_simd.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace symft {
namespace {

constexpr int kDefaultBatchShots = 2048;
constexpr std::size_t kDefaultBatchActiveAmplitudes = std::size_t{1} << 17;

#if defined(__clang__)
#define SYMFT_BATCH_SIMD_LOOP _Pragma("clang loop vectorize(enable) interleave(enable)")
#elif defined(__GNUC__)
#define SYMFT_BATCH_SIMD_LOOP _Pragma("GCC ivdep")
#else
#define SYMFT_BATCH_SIMD_LOOP
#endif

[[noreturn]] void fail(const std::string& message) {
    throw Error(message);
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

std::size_t active_length(int k) {
    if (k < 0 || k >= 62) {
        fail("active qubit count is too large for machine basis indices");
    }
    return std::size_t{1} << k;
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
    const double p = std::clamp(probability, 0.0, 1.0);
    if (p <= 0.0) {
        return false;
    }
    if (p >= 1.0) {
        return true;
    }
    return rand_float(rng_state) < p;
}

double sample_geometric_gap(std::uint64_t& rng_state, double probability) {
    if (!(probability > 0.0 && probability < 1.0)) {
        fail("geometric gap probability must be in (0, 1)");
    }
    const double u = std::max(rand_float(rng_state), std::numeric_limits<double>::min());
    const double gap = std::floor(std::log(u) / std::log1p(-probability));
    if (!std::isfinite(gap) || gap >= static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
        return static_cast<double>(std::numeric_limits<std::int64_t>::max());
    }
    return gap;
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

std::size_t batch_word_count(int shots) {
    if (shots <= 0) {
        return 0;
    }
    return static_cast<std::size_t>((shots + 63) >> 6);
}

std::uint64_t low_bits_mask(int nbits) {
    if (nbits <= 0) {
        return 0;
    }
    if (nbits >= 64) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return (std::uint64_t{1} << nbits) - 1;
}

std::uint64_t batch_live_word_mask(const BatchFactoredExecutorState& runtime, std::size_t word) {
    const int remaining = runtime.active_shots - static_cast<int>(word << 6);
    return low_bits_mask(remaining);
}

std::size_t runtime_batch_word_count(const BatchFactoredExecutorState& runtime) {
    return batch_word_count(runtime.active_shots);
}

std::uint64_t batch_shot_mask(int shot) {
    return std::uint64_t{1} << (shot & 63);
}

std::size_t batch_shot_word(int shot) {
    return static_cast<std::size_t>(shot >> 6);
}

bool batch_bit(const std::vector<std::uint64_t>& bits, int shot) {
    const std::size_t word = batch_shot_word(shot);
    return word < bits.size() && (bits[word] & batch_shot_mask(shot)) != 0;
}

void set_batch_bit(std::vector<std::uint64_t>& bits, int shot) {
    bits[batch_shot_word(shot)] |= batch_shot_mask(shot);
}

std::size_t batch_active_offset(const BatchFactoredExecutorState& runtime, std::size_t basis, int shot) {
    return basis * static_cast<std::size_t>(runtime.batches) + static_cast<std::size_t>(shot);
}

std::size_t batch_condition_offset(const BatchFactoredExecutorState& runtime, int condition, std::size_t word) {
    return static_cast<std::size_t>(condition - 1) * runtime.batch_words + word;
}

std::size_t batch_record_offset(const BatchFactoredExecutorState& runtime, int record, std::size_t word) {
    return static_cast<std::size_t>(record - 1) * runtime.batch_words + word;
}

void fill_batch_bits(std::vector<std::uint64_t>& bits, const BatchFactoredExecutorState& runtime, bool value) {
    const std::size_t nwords = runtime_batch_word_count(runtime);
    if (bits.size() < runtime.batch_words) {
        bits.resize(runtime.batch_words, 0);
    }
    const std::uint64_t fill = value ? std::numeric_limits<std::uint64_t>::max() : 0;
    for (std::size_t word = 0; word < nwords; ++word) {
        bits[word] = fill & batch_live_word_mask(runtime, word);
    }
    std::fill(bits.begin() + static_cast<std::ptrdiff_t>(nwords), bits.end(), 0);
}

void mask_batch_bits(std::vector<std::uint64_t>& bits, const BatchFactoredExecutorState& runtime) {
    const std::size_t nwords = runtime_batch_word_count(runtime);
    for (std::size_t word = 0; word < nwords; ++word) {
        bits[word] &= batch_live_word_mask(runtime, word);
    }
}

void check_batch_symbol_slot(const BatchFactoredExecutorState& runtime, int condition) {
    if (condition <= 0 || condition > runtime.nsymbols) {
        fail("symbolic condition exceeds batch executor symbol table");
    }
}

bool batch_symbol_assigned(const BatchFactoredExecutorState& runtime, int condition) {
    check_batch_symbol_slot(runtime, condition);
    const std::size_t word = symbol_word_index(condition);
    return word < runtime.assigned_words.size() && (runtime.assigned_words[word] & symbol_bit_mask(condition)) != 0;
}

void mark_batch_symbol_assigned_unchecked(BatchFactoredExecutorState& runtime, int condition) {
    runtime.assigned_words[symbol_word_index(condition)] |= symbol_bit_mask(condition);
}

void set_batch_symbol_false_unchecked(BatchFactoredExecutorState& runtime, int condition) {
    check_batch_symbol_slot(runtime, condition);
    mark_batch_symbol_assigned_unchecked(runtime, condition);
}

void set_batch_symbol_true_unchecked(BatchFactoredExecutorState& runtime, int condition, int shot) {
    check_batch_symbol_slot(runtime, condition);
    runtime.value_words[batch_condition_offset(runtime, condition, batch_shot_word(shot))] |= batch_shot_mask(shot);
}

void set_batch_symbol_true_all_unchecked(BatchFactoredExecutorState& runtime, int condition) {
    check_batch_symbol_slot(runtime, condition);
    mark_batch_symbol_assigned_unchecked(runtime, condition);
    const std::size_t nwords = runtime_batch_word_count(runtime);
    for (std::size_t word = 0; word < runtime.batch_words; ++word) {
        runtime.value_words[batch_condition_offset(runtime, condition, word)] =
            word < nwords ? batch_live_word_mask(runtime, word) : 0;
    }
}

void assign_batch_conditions_false(BatchFactoredExecutorState& runtime, const std::vector<int>& conditions) {
    for (int condition : conditions) {
        set_batch_symbol_false_unchecked(runtime, condition);
    }
}

bool any_batch_assigned(BatchFactoredExecutorState& runtime, const std::vector<int>& conditions) {
    for (int condition : conditions) {
        if (batch_symbol_assigned(runtime, condition)) {
            return true;
        }
    }
    return false;
}

bool any_batch_categorical_group_assigned(
    BatchFactoredExecutorState& runtime,
    const std::vector<std::vector<int>>& condition_sets) {
    for (const auto& conditions : condition_sets) {
        if (any_batch_assigned(runtime, conditions)) {
            return true;
        }
    }
    return false;
}

bool batch_symbol_matches_bits(
    const BatchFactoredExecutorState& runtime,
    int condition,
    const std::vector<std::uint64_t>& bits) {
    const std::size_t nwords = runtime_batch_word_count(runtime);
    if (bits.size() < nwords) {
        fail("batch bit vector is too short");
    }
    for (std::size_t word = 0; word < nwords; ++word) {
        const std::uint64_t mask = batch_live_word_mask(runtime, word);
        const std::uint64_t actual = runtime.value_words[batch_condition_offset(runtime, condition, word)];
        if (((actual ^ bits[word]) & mask) != 0) {
            return false;
        }
    }
    return true;
}

void copy_bits_to_batch_symbol_unchecked(
    BatchFactoredExecutorState& runtime,
    int condition,
    const std::vector<std::uint64_t>& bits) {
    const std::size_t nwords = runtime_batch_word_count(runtime);
    if (bits.size() < nwords) {
        fail("batch bit vector is too short");
    }
    for (std::size_t word = 0; word < runtime.batch_words; ++word) {
        runtime.value_words[batch_condition_offset(runtime, condition, word)] =
            word < nwords ? bits[word] & batch_live_word_mask(runtime, word) : 0;
    }
}

void assign_batch_symbol(BatchFactoredExecutorState& runtime, int condition, const std::vector<std::uint64_t>& bits) {
    check_batch_symbol_slot(runtime, condition);
    if (batch_symbol_assigned(runtime, condition)) {
        if (!batch_symbol_matches_bits(runtime, condition, bits)) {
            fail("symbolic condition was assigned inconsistent concrete batch values");
        }
        return;
    }
    mark_batch_symbol_assigned_unchecked(runtime, condition);
    copy_bits_to_batch_symbol_unchecked(runtime, condition, bits);
}

void assign_batch_symbol(
    BatchFactoredExecutorState& runtime,
    std::optional<int> condition,
    const std::vector<std::uint64_t>& bits) {
    if (condition) {
        assign_batch_symbol(runtime, *condition, bits);
    }
}

void eval_symbolic_bool_batch(
    std::vector<std::uint64_t>& out,
    const SymbolicBoolEvaluationPlan& plan,
    const BatchFactoredExecutorState& runtime) {
    fill_batch_bits(out, runtime, plan.constant);
    if (plan.conditions.empty()) {
        return;
    }
    if (!plan.word_indices.empty()) {
        const std::size_t max_word = static_cast<std::size_t>(plan.word_indices.back());
        if (max_word >= runtime.assigned_words.size()) {
            fail("symbolic condition expression has no concrete batch value");
        }
        std::uint64_t missing = 0;
        for (std::size_t i = 0; i < plan.word_indices.size(); ++i) {
            const std::size_t word = static_cast<std::size_t>(plan.word_indices[i]);
            missing |= plan.word_masks[i] & ~runtime.assigned_words[word];
        }
        if (missing != 0) {
            fail("symbolic condition expression has no concrete batch value");
        }
    } else {
        for (int condition : plan.conditions) {
            if (!batch_symbol_assigned(runtime, condition)) {
                fail("symbolic condition expression has no concrete batch value");
            }
        }
    }
    const std::size_t nwords = runtime_batch_word_count(runtime);
    for (int condition : plan.conditions) {
        for (std::size_t word = 0; word < nwords; ++word) {
            out[word] ^= runtime.value_words[batch_condition_offset(runtime, condition, word)];
        }
    }
    mask_batch_bits(out, runtime);
}

void ensure_batch_measurement_storage(BatchFactoredExecutorState& runtime, int record) {
    if (record <= runtime.nrecords) {
        return;
    }
    std::vector<std::uint64_t> next(static_cast<std::size_t>(record) * runtime.batch_words, 0);
    for (int old_record = 1; old_record <= runtime.nrecords; ++old_record) {
        const std::size_t old_base = batch_record_offset(runtime, old_record, 0);
        const std::size_t new_base = static_cast<std::size_t>(old_record - 1) * runtime.batch_words;
        std::copy(
            runtime.measurement_words.begin() + static_cast<std::ptrdiff_t>(old_base),
            runtime.measurement_words.begin() + static_cast<std::ptrdiff_t>(old_base + runtime.batch_words),
            next.begin() + static_cast<std::ptrdiff_t>(new_base));
    }
    runtime.nrecords = record;
    runtime.measurement_words = std::move(next);
}

void write_batch_measurement_record(
    BatchFactoredExecutorState& runtime,
    std::optional<int> record,
    const std::vector<std::uint64_t>& outcome_bits,
    std::optional<int> record_condition) {
    const std::size_t nwords = runtime_batch_word_count(runtime);
    if (record) {
        if (*record <= 0) {
            fail("measurement record id must be positive");
        }
        ensure_batch_measurement_storage(runtime, *record);
        for (std::size_t word = 0; word < runtime.batch_words; ++word) {
            runtime.measurement_words[batch_record_offset(runtime, *record, word)] =
                word < nwords ? outcome_bits[word] & batch_live_word_mask(runtime, word) : 0;
        }
    }
    assign_batch_symbol(runtime, record_condition, outcome_bits);
}

void assign_presampled_exogenous_batch(
    BatchFactoredExecutorState& runtime,
    const PresampledExogenous& samples) {
    if (runtime.active_shots > samples.nshots || runtime.nsymbols != samples.nsymbols ||
        samples.exogenous_assigned_words.size() != samples.nwords ||
        samples.value_words.size() != static_cast<std::size_t>(samples.nshots) * samples.nwords) {
        fail("presampled exogenous table does not match batch executor");
    }
    for (std::size_t symbol_word = 0; symbol_word < samples.nwords; ++symbol_word) {
        runtime.assigned_words[symbol_word] |= samples.exogenous_assigned_words[symbol_word];
    }
    for (int shot = 0; shot < runtime.active_shots; ++shot) {
        const std::size_t shot_word = batch_shot_word(shot);
        const std::uint64_t shot_mask = batch_shot_mask(shot);
        const std::size_t base = static_cast<std::size_t>(shot) * samples.nwords;
        for (std::size_t symbol_word = 0; symbol_word < samples.nwords; ++symbol_word) {
            std::uint64_t row = samples.value_words[base + symbol_word] &
                                samples.exogenous_assigned_words[symbol_word];
            while (row != 0) {
                const int bit = trailing_zeros64(row);
                const int condition = static_cast<int>(symbol_word * 64 + static_cast<std::size_t>(bit) + 1);
                if (condition <= runtime.nsymbols) {
                    runtime.value_words[batch_condition_offset(runtime, condition, shot_word)] |= shot_mask;
                }
                row &= row - 1;
            }
        }
    }
}

void sample_categorical_distribution_batch(
    BatchFactoredExecutorState& runtime,
    const std::vector<int>& conditions,
    int nbits,
    const std::vector<std::vector<std::uint64_t>>& assignments,
    const std::vector<double>& probabilities) {
    if (static_cast<int>(conditions.size()) != nbits) {
        fail("categorical condition count does not match assignment bit count");
    }
    bool all_assigned = true;
    bool any_assigned = false;
    for (int condition : conditions) {
        const bool assigned = batch_symbol_assigned(runtime, condition);
        all_assigned = all_assigned && assigned;
        any_assigned = any_assigned || assigned;
    }
    if (all_assigned) {
        return;
    }
    if (any_assigned) {
        fail("categorical symbolic distribution was only partially preassigned");
    }

    assign_batch_conditions_false(runtime, conditions);
    for (int shot = 0; shot < runtime.active_shots; ++shot) {
        const int row = sample_categorical_row(runtime.rng_state, probabilities);
        const auto& assignment = assignments[static_cast<std::size_t>(row)];
        for (std::size_t bit_idx = 0; bit_idx < conditions.size(); ++bit_idx) {
            if (packed_bit(assignment, static_cast<int>(bit_idx))) {
                set_batch_symbol_true_unchecked(runtime, conditions[bit_idx], shot);
            }
        }
    }
}

void sample_categorical_distribution_batch(
    BatchFactoredExecutorState& runtime,
    const SymbolicCategoricalDistribution& distribution) {
    sample_categorical_distribution_batch(
        runtime,
        distribution.conditions,
        distribution.nbits,
        distribution.assignments,
        distribution.probabilities);
}

void assign_batch_categorical_group_false(
    BatchFactoredExecutorState& runtime,
    const std::vector<std::vector<int>>& condition_sets) {
    for (const auto& conditions : condition_sets) {
        assign_batch_conditions_false(runtime, conditions);
    }
}

void sample_rare_categorical_group_batch(
    BatchFactoredExecutorState& runtime,
    const RareCategoricalSampleGroup& group) {
    if (any_batch_categorical_group_assigned(runtime, group.conditions)) {
        for (const auto& conditions : group.conditions) {
            sample_categorical_distribution_batch(runtime, conditions, group.nbits, group.assignments, group.probabilities);
        }
        return;
    }

    assign_batch_categorical_group_false(runtime, group.conditions);
    const int nsets = static_cast<int>(group.conditions.size());
    if (group.event_probability <= 0.0 || nsets == 0) {
        return;
    }
    const std::int64_t total_draws =
        static_cast<std::int64_t>(runtime.active_shots) * static_cast<std::int64_t>(nsets);
    std::int64_t draw = 0;
    while (true) {
        const auto gap = static_cast<std::int64_t>(sample_geometric_gap(runtime.rng_state, group.event_probability));
        if (gap >= total_draws - draw) {
            return;
        }
        draw += gap;
        const int shot = static_cast<int>(draw / nsets);
        const int set_idx = static_cast<int>(draw % nsets);
        const int row =
            group.event_rows[static_cast<std::size_t>(sample_categorical_row(runtime.rng_state, group.event_probabilities))];
        const auto& conditions = group.conditions[static_cast<std::size_t>(set_idx)];
        const auto& assignment = group.assignments[static_cast<std::size_t>(row)];
        for (std::size_t bit_idx = 0; bit_idx < conditions.size(); ++bit_idx) {
            if (packed_bit(assignment, static_cast<int>(bit_idx))) {
                set_batch_symbol_true_unchecked(runtime, conditions[bit_idx], shot);
            }
        }
        ++draw;
    }
}

void sample_bernoulli_condition_batch(BatchFactoredExecutorState& runtime, int condition, double probability) {
    if (batch_symbol_assigned(runtime, condition)) {
        return;
    }
    const double p = std::clamp(probability, 0.0, 1.0);
    if (p <= 0.0) {
        set_batch_symbol_false_unchecked(runtime, condition);
        return;
    }
    if (p >= 1.0) {
        set_batch_symbol_true_all_unchecked(runtime, condition);
        return;
    }

    set_batch_symbol_false_unchecked(runtime, condition);
    for (int shot = 0; shot < runtime.active_shots; ++shot) {
        if (sample_bernoulli(runtime.rng_state, p)) {
            set_batch_symbol_true_unchecked(runtime, condition, shot);
        }
    }
}

void sample_low_probability_bernoulli_group_batch(
    BatchFactoredExecutorState& runtime,
    const BernoulliSampleGroup& group) {
    if (any_batch_assigned(runtime, group.conditions)) {
        for (int condition : group.conditions) {
            sample_bernoulli_condition_batch(runtime, condition, group.probability);
        }
        return;
    }

    assign_batch_conditions_false(runtime, group.conditions);
    const int nconditions = static_cast<int>(group.conditions.size());
    if (group.probability <= 0.0 || nconditions == 0) {
        return;
    }
    const std::int64_t total_draws =
        static_cast<std::int64_t>(runtime.active_shots) * static_cast<std::int64_t>(nconditions);
    std::int64_t draw = 0;
    while (true) {
        const auto gap = static_cast<std::int64_t>(sample_geometric_gap(runtime.rng_state, group.probability));
        if (gap >= total_draws - draw) {
            return;
        }
        draw += gap;
        const int shot = static_cast<int>(draw / nconditions);
        const int condition_idx = static_cast<int>(draw % nconditions);
        set_batch_symbol_true_unchecked(runtime, group.conditions[static_cast<std::size_t>(condition_idx)], shot);
        ++draw;
    }
}

void sample_exogenous_symbols_batch(BatchFactoredExecutorState& runtime, const FactoredInstructionProgram& program) {
    for (const auto& distribution : program.sampled_categorical_distributions) {
        sample_categorical_distribution_batch(runtime, distribution);
    }
    for (const auto& group : program.sampled_rare_categorical_groups) {
        sample_rare_categorical_group_batch(runtime, group);
    }
    for (std::size_t i = 0; i < program.sampled_bernoulli_conditions.size(); ++i) {
        sample_bernoulli_condition_batch(
            runtime,
            program.sampled_bernoulli_conditions[i],
            program.sampled_bernoulli_probabilities[i]);
    }
    for (const auto& group : program.sampled_low_probability_bernoulli_groups) {
        sample_low_probability_bernoulli_group_batch(runtime, group);
    }
}

const std::vector<double>& fill_rotation_coefficients(
    BatchFactoredExecutorState& runtime,
    const std::vector<std::uint64_t>& sign_bits,
    double minus_coeff,
    double plus_coeff) {
    if (runtime.rotation_coefficients.size() < static_cast<std::size_t>(runtime.active_shots)) {
        runtime.rotation_coefficients.resize(static_cast<std::size_t>(runtime.active_shots), 0.0);
    }
    const std::size_t nwords = runtime_batch_word_count(runtime);
    for (std::size_t word = 0; word < nwords; ++word) {
        const std::uint64_t bits = word < sign_bits.size() ? sign_bits[word] : 0;
        const int base_shot = static_cast<int>(word << 6);
        const int live = std::min(64, runtime.active_shots - base_shot);
        SYMFT_BATCH_SIMD_LOOP
        for (int bit = 0; bit < live; ++bit) {
            runtime.rotation_coefficients[static_cast<std::size_t>(base_shot + bit)] =
                ((bits >> bit) & 1ULL) != 0 ? plus_coeff : minus_coeff;
        }
    }
    return runtime.rotation_coefficients;
}

void rotate_uniform_imag_pairs_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits) {
    const auto& coeffs = fill_rotation_coefficients(
        runtime,
        sign_bits,
        kernel.pair_left_minus_coefficients.front().imag(),
        kernel.pair_left_plus_coefficients.front().imag());
    batch_simd::dispatch_table().rotate_uniform_imag_pairs(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.batches),
        runtime.active_shots,
        kernel.pair_left_indices.data(),
        kernel.pair_right_indices.data(),
        kernel.pair_left_indices.size(),
        kernel.cos_theta,
        coeffs.data());
}

void rotate_real_pair_flip_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits) {
    const auto& coeffs = fill_rotation_coefficients(
        runtime,
        sign_bits,
        kernel.pair_left_minus_coefficients.front().real(),
        kernel.pair_left_plus_coefficients.front().real());
    batch_simd::dispatch_table().rotate_real_pair_flip(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.batches),
        runtime.active_shots,
        kernel.pair_left_indices.data(),
        kernel.pair_right_indices.data(),
        kernel.pair_left_indices.size(),
        kernel.cos_theta,
        coeffs.data(),
        kernel.action.zmask);
}

void rotate_pauli_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits) {
    if (kernel.action.nqubits != runtime.k) {
        fail("rotation kernel dimension does not match batch active state");
    }
    const double c = kernel.cos_theta;
    if (kernel.is_diagonal) {
        const std::size_t dim = active_length(runtime.k);
        for (std::size_t basis = 0; basis < dim; ++basis) {
            const Complex plus = kernel.diagonal_plus_coefficients[basis];
            const Complex minus = kernel.diagonal_minus_coefficients[basis];
            SYMFT_BATCH_SIMD_LOOP
            for (int shot = 0; shot < runtime.active_shots; ++shot) {
                const Complex coeff = batch_bit(sign_bits, shot) ? plus : minus;
                const double fr = c + coeff.real();
                const double fi = coeff.imag();
                const std::size_t off = batch_active_offset(runtime, basis, shot);
                const double r = runtime.active_re[off];
                const double i = runtime.active_im[off];
                runtime.active_re[off] = fr * r - fi * i;
                runtime.active_im[off] = fr * i + fi * r;
            }
        }
        return;
    }
    if (kernel.uniform_imag_pairs) {
        rotate_uniform_imag_pairs_batch(runtime, kernel, sign_bits);
        return;
    }
    if (kernel.real_pair_flip) {
        rotate_real_pair_flip_batch(runtime, kernel, sign_bits);
        return;
    }
    for (std::size_t idx = 0; idx < kernel.pair_left_indices.size(); ++idx) {
        const std::size_t i0 = kernel.pair_left_indices[idx];
        const std::size_t i1 = kernel.pair_right_indices[idx];
        const Complex left_plus = kernel.pair_left_plus_coefficients[idx];
        const Complex right_plus = kernel.pair_right_plus_coefficients[idx];
        const Complex left_minus = kernel.pair_left_minus_coefficients[idx];
        const Complex right_minus = kernel.pair_right_minus_coefficients[idx];
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < runtime.active_shots; ++shot) {
            const bool sign = batch_bit(sign_bits, shot);
            const Complex left_coeff = sign ? left_plus : left_minus;
            const Complex right_coeff = sign ? right_plus : right_minus;
            const std::size_t off0 = batch_active_offset(runtime, i0, shot);
            const std::size_t off1 = batch_active_offset(runtime, i1, shot);
            const double r0 = runtime.active_re[off0];
            const double im0 = runtime.active_im[off0];
            const double r1 = runtime.active_re[off1];
            const double im1 = runtime.active_im[off1];
            runtime.active_re[off0] = c * r0 + right_coeff.real() * r1 - right_coeff.imag() * im1;
            runtime.active_im[off0] = c * im0 + right_coeff.real() * im1 + right_coeff.imag() * r1;
            runtime.active_re[off1] = c * r1 + left_coeff.real() * r0 - left_coeff.imag() * im0;
            runtime.active_im[off1] = c * im1 + left_coeff.real() * im0 + left_coeff.imag() * r0;
        }
    }
}

void promote_first_dormant_rotation_batch(
    BatchFactoredExecutorState& runtime,
    double theta,
    const std::vector<std::uint64_t>& sign_bits) {
    if (runtime.ndormant <= 0) {
        fail("cannot promote a dormant qubit when none remain");
    }
    const std::size_t dim = active_length(runtime.k);
    const std::size_t promoted_dim = 2 * dim;
    if (runtime.active_re.size() < promoted_dim * static_cast<std::size_t>(runtime.batches)) {
        fail("batch active storage has too few columns for dormant promotion");
    }
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const auto& coeffs = fill_rotation_coefficients(runtime, sign_bits, -s, s);
    batch_simd::dispatch_table().promote_first_dormant_rotation(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.batches),
        runtime.active_shots,
        dim,
        c,
        coeffs.data());
    ++runtime.k;
    --runtime.ndormant;
}

void sample_batch_measurement_branches_from_true(
    BatchFactoredExecutorState& runtime,
    std::vector<std::uint64_t>& branch_bits,
    std::vector<double>& prob_true,
    std::vector<double>& invnorms) {
    fill_batch_bits(branch_bits, runtime, false);
    for (int shot = 0; shot < runtime.active_shots; ++shot) {
        const double pt = std::clamp(prob_true[static_cast<std::size_t>(shot)], 0.0, 1.0);
        const bool branch = sample_bernoulli(runtime.rng_state, pt);
        if (branch) {
            set_batch_bit(branch_bits, shot);
        }
        const double probability = branch ? pt : 1.0 - pt;
        if (probability <= 0.0) {
            fail("sampled an impossible active measurement branch");
        }
        invnorms[static_cast<std::size_t>(shot)] = 1.0 / std::sqrt(probability);
    }
}

void measure_active_last_z_batch(
    BatchFactoredExecutorState& runtime,
    int branch_condition,
    const SymbolicBoolEvaluationPlan& outcome_plan,
    std::optional<int> record,
    std::optional<int> record_condition) {
    if (runtime.k <= 0) {
        fail("cannot measure the last active qubit when k == 0");
    }
    const int new_k = runtime.k - 1;
    const std::size_t dim = active_length(new_k);
    batch_simd::dispatch_table().last_z_measure_true_prob(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.batches),
        runtime.active_shots,
        dim,
        runtime.branch_prob_true.data());
    sample_batch_measurement_branches_from_true(
        runtime,
        runtime.eval_scratch,
        runtime.branch_prob_true,
        runtime.branch_invnorms);
    const auto& branch_bits = runtime.eval_scratch;
    batch_simd::dispatch_table().last_z_project(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.batches),
        runtime.active_shots,
        dim,
        branch_bits.data(),
        runtime.branch_invnorms.data());
    runtime.k = new_k;
    ++runtime.ndormant;
    assign_batch_symbol(runtime, branch_condition, branch_bits);
    eval_symbolic_bool_batch(runtime.eval_scratch, outcome_plan, runtime);
    write_batch_measurement_record(runtime, record, runtime.eval_scratch, record_condition);
}

void measure_diagonal_active_pauli_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition,
    const SymbolicBoolEvaluationPlan& outcome_plan,
    std::optional<int> record,
    std::optional<int> record_condition) {
    const auto& source_false = kernel.source0_false;
    const auto& source_true = kernel.source0_true;
    const std::size_t out_dim = source_false.size();
    batch_simd::dispatch_table().diagonal_measure_true_prob(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.batches),
        runtime.active_shots,
        source_true.data(),
        out_dim,
        runtime.branch_prob_true.data());
    sample_batch_measurement_branches_from_true(
        runtime,
        runtime.eval_scratch,
        runtime.branch_prob_true,
        runtime.branch_invnorms);
    const auto& branch_bits = runtime.eval_scratch;
    batch_simd::dispatch_table().diagonal_project(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.batches),
        runtime.active_shots,
        source_false.data(),
        source_true.data(),
        out_dim,
        branch_bits.data(),
        runtime.branch_invnorms.data());
    --runtime.k;
    ++runtime.ndormant;
    assign_batch_symbol(runtime, branch_condition, branch_bits);
    eval_symbolic_bool_batch(runtime.eval_scratch, outcome_plan, runtime);
    write_batch_measurement_record(runtime, record, runtime.eval_scratch, record_condition);
}

void measure_nondiagonal_active_pauli_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition,
    const SymbolicBoolEvaluationPlan& outcome_plan,
    std::optional<int> record,
    std::optional<int> record_condition) {
    const std::size_t out_dim = kernel.source0_false.size();
    batch_simd::dispatch_table().nondiagonal_measure_true_prob(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.batches),
        runtime.active_shots,
        kernel.source0_false.data(),
        kernel.source1_false.data(),
        kernel.coeff1_false.data(),
        out_dim,
        runtime.branch_prob_true.data());
    sample_batch_measurement_branches_from_true(
        runtime,
        runtime.eval_scratch,
        runtime.branch_prob_true,
        runtime.branch_invnorms);
    const auto& branch_bits = runtime.eval_scratch;
    batch_simd::dispatch_table().nondiagonal_project(
        runtime.active_re.data(),
        runtime.active_im.data(),
        runtime.scratch_re.data(),
        runtime.scratch_im.data(),
        static_cast<std::size_t>(runtime.batches),
        runtime.active_shots,
        kernel.source0_false.data(),
        kernel.source1_false.data(),
        kernel.coeff1_false.data(),
        out_dim,
        branch_bits.data(),
        runtime.branch_invnorms.data());
    const std::size_t active_prefix_size = out_dim * static_cast<std::size_t>(runtime.batches);
    std::copy_n(runtime.scratch_re.data(), active_prefix_size, runtime.active_re.data());
    std::copy_n(runtime.scratch_im.data(), active_prefix_size, runtime.active_im.data());
    --runtime.k;
    ++runtime.ndormant;
    assign_batch_symbol(runtime, branch_condition, branch_bits);
    eval_symbolic_bool_batch(runtime.eval_scratch, outcome_plan, runtime);
    write_batch_measurement_record(runtime, record, runtime.eval_scratch, record_condition);
}

void measure_precomputed_active_pauli_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition,
    const SymbolicBoolEvaluationPlan& outcome_plan,
    std::optional<int> record,
    std::optional<int> record_condition) {
    if (runtime.k <= 0) {
        fail("cannot measure an active Pauli when k == 0");
    }
    if (kernel.action.nqubits != runtime.k) {
        fail("measurement kernel dimension does not match batch active state");
    }
    if (kernel.is_diagonal) {
        measure_diagonal_active_pauli_batch(runtime, kernel, branch_condition, outcome_plan, record, record_condition);
    } else {
        measure_nondiagonal_active_pauli_batch(runtime, kernel, branch_condition, outcome_plan, record, record_condition);
    }
}

void execute_batch_instruction(BatchFactoredExecutorState& runtime, const ApplyPrecomputedActivePauliRotation& instruction) {
    eval_symbolic_bool_batch(runtime.eval_scratch, instruction.sign_plan, runtime);
    rotate_pauli_batch(runtime, instruction.rotation_kernel, runtime.eval_scratch);
}

void execute_batch_instruction(BatchFactoredExecutorState& runtime, const PromoteDormantRotation& instruction) {
    eval_symbolic_bool_batch(runtime.eval_scratch, instruction.sign_plan, runtime);
    promote_first_dormant_rotation_batch(runtime, instruction.theta, runtime.eval_scratch);
}

void execute_batch_instruction(BatchFactoredExecutorState& runtime, const RecordMeasurement& instruction) {
    eval_symbolic_bool_batch(runtime.eval_scratch, instruction.outcome_plan, runtime);
    write_batch_measurement_record(runtime, instruction.record, runtime.eval_scratch, instruction.record_condition);
}

void execute_batch_instruction(BatchFactoredExecutorState& runtime, const MeasureActiveLastZ& instruction) {
    measure_active_last_z_batch(
        runtime,
        instruction.branch,
        instruction.outcome_plan,
        instruction.record,
        instruction.record_condition);
}

void execute_batch_instruction(BatchFactoredExecutorState& runtime, const MeasurePrecomputedActivePauli& instruction) {
    measure_precomputed_active_pauli_batch(
        runtime,
        instruction.kernel,
        instruction.branch,
        instruction.outcome_plan,
        instruction.record,
        instruction.record_condition);
}

void execute_batch_instruction(BatchFactoredExecutorState& runtime, const IntroduceDormantMeasurementBranch& instruction) {
    fill_batch_bits(runtime.eval_scratch, runtime, false);
    for (int shot = 0; shot < runtime.active_shots; ++shot) {
        if (sample_bernoulli(runtime.rng_state, 0.5)) {
            set_batch_bit(runtime.eval_scratch, shot);
        }
    }
    const auto& branch_bits = runtime.eval_scratch;
    assign_batch_symbol(runtime, instruction.branch, branch_bits);
    eval_symbolic_bool_batch(runtime.eval_scratch, instruction.outcome_plan, runtime);
    write_batch_measurement_record(runtime, instruction.record, runtime.eval_scratch, instruction.record_condition);
}

void execute_batch_instruction(BatchFactoredExecutorState& runtime, const FactoredInstruction& instruction) {
    std::visit([&](const auto& inst) { execute_batch_instruction(runtime, inst); }, instruction);
}

} // namespace

int default_batch_count(int max_k) {
    const std::size_t dim = active_length(max_k);
    const std::size_t count = std::min<std::size_t>(
        kDefaultBatchShots,
        std::max<std::size_t>(1, kDefaultBatchActiveAmplitudes / dim));
    return static_cast<int>(count);
}

const char* active_batch_simd_backend() {
    return batch_simd::dispatch_name();
}

BatchFactoredExecutorState::BatchFactoredExecutorState(
    const FactoredInstructionProgram& program,
    int batches_,
    std::uint64_t seed)
    : batches(batches_ > 0 ? batches_ : default_batch_count(program.max_k)),
      rng_state(seed) {
    reset_batch_executor(*this, program, batches);
}

void reset_batch_executor(BatchFactoredExecutorState& runtime, const FactoredInstructionProgram& program, int shots) {
    if (shots < 0 || shots > runtime.batches) {
        fail("active batch shot count is out of range");
    }
    runtime.n = program.n;
    runtime.k = program.initial_k;
    runtime.ndormant = program.n - program.initial_k;
    runtime.active_shots = shots;
    runtime.nsymbols = program.nsymbols;
    runtime.nrecords = program.nrecords;
    runtime.max_k = program.max_k;
    runtime.batch_words = batch_word_count(runtime.batches);

    const std::size_t max_dim = active_length(program.max_k);
    const std::size_t active_size = max_dim * static_cast<std::size_t>(runtime.batches);
    if (runtime.active_re.size() < active_size) {
        runtime.active_re.resize(active_size, 0.0);
    }
    if (runtime.active_im.size() < active_size) {
        runtime.active_im.resize(active_size, 0.0);
    }
    if (runtime.scratch_re.size() < active_size) {
        runtime.scratch_re.resize(active_size, 0.0);
    }
    if (runtime.scratch_im.size() < active_size) {
        runtime.scratch_im.resize(active_size, 0.0);
    }

    const std::size_t symbol_size = static_cast<std::size_t>(program.nsymbols) * runtime.batch_words;
    if (runtime.value_words.size() != symbol_size) {
        runtime.value_words.resize(symbol_size, 0);
    }
    std::fill(runtime.value_words.begin(), runtime.value_words.end(), 0);
    const std::size_t assigned_size = symbol_word_count(program.nsymbols);
    if (runtime.assigned_words.size() != assigned_size) {
        runtime.assigned_words.resize(assigned_size, 0);
    }
    std::fill(runtime.assigned_words.begin(), runtime.assigned_words.end(), 0);
    const std::size_t measurement_size = static_cast<std::size_t>(program.nrecords) * runtime.batch_words;
    if (runtime.measurement_words.size() != measurement_size) {
        runtime.measurement_words.resize(measurement_size, 0);
    }
    std::fill(runtime.measurement_words.begin(), runtime.measurement_words.end(), 0);
    if (runtime.eval_scratch.size() != runtime.batch_words) {
        runtime.eval_scratch.resize(runtime.batch_words, 0);
    }
    std::fill(runtime.eval_scratch.begin(), runtime.eval_scratch.end(), 0);
    if (runtime.rotation_coefficients.size() < static_cast<std::size_t>(runtime.batches)) {
        runtime.rotation_coefficients.resize(static_cast<std::size_t>(runtime.batches), 0.0);
    }
    if (runtime.branch_prob_true.size() < static_cast<std::size_t>(runtime.batches)) {
        runtime.branch_prob_true.resize(static_cast<std::size_t>(runtime.batches), 0.0);
    }
    if (runtime.branch_invnorms.size() < static_cast<std::size_t>(runtime.batches)) {
        runtime.branch_invnorms.resize(static_cast<std::size_t>(runtime.batches), 0.0);
    }

    if (program.initial_active.k != program.initial_k) {
        fail("program initial active state dimension mismatch");
    }
    const std::size_t dim = program.initial_active.dim();
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const Complex amp = program.initial_active.alpha[basis];
        const std::size_t base = basis * static_cast<std::size_t>(runtime.batches);
        for (int shot = 0; shot < runtime.active_shots; ++shot) {
            runtime.active_re[base + static_cast<std::size_t>(shot)] = amp.real();
            runtime.active_im[base + static_cast<std::size_t>(shot)] = amp.imag();
        }
    }
}

void execute_batch_in_place(BatchFactoredExecutorState& runtime, const FactoredInstructionProgram& program) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
        fail("batch executor state does not match program");
    }
    if (runtime.active_shots == 0) {
        return;
    }
    sample_exogenous_symbols_batch(runtime, program);
    for (const auto& instruction : program.instructions) {
        execute_batch_instruction(runtime, instruction);
    }
}

void execute_batch_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExogenous& samples) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
        fail("batch executor state does not match program");
    }
    if (runtime.active_shots == 0) {
        return;
    }
    assign_presampled_exogenous_batch(runtime, samples);
    for (const auto& instruction : program.instructions) {
        execute_batch_instruction(runtime, instruction);
    }
}

std::vector<std::vector<std::uint64_t>> sample_measurements_batch(
    const FactoredInstructionProgram& program,
    int shots,
    int batches,
    std::uint64_t seed) {
    if (shots < 0) {
        fail("shot count must be nonnegative");
    }
    BatchFactoredExecutorState runtime(program, batches, seed);
    std::vector<std::vector<std::uint64_t>> out(
        static_cast<std::size_t>(shots),
        std::vector<std::uint64_t>(symbol_word_count(program.nrecords), 0));

    int offset = 0;
    while (offset < shots) {
        const int block = std::min(runtime.batches, shots - offset);
        reset_batch_executor(runtime, program, block);
        execute_batch_in_place(runtime, program);
        for (int shot = 0; shot < block; ++shot) {
            auto& row = out[static_cast<std::size_t>(offset + shot)];
            for (int record = 1; record <= runtime.nrecords; ++record) {
                const std::size_t word = batch_shot_word(shot);
                const bool bit = (runtime.measurement_words[batch_record_offset(runtime, record, word)] &
                                  batch_shot_mask(shot)) != 0;
                if (bit) {
                    const int record_bit = record - 1;
                    row[static_cast<std::size_t>(record_bit >> 6)] |=
                        std::uint64_t{1} << (record_bit & 63);
                }
            }
        }
        offset += block;
    }
    return out;
}

} // namespace symft

#undef SYMFT_BATCH_SIMD_LOOP
