#include "sampler/single_shot.hpp"

#include "core/internal.hpp"
#include "sampler/random.hpp"

#include <cstdint>
#include <vector>

namespace symft {
using namespace detail;

namespace {

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

} // namespace

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

} // namespace symft
