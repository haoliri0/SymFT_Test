#pragma once

#include "sampler/single_shot.hpp"
#include "simd/batch_simd.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace symft {

inline constexpr int kDefaultBatchShots = 2048;
inline constexpr std::size_t kDefaultBatchActiveAmplitudes = std::size_t{1} << 16;
inline constexpr std::size_t kXmaskRotationPairThreshold = 64;
inline constexpr std::size_t kBatchScalarSymbolicEvalThreshold = 32;

#if defined(__clang__)
#define SYMFT_BATCH_SIMD_LOOP _Pragma("clang loop vectorize(enable) interleave(enable)")
#elif defined(__GNUC__)
#define SYMFT_BATCH_SIMD_LOOP _Pragma("GCC ivdep")
#else
#define SYMFT_BATCH_SIMD_LOOP
#endif

enum class BatchSignMode {
    AllMinus,
    AllPlus,
    Mixed,
};

[[noreturn]] inline void fail(const std::string& message) {
    throw Error(message);
}

inline int trailing_zeros64(std::uint64_t value) {
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

inline std::size_t active_length(int k) {
    if (k < 0 || k >= 62) {
        fail("active qubit count is too large for machine basis indices");
    }
    return std::size_t{1} << k;
}

inline std::uint64_t symbol_bit_mask(int condition) {
    if (condition <= 0) {
        fail("condition id must be positive");
    }
    return std::uint64_t{1} << ((condition - 1) & 63);
}

inline std::size_t symbol_word_index(int condition) {
    if (condition <= 0) {
        fail("condition id must be positive");
    }
    return static_cast<std::size_t>((condition - 1) >> 6);
}

inline std::size_t symbol_word_count(int nsymbols) {
    if (nsymbols <= 0) {
        return 0;
    }
    return static_cast<std::size_t>((nsymbols + 63) >> 6);
}

inline std::uint64_t next_random_u64(std::uint64_t& state) {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

inline double rand_float(std::uint64_t& state) {
    return static_cast<double>(next_random_u64(state) >> 11) * 0x1.0p-53;
}

inline bool sample_bernoulli(std::uint64_t& rng_state, double probability) {
    const double p = std::clamp(probability, 0.0, 1.0);
    if (p <= 0.0) {
        return false;
    }
    if (p >= 1.0) {
        return true;
    }
    return rand_float(rng_state) < p;
}

inline double sample_geometric_gap(std::uint64_t& rng_state, double probability) {
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

inline int sample_categorical_row(std::uint64_t& rng_state, const std::vector<double>& probabilities) {
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

inline std::size_t batch_word_count(int shots) {
    if (shots <= 0) {
        return 0;
    }
    return static_cast<std::size_t>((shots + 63) >> 6);
}

inline std::uint64_t low_bits_mask(int nbits) {
    if (nbits <= 0) {
        return 0;
    }
    if (nbits >= 64) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return (std::uint64_t{1} << nbits) - 1;
}

inline std::uint64_t batch_live_word_mask(const BatchFactoredExecutorState& runtime, std::size_t word) {
    const int remaining = runtime.active_shots - static_cast<int>(word << 6);
    return low_bits_mask(remaining);
}

inline std::size_t runtime_batch_word_count(const BatchFactoredExecutorState& runtime) {
    return batch_word_count(runtime.active_shots);
}

inline std::uint64_t batch_shot_mask(int shot) {
    return std::uint64_t{1} << (shot & 63);
}

inline std::size_t batch_shot_word(int shot) {
    return static_cast<std::size_t>(shot >> 6);
}

inline void set_batch_bit(std::vector<std::uint64_t>& bits, int shot) {
    bits[batch_shot_word(shot)] |= batch_shot_mask(shot);
}

inline std::size_t batch_active_offset(const BatchFactoredExecutorState& runtime, std::size_t basis, int shot) {
    return basis * static_cast<std::size_t>(runtime.batches) + static_cast<std::size_t>(shot);
}

inline std::size_t batch_condition_offset(const BatchFactoredExecutorState& runtime, int condition, std::size_t word) {
    return static_cast<std::size_t>(condition - 1) * runtime.batch_words + word;
}

inline std::size_t batch_record_offset(const BatchFactoredExecutorState& runtime, int record, std::size_t word) {
    return static_cast<std::size_t>(record - 1) * runtime.batch_words + word;
}

inline void fill_batch_bits(std::vector<std::uint64_t>& bits, const BatchFactoredExecutorState& runtime, bool value) {
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

inline void mask_batch_bits(std::vector<std::uint64_t>& bits, const BatchFactoredExecutorState& runtime) {
    const std::size_t nwords = runtime_batch_word_count(runtime);
    if (nwords == 0) {
        return;
    }
    const int live_bits = runtime.active_shots & 63;
    if (live_bits != 0) {
        bits[nwords - 1] &= low_bits_mask(live_bits);
    }
}

inline void invert_batch_bits(std::vector<std::uint64_t>& bits, const BatchFactoredExecutorState& runtime) {
    const std::size_t nwords = runtime_batch_word_count(runtime);
    for (std::size_t word = 0; word < nwords; ++word) {
        bits[word] = ~bits[word] & batch_live_word_mask(runtime, word);
    }
    std::fill(bits.begin() + static_cast<std::ptrdiff_t>(nwords), bits.end(), 0);
}

inline void fill_batch_random_half_bits(std::vector<std::uint64_t>& bits, BatchFactoredExecutorState& runtime) {
    const std::size_t nwords = runtime_batch_word_count(runtime);
    if (bits.size() < runtime.batch_words) {
        bits.resize(runtime.batch_words, 0);
    }
    if ((runtime.active_shots & 63) == 0) {
        for (std::size_t word = 0; word < nwords; ++word) {
            bits[word] = next_random_u64(runtime.rng_state);
        }
    } else {
        for (std::size_t word = 0; word < nwords; ++word) {
            bits[word] = next_random_u64(runtime.rng_state) & batch_live_word_mask(runtime, word);
        }
    }
    std::fill(bits.begin() + static_cast<std::ptrdiff_t>(nwords), bits.end(), 0);
}



void assign_batch_symbol(BatchFactoredExecutorState& runtime, int condition, const std::vector<std::uint64_t>& bits);
void assign_batch_symbol(
    BatchFactoredExecutorState& runtime,
    std::optional<int> condition,
    const std::vector<std::uint64_t>& bits);
void eval_symbolic_bool_batch(
    std::vector<std::uint64_t>& out,
    const SymbolicBoolEvaluationPlan& plan,
    const BatchFactoredExecutorState& runtime);
void write_batch_measurement_record(
    BatchFactoredExecutorState& runtime,
    std::optional<int> record,
    const std::vector<std::uint64_t>& outcome_bits,
    std::optional<int> record_condition);
void assign_presampled_exogenous_batch(BatchFactoredExecutorState& runtime, const PresampledExogenous& samples);
void sample_exogenous_symbols_batch(BatchFactoredExecutorState& runtime, const FactoredInstructionProgram& program);
void rotate_pauli_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits);
void apply_active_basis_change_batch(BatchFactoredExecutorState& runtime, char kind, int q);
void promote_first_dormant_rotation_batch(
    BatchFactoredExecutorState& runtime,
    double theta,
    const std::vector<std::uint64_t>& sign_bits);
void measure_active_last_z_batch(
    BatchFactoredExecutorState& runtime,
    int branch_condition,
    const SymbolicBoolEvaluationPlan& outcome_plan,
    std::optional<int> record,
    std::optional<int> record_condition);
void measure_precomputed_active_pauli_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition,
    const SymbolicBoolEvaluationPlan& outcome_plan,
    std::optional<int> record,
    std::optional<int> record_condition);

} // namespace symft
