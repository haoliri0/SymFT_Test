#pragma once

#include "sampler/active_internal.hpp"
#include "sampler/batch_sampler.hpp"
#include "sampler/exogenous.hpp"
#include "sampler/random.hpp"
#include "sampler/single_shot.hpp"
#include "simd/batch_simd.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace symft {

struct BatchThreadPool {
    explicit BatchThreadPool(int threads);
    ~BatchThreadPool();

    BatchThreadPool(const BatchThreadPool&) = delete;
    BatchThreadPool& operator=(const BatchThreadPool&) = delete;

    int size() const;
    void parallel_for(
        int workers,
        std::size_t items,
        const std::function<void(int, std::size_t, std::size_t)>& fn);

private:
    void worker_loop(int worker_id);

    int thread_count_ = 1;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable task_cv_;
    std::condition_variable done_cv_;
    std::function<void(int)> task_;
    std::exception_ptr error_;
    int generation_ = 0;
    int completed_generation_ = 0;
    int pending_ = 0;
    bool stop_ = false;
};

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

using detail::active_length;
using detail::fail;
using detail::symbol_bit_mask;
using detail::symbol_word_count;
using detail::symbol_word_index;
using detail::trailing_zeros64;

inline int normalized_batch_threads(int threads) {
    return std::max(1, threads);
}

inline int batch_parallel_worker_count(
    const BatchFactoredExecutorState& runtime,
    std::size_t items,
    std::size_t min_items_per_worker) {
    const int requested = normalized_batch_threads(runtime.threads);
    if (requested <= 1 || items == 0) {
        return 1;
    }
    const std::size_t grain = std::max<std::size_t>(1, min_items_per_worker);
    const std::size_t shots = static_cast<std::size_t>(std::max(1, runtime.active_shots));
    const std::size_t work = items > std::numeric_limits<std::size_t>::max() / shots
                                 ? std::numeric_limits<std::size_t>::max()
                                 : items * shots;
    const std::size_t item_limited = std::max<std::size_t>(1, work / grain);
    return std::max(1, std::min<int>(requested, static_cast<int>(std::min<std::size_t>(item_limited, items))));
}

template <class Fn>
void batch_parallel_for_workers(
    const BatchFactoredExecutorState& runtime,
    int workers,
    std::size_t items,
    Fn&& fn) {
    if (workers <= 1 || items == 0) {
        fn(0, std::size_t{0}, items);
        return;
    }
    if (runtime.thread_pool) {
        runtime.thread_pool->parallel_for(
            workers,
            items,
            [&](int worker, std::size_t first, std::size_t last) {
                fn(worker, first, last);
            });
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers - 1));
    std::exception_ptr error;
    std::mutex error_mutex;
    auto run_range = [&](int worker) {
        const std::size_t first = items * static_cast<std::size_t>(worker) / static_cast<std::size_t>(workers);
        const std::size_t last = items * static_cast<std::size_t>(worker + 1) / static_cast<std::size_t>(workers);
        try {
            fn(worker, first, last);
        } catch (...) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!error) {
                error = std::current_exception();
            }
        }
    };
    for (int worker = 1; worker < workers; ++worker) {
        threads.emplace_back(run_range, worker);
    }
    run_range(0);
    for (auto& thread : threads) {
        thread.join();
    }
    if (error) {
        std::rethrow_exception(error);
    }
}

template <class Fn>
void batch_parallel_for(
    const BatchFactoredExecutorState& runtime,
    std::size_t items,
    std::size_t min_items_per_worker,
    Fn&& fn) {
    const int workers = batch_parallel_worker_count(runtime, items, min_items_per_worker);
    batch_parallel_for_workers(runtime, workers, items, std::forward<Fn>(fn));
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
