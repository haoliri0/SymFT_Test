#include "sampler/prepared_sampler.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

namespace symft {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_between(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double>(stop - start).count();
}

std::uint64_t ceil_div_u64(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) {
        throw std::runtime_error("division by zero in chunk sizing");
    }
    return numerator / denominator + (numerator % denominator != 0 ? 1 : 0);
}

std::uint64_t block_seed(std::uint64_t base, std::uint64_t stream_id, std::uint64_t block_index) {
    return base ^
           (std::uint64_t{0x9e3779b97f4a7c15} * (block_index + 1)) ^
           (std::uint64_t{0xbf58476d1ce4e5b9} * (stream_id + 1));
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

std::size_t batch_word_count(int shots) {
    return shots <= 0 ? 0 : static_cast<std::size_t>((shots + 63) >> 6);
}

std::uint64_t low_bits_mask(int nbits) {
    if (nbits <= 0) {
        return 0;
    }
    if (nbits >= 64) {
        return ~std::uint64_t{0};
    }
    return (std::uint64_t{1} << nbits) - 1;
}

std::uint64_t live_word_mask(int shots, std::size_t word) {
    return low_bits_mask(shots - static_cast<int>(word << 6));
}

std::size_t measurement_offset(std::size_t stride_words, int record, std::size_t word) {
    if (record <= 0) {
        throw std::runtime_error("record ids must be positive");
    }
    return static_cast<std::size_t>(record - 1) * stride_words + word;
}

void xor_records_into(
    std::vector<std::uint64_t>& out,
    const std::vector<std::uint64_t>& measurements,
    std::size_t stride_words,
    std::size_t nwords,
    const std::vector<int>& records) {
    std::fill(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(nwords), 0);
    for (int record : records) {
        for (std::size_t word = 0; word < nwords; ++word) {
            out[word] ^= measurements[measurement_offset(stride_words, record, word)];
        }
    }
}

bool record_parity(const std::vector<std::uint64_t>& words, const std::vector<int>& records) {
    bool parity = false;
    for (int record : records) {
        if (record <= 0) {
            throw std::runtime_error("record ids must be positive");
        }
        parity ^= packed_bit(words, record - 1);
    }
    return parity;
}

void accumulate_accepted_single_counts(
    CircuitSamplingCounts& counts,
    const std::vector<std::uint64_t>& measurement_words,
    const std::vector<std::vector<int>>& logical_records) {
    ++counts.accepted;
    bool logical = false;
    for (const auto& records : logical_records) {
        logical ^= record_parity(measurement_words, records);
    }
    if (logical) {
        ++counts.logical_errors;
    }
}

void accumulate_single_counts(
    CircuitSamplingCounts& counts,
    const std::vector<std::uint64_t>& measurement_words,
    const std::vector<std::uint64_t>& detector_words,
    int ndetectors,
    const std::vector<std::vector<int>>& logical_records) {
    for (int detector = 0; detector < ndetectors; ++detector) {
        if (packed_bit(detector_words, detector)) {
            ++counts.discarded;
            return;
        }
    }
    accumulate_accepted_single_counts(counts, measurement_words, logical_records);
}

void accumulate_block_counts(
    CircuitSamplingCounts& counts,
    const BatchFactoredExecutorState& runtime,
    int block,
    const std::vector<std::vector<int>>& logical_records,
    std::vector<std::uint64_t>& discard_bits,
    std::vector<std::uint64_t>& logical_bits,
    std::vector<std::uint64_t>& scratch) {
    const std::size_t stride_words = runtime.batch_words;
    const std::size_t nwords = batch_word_count(block);
    std::fill(discard_bits.begin(), discard_bits.begin() + static_cast<std::ptrdiff_t>(nwords), 0);
    std::fill(logical_bits.begin(), logical_bits.begin() + static_cast<std::ptrdiff_t>(nwords), 0);

    for (int detector = 1; detector <= runtime.ndetectors; ++detector) {
        const std::size_t base = static_cast<std::size_t>(detector - 1) * stride_words;
        for (std::size_t word = 0; word < nwords; ++word) {
            discard_bits[word] |= runtime.detector_words[base + word];
        }
    }

    for (const auto& records : logical_records) {
        xor_records_into(scratch, runtime.measurement_words, stride_words, nwords, records);
        for (std::size_t word = 0; word < nwords; ++word) {
            logical_bits[word] ^= scratch[word];
        }
    }

    for (std::size_t word = 0; word < nwords; ++word) {
        const std::uint64_t live = live_word_mask(block, word);
        const std::uint64_t discarded_word = discard_bits[word] & live;
        const std::uint64_t accepted_word = (~discarded_word) & live;
        counts.discarded += static_cast<std::uint64_t>(popcount64(discarded_word));
        counts.accepted += static_cast<std::uint64_t>(popcount64(accepted_word));
        counts.logical_errors += static_cast<std::uint64_t>(popcount64(logical_bits[word] & accepted_word));
    }
}

void accumulate_logical_counts_for_survivors(
    CircuitSamplingCounts& counts,
    const BatchFactoredExecutorState& runtime,
    const std::vector<std::vector<int>>& logical_records,
    std::vector<std::uint64_t>& logical_bits,
    std::vector<std::uint64_t>& scratch) {
    counts.accepted += static_cast<std::uint64_t>(runtime.active_shots);
    if (runtime.active_shots == 0) {
        return;
    }
    const std::size_t nwords = batch_word_count(runtime.active_shots);
    std::fill(logical_bits.begin(), logical_bits.begin() + static_cast<std::ptrdiff_t>(nwords), 0);
    for (const auto& records : logical_records) {
        xor_records_into(scratch, runtime.measurement_words, runtime.batch_words, nwords, records);
        for (std::size_t word = 0; word < nwords; ++word) {
            logical_bits[word] ^= scratch[word] & live_word_mask(runtime.active_shots, word);
        }
    }
    for (std::size_t word = 0; word < nwords; ++word) {
        counts.logical_errors += static_cast<std::uint64_t>(
            popcount64(logical_bits[word] & live_word_mask(runtime.active_shots, word)));
    }
}

CircuitSamplingInfo make_info(
    const FactoredInstructionProgram& program,
    int observable,
    int observable_includes,
    const CircuitSamplingOptions& options,
    int batch_size,
    int sample_chunk_shots,
    int threads) {
    CircuitSamplingInfo info;
    info.n = program.n;
    info.records = program.nrecords;
    info.detectors = program.ndetectors;
    info.observable_includes = observable_includes;
    info.observable = observable;
    info.max_k = program.max_k;
    info.batch_size = batch_size;
    info.sample_chunk_shots = sample_chunk_shots;
    info.threads = threads;
    info.detector_postselection = options.postselect_detectors;
    info.batch_mask_threshold_denominator =
        options.postselect_detectors ? options.batch_mask_threshold_denominator : 0;
    return info;
}

int sample_chunk_or_default(int requested) {
    return requested > 0 ? requested : default_single_shot_sample_chunk_shots();
}

int batch_size_or_default(
    int requested,
    const FactoredInstructionProgram& program,
    bool postselect_detectors) {
    if (requested > 0) {
        return requested;
    }
    return postselect_detectors
               ? default_postselected_batch_count(program.max_k)
               : default_batch_count(program.max_k);
}

std::size_t assigned_condition_count(const std::vector<std::uint64_t>& assigned_words) {
    std::size_t out = 0;
    for (std::uint64_t word : assigned_words) {
        out += static_cast<std::size_t>(popcount64(word));
    }
    return out;
}

bool introduce_branch_outcome_is_direct(const IntroduceDormantMeasurementBranch& instruction) {
    return instruction.outcome_plan.conditions.size() == 1 &&
           instruction.outcome_plan.conditions.front() == instruction.branch;
}

std::size_t direct_symbolic_eval_word_terms(const FactoredInstruction& instruction) {
    return std::visit(
        [](const auto& inst) -> std::size_t {
            using T = std::decay_t<decltype(inst)>;
            if constexpr (
                std::is_same_v<T, ApplyPrecomputedActivePauliRotation> ||
                std::is_same_v<T, PromoteDormantRotation>) {
                return 1 + inst.sign_plan.conditions.size();
            } else if constexpr (std::is_same_v<T, IntroduceDormantMeasurementBranch>) {
                return introduce_branch_outcome_is_direct(inst) ? 0 : 1 + inst.outcome_plan.conditions.size();
            } else {
                return 1 + inst.outcome_plan.conditions.size();
            }
        },
        instruction);
}

std::size_t presampled_expression_eval_word_terms(
    const FactoredInstruction& instruction,
    const PresampledExpression& expression) {
    return std::visit(
        [&](const auto& inst) -> std::size_t {
            using T = std::decay_t<decltype(inst)>;
            if constexpr (std::is_same_v<T, RecordDetector>) {
                if (!inst.records.empty()) {
                    return inst.records.size();
                }
                if (inst.outcome.conditions.empty()) {
                    return 1;
                }
            } else if constexpr (std::is_same_v<T, IntroduceDormantMeasurementBranch>) {
                if (introduce_branch_outcome_is_direct(inst)) {
                    return 0;
                }
            }
            return 1 + expression.residual_plan.conditions.size();
        },
        instruction);
}

std::size_t presampled_expression_chunk_word_terms(const PresampledExpressionPlan& expression_plan) {
    std::size_t terms = 0;
    for (const auto& expression : expression_plan.block_expressions) {
        terms += 1;
        if (expression.parent_block_expression_index >= 0) {
            terms += expression.parent_delta_constant ? 1 : 0;
            terms += expression.parent_delta_exogenous_conditions.size();
        } else {
            terms += expression.exogenous_conditions.size();
        }
    }
    return terms;
}

bool should_use_presampled_batch_expressions(
    const FactoredInstructionProgram& program,
    const PresampledExpressionPlan& expression_plan,
    const PackedPresampledExogenous& samples,
    int sample_chunk_shots,
    int batch_size) {
    const int typical_block = std::min(sample_chunk_shots, batch_size);
    if (typical_block <= 0) {
        return false;
    }
    const std::uint64_t blocks_per_chunk = ceil_div_u64(
        static_cast<std::uint64_t>(sample_chunk_shots),
        static_cast<std::uint64_t>(batch_size));
    if (blocks_per_chunk <= 1) {
        return false;
    }
    const std::size_t block_words = batch_word_count(typical_block);
    const std::size_t chunk_words = batch_word_count(sample_chunk_shots);

    std::size_t direct_terms = assigned_condition_count(samples.exogenous_assigned_words);
    for (const auto& instruction : program.instructions) {
        direct_terms += direct_symbolic_eval_word_terms(instruction);
    }

    std::size_t presampled_block_terms = expression_plan.block_expressions.size();
    for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
        presampled_block_terms += presampled_expression_eval_word_terms(
            program.instructions[idx],
            expression_plan.instruction_expressions[idx]);
    }

    const std::uint64_t direct_cost =
        blocks_per_chunk * static_cast<std::uint64_t>(block_words) * static_cast<std::uint64_t>(direct_terms);
    const std::uint64_t presampled_cost =
        static_cast<std::uint64_t>(chunk_words) *
            static_cast<std::uint64_t>(presampled_expression_chunk_word_terms(expression_plan)) +
        blocks_per_chunk * static_cast<std::uint64_t>(block_words) *
            static_cast<std::uint64_t>(presampled_block_terms);

    return static_cast<long double>(presampled_cost) * 4.0L <
           static_cast<long double>(direct_cost);
}

} // namespace

std::vector<std::vector<int>> logical_records_for_observable(
    const std::vector<CircuitObservableInclude>& observables,
    int observable) {
    std::vector<std::vector<int>> out;
    for (const auto& include : observables) {
        if (include.index == observable) {
            out.push_back(include.records);
        }
    }
    return out;
}

CircuitSamplingInput make_circuit_sampling_input(
    FactoredInstructionProgram program,
    std::vector<std::vector<int>> logical_records,
    int observable,
    int observable_includes,
    CircuitSamplingTiming preprocessing_timing) {
    CircuitSamplingInput input;
    input.program = std::move(program);
    input.logical_records = std::move(logical_records);
    input.observable = observable;
    input.observable_includes = observable_includes;
    input.preprocessing_timing = preprocessing_timing;
    return input;
}

PreparedCircuitSingleShotSampler::PreparedCircuitSingleShotSampler(
    FactoredInstructionProgram program,
    std::vector<std::vector<int>> logical_records,
    CircuitSamplingOptions options)
    : PreparedCircuitSingleShotSampler(
          make_circuit_sampling_input(
              std::move(program),
              std::move(logical_records),
              options.observable),
          options) {}

PreparedCircuitSingleShotSampler::PreparedCircuitSingleShotSampler(
    CircuitSamplingInput input,
    CircuitSamplingOptions options)
    : options_(options) {
    program_ = std::move(input.program);
    logical_records_ = std::move(input.logical_records);
    preprocessing_timing_ = input.preprocessing_timing;

    options_.sample_chunk_shots = sample_chunk_or_default(options_.sample_chunk_shots);
    info_ = make_info(
        program_,
        input.observable,
        input.observable_includes,
        options_,
        0,
        options_.sample_chunk_shots,
        1);
    runtime_.emplace(program_);
    prepare_presampled_exogenous_packed(samples_, program_);
    prepare_presampled_expression_plan(expression_plan_, program_, samples_);
}

CircuitSamplingRunResult PreparedCircuitSingleShotSampler::sample(std::uint64_t shots) {
    return sample(shots, next_stream_id_++);
}

CircuitSamplingRunResult PreparedCircuitSingleShotSampler::sample(
    std::uint64_t shots,
    std::uint64_t stream_id) {
    CircuitSamplingRunResult result;
    result.active_threads = 1;
    auto& runtime = *runtime_;
    const std::uint64_t nchunks = ceil_div_u64(
        shots,
        static_cast<std::uint64_t>(options_.sample_chunk_shots));

    const auto sample_start = Clock::now();
    for (std::uint64_t chunk_index = 0; chunk_index < nchunks; ++chunk_index) {
        const std::uint64_t offset = chunk_index * static_cast<std::uint64_t>(options_.sample_chunk_shots);
        const int chunk = static_cast<int>(std::min<std::uint64_t>(
            static_cast<std::uint64_t>(options_.sample_chunk_shots),
            shots - offset));

        const auto presample_start = Clock::now();
        resample_prepared_exogenous_packed_in_place(
            samples_,
            program_,
            chunk,
            block_seed(0x7eed0000ULL, stream_id, chunk_index));
        evaluate_presampled_expression_block(expression_block_, expression_plan_, samples_);
        const auto presample_stop = Clock::now();

        const auto execute_start = Clock::now();
        runtime.rng_state = block_seed(0x5eed1234ULL, stream_id, chunk_index);
        for (int shot = 0; shot < chunk; ++shot) {
            reset_executor(runtime, program_, !options_.postselect_detectors);
            if (options_.postselect_detectors) {
                const bool survived = execute_postselected_in_place(
                    runtime,
                    program_,
                    expression_plan_,
                    expression_block_,
                    shot);
                if (!survived) {
                    ++result.counts.discarded;
                } else {
                    accumulate_accepted_single_counts(
                        result.counts,
                        runtime.measurement_words,
                        logical_records_);
                }
                continue;
            }
            execute_in_place(runtime, program_, expression_plan_, expression_block_, shot);
            accumulate_single_counts(
                result.counts,
                runtime.measurement_words,
                runtime.detector_words,
                program_.ndetectors,
                logical_records_);
        }
        result.timing.execute_s += seconds_between(execute_start, Clock::now());
        result.counts.shots += static_cast<std::uint64_t>(chunk);
        result.timing.presample_s += seconds_between(presample_start, presample_stop);
    }
    result.timing.sample_s = seconds_between(sample_start, Clock::now());
    return result;
}

struct PreparedCircuitBatchSampler::WorkerContext {
    CircuitSamplingCounts counts;
    CircuitSamplingTiming timing;
    BatchFactoredExecutorState runtime;
    std::vector<std::uint64_t> discard_bits;
    std::vector<std::uint64_t> logical_bits;
    std::vector<std::uint64_t> scratch;
    BatchDetectorPostselectionScratch postselection_scratch;
    PackedPresampledExogenous samples;
    PresampledExpressionPlan expression_plan;
    PresampledExpressionBlock expression_block;

    WorkerContext(
        const FactoredInstructionProgram& program,
        int batch_size,
        std::uint64_t seed)
        : runtime(program, batch_size, seed),
          discard_bits(runtime.batch_words, 0),
          logical_bits(runtime.batch_words, 0),
          scratch(runtime.batch_words, 0) {}
};

PreparedCircuitBatchSampler::PreparedCircuitBatchSampler(
    FactoredInstructionProgram program,
    std::vector<std::vector<int>> logical_records,
    CircuitSamplingOptions options)
    : PreparedCircuitBatchSampler(
          make_circuit_sampling_input(
              std::move(program),
              std::move(logical_records),
              options.observable),
          options) {}

PreparedCircuitBatchSampler::PreparedCircuitBatchSampler(
    CircuitSamplingInput input,
    CircuitSamplingOptions options)
    : options_(options) {
    program_ = std::move(input.program);
    logical_records_ = std::move(input.logical_records);
    preprocessing_timing_ = input.preprocessing_timing;

    options_.batch_size = batch_size_or_default(
        options_.batch_size,
        program_,
        options_.postselect_detectors);
    options_.sample_chunk_shots = sample_chunk_or_default(options_.sample_chunk_shots);
    options_.threads = std::max(1, options_.threads);
    postselection_options_.mask_dead_shots_min_fraction_denominator =
        options_.batch_mask_threshold_denominator;
    postselection_options_.retained_record_uses = &logical_records_;

    PackedPresampledExogenous selector_samples;
    PresampledExpressionPlan selector_expression_plan;
    prepare_presampled_exogenous_packed(selector_samples, program_);
    prepare_presampled_expression_plan(selector_expression_plan, program_, selector_samples);
    use_presampled_batch_expressions_ =
        options_.postselect_detectors ||
        should_use_presampled_batch_expressions(
            program_,
            selector_expression_plan,
            selector_samples,
            options_.sample_chunk_shots,
            options_.batch_size);

    info_ = make_info(
        program_,
        input.observable,
        input.observable_includes,
        options_,
        options_.batch_size,
        options_.sample_chunk_shots,
        options_.threads);

    workers_.reserve(static_cast<std::size_t>(options_.threads));
    for (int worker_id = 0; worker_id < options_.threads; ++worker_id) {
        auto context = std::make_unique<WorkerContext>(
            program_,
            options_.batch_size,
            block_seed(0x5eed1234ULL, 0, static_cast<std::uint64_t>(worker_id)));
        if (options_.postselect_detectors) {
            prepare_batch_detector_postselection_scratch(
                context->postselection_scratch,
                context->runtime,
                program_,
                postselection_options_);
        }
        prepare_presampled_exogenous_packed(context->samples, program_);
        if (use_presampled_batch_expressions_) {
            prepare_presampled_expression_plan(context->expression_plan, program_, context->samples);
        }
        workers_.push_back(std::move(context));
    }
}

PreparedCircuitBatchSampler::~PreparedCircuitBatchSampler() = default;
PreparedCircuitBatchSampler::PreparedCircuitBatchSampler(PreparedCircuitBatchSampler&&) noexcept = default;
PreparedCircuitBatchSampler& PreparedCircuitBatchSampler::operator=(PreparedCircuitBatchSampler&&) noexcept = default;

CircuitSamplingRunResult PreparedCircuitBatchSampler::sample(std::uint64_t shots) {
    return sample(shots, next_stream_id_++);
}

CircuitSamplingRunResult PreparedCircuitBatchSampler::sample(
    std::uint64_t shots,
    std::uint64_t stream_id) {
    CircuitSamplingRunResult result;
    const std::uint64_t nchunks = ceil_div_u64(
        shots,
        static_cast<std::uint64_t>(options_.sample_chunk_shots));
    const std::uint64_t blocks_per_chunk = ceil_div_u64(
        static_cast<std::uint64_t>(options_.sample_chunk_shots),
        static_cast<std::uint64_t>(options_.batch_size));
    const int active_threads = std::min<int>(
        options_.threads,
        static_cast<int>(std::max<std::uint64_t>(1, nchunks)));
    result.active_threads = active_threads;

    const auto sample_start = Clock::now();
    auto run_worker = [&](int worker_id) {
        auto& context = *workers_[static_cast<std::size_t>(worker_id)];
        context.counts = {};
        context.timing = {};
        for (std::uint64_t chunk_index = static_cast<std::uint64_t>(worker_id);
             chunk_index < nchunks;
             chunk_index += static_cast<std::uint64_t>(active_threads)) {
            const std::uint64_t chunk_offset =
                chunk_index * static_cast<std::uint64_t>(options_.sample_chunk_shots);
            const int chunk_shots = static_cast<int>(std::min<std::uint64_t>(
                static_cast<std::uint64_t>(options_.sample_chunk_shots),
                shots - chunk_offset));

            const auto presample_start = Clock::now();
            resample_prepared_exogenous_packed_in_place(
                context.samples,
                program_,
                chunk_shots,
                block_seed(0x7eed0000ULL, stream_id, chunk_index));
            if (use_presampled_batch_expressions_) {
                evaluate_presampled_expression_block(
                    context.expression_block,
                    context.expression_plan,
                    context.samples);
            }
            const auto presample_stop = Clock::now();
            context.timing.presample_s += seconds_between(presample_start, presample_stop);

            const auto execute_start = Clock::now();
            for (int chunk_local_offset = 0, local_block_index = 0;
                 chunk_local_offset < chunk_shots;
                 chunk_local_offset += options_.batch_size, ++local_block_index) {
                const int block = std::min(options_.batch_size, chunk_shots - chunk_local_offset);
                const std::uint64_t block_index =
                    chunk_index * blocks_per_chunk + static_cast<std::uint64_t>(local_block_index);
                reset_batch_executor(
                    context.runtime,
                    program_,
                    block,
                    !options_.postselect_detectors);
                context.runtime.rng_state = block_seed(0x5eed1234ULL, stream_id, block_index);
                if (options_.postselect_detectors) {
                    const auto postselection_result = execute_batch_postselected_in_place(
                        context.runtime,
                        program_,
                        context.expression_plan,
                        context.expression_block,
                        chunk_local_offset,
                        context.postselection_scratch,
                        postselection_options_);
                    context.counts.discarded +=
                        static_cast<std::uint64_t>(postselection_result.discarded);
                } else {
                    if (use_presampled_batch_expressions_) {
                        execute_batch_in_place(
                            context.runtime,
                            program_,
                            context.expression_plan,
                            context.expression_block,
                            chunk_local_offset);
                    } else {
                        execute_batch_in_place(
                            context.runtime,
                            program_,
                            context.samples,
                            chunk_local_offset);
                    }
                }
                if (options_.postselect_detectors) {
                    accumulate_logical_counts_for_survivors(
                        context.counts,
                        context.runtime,
                        logical_records_,
                        context.logical_bits,
                        context.scratch);
                } else {
                    accumulate_block_counts(
                        context.counts,
                        context.runtime,
                        block,
                        logical_records_,
                        context.discard_bits,
                        context.logical_bits,
                        context.scratch);
                }
                context.counts.shots += static_cast<std::uint64_t>(block);
            }
            context.timing.execute_s += seconds_between(execute_start, Clock::now());
        }
    };

    if (active_threads == 1) {
        run_worker(0);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(active_threads));
        for (int worker_id = 0; worker_id < active_threads; ++worker_id) {
            threads.emplace_back(run_worker, worker_id);
        }
        for (auto& worker : threads) {
            worker.join();
        }
    }

    for (int worker_id = 0; worker_id < active_threads; ++worker_id) {
        const auto& context = workers_[static_cast<std::size_t>(worker_id)];
        result.counts.shots += context->counts.shots;
        result.counts.discarded += context->counts.discarded;
        result.counts.accepted += context->counts.accepted;
        result.counts.logical_errors += context->counts.logical_errors;
        result.timing.presample_s += context->timing.presample_s;
        result.timing.execute_s += context->timing.execute_s;
        result.timing.accumulate_s += context->timing.accumulate_s;
    }
    result.timing.sample_s = seconds_between(sample_start, Clock::now());
    return result;
}

} // namespace symft
