#include "rate_bench/rate_bench.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <variant>

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
#include <immintrin.h>
#endif

namespace symft_rate_bench {

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
        return std::numeric_limits<std::uint64_t>::max();
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

void accumulate_block_counts(
    RateCounts& counts,
    const symft::BatchFactoredExecutorState& runtime,
    int block,
    const std::vector<symft::StimDetector>& detectors,
    const std::vector<std::vector<int>>& logical_records,
    std::vector<std::uint64_t>& discard_bits,
    std::vector<std::uint64_t>& logical_bits,
    std::vector<std::uint64_t>& scratch) {
    const std::size_t stride_words = runtime.batch_words;
    const std::size_t nwords = batch_word_count(block);
    std::fill(discard_bits.begin(), discard_bits.begin() + static_cast<std::ptrdiff_t>(nwords), 0);
    std::fill(logical_bits.begin(), logical_bits.begin() + static_cast<std::ptrdiff_t>(nwords), 0);

    for (const auto& detector : detectors) {
        xor_records_into(scratch, runtime.measurement_words, stride_words, nwords, detector.records);
        for (std::size_t word = 0; word < nwords; ++word) {
            discard_bits[word] |= scratch[word];
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

std::optional<int> instruction_record(const symft::FactoredInstruction& instruction) {
    return std::visit(
        [](const auto& inst) -> std::optional<int> {
            using T = std::decay_t<decltype(inst)>;
            if constexpr (
                std::is_same_v<T, symft::RecordMeasurement> ||
                std::is_same_v<T, symft::MeasureActiveLastZ> ||
                std::is_same_v<T, symft::MeasurePrecomputedActivePauli> ||
                std::is_same_v<T, symft::IntroduceDormantMeasurementBranch>) {
                return inst.record;
            } else {
                return std::nullopt;
            }
        },
        instruction);
}

std::vector<int> instruction_records(const symft::FactoredInstructionProgram& program) {
    std::vector<int> out;
    out.reserve(program.instructions.size());
    for (const auto& instruction : program.instructions) {
        const auto record = instruction_record(instruction);
        out.push_back(record.value_or(0));
    }
    return out;
}

void mark_condition_uses(
    std::vector<int>& last_use,
    const symft::SymbolicBoolEvaluationPlan& plan,
    int instruction_index) {
    for (int condition : plan.conditions) {
        if (condition > 0 && condition < static_cast<int>(last_use.size())) {
            last_use[static_cast<std::size_t>(condition)] =
                std::max(last_use[static_cast<std::size_t>(condition)], instruction_index);
        }
    }
}

std::vector<int> condition_last_uses(const symft::FactoredInstructionProgram& program) {
    std::vector<int> last_use(static_cast<std::size_t>(program.nsymbols + 1), -1);
    for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
        const int instruction_index = static_cast<int>(idx);
        std::visit(
            [&](const auto& inst) {
                using T = std::decay_t<decltype(inst)>;
                if constexpr (
                    std::is_same_v<T, symft::ApplyPrecomputedActivePauliRotation> ||
                    std::is_same_v<T, symft::PromoteDormantRotation>) {
                    mark_condition_uses(last_use, inst.sign_plan, instruction_index);
                } else if constexpr (
                    std::is_same_v<T, symft::RecordMeasurement> ||
                    std::is_same_v<T, symft::MeasureActiveLastZ> ||
                    std::is_same_v<T, symft::MeasurePrecomputedActivePauli> ||
                    std::is_same_v<T, symft::IntroduceDormantMeasurementBranch>) {
                    mark_condition_uses(last_use, inst.outcome_plan, instruction_index);
                }
            },
            program.instructions[idx]);
    }
    return last_use;
}

std::vector<std::vector<std::vector<int>>> detectors_by_max_record(
    const std::vector<symft::StimDetector>& detectors,
    int nrecords) {
    std::vector<std::vector<std::vector<int>>> out(static_cast<std::size_t>(nrecords + 1));
    for (const auto& detector : detectors) {
        if (detector.records.empty()) {
            continue;
        }
        int max_record = 0;
        for (int record : detector.records) {
            if (record <= 0 || record > nrecords) {
                throw std::runtime_error("detector references an out-of-range measurement record");
            }
            max_record = std::max(max_record, record);
        }
        out[static_cast<std::size_t>(max_record)].push_back(detector.records);
    }
    return out;
}

bool packed_batch_bit(const std::vector<std::uint64_t>& words, int shot) {
    return (words[static_cast<std::size_t>(shot >> 6)] & (std::uint64_t{1} << (shot & 63))) != 0;
}

void set_packed_batch_bit(std::vector<std::uint64_t>& words, int shot) {
    words[static_cast<std::size_t>(shot >> 6)] |= std::uint64_t{1} << (shot & 63);
}

int count_live_bits(const std::vector<std::uint64_t>& bits, int shots) {
    int count = 0;
    const std::size_t nwords = batch_word_count(shots);
    for (std::size_t word = 0; word < nwords; ++word) {
        count += popcount64(bits[word] & live_word_mask(shots, word));
    }
    return count;
}

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386))
__attribute__((target("bmi2")))
std::uint64_t compress_bits_bmi2(std::uint64_t bits, std::uint64_t keep_mask) {
    return _pext_u64(bits, keep_mask);
}
#endif

std::uint64_t compress_bits_portable(std::uint64_t bits, std::uint64_t keep_mask) {
    std::uint64_t out = 0;
    std::uint64_t dest = 1;
    while (keep_mask != 0) {
        const std::uint64_t bit = keep_mask & (~keep_mask + 1);
        if ((bits & bit) != 0) {
            out |= dest;
        }
        keep_mask &= keep_mask - 1;
        dest <<= 1;
    }
    return out;
}

std::uint64_t compress_bits(std::uint64_t bits, std::uint64_t keep_mask) {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386))
    static const bool has_bmi2 = __builtin_cpu_supports("bmi2");
    if (has_bmi2) {
        return compress_bits_bmi2(bits, keep_mask);
    }
#endif
    return compress_bits_portable(bits, keep_mask);
}

void append_compressed_bits(
    std::vector<std::uint64_t>& out,
    int& dest_bit,
    std::uint64_t compressed,
    int nbits) {
    if (nbits == 0) {
        return;
    }
    const std::size_t word = static_cast<std::size_t>(dest_bit >> 6);
    const int shift = dest_bit & 63;
    out[word] |= compressed << shift;
    if (shift != 0 && nbits > 64 - shift) {
        out[word + 1] |= compressed >> (64 - shift);
    }
    dest_bit += nbits;
}

void compact_packed_columns(
    std::vector<std::uint64_t>& columns,
    int column_count,
    std::size_t stride_words,
    int old_shots,
    int survivor_count,
    const std::vector<std::uint64_t>& keep_bits,
    std::vector<std::uint64_t>& scratch) {
    if (column_count <= 0 || stride_words == 0) {
        return;
    }
    if (old_shots <= 64) {
        const std::uint64_t keep_mask = keep_bits.empty() ? 0 : keep_bits[0] & live_word_mask(old_shots, 0);
        for (int column = 0; column < column_count; ++column) {
            const std::size_t base = static_cast<std::size_t>(column) * stride_words;
            columns[base] = compress_bits(columns[base], keep_mask);
            for (std::size_t word = 1; word < stride_words; ++word) {
                columns[base + word] = 0;
            }
        }
        return;
    }
    if (scratch.size() < stride_words) {
        scratch.resize(stride_words, 0);
    }
    const std::size_t nwords = batch_word_count(old_shots);
    for (int column = 0; column < column_count; ++column) {
        std::fill(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(stride_words), 0);
        const std::size_t base = static_cast<std::size_t>(column) * stride_words;
        int dest_bit = 0;
        for (std::size_t word = 0; word < nwords; ++word) {
            const std::uint64_t keep_mask = keep_bits[word] & live_word_mask(old_shots, word);
            if (keep_mask == 0) {
                continue;
            }
            const int kept = popcount64(keep_mask);
            append_compressed_bits(
                scratch,
                dest_bit,
                compress_bits(columns[base + word], keep_mask),
                kept);
        }
        if (dest_bit != survivor_count) {
            throw std::runtime_error("internal survivor compaction count mismatch");
        }
        std::copy_n(scratch.data(), stride_words, columns.data() + base);
    }
}

void compact_live_symbol_columns(
    std::vector<std::uint64_t>& columns,
    const std::vector<int>& condition_last_use,
    int instruction_index,
    std::size_t stride_words,
    int old_shots,
    int survivor_count,
    const std::vector<std::uint64_t>& keep_bits,
    std::vector<std::uint64_t>& scratch) {
    if (stride_words == 0 || old_shots == 0 || condition_last_use.size() <= 1) {
        return;
    }
    if (old_shots <= 64) {
        const std::uint64_t keep_mask = keep_bits.empty() ? 0 : keep_bits[0] & live_word_mask(old_shots, 0);
        for (int condition = 1; condition < static_cast<int>(condition_last_use.size()); ++condition) {
            if (condition_last_use[static_cast<std::size_t>(condition)] <= instruction_index) {
                continue;
            }
            const std::size_t base = static_cast<std::size_t>(condition - 1) * stride_words;
            columns[base] = compress_bits(columns[base], keep_mask);
            for (std::size_t word = 1; word < stride_words; ++word) {
                columns[base + word] = 0;
            }
        }
        return;
    }
    if (scratch.size() < stride_words) {
        scratch.resize(stride_words, 0);
    }
    const std::size_t nwords = batch_word_count(old_shots);
    for (int condition = 1; condition < static_cast<int>(condition_last_use.size()); ++condition) {
        if (condition_last_use[static_cast<std::size_t>(condition)] <= instruction_index) {
            continue;
        }
        std::fill(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(stride_words), 0);
        const std::size_t base = static_cast<std::size_t>(condition - 1) * stride_words;
        int dest_bit = 0;
        for (std::size_t word = 0; word < nwords; ++word) {
            const std::uint64_t keep_mask = keep_bits[word] & live_word_mask(old_shots, word);
            if (keep_mask == 0) {
                continue;
            }
            const int kept = popcount64(keep_mask);
            append_compressed_bits(
                scratch,
                dest_bit,
                compress_bits(columns[base + word], keep_mask),
                kept);
        }
        if (dest_bit != survivor_count) {
            throw std::runtime_error("internal symbol compaction count mismatch");
        }
        std::copy_n(scratch.data(), stride_words, columns.data() + base);
    }
}

void compact_active_columns(
    symft::BatchFactoredExecutorState& runtime,
    int old_shots,
    int survivor_count,
    const std::vector<std::uint64_t>& keep_bits) {
    const std::size_t dim = std::size_t{1} << runtime.k;
    if (old_shots <= 64) {
        const std::uint64_t keep_mask = keep_bits.empty() ? 0 : keep_bits[0] & live_word_mask(old_shots, 0);
        for (std::size_t basis = 0; basis < dim; ++basis) {
            const std::size_t base = basis * static_cast<std::size_t>(runtime.batches);
            int dest = 0;
            for (int src = 0; src < old_shots; ++src) {
                if ((keep_mask & (std::uint64_t{1} << src)) == 0) {
                    continue;
                }
                if (dest != src) {
                    runtime.active_re[base + static_cast<std::size_t>(dest)] =
                        runtime.active_re[base + static_cast<std::size_t>(src)];
                    runtime.active_im[base + static_cast<std::size_t>(dest)] =
                        runtime.active_im[base + static_cast<std::size_t>(src)];
                }
                ++dest;
            }
            if (dest != survivor_count) {
                throw std::runtime_error("internal active compaction count mismatch");
            }
        }
        return;
    }
    const std::size_t nwords = batch_word_count(old_shots);
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const std::size_t base = basis * static_cast<std::size_t>(runtime.batches);
        int dest = 0;
        for (std::size_t word = 0; word < nwords; ++word) {
            std::uint64_t mask = keep_bits[word] & live_word_mask(old_shots, word);
            while (mask != 0) {
                const int bit = __builtin_ctzll(mask);
                const int src = static_cast<int>(word << 6) + bit;
                if (dest != src) {
                    runtime.active_re[base + static_cast<std::size_t>(dest)] =
                        runtime.active_re[base + static_cast<std::size_t>(src)];
                    runtime.active_im[base + static_cast<std::size_t>(dest)] =
                        runtime.active_im[base + static_cast<std::size_t>(src)];
                }
                ++dest;
                mask &= mask - 1;
            }
        }
        if (dest != survivor_count) {
            throw std::runtime_error("internal active compaction count mismatch");
        }
    }
}

void compact_surviving_shots(
    symft::BatchFactoredExecutorState& runtime,
    const std::vector<std::uint64_t>& keep_bits,
    const std::vector<int>& condition_last_use,
    int instruction_index,
    std::vector<std::uint64_t>& scratch) {
    const int old_shots = runtime.active_shots;
    const int survivor_count = count_live_bits(keep_bits, old_shots);
    if (survivor_count == old_shots) {
        return;
    }
    compact_active_columns(runtime, old_shots, survivor_count, keep_bits);
    compact_live_symbol_columns(
        runtime.value_words,
        condition_last_use,
        instruction_index,
        runtime.batch_words,
        old_shots,
        survivor_count,
        keep_bits,
        scratch);
    compact_packed_columns(
        runtime.measurement_words,
        runtime.nrecords,
        runtime.batch_words,
        old_shots,
        survivor_count,
        keep_bits,
        scratch);
    runtime.active_shots = survivor_count;
}

void compact_dead_shots_if_needed(
    symft::BatchFactoredExecutorState& runtime,
    std::vector<std::uint64_t>& dead_bits,
    std::vector<std::uint64_t>& keep_bits,
    const std::vector<int>& condition_last_use,
    int instruction_index,
    std::vector<std::uint64_t>& compact_scratch,
    bool force) {
    if (runtime.active_shots == 0) {
        return;
    }
    const int dead_count = count_live_bits(dead_bits, runtime.active_shots);
    if (dead_count == 0) {
        return;
    }
    if (!force && dead_count != runtime.active_shots && dead_count * 4 < runtime.active_shots) {
        return;
    }
    const std::size_t nwords = batch_word_count(runtime.active_shots);
    for (std::size_t word = 0; word < nwords; ++word) {
        keep_bits[word] = (~dead_bits[word]) & live_word_mask(runtime.active_shots, word);
    }
    std::fill(keep_bits.begin() + static_cast<std::ptrdiff_t>(nwords), keep_bits.end(), 0);
    compact_surviving_shots(runtime, keep_bits, condition_last_use, instruction_index, compact_scratch);
    std::fill(dead_bits.begin(), dead_bits.end(), 0);
}

void apply_postselection_checks_for_record(
    RateCounts& counts,
    symft::BatchFactoredExecutorState& runtime,
    const std::vector<std::vector<int>>& detectors,
    std::vector<std::uint64_t>& detector_bits,
    std::vector<std::uint64_t>& dead_bits,
    std::vector<std::uint64_t>& keep_bits,
    const std::vector<int>& condition_last_use,
    int instruction_index,
    std::vector<std::uint64_t>& scratch,
    std::vector<std::uint64_t>& compact_scratch) {
    if (detectors.empty() || runtime.active_shots == 0) {
        return;
    }
    const std::size_t nwords = batch_word_count(runtime.active_shots);
    std::fill(detector_bits.begin(), detector_bits.begin() + static_cast<std::ptrdiff_t>(nwords), 0);
    for (const auto& records : detectors) {
        xor_records_into(scratch, runtime.measurement_words, runtime.batch_words, nwords, records);
        for (std::size_t word = 0; word < nwords; ++word) {
            detector_bits[word] |= scratch[word] & live_word_mask(runtime.active_shots, word);
        }
    }
    int discarded_now = 0;
    for (std::size_t word = 0; word < nwords; ++word) {
        const std::uint64_t live = live_word_mask(runtime.active_shots, word);
        const std::uint64_t newly_dead = detector_bits[word] & ~dead_bits[word] & live;
        discarded_now += popcount64(newly_dead);
        dead_bits[word] |= detector_bits[word] & live;
    }
    if (discarded_now == 0) {
        return;
    }
    counts.discarded += static_cast<std::uint64_t>(discarded_now);
    compact_dead_shots_if_needed(
        runtime,
        dead_bits,
        keep_bits,
        condition_last_use,
        instruction_index,
        compact_scratch,
        false);
}

void accumulate_logical_counts_for_survivors(
    RateCounts& counts,
    const symft::BatchFactoredExecutorState& runtime,
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

void execute_postselected_block(
    RateCounts& counts,
    symft::BatchFactoredExecutorState& runtime,
    const symft::FactoredInstructionProgram& program,
    const symft::PresampledExogenous& samples,
    const std::vector<int>& instruction_records_by_index,
    const std::vector<int>& condition_last_use,
    const std::vector<std::vector<std::vector<int>>>& detectors_by_record,
    const std::vector<std::vector<int>>& logical_records,
    std::vector<std::uint64_t>& detector_bits,
    std::vector<std::uint64_t>& dead_bits,
    std::vector<std::uint64_t>& keep_bits,
    std::vector<std::uint64_t>& scratch,
    std::vector<std::uint64_t>& compact_scratch) {
    std::fill(dead_bits.begin(), dead_bits.end(), 0);
    symft::assign_presampled_exogenous_batch_in_place(runtime, samples);
    for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
        if (runtime.active_shots == 0) {
            break;
        }
        const auto& instruction = program.instructions[idx];
        symft::execute_batch_instruction_in_place(runtime, instruction);
        const int record = instruction_records_by_index[idx];
        if (record <= 0 || record >= static_cast<int>(detectors_by_record.size())) {
            continue;
        }
        apply_postselection_checks_for_record(
            counts,
            runtime,
            detectors_by_record[static_cast<std::size_t>(record)],
            detector_bits,
            dead_bits,
            keep_bits,
            condition_last_use,
            static_cast<int>(idx),
            scratch,
            compact_scratch);
    }
    compact_dead_shots_if_needed(
        runtime,
        dead_bits,
        keep_bits,
        condition_last_use,
        static_cast<int>(program.instructions.size()),
        compact_scratch,
        true);
    accumulate_logical_counts_for_survivors(counts, runtime, logical_records, keep_bits, scratch);
}

std::uint64_t block_seed(std::uint64_t base, int repeat, std::uint64_t block_index) {
    return base ^
           (std::uint64_t{0x9e3779b97f4a7c15} * (block_index + 1)) ^
           (std::uint64_t{0xbf58476d1ce4e5b9} * static_cast<std::uint64_t>(repeat + 1));
}

std::vector<std::vector<int>> logical_records_for_observable(
    const std::vector<symft::StimObservableInclude>& observables,
    int observable) {
    std::vector<std::vector<int>> out;
    for (const auto& include : observables) {
        if (include.index == observable) {
            out.push_back(include.records);
        }
    }
    return out;
}

bool record_parity(const std::vector<std::uint64_t>& words, const std::vector<int>& records) {
    bool parity = false;
    for (int record : records) {
        if (record <= 0) {
            throw std::runtime_error("record ids must be positive");
        }
        parity ^= symft::packed_bit(words, record - 1);
    }
    return parity;
}

void accumulate_single_counts(
    RateCounts& counts,
    const std::vector<std::uint64_t>& measurement_words,
    const std::vector<symft::StimDetector>& detectors,
    const std::vector<std::vector<int>>& logical_records) {
    bool discarded = false;
    for (const auto& detector : detectors) {
        discarded = discarded || record_parity(measurement_words, detector.records);
    }
    if (discarded) {
        ++counts.discarded;
        return;
    }
    ++counts.accepted;
    bool logical = false;
    for (const auto& records : logical_records) {
        logical ^= record_parity(measurement_words, records);
    }
    if (logical) {
        ++counts.logical_errors;
    }
}

bool any_detector_fires(
    const std::vector<std::uint64_t>& measurement_words,
    const std::vector<std::vector<int>>& detectors) {
    for (const auto& records : detectors) {
        if (record_parity(measurement_words, records)) {
            return true;
        }
    }
    return false;
}

} // namespace symft_rate_bench
