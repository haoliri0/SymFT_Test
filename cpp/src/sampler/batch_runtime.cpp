#include "batch_internal.hpp"

#include <algorithm>

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
#include <immintrin.h>
#endif

namespace symft {

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
    fill_batch_random_half_bits(runtime.eval_scratch, runtime);
    const auto& branch_bits = runtime.eval_scratch;
    assign_batch_symbol(runtime, instruction.branch, branch_bits);
    if (instruction.outcome_plan.conditions.size() == 1 &&
        instruction.outcome_plan.conditions.front() == instruction.branch) {
        if (instruction.outcome_plan.constant) {
            invert_batch_bits(runtime.eval_scratch, runtime);
        }
        write_batch_measurement_record(runtime, instruction.record, runtime.eval_scratch, instruction.record_condition);
        return;
    }
    eval_symbolic_bool_batch(runtime.eval_scratch, instruction.outcome_plan, runtime);
    write_batch_measurement_record(runtime, instruction.record, runtime.eval_scratch, instruction.record_condition);
}

void execute_batch_instruction(BatchFactoredExecutorState& runtime, const FactoredInstruction& instruction) {
    std::visit([&](const auto& inst) { execute_batch_instruction(runtime, inst); }, instruction);
}

namespace {

inline constexpr int kExpensivePureCompactionDenominator = 64;

std::uint64_t live_word_mask_for_shots(int shots, std::size_t word) {
    return low_bits_mask(shots - static_cast<int>(word << 6));
}

void ensure_word_scratch(std::vector<std::uint64_t>& words, std::size_t nwords) {
    if (words.size() < nwords) {
        words.resize(nwords, 0);
    }
}

void xor_records_into(
    std::vector<std::uint64_t>& out,
    const BatchFactoredExecutorState& runtime,
    std::size_t nwords,
    const std::vector<int>& records) {
    ensure_word_scratch(out, runtime.batch_words);
    std::fill(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(nwords), 0);
    for (int record : records) {
        if (record <= 0 || record > runtime.nrecords) {
            fail("detector references an out-of-range measurement record");
        }
        for (std::size_t word = 0; word < nwords; ++word) {
            out[word] ^= runtime.measurement_words[batch_record_offset(runtime, record, word)];
        }
    }
}

int count_live_bits(const std::vector<std::uint64_t>& bits, int shots) {
    int count = 0;
    const std::size_t nwords = batch_word_count(shots);
    for (std::size_t word = 0; word < nwords; ++word) {
        count += detail::popcount64(bits[word] & live_word_mask_for_shots(shots, word));
    }
    return count;
}

template <typename Fn>
void for_each_live_assigned_condition(
    const std::vector<std::uint64_t>& assigned_words,
    const std::vector<int>& condition_last_use,
    int instruction_index,
    Fn&& fn) {
    for (std::size_t word = 0; word < assigned_words.size(); ++word) {
        std::uint64_t bits = assigned_words[word];
        while (bits != 0) {
            const int bit = trailing_zeros64(bits);
            const int condition = static_cast<int>(word << 6) + bit + 1;
            if (condition < static_cast<int>(condition_last_use.size()) &&
                condition_last_use[static_cast<std::size_t>(condition)] > instruction_index) {
                fn(condition);
            }
            bits &= bits - 1;
        }
    }
}

bool instruction_is_pure_over_dead(const ApplyPrecomputedActivePauliRotation&) {
    return true;
}

bool instruction_is_pure_over_dead(const PromoteDormantRotation&) {
    return true;
}

bool instruction_is_pure_over_dead(const RecordMeasurement&) {
    return true;
}

bool instruction_is_pure_over_dead(const MeasureActiveLastZ&) {
    return false;
}

bool instruction_is_pure_over_dead(const MeasurePrecomputedActivePauli&) {
    return false;
}

bool instruction_is_pure_over_dead(const IntroduceDormantMeasurementBranch&) {
    return false;
}

bool instruction_is_pure_over_dead(const FactoredInstruction& instruction) {
    return std::visit([](const auto& inst) { return instruction_is_pure_over_dead(inst); }, instruction);
}

bool instruction_is_expensive_over_dead(const ApplyPrecomputedActivePauliRotation&) {
    return true;
}

bool instruction_is_expensive_over_dead(const PromoteDormantRotation&) {
    return true;
}

bool instruction_is_expensive_over_dead(const RecordMeasurement&) {
    return false;
}

bool instruction_is_expensive_over_dead(const MeasureActiveLastZ&) {
    return true;
}

bool instruction_is_expensive_over_dead(const MeasurePrecomputedActivePauli&) {
    return true;
}

bool instruction_is_expensive_over_dead(const IntroduceDormantMeasurementBranch&) {
    return true;
}

bool instruction_is_expensive_over_dead(const FactoredInstruction& instruction) {
    return std::visit([](const auto& inst) { return instruction_is_expensive_over_dead(inst); }, instruction);
}

bool should_compact_dead_before_instruction(
    const BatchFactoredExecutorState& runtime,
    const BatchDetectorPostselectionScratch& scratch,
    const BatchDetectorPostselectionOptions& options,
    const FactoredInstruction& instruction) {
    if (runtime.active_shots == 0) {
        return false;
    }
    if (scratch.dead_count == 0) {
        return false;
    }
    if (!instruction_is_pure_over_dead(instruction)) {
        return true;
    }
    int denominator = std::max(1, options.mask_dead_shots_min_fraction_denominator);
    if (instruction_is_expensive_over_dead(instruction)) {
        denominator = std::max(denominator, kExpensivePureCompactionDenominator);
    }
    return scratch.dead_count * denominator >= runtime.active_shots;
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

void compact_live_measurement_columns(
    std::vector<std::uint64_t>& columns,
    const std::vector<int>& record_last_use,
    int instruction_index,
    bool include_current_use,
    std::size_t stride_words,
    int old_shots,
    int survivor_count,
    const std::vector<std::uint64_t>& keep_bits,
    std::vector<std::uint64_t>& scratch) {
    if (stride_words == 0 || old_shots == 0 || record_last_use.size() <= 1) {
        return;
    }
    const auto record_is_live = [&](int record) {
        const int last_use = record_last_use[static_cast<std::size_t>(record)];
        return include_current_use ? last_use >= instruction_index : last_use > instruction_index;
    };
    if (old_shots <= 64) {
        const std::uint64_t keep_mask = keep_bits.empty() ? 0 : keep_bits[0] & live_word_mask_for_shots(old_shots, 0);
        for (int record = 1; record < static_cast<int>(record_last_use.size()); ++record) {
            if (!record_is_live(record)) {
                continue;
            }
            const std::size_t base = static_cast<std::size_t>(record - 1) * stride_words;
            columns[base] = compress_bits(columns[base], keep_mask);
            for (std::size_t word = 1; word < stride_words; ++word) {
                columns[base + word] = 0;
            }
        }
        return;
    }
    ensure_word_scratch(scratch, stride_words);
    const std::size_t nwords = batch_word_count(old_shots);
    for (int record = 1; record < static_cast<int>(record_last_use.size()); ++record) {
        if (!record_is_live(record)) {
            continue;
        }
        std::fill(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(stride_words), 0);
        const std::size_t base = static_cast<std::size_t>(record - 1) * stride_words;
        int dest_bit = 0;
        for (std::size_t word = 0; word < nwords; ++word) {
            const std::uint64_t keep_mask = keep_bits[word] & live_word_mask_for_shots(old_shots, word);
            if (keep_mask == 0) {
                continue;
            }
            const int kept = detail::popcount64(keep_mask);
            append_compressed_bits(
                scratch,
                dest_bit,
                compress_bits(columns[base + word], keep_mask),
                kept);
        }
        if (dest_bit != survivor_count) {
            fail("internal measurement compaction count mismatch");
        }
        std::copy_n(scratch.data(), stride_words, columns.data() + base);
    }
}

void compact_live_symbol_columns(
    std::vector<std::uint64_t>& columns,
    const std::vector<std::uint64_t>& assigned_words,
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
        const std::uint64_t keep_mask = keep_bits.empty() ? 0 : keep_bits[0] & live_word_mask_for_shots(old_shots, 0);
        for_each_live_assigned_condition(assigned_words, condition_last_use, instruction_index, [&](int condition) {
            const std::size_t base = static_cast<std::size_t>(condition - 1) * stride_words;
            columns[base] = compress_bits(columns[base], keep_mask);
            for (std::size_t word = 1; word < stride_words; ++word) {
                columns[base + word] = 0;
            }
        });
        return;
    }
    ensure_word_scratch(scratch, stride_words);
    const std::size_t nwords = batch_word_count(old_shots);
    for_each_live_assigned_condition(assigned_words, condition_last_use, instruction_index, [&](int condition) {
        std::fill(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(stride_words), 0);
        const std::size_t base = static_cast<std::size_t>(condition - 1) * stride_words;
        int dest_bit = 0;
        for (std::size_t word = 0; word < nwords; ++word) {
            const std::uint64_t keep_mask = keep_bits[word] & live_word_mask_for_shots(old_shots, word);
            if (keep_mask == 0) {
                continue;
            }
            const int kept = detail::popcount64(keep_mask);
            append_compressed_bits(
                scratch,
                dest_bit,
                compress_bits(columns[base + word], keep_mask),
                kept);
        }
        if (dest_bit != survivor_count) {
            fail("internal symbol compaction count mismatch");
        }
        std::copy_n(scratch.data(), stride_words, columns.data() + base);
    });
}

void compact_live_expression_columns(
    std::vector<std::uint64_t>& columns,
    const std::vector<int>& expression_last_use,
    int instruction_index,
    std::size_t stride_words,
    int old_shots,
    int survivor_count,
    const std::vector<std::uint64_t>& keep_bits,
    std::vector<std::uint64_t>& scratch) {
    const std::size_t expression_count = expression_last_use.size();
    if (stride_words == 0 || old_shots == 0 || expression_count == 0) {
        return;
    }
    if (old_shots <= 64) {
        const std::uint64_t keep_mask = keep_bits.empty() ? 0 : keep_bits[0] & live_word_mask_for_shots(old_shots, 0);
        for (std::size_t expression = 0; expression < expression_count; ++expression) {
            if (expression_last_use[expression] <= instruction_index) {
                continue;
            }
            const std::size_t base = expression * stride_words;
            columns[base] = compress_bits(columns[base], keep_mask);
            for (std::size_t word = 1; word < stride_words; ++word) {
                columns[base + word] = 0;
            }
        }
        return;
    }
    ensure_word_scratch(scratch, stride_words);
    const std::size_t nwords = batch_word_count(old_shots);
    for (std::size_t expression = 0; expression < expression_count; ++expression) {
        if (expression_last_use[expression] <= instruction_index) {
            continue;
        }
        std::fill(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(stride_words), 0);
        const std::size_t base = expression * stride_words;
        int dest_bit = 0;
        for (std::size_t word = 0; word < nwords; ++word) {
            const std::uint64_t keep_mask = keep_bits[word] & live_word_mask_for_shots(old_shots, word);
            if (keep_mask == 0) {
                continue;
            }
            const int kept = detail::popcount64(keep_mask);
            append_compressed_bits(
                scratch,
                dest_bit,
                compress_bits(columns[base + word], keep_mask),
                kept);
        }
        if (dest_bit != survivor_count) {
            fail("internal expression compaction count mismatch");
        }
        std::copy_n(scratch.data(), stride_words, columns.data() + base);
    }
}

void collect_live_sources(
    std::vector<int>& live_sources,
    int old_shots,
    int survivor_count,
    const std::vector<std::uint64_t>& keep_bits) {
    live_sources.resize(static_cast<std::size_t>(survivor_count));
    int dest = 0;
    if (old_shots <= 64) {
        const std::uint64_t keep_mask = keep_bits.empty() ? 0 : keep_bits[0] & live_word_mask_for_shots(old_shots, 0);
        for (int src = 0; src < old_shots; ++src) {
            if ((keep_mask & (std::uint64_t{1} << src)) != 0) {
                live_sources[static_cast<std::size_t>(dest++)] = src;
            }
        }
    } else {
        const std::size_t nwords = batch_word_count(old_shots);
        for (std::size_t word = 0; word < nwords; ++word) {
            std::uint64_t mask = keep_bits[word] & live_word_mask_for_shots(old_shots, word);
            while (mask != 0) {
                const int bit = trailing_zeros64(mask);
                live_sources[static_cast<std::size_t>(dest++)] = static_cast<int>(word << 6) + bit;
                mask &= mask - 1;
            }
        }
    }
    if (dest != survivor_count) {
        fail("internal live source compaction count mismatch");
    }
}

int compacted_active_pitch(int survivor_count, int capacity) {
    if (survivor_count <= 0) {
        return 1;
    }
    constexpr int kActivePitchAlignment = 4;
    const int aligned = ((survivor_count + kActivePitchAlignment - 1) / kActivePitchAlignment) *
                        kActivePitchAlignment;
    return std::min(capacity, aligned);
}

void compact_active_columns(
    BatchFactoredExecutorState& runtime,
    int survivor_count,
    const std::vector<int>& live_sources) {
    const int old_pitch_int = runtime.active_pitch > 0 ? runtime.active_pitch : runtime.batches;
    const int new_pitch_int = compacted_active_pitch(survivor_count, runtime.batches);
    int first_moved = 0;
    while (first_moved < survivor_count &&
           live_sources[static_cast<std::size_t>(first_moved)] == first_moved) {
        ++first_moved;
    }
    if (first_moved == survivor_count && new_pitch_int == old_pitch_int) {
        return;
    }

    const std::size_t old_pitch = static_cast<std::size_t>(old_pitch_int);
    const std::size_t new_pitch = static_cast<std::size_t>(new_pitch_int);
    const std::size_t dim = active_length(runtime.k);
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const std::size_t old_base = basis * old_pitch;
        const std::size_t new_base = basis * new_pitch;
        const int first_dst = new_pitch_int == old_pitch_int ? first_moved : 0;
        for (int dst = first_dst; dst < survivor_count; ++dst) {
            const int src = live_sources[static_cast<std::size_t>(dst)];
            runtime.active_re[new_base + static_cast<std::size_t>(dst)] =
                runtime.active_re[old_base + static_cast<std::size_t>(src)];
            runtime.active_im[new_base + static_cast<std::size_t>(dst)] =
                runtime.active_im[old_base + static_cast<std::size_t>(src)];
        }
    }
    runtime.active_pitch = new_pitch_int;
}

void compact_surviving_shots(
    BatchFactoredExecutorState& runtime,
    const std::vector<std::uint64_t>& keep_bits,
    const std::vector<int>& condition_last_use,
    const std::vector<int>& record_last_use,
    int instruction_index,
    bool include_current_record_use,
    std::vector<std::uint64_t>& scratch,
    std::vector<int>& live_sources,
    std::vector<std::uint64_t>* expression_words,
    const std::vector<int>* expression_last_use) {
    const int old_shots = runtime.active_shots;
    const int survivor_count = count_live_bits(keep_bits, old_shots);
    if (survivor_count == old_shots) {
        return;
    }
    collect_live_sources(live_sources, old_shots, survivor_count, keep_bits);
    compact_active_columns(runtime, survivor_count, live_sources);
    if (expression_words != nullptr) {
        if (expression_last_use == nullptr) {
            fail("internal expression compaction last-use table is missing");
        }
        compact_live_expression_columns(
            *expression_words,
            *expression_last_use,
            instruction_index,
            runtime.batch_words,
            old_shots,
            survivor_count,
            keep_bits,
            scratch);
    }
    compact_live_symbol_columns(
        runtime.value_words,
        runtime.assigned_words,
        condition_last_use,
        instruction_index,
        runtime.batch_words,
        old_shots,
        survivor_count,
        keep_bits,
        scratch);
    compact_live_measurement_columns(
        runtime.measurement_words,
        record_last_use,
        instruction_index,
        include_current_record_use,
        runtime.batch_words,
        old_shots,
        survivor_count,
        keep_bits,
        scratch);
    runtime.active_shots = survivor_count;
}

void compact_dead_shots_if_needed(
    BatchFactoredExecutorState& runtime,
    BatchDetectorPostselectionScratch& scratch,
    const std::vector<int>& condition_last_use,
    const std::vector<int>& record_last_use,
    int instruction_index,
    bool include_current_record_use,
    std::vector<std::uint64_t>* expression_words = nullptr,
    const std::vector<int>* expression_last_use = nullptr) {
    if (runtime.active_shots == 0) {
        return;
    }
    const int dead_count = scratch.dead_count;
    if (dead_count == 0) {
        return;
    }
    const std::size_t nwords = batch_word_count(runtime.active_shots);
    for (std::size_t word = 0; word < nwords; ++word) {
        scratch.keep_bits[word] = (~scratch.dead_bits[word]) & live_word_mask_for_shots(runtime.active_shots, word);
    }
    std::fill(
        scratch.keep_bits.begin() + static_cast<std::ptrdiff_t>(nwords),
        scratch.keep_bits.end(),
        0);
    compact_surviving_shots(
        runtime,
        scratch.keep_bits,
        condition_last_use,
        record_last_use,
        instruction_index,
        include_current_record_use,
        scratch.compact_scratch,
        scratch.live_sources,
        expression_words,
        expression_last_use);
    std::fill(scratch.dead_bits.begin(), scratch.dead_bits.end(), 0);
    scratch.dead_count = 0;
}

int apply_postselection_checks_for_record(
    BatchFactoredExecutorState& runtime,
    const std::vector<std::vector<int>>& detectors,
    BatchDetectorPostselectionScratch& scratch) {
    if (detectors.empty() || runtime.active_shots == 0) {
        return 0;
    }
    const std::size_t nwords = batch_word_count(runtime.active_shots);
    if (nwords == 1) {
        const std::uint64_t live = live_word_mask_for_shots(runtime.active_shots, 0);
        std::uint64_t detector_bits = 0;
        for (const auto& records : detectors) {
            std::uint64_t bits = 0;
            for (int record : records) {
                if (record <= 0 || record > runtime.nrecords) {
                    fail("detector references an out-of-range measurement record");
                }
                bits ^= runtime.measurement_words[batch_record_offset(runtime, record, 0)];
            }
            detector_bits |= bits & live;
        }
        scratch.detector_bits[0] = detector_bits;
        const std::uint64_t newly_dead = detector_bits & ~scratch.dead_bits[0] & live;
        const int discarded_now = detail::popcount64(newly_dead);
        if (discarded_now == 0) {
            return 0;
        }
        scratch.dead_bits[0] |= detector_bits & live;
        scratch.dead_count += discarded_now;
        return discarded_now;
    }
    std::fill(
        scratch.detector_bits.begin(),
        scratch.detector_bits.begin() + static_cast<std::ptrdiff_t>(nwords),
        0);
    for (const auto& records : detectors) {
        xor_records_into(scratch.scratch, runtime, nwords, records);
        for (std::size_t word = 0; word < nwords; ++word) {
            scratch.detector_bits[word] |=
                scratch.scratch[word] & live_word_mask_for_shots(runtime.active_shots, word);
        }
    }
    int discarded_now = 0;
    for (std::size_t word = 0; word < nwords; ++word) {
        const std::uint64_t live = live_word_mask_for_shots(runtime.active_shots, word);
        const std::uint64_t newly_dead = scratch.detector_bits[word] & ~scratch.dead_bits[word] & live;
        discarded_now += detail::popcount64(newly_dead);
        scratch.dead_bits[word] |= scratch.detector_bits[word] & live;
    }
    if (discarded_now == 0) {
        return 0;
    }
    scratch.dead_count += discarded_now;
    return discarded_now;
}

void validate_batch_postselection_plan(
    const FactoredInstructionProgram& program,
    const BatchDetectorPostselectionPlan& postselection) {
    if (postselection.instruction_records_by_index.size() != program.instructions.size()) {
        fail("batch postselection instruction record table does not match program");
    }
    if (postselection.condition_last_use_by_index.size() != static_cast<std::size_t>(program.nsymbols + 1)) {
        fail("batch postselection condition last-use table does not match program");
    }
    if (postselection.record_last_use_by_index.size() != static_cast<std::size_t>(program.nrecords + 1)) {
        fail("batch postselection record last-use table does not match program");
    }
    if (postselection.detectors_by_record.size() < static_cast<std::size_t>(program.nrecords + 1)) {
        fail("batch postselection detector table does not match program");
    }
}

std::uint64_t expression_slice_word(
    const PresampledExpressionBlock& block,
    int block_expression_index,
    int first_sample_shot,
    int active_shots,
    std::size_t dest_word) {
    const int bit_offset = first_sample_shot & 63;
    const std::size_t src_word = static_cast<std::size_t>(first_sample_shot >> 6) + dest_word;
    const std::size_t expression_index = static_cast<std::size_t>(block_expression_index);
    std::uint64_t out = 0;
    if (src_word < block.shot_words) {
        out = block.expression_words[presampled_expression_block_offset(block, expression_index, src_word)] >> bit_offset;
    }
    if (bit_offset != 0 && src_word + 1 < block.shot_words) {
        out |= block.expression_words[presampled_expression_block_offset(block, expression_index, src_word + 1)]
               << (64 - bit_offset);
    }
    return out & live_word_mask_for_shots(active_shots, dest_word);
}

struct BatchExpressionEvaluator {
    const PresampledExpressionPlan& expression_plan;
    const std::vector<std::uint64_t>& expression_words;
    std::size_t expression_stride_words = 0;
    std::vector<std::uint64_t>& out;
    std::vector<std::uint64_t>& residual_scratch;

    const std::vector<std::uint64_t>& eval(
        std::size_t instruction_index,
        const BatchFactoredExecutorState& runtime) const {
        if (instruction_index >= expression_plan.instruction_expressions.size()) {
            fail("batch presampled expression plan does not match program");
        }
        const auto& expression = expression_plan.instruction_expressions[instruction_index];
        if (expression.block_expression_index < 0 ||
            expression.block_expression_index >= static_cast<int>(expression_plan.block_expressions.size())) {
            fail("batch presampled expression references an out-of-range block expression");
        }
        const int block_expression_index = expression.block_expression_index;
        const std::size_t nwords = runtime_batch_word_count(runtime);
        const std::size_t base = static_cast<std::size_t>(block_expression_index) * expression_stride_words;
        if (expression_words.size() < base + expression_stride_words) {
            fail("batch presampled expression workspace is too short");
        }
        if (out.size() < runtime.batch_words) {
            out.resize(runtime.batch_words, 0);
        }
        for (std::size_t word = 0; word < nwords; ++word) {
            out[word] = expression_words[base + word] & live_word_mask_for_shots(runtime.active_shots, word);
        }
        std::fill(out.begin() + static_cast<std::ptrdiff_t>(nwords), out.end(), 0);
        if (!expression.residual_plan.conditions.empty()) {
            eval_symbolic_bool_batch(residual_scratch, expression.residual_plan, runtime);
            for (std::size_t word = 0; word < nwords; ++word) {
                out[word] = (out[word] ^ residual_scratch[word]) & live_word_mask_for_shots(runtime.active_shots, word);
            }
        }
        return out;
    }
};

void execute_batch_instruction_presampled(
    BatchFactoredExecutorState& runtime,
    const ApplyPrecomputedActivePauliRotation& instruction,
    const BatchExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    const auto& sign_bits = evaluator.eval(instruction_index, runtime);
    rotate_pauli_batch(runtime, instruction.rotation_kernel, sign_bits);
}

void execute_batch_instruction_presampled(
    BatchFactoredExecutorState& runtime,
    const PromoteDormantRotation& instruction,
    const BatchExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    const auto& sign_bits = evaluator.eval(instruction_index, runtime);
    promote_first_dormant_rotation_batch(runtime, instruction.theta, sign_bits);
}

void execute_batch_instruction_presampled(
    BatchFactoredExecutorState& runtime,
    const RecordMeasurement& instruction,
    const BatchExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    const auto& outcome_bits = evaluator.eval(instruction_index, runtime);
    write_batch_measurement_record(runtime, instruction.record, outcome_bits, instruction.record_condition);
}

void execute_batch_instruction_presampled(
    BatchFactoredExecutorState& runtime,
    const MeasureActiveLastZ& instruction,
    const BatchExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    measure_active_last_z_branch_batch(runtime, instruction.branch);
    const auto& outcome_bits = evaluator.eval(instruction_index, runtime);
    write_batch_measurement_record(runtime, instruction.record, outcome_bits, instruction.record_condition);
}

void execute_batch_instruction_presampled(
    BatchFactoredExecutorState& runtime,
    const MeasurePrecomputedActivePauli& instruction,
    const BatchExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    measure_precomputed_active_pauli_branch_batch(runtime, instruction.kernel, instruction.branch);
    const auto& outcome_bits = evaluator.eval(instruction_index, runtime);
    write_batch_measurement_record(runtime, instruction.record, outcome_bits, instruction.record_condition);
}

void execute_batch_instruction_presampled(
    BatchFactoredExecutorState& runtime,
    const IntroduceDormantMeasurementBranch& instruction,
    const BatchExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    fill_batch_random_half_bits(runtime.eval_scratch, runtime);
    const auto& branch_bits = runtime.eval_scratch;
    assign_batch_symbol(runtime, instruction.branch, branch_bits);
    const auto& outcome_bits = evaluator.eval(instruction_index, runtime);
    write_batch_measurement_record(runtime, instruction.record, outcome_bits, instruction.record_condition);
}

void execute_batch_instruction_presampled(
    BatchFactoredExecutorState& runtime,
    const FactoredInstruction& instruction,
    const BatchExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    std::visit(
        [&](const auto& inst) {
            execute_batch_instruction_presampled(runtime, inst, evaluator, instruction_index);
        },
        instruction);
}

void initialize_expression_workspace(
    std::vector<std::uint64_t>& expression_words,
    const PresampledExpressionPlan& expression_plan,
    const PresampledExpressionBlock& expression_block,
    std::size_t stride_words,
    int active_shots,
    int first_sample_shot) {
    if (first_sample_shot < 0 || active_shots > expression_block.nshots - first_sample_shot) {
        fail("batch presampled expression source range is out of bounds");
    }
    const std::size_t expression_count = expression_plan.block_expressions.size();
    const std::size_t total_words = expression_count * stride_words;
    if (expression_words.size() < total_words) {
        expression_words.resize(total_words, 0);
    }
    for (std::size_t expression = 0; expression < expression_count; ++expression) {
        const std::size_t base = expression * stride_words;
        for (std::size_t word = 0; word < stride_words; ++word) {
            expression_words[base + word] = expression_slice_word(
                expression_block,
                static_cast<int>(expression),
                first_sample_shot,
                active_shots,
                word);
        }
    }
}

BatchDetectorPostselectionResult execute_batch_postselected_with_expressions(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExpressionPlan& expression_plan,
    const PresampledExpressionBlock& expression_block,
    int first_sample_shot,
    const BatchDetectorPostselectionPlan& postselection,
    BatchDetectorPostselectionScratch& scratch,
    BatchDetectorPostselectionOptions options) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
        fail("batch executor state does not match program");
    }
    if (expression_plan.instruction_expressions.size() != program.instructions.size()) {
        fail("batch presampled expression plan does not match program");
    }
    if (expression_plan.block_expression_last_use_by_index.size() != expression_plan.block_expressions.size()) {
        fail("batch presampled expression last-use table does not match expression plan");
    }
    validate_batch_postselection_plan(program, postselection);
    prepare_batch_detector_postselection_scratch(scratch, runtime);
    std::fill(scratch.dead_bits.begin(), scratch.dead_bits.end(), 0);
    scratch.dead_count = 0;
    if (runtime.active_shots == 0) {
        return {};
    }
    initialize_expression_workspace(
        scratch.expression_words,
        expression_plan,
        expression_block,
        runtime.batch_words,
        runtime.active_shots,
        first_sample_shot);
    int discarded = 0;
    BatchExpressionEvaluator evaluator{
        expression_plan,
        scratch.expression_words,
        runtime.batch_words,
        runtime.eval_scratch,
        scratch.scratch};

    for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
        if (runtime.active_shots == 0) {
            break;
        }
        if (should_compact_dead_before_instruction(runtime, scratch, options, program.instructions[idx])) {
            compact_dead_shots_if_needed(
                runtime,
                scratch,
                postselection.condition_last_use_by_index,
                postselection.record_last_use_by_index,
                static_cast<int>(idx) - 1,
                false,
                &scratch.expression_words,
                &expression_plan.block_expression_last_use_by_index);
            if (runtime.active_shots == 0) {
                break;
            }
        }
        execute_batch_instruction_presampled(runtime, program.instructions[idx], evaluator, idx);
        const int record = postselection.instruction_records_by_index[idx];
        if (record <= 0) {
            continue;
        }
        if (record >= static_cast<int>(postselection.detectors_by_record.size())) {
            fail("batch postselection instruction record table references an out-of-range record");
        }
        const auto& detectors = postselection.detectors_by_record[static_cast<std::size_t>(record)];
        const int discarded_now = apply_postselection_checks_for_record(
            runtime,
            detectors,
            scratch);
        discarded += discarded_now;
    }
    compact_dead_shots_if_needed(
        runtime,
        scratch,
        postselection.condition_last_use_by_index,
        postselection.record_last_use_by_index,
        static_cast<int>(program.instructions.size()),
        true,
        &scratch.expression_words,
        &expression_plan.block_expression_last_use_by_index);
    return {discarded, runtime.active_shots};
}

} // namespace


int default_batch_count(int max_k) {
    const std::size_t dim = active_length(max_k);
    const std::size_t count = std::min<std::size_t>(
        kDefaultBatchShots,
        std::max<std::size_t>(1, kDefaultBatchActiveAmplitudes / dim));
    return static_cast<int>(count);
}

int default_postselected_batch_count(int max_k) {
    constexpr int kPostselectedBatchShots = 256;
    constexpr std::size_t kPostselectedBatchActiveAmplitudes = std::size_t{1} << 18;
    const std::size_t dim = active_length(max_k);
    const std::size_t count = std::min<std::size_t>(
        kPostselectedBatchShots,
        std::max<std::size_t>(1, kPostselectedBatchActiveAmplitudes / dim));
    return static_cast<int>(count);
}

const char* active_batch_backend() {
    return batch_simd::scalar_table().name;
}

void assign_presampled_exogenous_batch_in_place(
    BatchFactoredExecutorState& runtime,
    const PresampledExogenous& samples) {
    assign_presampled_exogenous_batch(runtime, samples);
}

void assign_presampled_exogenous_batch_in_place(
    BatchFactoredExecutorState& runtime,
    const PackedPresampledExogenous& samples,
    int first_sample_shot) {
    assign_presampled_exogenous_batch(runtime, samples, first_sample_shot);
}

void execute_batch_instruction_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstruction& instruction) {
    execute_batch_instruction(runtime, instruction);
}

BatchFactoredExecutorState::BatchFactoredExecutorState(
    const FactoredInstructionProgram& program,
    int batches_,
    std::uint64_t seed)
    : batches(batches_ > 0 ? batches_ : default_batch_count(program.max_k)),
      rng_state(seed) {
    reset_batch_executor(*this, program, batches);
}

void reset_batch_executor(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    int shots,
    bool clear_symbol_values) {
    if (shots < 0 || shots > runtime.batches) {
        fail("active batch shot count is out of range");
    }
    runtime.n = program.n;
    runtime.k = program.initial_k;
    runtime.ndormant = program.n - program.initial_k;
    runtime.active_shots = shots;
    runtime.active_pitch = runtime.batches;
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
    if (clear_symbol_values) {
        std::fill(runtime.value_words.begin(), runtime.value_words.end(), 0);
    }
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

    const std::size_t dim = active_length(program.initial_k);
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const std::size_t base = basis * static_cast<std::size_t>(runtime.active_pitch);
        const double re = basis == 0 ? 1.0 : 0.0;
        for (int shot = 0; shot < runtime.active_shots; ++shot) {
            runtime.active_re[base + static_cast<std::size_t>(shot)] = re;
            runtime.active_im[base + static_cast<std::size_t>(shot)] = 0.0;
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

void execute_batch_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PackedPresampledExogenous& samples,
    int first_sample_shot) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
        fail("batch executor state does not match program");
    }
    if (runtime.active_shots == 0) {
        return;
    }
    assign_presampled_exogenous_batch(runtime, samples, first_sample_shot);
    for (const auto& instruction : program.instructions) {
        execute_batch_instruction(runtime, instruction);
    }
}

void prepare_batch_detector_postselection_scratch(
    BatchDetectorPostselectionScratch& scratch,
    const BatchFactoredExecutorState& runtime) {
    const std::size_t nwords = runtime.batch_words;
    ensure_word_scratch(scratch.detector_bits, nwords);
    ensure_word_scratch(scratch.dead_bits, nwords);
    ensure_word_scratch(scratch.keep_bits, nwords);
    ensure_word_scratch(scratch.scratch, nwords);
    ensure_word_scratch(scratch.compact_scratch, nwords);
    if (scratch.live_sources.capacity() < static_cast<std::size_t>(runtime.batches)) {
        scratch.live_sources.reserve(static_cast<std::size_t>(runtime.batches));
    }
}

BatchDetectorPostselectionResult execute_batch_postselected_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExpressionPlan& expression_plan,
    const PresampledExpressionBlock& expression_block,
    int first_sample_shot,
    const BatchDetectorPostselectionPlan& postselection,
    BatchDetectorPostselectionScratch& scratch,
    BatchDetectorPostselectionOptions options) {
    return execute_batch_postselected_with_expressions(
        runtime,
        program,
        expression_plan,
        expression_block,
        first_sample_shot,
        postselection,
        scratch,
        options);
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
