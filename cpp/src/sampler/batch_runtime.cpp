#include "batch_internal.hpp"

#include <algorithm>
#include <utility>

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
    ensure_word_scratch(scratch, stride_words);
    const std::size_t nwords = batch_word_count(old_shots);
    for (int condition = 1; condition < static_cast<int>(condition_last_use.size()); ++condition) {
        if (condition_last_use[static_cast<std::size_t>(condition)] <= instruction_index) {
            continue;
        }
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
    }
}

void compact_active_columns(
    BatchFactoredExecutorState& runtime,
    int old_shots,
    int survivor_count,
    const std::vector<std::uint64_t>& keep_bits) {
    const std::size_t dim = active_length(runtime.k);
    if (old_shots <= 64) {
        const std::uint64_t keep_mask = keep_bits.empty() ? 0 : keep_bits[0] & live_word_mask_for_shots(old_shots, 0);
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
                fail("internal active compaction count mismatch");
            }
        }
        return;
    }
    const std::size_t nwords = batch_word_count(old_shots);
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const std::size_t base = basis * static_cast<std::size_t>(runtime.batches);
        int dest = 0;
        for (std::size_t word = 0; word < nwords; ++word) {
            std::uint64_t mask = keep_bits[word] & live_word_mask_for_shots(old_shots, word);
            while (mask != 0) {
                const int bit = trailing_zeros64(mask);
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
            fail("internal active compaction count mismatch");
        }
    }
}

void compact_surviving_shots(
    BatchFactoredExecutorState& runtime,
    const std::vector<std::uint64_t>& keep_bits,
    const std::vector<int>& condition_last_use,
    const std::vector<int>& record_last_use,
    int instruction_index,
    bool include_current_record_use,
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
    std::vector<std::uint64_t>& dead_bits,
    std::vector<std::uint64_t>& keep_bits,
    const std::vector<int>& condition_last_use,
    const std::vector<int>& record_last_use,
    int instruction_index,
    bool include_current_record_use,
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
        keep_bits[word] = (~dead_bits[word]) & live_word_mask_for_shots(runtime.active_shots, word);
    }
    std::fill(keep_bits.begin() + static_cast<std::ptrdiff_t>(nwords), keep_bits.end(), 0);
    compact_surviving_shots(
        runtime,
        keep_bits,
        condition_last_use,
        record_last_use,
        instruction_index,
        include_current_record_use,
        compact_scratch);
    std::fill(dead_bits.begin(), dead_bits.end(), 0);
}

int apply_postselection_checks_for_record(
    BatchFactoredExecutorState& runtime,
    const std::vector<std::vector<int>>& detectors,
    BatchDetectorPostselectionScratch& scratch,
    const std::vector<int>& condition_last_use,
    const std::vector<int>& record_last_use,
    int instruction_index) {
    if (detectors.empty() || runtime.active_shots == 0) {
        return 0;
    }
    const std::size_t nwords = batch_word_count(runtime.active_shots);
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
    compact_dead_shots_if_needed(
        runtime,
        scratch.dead_bits,
        scratch.keep_bits,
        condition_last_use,
        record_last_use,
        instruction_index,
        false,
        scratch.compact_scratch,
        false);
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

} // namespace


int default_batch_count(int max_k) {
    const std::size_t dim = active_length(max_k);
    const std::size_t count = std::min<std::size_t>(
        kDefaultBatchShots,
        std::max<std::size_t>(1, kDefaultBatchActiveAmplitudes / dim));
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

    const std::size_t dim = active_length(program.initial_k);
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const std::size_t base = basis * static_cast<std::size_t>(runtime.batches);
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
}

BatchDetectorPostselectionResult execute_batch_postselected_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExogenous& samples,
    const BatchDetectorPostselectionPlan& postselection,
    BatchDetectorPostselectionScratch& scratch) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
        fail("batch executor state does not match program");
    }
    validate_batch_postselection_plan(program, postselection);
    prepare_batch_detector_postselection_scratch(scratch, runtime);
    std::fill(scratch.dead_bits.begin(), scratch.dead_bits.end(), 0);
    if (runtime.active_shots == 0) {
        return {};
    }

    int discarded = 0;
    assign_presampled_exogenous_batch(runtime, samples);
    for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
        if (runtime.active_shots == 0) {
            break;
        }
        execute_batch_instruction(runtime, program.instructions[idx]);
        const int record = postselection.instruction_records_by_index[idx];
        if (record <= 0) {
            continue;
        }
        if (record >= static_cast<int>(postselection.detectors_by_record.size())) {
            fail("batch postselection instruction record table references an out-of-range record");
        }
        const auto& detectors = postselection.detectors_by_record[static_cast<std::size_t>(record)];
        discarded += apply_postselection_checks_for_record(
            runtime,
            detectors,
            scratch,
            postselection.condition_last_use_by_index,
            postselection.record_last_use_by_index,
            static_cast<int>(idx));
    }
    compact_dead_shots_if_needed(
        runtime,
        scratch.dead_bits,
        scratch.keep_bits,
        postselection.condition_last_use_by_index,
        postselection.record_last_use_by_index,
        static_cast<int>(program.instructions.size()),
        true,
        scratch.compact_scratch,
        true);
    return {discarded, runtime.active_shots};
}

BatchDetectorPostselectionResult execute_batch_postselected_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PackedPresampledExogenous& samples,
    int first_sample_shot,
    const BatchDetectorPostselectionPlan& postselection,
    BatchDetectorPostselectionScratch& scratch) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
        fail("batch executor state does not match program");
    }
    validate_batch_postselection_plan(program, postselection);
    prepare_batch_detector_postselection_scratch(scratch, runtime);
    std::fill(scratch.dead_bits.begin(), scratch.dead_bits.end(), 0);
    if (runtime.active_shots == 0) {
        return {};
    }

    int discarded = 0;
    assign_presampled_exogenous_batch(runtime, samples, first_sample_shot);
    for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
        if (runtime.active_shots == 0) {
            break;
        }
        execute_batch_instruction(runtime, program.instructions[idx]);
        const int record = postselection.instruction_records_by_index[idx];
        if (record <= 0) {
            continue;
        }
        if (record >= static_cast<int>(postselection.detectors_by_record.size())) {
            fail("batch postselection instruction record table references an out-of-range record");
        }
        const auto& detectors = postselection.detectors_by_record[static_cast<std::size_t>(record)];
        discarded += apply_postselection_checks_for_record(
            runtime,
            detectors,
            scratch,
            postselection.condition_last_use_by_index,
            postselection.record_last_use_by_index,
            static_cast<int>(idx));
    }
    compact_dead_shots_if_needed(
        runtime,
        scratch.dead_bits,
        scratch.keep_bits,
        postselection.condition_last_use_by_index,
        postselection.record_last_use_by_index,
        static_cast<int>(program.instructions.size()),
        true,
        scratch.compact_scratch,
        true);
    return {discarded, runtime.active_shots};
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
