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
    const int denominator = std::max(1, options.dense_over_dead_max_fraction_denominator);
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

void compact_active_columns(
    BatchFactoredExecutorState& runtime,
    int survivor_count,
    const std::vector<int>& live_sources) {
    int first_moved = 0;
    while (first_moved < survivor_count &&
           live_sources[static_cast<std::size_t>(first_moved)] == first_moved) {
        ++first_moved;
    }
    if (first_moved == survivor_count) {
        return;
    }

    const std::size_t pitch = static_cast<std::size_t>(runtime.batches);
    const std::size_t dim = active_length(runtime.k);
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const std::size_t base = basis * pitch;
        for (int dst = first_moved; dst < survivor_count; ++dst) {
            const int src = live_sources[static_cast<std::size_t>(dst)];
            runtime.active_re[base + static_cast<std::size_t>(dst)] =
                runtime.active_re[base + static_cast<std::size_t>(src)];
            runtime.active_im[base + static_cast<std::size_t>(dst)] =
                runtime.active_im[base + static_cast<std::size_t>(src)];
        }
    }
}

bool batch_bit_is_set(const std::vector<std::uint64_t>& bits, int shot) {
    return (bits[batch_shot_word(shot)] & batch_shot_mask(shot)) != 0;
}

bool column_bit_is_set(const std::vector<std::uint64_t>& columns, std::size_t base, int shot) {
    return (columns[base + batch_shot_word(shot)] & batch_shot_mask(shot)) != 0;
}

void copy_column_bit(std::vector<std::uint64_t>& columns, std::size_t base, int dst, int src) {
    const std::size_t dst_word = batch_shot_word(dst);
    const std::uint64_t dst_mask = batch_shot_mask(dst);
    if (column_bit_is_set(columns, base, src)) {
        columns[base + dst_word] |= dst_mask;
    } else {
        columns[base + dst_word] &= ~dst_mask;
    }
}

void collect_tail_fill_moves(
    std::vector<int>& moves,
    int old_shots,
    int survivor_count,
    const std::vector<std::uint64_t>& dead_bits) {
    moves.clear();
    int dst = 0;
    int src = old_shots - 1;
    while (dst < survivor_count) {
        while (dst < survivor_count && !batch_bit_is_set(dead_bits, dst)) {
            ++dst;
        }
        if (dst >= survivor_count) {
            break;
        }
        while (src >= survivor_count && batch_bit_is_set(dead_bits, src)) {
            --src;
        }
        if (src < survivor_count) {
            fail("internal tail-fill compaction source mismatch");
        }
        moves.push_back(dst);
        moves.push_back(src);
        ++dst;
        --src;
    }
}

void tail_fill_active_columns(BatchFactoredExecutorState& runtime, const std::vector<int>& moves) {
    if (moves.empty()) {
        return;
    }
    const std::size_t pitch = static_cast<std::size_t>(runtime.batches);
    const std::size_t dim = active_length(runtime.k);
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const std::size_t base = basis * pitch;
        for (std::size_t move = 0; move < moves.size(); move += 2) {
            const int dst = moves[move];
            const int src = moves[move + 1];
            runtime.active_re[base + static_cast<std::size_t>(dst)] =
                runtime.active_re[base + static_cast<std::size_t>(src)];
            runtime.active_im[base + static_cast<std::size_t>(dst)] =
                runtime.active_im[base + static_cast<std::size_t>(src)];
        }
    }
}

void copy_active_lanes_between_pages(
    BatchFactoredExecutorState& dst,
    int dst0,
    const BatchFactoredExecutorState& src,
    int src0,
    int count) {
    if (count <= 0) {
        return;
    }
    if (dst.k != src.k || dst.batches != src.batches) {
        fail("cannot coalesce batch pages with incompatible active layouts");
    }
    const std::size_t pitch = static_cast<std::size_t>(dst.batches);
    const std::size_t dim = active_length(dst.k);
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const std::size_t base = basis * pitch;
        std::copy_n(
            src.active_re.data() + base + static_cast<std::size_t>(src0),
            count,
            dst.active_re.data() + base + static_cast<std::size_t>(dst0));
        std::copy_n(
            src.active_im.data() + base + static_cast<std::size_t>(src0),
            count,
            dst.active_im.data() + base + static_cast<std::size_t>(dst0));
    }
}

void copy_bit_column_lanes_between_pages(
    std::vector<std::uint64_t>& dst_columns,
    const std::vector<std::uint64_t>& src_columns,
    std::size_t dst_base,
    std::size_t src_base,
    int dst0,
    int src0,
    int count) {
    if (count <= 0) {
        return;
    }
    const std::uint64_t bits = (src_columns[src_base] >> src0) & low_bits_mask(count);
    const std::uint64_t mask = low_bits_mask(count) << dst0;
    dst_columns[dst_base] = (dst_columns[dst_base] & ~mask) | (bits << dst0);
}

void copy_symbol_lanes_between_pages(
    BatchFactoredExecutorState& dst,
    const BatchFactoredExecutorState& src,
    const std::vector<int>& conditions,
    int dst0,
    int src0,
    int count) {
    if (conditions.empty() || count <= 0) {
        return;
    }
    for (int condition : conditions) {
        copy_bit_column_lanes_between_pages(
            dst.value_words,
            src.value_words,
            batch_condition_offset(dst, condition, 0),
            batch_condition_offset(src, condition, 0),
            dst0,
            src0,
            count);
    }
}

void copy_measurement_lanes_between_pages(
    BatchFactoredExecutorState& dst,
    const BatchFactoredExecutorState& src,
    const std::vector<int>& records,
    int dst0,
    int src0,
    int count) {
    if (records.empty() || count <= 0) {
        return;
    }
    for (int record : records) {
        copy_bit_column_lanes_between_pages(
            dst.measurement_words,
            src.measurement_words,
            batch_record_offset(dst, record, 0),
            batch_record_offset(src, record, 0),
            dst0,
            src0,
            count);
    }
}

void move_lanes_between_pages(
    BatchFactoredExecutorState& dst,
    int dst0,
    BatchFactoredExecutorState& src,
    int src0,
    int count,
    const BatchPostselectionBoundaryPlan& transfer_plan) {
    if (count <= 0) {
        return;
    }
    copy_active_lanes_between_pages(dst, dst0, src, src0, count);
    copy_symbol_lanes_between_pages(
        dst,
        src,
        transfer_plan.transfer_conditions,
        dst0,
        src0,
        count);
    copy_measurement_lanes_between_pages(
        dst,
        src,
        transfer_plan.transfer_records,
        dst0,
        src0,
        count);
}

void coalesce_partial_pages(
    std::vector<BatchFactoredExecutorState>& pages,
    const BatchPostselectionBoundaryPlan& transfer_plan) {
    int total_live = 0;
    int nonempty_pages = 0;
    int page_capacity = 0;
    for (const auto& page : pages) {
        if (page_capacity == 0) {
            page_capacity = page.batches;
        }
        if (page.active_shots == 0) {
            continue;
        }
        total_live += page.active_shots;
        ++nonempty_pages;
    }
    if (total_live == 0 || page_capacity <= 0) {
        return;
    }
    const int target_pages = (total_live + page_capacity - 1) / page_capacity;
    const int saved_pages = nonempty_pages - target_pages;
    const int min_saved_pages = std::max(2, static_cast<int>(pages.size()) / 2);
    if (saved_pages < min_saved_pages) {
        return;
    }

    int tail_page = -1;
    for (int page_index = 0; page_index < static_cast<int>(pages.size()); ++page_index) {
        auto& page = pages[static_cast<std::size_t>(page_index)];
        if (page.active_shots == 0) {
            continue;
        }
        if (page.active_shots == page.batches) {
            continue;
        }
        if (tail_page < 0) {
            tail_page = page_index;
            continue;
        }
        auto& dst = pages[static_cast<std::size_t>(tail_page)];
        auto& src = page;
        const int room = dst.batches - dst.active_shots;
        const int moved = std::min(room, src.active_shots);
        const int src0 = src.active_shots - moved;
        const int dst0 = dst.active_shots;
        move_lanes_between_pages(
            dst,
            dst0,
            src,
            src0,
            moved,
            transfer_plan);
        dst.active_shots += moved;
        src.active_shots -= moved;
        if (dst.active_shots == dst.batches) {
            tail_page = src.active_shots > 0 ? page_index : -1;
        } else if (src.active_shots > 0) {
            fail("internal page coalescing did not fill the tail page");
        }
    }
}

BatchPostselectionBoundaryPlan make_batch_postselection_boundary_plan(
    const BatchDetectorPostselectionPlan& postselection,
    int instruction_index,
    bool include_current_record_use) {
    BatchPostselectionBoundaryPlan plan;
    plan.instruction_index = static_cast<std::size_t>(instruction_index);
    if (postselection.condition_last_use_by_index.size() > 1) {
        plan.transfer_conditions.reserve(postselection.condition_last_use_by_index.size() - 1);
        for (int condition = 1;
             condition < static_cast<int>(postselection.condition_last_use_by_index.size());
             ++condition) {
            if (postselection.condition_last_use_by_index[static_cast<std::size_t>(condition)] >
                instruction_index) {
                plan.transfer_conditions.push_back(condition);
            }
        }
    }
    if (postselection.record_last_use_by_index.size() > 1) {
        plan.transfer_records.reserve(postselection.record_last_use_by_index.size() - 1);
        for (int record = 1;
             record < static_cast<int>(postselection.record_last_use_by_index.size());
             ++record) {
            const int last_use = postselection.record_last_use_by_index[static_cast<std::size_t>(record)];
            const bool live = include_current_record_use ? last_use >= instruction_index : last_use > instruction_index;
            if (live) {
                plan.transfer_records.push_back(record);
            }
        }
    }
    return plan;
}

void tail_fill_symbol_columns(
    std::vector<std::uint64_t>& columns,
    const std::vector<int>& condition_last_use,
    int instruction_index,
    std::size_t stride_words,
    const std::vector<int>& moves) {
    if (stride_words == 0 || condition_last_use.size() <= 1 || moves.empty()) {
        return;
    }
    for (int condition = 1; condition < static_cast<int>(condition_last_use.size()); ++condition) {
        if (condition_last_use[static_cast<std::size_t>(condition)] <= instruction_index) {
            continue;
        }
        const std::size_t base = static_cast<std::size_t>(condition - 1) * stride_words;
        for (std::size_t move = 0; move < moves.size(); move += 2) {
            copy_column_bit(columns, base, moves[move], moves[move + 1]);
        }
    }
}

void tail_fill_measurement_columns(
    std::vector<std::uint64_t>& columns,
    const std::vector<int>& record_last_use,
    int instruction_index,
    bool include_current_use,
    std::size_t stride_words,
    const std::vector<int>& moves) {
    if (stride_words == 0 || record_last_use.size() <= 1 || moves.empty()) {
        return;
    }
    const auto record_is_live = [&](int record) {
        const int last_use = record_last_use[static_cast<std::size_t>(record)];
        return include_current_use ? last_use >= instruction_index : last_use > instruction_index;
    };
    for (int record = 1; record < static_cast<int>(record_last_use.size()); ++record) {
        if (!record_is_live(record)) {
            continue;
        }
        const std::size_t base = static_cast<std::size_t>(record - 1) * stride_words;
        for (std::size_t move = 0; move < moves.size(); move += 2) {
            copy_column_bit(columns, base, moves[move], moves[move + 1]);
        }
    }
}

void tail_fill_compact_surviving_shots(
    BatchFactoredExecutorState& runtime,
    const std::vector<std::uint64_t>& dead_bits,
    int dead_count,
    const std::vector<int>& condition_last_use,
    const std::vector<int>& record_last_use,
    int instruction_index,
    bool include_current_record_use,
    std::vector<int>& moves) {
    const int old_shots = runtime.active_shots;
    const int survivor_count = old_shots - dead_count;
    if (survivor_count <= 0) {
        runtime.active_shots = 0;
        return;
    }
    collect_tail_fill_moves(moves, old_shots, survivor_count, dead_bits);
    tail_fill_active_columns(runtime, moves);
    tail_fill_symbol_columns(
        runtime.value_words,
        condition_last_use,
        instruction_index,
        runtime.batch_words,
        moves);
    tail_fill_measurement_columns(
        runtime.measurement_words,
        record_last_use,
        instruction_index,
        include_current_record_use,
        runtime.batch_words,
        moves);
    runtime.active_shots = survivor_count;
}

void compact_surviving_shots(
    BatchFactoredExecutorState& runtime,
    const std::vector<std::uint64_t>& keep_bits,
    const std::vector<int>& condition_last_use,
    const std::vector<int>& record_last_use,
    int instruction_index,
    bool include_current_record_use,
    std::vector<std::uint64_t>& scratch,
    std::vector<int>& live_sources) {
    const int old_shots = runtime.active_shots;
    const int survivor_count = count_live_bits(keep_bits, old_shots);
    if (survivor_count == old_shots) {
        return;
    }
    collect_live_sources(live_sources, old_shots, survivor_count, keep_bits);
    compact_active_columns(runtime, survivor_count, live_sources);
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

bool should_use_tail_fill_compaction(
    const BatchDetectorPostselectionOptions& options,
    int dead_count,
    int active_shots) {
    if (dead_count <= 0 || dead_count >= active_shots) {
        return false;
    }
    if (active_shots <= 64) {
        return false;
    }
    const int denominator = options.unordered_tail_fill_max_dead_fraction_denominator;
    return denominator > 0 && dead_count * denominator <= active_shots;
}

void compact_dead_shots_if_needed(
    BatchFactoredExecutorState& runtime,
    BatchDetectorPostselectionScratch& scratch,
    const std::vector<int>& condition_last_use,
    const std::vector<int>& record_last_use,
    int instruction_index,
    bool include_current_record_use,
    const BatchDetectorPostselectionOptions& options,
    bool force) {
    if (runtime.active_shots == 0) {
        return;
    }
    const int dead_count = scratch.dead_count;
    if (dead_count == 0) {
        return;
    }
    if (!force) {
        return;
    }
    if (should_use_tail_fill_compaction(options, dead_count, runtime.active_shots)) {
        tail_fill_compact_surviving_shots(
            runtime,
            scratch.dead_bits,
            dead_count,
            condition_last_use,
            record_last_use,
            instruction_index,
            include_current_record_use,
            scratch.tail_fill_moves);
        std::fill(scratch.dead_bits.begin(), scratch.dead_bits.end(), 0);
        scratch.dead_count = 0;
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
        scratch.live_sources);
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

void prepare_batch_detector_postselection_boundaries(BatchDetectorPostselectionPlan& postselection) {
    postselection.boundary_plans.clear();
    postselection.boundary_plans.reserve(postselection.instruction_records_by_index.size());
    for (std::size_t idx = 0; idx < postselection.instruction_records_by_index.size(); ++idx) {
        const int record = postselection.instruction_records_by_index[idx];
        if (record <= 0) {
            continue;
        }
        if (record >= static_cast<int>(postselection.detectors_by_record.size())) {
            fail("batch postselection instruction record table references an out-of-range record");
        }
        if (postselection.detectors_by_record[static_cast<std::size_t>(record)].empty()) {
            continue;
        }
        postselection.boundary_plans.push_back(make_batch_postselection_boundary_plan(
            postselection,
            static_cast<int>(idx),
            false));
    }
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
    if (scratch.live_sources.capacity() < static_cast<std::size_t>(runtime.batches)) {
        scratch.live_sources.reserve(static_cast<std::size_t>(runtime.batches));
    }
    if (scratch.tail_fill_moves.capacity() < static_cast<std::size_t>(runtime.batches)) {
        scratch.tail_fill_moves.reserve(static_cast<std::size_t>(runtime.batches));
    }
}

BatchDetectorPostselectionResult execute_batch_postselected_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExogenous& samples,
    const BatchDetectorPostselectionPlan& postselection,
    BatchDetectorPostselectionScratch& scratch,
    BatchDetectorPostselectionOptions options) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
        fail("batch executor state does not match program");
    }
    validate_batch_postselection_plan(program, postselection);
    prepare_batch_detector_postselection_scratch(scratch, runtime);
    std::fill(scratch.dead_bits.begin(), scratch.dead_bits.end(), 0);
    scratch.dead_count = 0;
    if (runtime.active_shots == 0) {
        return {};
    }

    int discarded = 0;
    assign_presampled_exogenous_batch(runtime, samples);
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
                options,
                true);
            if (runtime.active_shots == 0) {
                break;
            }
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
        options,
        true);
    return {discarded, runtime.active_shots};
}

BatchDetectorPostselectionResult execute_batch_postselected_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PackedPresampledExogenous& samples,
    int first_sample_shot,
    const BatchDetectorPostselectionPlan& postselection,
    BatchDetectorPostselectionScratch& scratch,
    BatchDetectorPostselectionOptions options) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
        fail("batch executor state does not match program");
    }
    validate_batch_postselection_plan(program, postselection);
    prepare_batch_detector_postselection_scratch(scratch, runtime);
    std::fill(scratch.dead_bits.begin(), scratch.dead_bits.end(), 0);
    scratch.dead_count = 0;
    if (runtime.active_shots == 0) {
        return {};
    }

    int discarded = 0;
    assign_presampled_exogenous_batch(runtime, samples, first_sample_shot);
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
                options,
                true);
            if (runtime.active_shots == 0) {
                break;
            }
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
        options,
        true);
    return {discarded, runtime.active_shots};
}

BatchDetectorPostselectionResult execute_batch_postselected_page_pool_in_place(
    std::vector<BatchFactoredExecutorState>& pages,
    const FactoredInstructionProgram& program,
    const BatchDetectorPostselectionPlan& postselection,
    std::vector<BatchDetectorPostselectionScratch>& scratches,
    BatchDetectorPostselectionOptions options) {
    validate_batch_postselection_plan(program, postselection);
    if (scratches.size() < pages.size()) {
        scratches.resize(pages.size());
    }
    for (std::size_t page_index = 0; page_index < pages.size(); ++page_index) {
        auto& page = pages[page_index];
        if (page.n != program.n || page.k + page.ndormant != page.n) {
            fail("batch page executor state does not match program");
        }
        if (page.batches != 64 || page.batch_words != 1) {
            fail("paged batch postselection requires 64-shot physical pages");
        }
        prepare_batch_detector_postselection_scratch(scratches[page_index], page);
        std::fill(scratches[page_index].dead_bits.begin(), scratches[page_index].dead_bits.end(), 0);
        scratches[page_index].dead_count = 0;
    }

    std::vector<BatchPostselectionBoundaryPlan> local_boundary_plans;
    const auto* boundary_plans = &postselection.boundary_plans;
    if (boundary_plans->empty()) {
        local_boundary_plans.reserve(program.instructions.size());
        for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
            const int record = postselection.instruction_records_by_index[idx];
            if (record <= 0) {
                continue;
            }
            if (record >= static_cast<int>(postselection.detectors_by_record.size())) {
                fail("batch postselection instruction record table references an out-of-range record");
            }
            if (!postselection.detectors_by_record[static_cast<std::size_t>(record)].empty()) {
                local_boundary_plans.push_back(make_batch_postselection_boundary_plan(
                    postselection,
                    static_cast<int>(idx),
                    false));
            }
        }
        boundary_plans = &local_boundary_plans;
    }

    int discarded = 0;
    std::size_t segment_start = 0;
    for (std::size_t boundary_pos = 0; boundary_pos < boundary_plans->size(); ++boundary_pos) {
        const auto& boundary_plan = (*boundary_plans)[boundary_pos];
        const std::size_t boundary_idx = boundary_plan.instruction_index;
        bool discarded_at_boundary = false;
        for (std::size_t page_index = 0; page_index < pages.size(); ++page_index) {
            auto& page = pages[page_index];
            if (page.active_shots == 0) {
                continue;
            }
            auto& scratch = scratches[page_index];
            for (std::size_t idx = segment_start; idx <= boundary_idx; ++idx) {
                execute_batch_instruction(page, program.instructions[idx]);
            }
            const int record = postselection.instruction_records_by_index[boundary_idx];
            if (record >= static_cast<int>(postselection.detectors_by_record.size())) {
                fail("batch postselection instruction record table references an out-of-range record");
            }
            const auto& detectors = postselection.detectors_by_record[static_cast<std::size_t>(record)];
            const int discarded_now = apply_postselection_checks_for_record(page, detectors, scratch);
            discarded += discarded_now;
            discarded_at_boundary = discarded_at_boundary || discarded_now != 0;
        }
        if (discarded_at_boundary) {
            for (std::size_t page_index = 0; page_index < pages.size(); ++page_index) {
                auto& page = pages[page_index];
                if (page.active_shots == 0) {
                    continue;
                }
                compact_dead_shots_if_needed(
                    page,
                    scratches[page_index],
                    postselection.condition_last_use_by_index,
                    postselection.record_last_use_by_index,
                    static_cast<int>(boundary_idx),
                    false,
                    options,
                    true);
            }
            coalesce_partial_pages(
                pages,
                boundary_plan);
        }
        segment_start = boundary_idx + 1;
    }

    if (segment_start < program.instructions.size()) {
        for (std::size_t page_index = 0; page_index < pages.size(); ++page_index) {
            auto& page = pages[page_index];
            if (page.active_shots == 0) {
                continue;
            }
            for (std::size_t idx = segment_start; idx < program.instructions.size(); ++idx) {
                execute_batch_instruction(page, program.instructions[idx]);
            }
        }
    }

    int accepted = 0;
    for (std::size_t page_index = 0; page_index < pages.size(); ++page_index) {
        auto& page = pages[page_index];
        compact_dead_shots_if_needed(
            page,
            scratches[page_index],
            postselection.condition_last_use_by_index,
            postselection.record_last_use_by_index,
            static_cast<int>(program.instructions.size()),
            true,
            options,
            true);
        accepted += page.active_shots;
    }
    return {discarded, accepted};
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
