#include "symft/symft.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <variant>
#include <vector>

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
#include <immintrin.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct RateCounts {
    std::uint64_t shots = 0;
    std::uint64_t discarded = 0;
    std::uint64_t accepted = 0;
    std::uint64_t logical_errors = 0;
};

struct WorkerResult {
    RateCounts counts;
    double presample_s = 0.0;
    double execute_s = 0.0;
    double accumulate_s = 0.0;
};

enum class SamplerMode {
    Single,
    Batch,
    Both,
};

struct BenchResult {
    std::string path;
    std::string sampler;
    std::uint64_t requested_shots = 0;
    RateCounts counts;
    int n = 0;
    int records = 0;
    int detectors = 0;
    int observable_includes = 0;
    int observable = 0;
    int max_k = 0;
    int block_shots = 0;
    int repeats = 1;
    int threads = 1;
    int requested_threads = 1;
    bool detector_postselection = false;
    std::string exogenous_mode;
    std::string rng_streams = "split_exogenous_branch";
    std::string phase_timing = "wall";
    double parse_s = 0.0;
    double plan_s = 0.0;
    double presample_s = 0.0;
    double execute_s = 0.0;
    double accumulate_s = 0.0;
    double sample_s = 0.0;
};

double seconds_between(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double>(stop - start).count();
}

std::uint64_t parse_u64(const char* raw, const char* name) {
    if (raw[0] == '-') {
        throw std::runtime_error(std::string(name) + " must be nonnegative");
    }
    errno = 0;
    char* end = nullptr;
    const auto value = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' || errno == ERANGE) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + raw);
    }
    return static_cast<std::uint64_t>(value);
}

int parse_positive_int(const char* raw, const char* name) {
    const std::uint64_t value = parse_u64(raw, name);
    if (value == 0 || value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + " must be in 1:Int32Max");
    }
    return static_cast<int>(value);
}

int parse_nonnegative_int(const char* raw, const char* name) {
    const std::uint64_t value = parse_u64(raw, name);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + " must fit in Int32Max");
    }
    return static_cast<int>(value);
}

bool parse_bool_flag(const char* raw, const char* name) {
    const std::string value(raw);
    if (value == "1" || value == "true" || value == "yes" || value == "on" || value == "postselect") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off" || value == "none") {
        return false;
    }
    throw std::runtime_error(std::string("invalid ") + name + ": " + raw);
}

int parse_block_shots(const char* raw) {
    if (std::string(raw) == "auto") {
        return 0;
    }
    return parse_positive_int(raw, "block_shots");
}

int parse_threads(const char* raw) {
    const std::string value(raw);
    if (value == "auto") {
        const unsigned int detected = std::thread::hardware_concurrency();
        return std::max(1, static_cast<int>(detected == 0 ? 1 : detected));
    }
    return parse_positive_int(raw, "threads");
}

SamplerMode parse_sampler_mode(const char* raw) {
    const std::string value(raw);
    if (value == "single" || value == "single-shot" || value == "single_shot") {
        return SamplerMode::Single;
    }
    if (value == "batch" || value == "batched") {
        return SamplerMode::Batch;
    }
    if (value == "both" || value == "all") {
        return SamplerMode::Both;
    }
    throw std::runtime_error(std::string("invalid sampler: ") + raw);
}

struct Options {
    std::string path = "d3.stim";
    std::uint64_t shots = 100000000ULL;
    int block_shots = 0;
    int repeats = 1;
    int observable = 0;
    int threads = 1;
    bool postselect_detectors = false;
    SamplerMode sampler = SamplerMode::Batch;
};

void print_usage(const char* argv0) {
    std::cerr
        << "usage: " << argv0 << " [circuit.stim] [shots] [batch_size] [repeats] [observable] [postselect_detectors]\n"
        << "       " << argv0 << " --circuit d5.stim --shots 1000000 --batch-size 64 --postselect-detectors\n"
        << "       " << argv0 << " --sampler both --circuit d5.stim --shots 1000000 --batch-size auto\n"
        << "\n"
        << "options:\n"
        << "  --circuit PATH, --file PATH        Stim circuit path\n"
        << "  --shots N                         Number of shots\n"
        << "  --sampler single|batch|both       Sampler to benchmark; default: batch\n"
        << "  --batch-size N|auto               Batch size; --block-shots is accepted as an alias\n"
        << "  --repeats N                       Number of repeats\n"
        << "  --observable N                    Observable index\n"
        << "  --threads N|auto                  Parallel worker threads over independent batches\n"
        << "  --postselect-detectors            Abort and compact shots after fired detectors\n"
        << "  --no-postselect-detectors         Disable detector postselection abort\n";
}

std::string require_option_value(
    const std::string& option,
    const std::string& inline_value,
    int& idx,
    int argc,
    char** argv) {
    if (!inline_value.empty()) {
        return inline_value;
    }
    if (idx + 1 >= argc) {
        throw std::runtime_error(option + " requires a value");
    }
    return argv[++idx];
}

void apply_positional_option(Options& options, int position, const char* value) {
    switch (position) {
    case 0:
        options.path = value;
        break;
    case 1:
        options.shots = parse_u64(value, "shots");
        break;
    case 2:
        options.block_shots = parse_block_shots(value);
        break;
    case 3:
        options.repeats = parse_positive_int(value, "repeats");
        break;
    case 4:
        options.observable = parse_nonnegative_int(value, "observable");
        break;
    case 5:
        options.postselect_detectors = parse_bool_flag(value, "postselect_detectors");
        break;
    default:
        throw std::runtime_error("too many positional arguments");
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    int positional = 0;
    for (int idx = 1; idx < argc; ++idx) {
        std::string arg(argv[idx]);
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg.rfind("--", 0) != 0) {
            apply_positional_option(options, positional++, argv[idx]);
            continue;
        }

        std::string name = arg;
        std::string value;
        const std::size_t eq = arg.find('=');
        if (eq != std::string::npos) {
            name = arg.substr(0, eq);
            value = arg.substr(eq + 1);
        }

        if (name == "--circuit" || name == "--file") {
            options.path = require_option_value(name, value, idx, argc, argv);
        } else if (name == "--shots") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.shots = parse_u64(raw.c_str(), "shots");
        } else if (name == "--sampler") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.sampler = parse_sampler_mode(raw.c_str());
        } else if (name == "--batch-size" || name == "--block-shots") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.block_shots = parse_block_shots(raw.c_str());
        } else if (name == "--repeats") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.repeats = parse_positive_int(raw.c_str(), "repeats");
        } else if (name == "--observable") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.observable = parse_nonnegative_int(raw.c_str(), "observable");
        } else if (name == "--threads") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.threads = parse_threads(raw.c_str());
        } else if (name == "--postselect-detectors" || name == "--detector-postselection") {
            options.postselect_detectors = value.empty() ? true : parse_bool_flag(value.c_str(), "postselect_detectors");
        } else if (name == "--no-postselect-detectors") {
            options.postselect_detectors = false;
        } else {
            throw std::runtime_error("unknown option: " + name);
        }
    }
    return options;
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

BenchResult run_single_sampler(const Options& options) {
    BenchResult result;
    result.path = options.path;
    result.sampler = "single";
    result.requested_shots = options.shots;
    result.repeats = options.repeats;
    result.observable = options.observable;
    result.threads = 1;
    result.requested_threads = options.threads;
    result.detector_postselection = options.postselect_detectors;
    result.exogenous_mode = "presampled_streaming";
    result.phase_timing = "wall";

    double parse_total = 0.0;
    double plan_total = 0.0;
    double presample_total = 0.0;
    double execute_total = 0.0;
    double accumulate_total = 0.0;
    double sample_wall_total = 0.0;

    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        const auto parse_start = Clock::now();
        const auto parsed = symft::parse_stim_file(options.path);
        const auto parse_stop = Clock::now();

        const auto plan_start = Clock::now();
        symft::PendingFactoredState pending(parsed.state);
        const auto program = symft::plan_factored_updates(pending);
        const auto plan_stop = Clock::now();

        result.n = parsed.state.n;
        result.records = program.nrecords;
        result.max_k = program.max_k;
        result.detectors = static_cast<int>(parsed.detectors.size());
        result.observable_includes = static_cast<int>(parsed.observables.size());
        result.block_shots = options.block_shots > 0 ? options.block_shots : symft::default_batch_count(program.max_k);

        const auto logical_records = logical_records_for_observable(parsed.observables, options.observable);
        const std::uint64_t nblocks =
            (options.shots + static_cast<std::uint64_t>(result.block_shots) - 1) /
            static_cast<std::uint64_t>(result.block_shots);
        symft::FactoredExecutorState runtime(program, block_seed(0x5eed1234ULL, repeat, 0));

        const auto sample_wall_start = Clock::now();
        RateCounts counts;
        for (std::uint64_t block_index = 0; block_index < nblocks; ++block_index) {
            const std::uint64_t offset = block_index * static_cast<std::uint64_t>(result.block_shots);
            const int block = static_cast<int>(std::min<std::uint64_t>(
                static_cast<std::uint64_t>(result.block_shots),
                options.shots - offset));

            const auto presample_start = Clock::now();
            const auto samples = symft::presample_exogenous(
                program,
                block,
                block_seed(0x7eed0000ULL, repeat, block_index));
            const auto presample_stop = Clock::now();

            runtime.rng_state = block_seed(0x5eed1234ULL, repeat, block_index);
            for (int shot = 0; shot < block; ++shot) {
                const auto execute_start = Clock::now();
                symft::reset_executor(runtime, program);
                symft::execute_in_place(runtime, program, samples, shot);
                const auto execute_stop = Clock::now();
                accumulate_single_counts(counts, runtime.measurement_words, parsed.detectors, logical_records);
                const auto accumulate_stop = Clock::now();
                execute_total += seconds_between(execute_start, execute_stop);
                accumulate_total += seconds_between(execute_stop, accumulate_stop);
            }
            counts.shots += static_cast<std::uint64_t>(block);
            presample_total += seconds_between(presample_start, presample_stop);
        }
        const auto sample_wall_stop = Clock::now();

        result.counts.shots += counts.shots;
        result.counts.discarded += counts.discarded;
        result.counts.accepted += counts.accepted;
        result.counts.logical_errors += counts.logical_errors;
        parse_total += seconds_between(parse_start, parse_stop);
        plan_total += seconds_between(plan_start, plan_stop);
        sample_wall_total += seconds_between(sample_wall_start, sample_wall_stop);
    }

    const double inv_repeats = 1.0 / static_cast<double>(options.repeats);
    result.parse_s = parse_total * inv_repeats;
    result.plan_s = plan_total * inv_repeats;
    result.presample_s = presample_total * inv_repeats;
    result.execute_s = execute_total * inv_repeats;
    result.accumulate_s = accumulate_total * inv_repeats;
    result.sample_s = sample_wall_total * inv_repeats;
    return result;
}

BenchResult run_batch_sampler(const Options& options) {
    BenchResult result;
    result.path = options.path;
    result.requested_shots = options.shots;
    result.repeats = options.repeats;
    result.observable = options.observable;
    result.requested_threads = options.threads;
    result.detector_postselection = options.postselect_detectors;
    result.exogenous_mode = "presampled_streaming";
    result.rng_streams = "split_exogenous_branch";

    double parse_total = 0.0;
    double plan_total = 0.0;
    double presample_total = 0.0;
    double execute_total = 0.0;
    double accumulate_total = 0.0;
    double sample_wall_total = 0.0;
    int active_threads_used = 1;

    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        const auto parse_start = Clock::now();
        const auto parsed = symft::parse_stim_file(options.path);
        const auto parse_stop = Clock::now();

        const auto plan_start = Clock::now();
        symft::PendingFactoredState pending(parsed.state);
        const auto program = symft::plan_factored_updates(pending);
        const auto plan_stop = Clock::now();

        result.n = parsed.state.n;
        result.records = program.nrecords;
        result.max_k = program.max_k;
        result.block_shots = options.block_shots > 0 ? options.block_shots : symft::default_batch_count(program.max_k);
        result.detectors = static_cast<int>(parsed.detectors.size());
        result.observable_includes = static_cast<int>(parsed.observables.size());

        const auto logical_records = logical_records_for_observable(parsed.observables, options.observable);
        const auto postselection_detectors_by_record = detectors_by_max_record(parsed.detectors, program.nrecords);
        const auto instruction_records_by_index = instruction_records(program);
        const auto condition_last_use_by_index = condition_last_uses(program);

        const std::uint64_t nblocks =
            (options.shots + static_cast<std::uint64_t>(result.block_shots) - 1) /
            static_cast<std::uint64_t>(result.block_shots);
        const int active_threads = std::min<int>(
            options.threads,
            static_cast<int>(std::max<std::uint64_t>(1, nblocks)));
        active_threads_used = std::max(active_threads_used, active_threads);
        std::vector<WorkerResult> worker_results(static_cast<std::size_t>(active_threads));

        const auto sample_wall_start = Clock::now();
        auto run_worker = [&](int worker_id) {
            WorkerResult local;
            symft::BatchFactoredExecutorState runtime(
                program,
                result.block_shots,
                block_seed(0x5eed1234ULL, repeat, static_cast<std::uint64_t>(worker_id)));
            std::vector<std::uint64_t> discard_bits(runtime.batch_words, 0);
            std::vector<std::uint64_t> dead_bits(runtime.batch_words, 0);
            std::vector<std::uint64_t> logical_bits(runtime.batch_words, 0);
            std::vector<std::uint64_t> scratch(runtime.batch_words, 0);
            std::vector<std::uint64_t> compact_scratch(runtime.batch_words, 0);
            for (std::uint64_t block_index = static_cast<std::uint64_t>(worker_id);
                 block_index < nblocks;
                 block_index += static_cast<std::uint64_t>(active_threads)) {
                const std::uint64_t offset = block_index * static_cast<std::uint64_t>(result.block_shots);
                const int block = static_cast<int>(std::min<std::uint64_t>(
                    static_cast<std::uint64_t>(result.block_shots),
                    options.shots - offset));

                const auto presample_start = Clock::now();
                const auto samples = symft::presample_exogenous(
                    program,
                    block,
                    block_seed(0x7eed0000ULL, repeat, block_index));
                const auto presample_stop = Clock::now();

                symft::reset_batch_executor(runtime, program, block);
                runtime.rng_state = block_seed(0x5eed1234ULL, repeat, block_index);
                if (options.postselect_detectors) {
                    execute_postselected_block(
                        local.counts,
                        runtime,
                        program,
                        samples,
                        instruction_records_by_index,
                        condition_last_use_by_index,
                        postselection_detectors_by_record,
                        logical_records,
                        discard_bits,
                        dead_bits,
                        logical_bits,
                        scratch,
                        compact_scratch);
                } else {
                    symft::execute_batch_in_place(runtime, program, samples);
                }
                const auto execute_stop = Clock::now();
                if (!options.postselect_detectors) {
                    accumulate_block_counts(
                        local.counts,
                        runtime,
                        block,
                        parsed.detectors,
                        logical_records,
                        discard_bits,
                        logical_bits,
                        scratch);
                }
                const auto accumulate_stop = Clock::now();
                local.counts.shots += static_cast<std::uint64_t>(block);
                local.presample_s += seconds_between(presample_start, presample_stop);
                local.execute_s += seconds_between(presample_stop, execute_stop);
                local.accumulate_s += seconds_between(execute_stop, accumulate_stop);
            }
            worker_results[static_cast<std::size_t>(worker_id)] = local;
        };
        if (active_threads == 1) {
            run_worker(0);
        } else {
            std::vector<std::thread> workers;
            workers.reserve(static_cast<std::size_t>(active_threads));
            for (int worker_id = 0; worker_id < active_threads; ++worker_id) {
                workers.emplace_back(run_worker, worker_id);
            }
            for (auto& worker : workers) {
                worker.join();
            }
        }
        const auto sample_wall_stop = Clock::now();

        for (const auto& worker : worker_results) {
            result.counts.shots += worker.counts.shots;
            result.counts.discarded += worker.counts.discarded;
            result.counts.accepted += worker.counts.accepted;
            result.counts.logical_errors += worker.counts.logical_errors;
            presample_total += worker.presample_s;
            execute_total += worker.execute_s;
            accumulate_total += worker.accumulate_s;
        }
        parse_total += seconds_between(parse_start, parse_stop);
        plan_total += seconds_between(plan_start, plan_stop);
        sample_wall_total += seconds_between(sample_wall_start, sample_wall_stop);
    }

    const double inv_repeats = 1.0 / static_cast<double>(options.repeats);
    result.sampler = options.postselect_detectors ? "batch_postselected" : "batch";
    result.threads = active_threads_used;
    result.phase_timing = active_threads_used > 1 ? "worker_sum" : "wall";
    result.parse_s = parse_total * inv_repeats;
    result.plan_s = plan_total * inv_repeats;
    result.presample_s = presample_total * inv_repeats;
    result.execute_s = execute_total * inv_repeats;
    result.accumulate_s = accumulate_total * inv_repeats;
    result.sample_s = sample_wall_total * inv_repeats;
    return result;
}

void print_result(const BenchResult& result) {
    const double shots_per_s = result.sample_s > 0.0 ? static_cast<double>(result.requested_shots) / result.sample_s : 0.0;
    const double discard_rate = result.counts.shots == 0
                                    ? std::numeric_limits<double>::quiet_NaN()
                                    : static_cast<double>(result.counts.discarded) / static_cast<double>(result.counts.shots);
    const double logical_rate = result.counts.accepted == 0
                                    ? std::numeric_limits<double>::quiet_NaN()
                                    : static_cast<double>(result.counts.logical_errors) / static_cast<double>(result.counts.accepted);

    std::cout << "file " << result.path << "\n";
    std::cout << "qubits " << result.n << "\n";
    std::cout << "records " << result.records << "\n";
    std::cout << "detectors " << result.detectors << "\n";
    std::cout << "observable_includes " << result.observable_includes << "\n";
    std::cout << "observable " << result.observable << "\n";
    std::cout << "max_active_qubits " << result.max_k << "\n";
    std::cout << "simd_backend " << symft::active_simd_backend() << "\n";
    std::cout << "batch_backend " << symft::active_batch_backend() << "\n";
    std::cout << "shots " << result.requested_shots << "\n";
    std::cout << "sampled_shots " << result.counts.shots << "\n";
    std::cout << "batch_size " << result.block_shots << "\n";
    std::cout << "block_shots " << result.block_shots << "\n";
    std::cout << "repeats " << result.repeats << "\n";
    std::cout << "threads " << result.threads << "\n";
    std::cout << "requested_threads " << result.requested_threads << "\n";
    std::cout << "sampler " << result.sampler << "\n";
    std::cout << "detector_postselection " << (result.detector_postselection ? "enabled" : "disabled") << "\n";
    std::cout << "exogenous_mode " << result.exogenous_mode << "\n";
    std::cout << "rng_streams " << result.rng_streams << "\n";
    std::cout << "phase_timing " << result.phase_timing << "\n";
    std::cout << "parse_s_avg " << result.parse_s << "\n";
    std::cout << "plan_s_avg " << result.plan_s << "\n";
    std::cout << "presample_s_avg " << result.presample_s << "\n";
    std::cout << "execute_s_avg " << result.execute_s << "\n";
    std::cout << "accumulate_s_avg " << result.accumulate_s << "\n";
    std::cout << "sample_wall_s_avg " << result.sample_s << "\n";
    std::cout << "sample_s_avg " << result.sample_s << "\n";
    std::cout << "sample_shots_per_s " << shots_per_s << "\n";
    std::cout << "discarded " << result.counts.discarded << "\n";
    std::cout << "accepted " << result.counts.accepted << "\n";
    std::cout << "logical_errors " << result.counts.logical_errors << "\n";
    std::cout << "discard_rate " << discard_rate << "\n";
    std::cout << "logical_error_rate " << logical_rate << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.sampler == SamplerMode::Single) {
            print_result(run_single_sampler(options));
        } else if (options.sampler == SamplerMode::Batch) {
            print_result(run_batch_sampler(options));
        } else {
            print_result(run_single_sampler(options));
            std::cout << "\n";
            print_result(run_batch_sampler(options));
        }
    } catch (const std::exception& ex) {
        std::cerr << "symft_rate_bench: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
