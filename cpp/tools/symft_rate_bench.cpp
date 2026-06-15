#include "symft/symft.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct RateCounts {
    std::uint64_t shots = 0;
    std::uint64_t discarded = 0;
    std::uint64_t accepted = 0;
    std::uint64_t logical_errors = 0;
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

int parse_block_shots(const char* raw) {
    if (std::string(raw) == "auto") {
        return 0;
    }
    return parse_positive_int(raw, "block_shots");
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

} // namespace

int main(int argc, char** argv) {
    if (argc > 6 || (argc >= 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))) {
        std::cerr << "usage: symft_rate_bench [circuit.stim=d3.stim] [shots=100000000] [block_shots=auto] [repeats=1] [observable=0]\n";
        return argc > 6 ? 2 : 0;
    }

    try {
        const std::string path = argc >= 2 ? argv[1] : "d3.stim";
        const std::uint64_t shots = argc >= 3 ? parse_u64(argv[2], "shots") : 100000000ULL;
        const int requested_block_shots = argc >= 4 ? parse_block_shots(argv[3]) : 0;
        const int repeats = argc >= 5 ? parse_positive_int(argv[4], "repeats") : 1;
        const int observable = argc >= 6 ? parse_nonnegative_int(argv[5], "observable") : 0;

        double parse_total = 0.0;
        double plan_total = 0.0;
        double presample_total = 0.0;
        double execute_total = 0.0;
        double accumulate_total = 0.0;
        RateCounts total_counts;
        int n = 0;
        int records = 0;
        int max_k = 0;
        int block_shots = 0;
        int detectors_count = 0;
        int observable_includes_count = 0;

        for (int repeat = 0; repeat < repeats; ++repeat) {
            const auto parse_start = Clock::now();
            const auto parsed = symft::parse_stim_file(path);
            const auto parse_stop = Clock::now();

            const auto plan_start = Clock::now();
            symft::PendingFactoredState pending(parsed.state);
            const auto program = symft::plan_factored_updates(pending);
            const auto plan_stop = Clock::now();

            n = parsed.state.n;
            records = program.nrecords;
            max_k = program.max_k;
            block_shots = requested_block_shots > 0 ? requested_block_shots : symft::default_batch_count(program.max_k);
            detectors_count = static_cast<int>(parsed.detectors.size());
            observable_includes_count = static_cast<int>(parsed.observables.size());

            std::vector<std::vector<int>> logical_records;
            for (const auto& include : parsed.observables) {
                if (include.index == observable) {
                    logical_records.push_back(include.records);
                }
            }

            RateCounts counts;
            counts.shots = shots;
            symft::BatchFactoredExecutorState runtime(
                program,
                block_shots,
                0x5eed1234ULL + static_cast<std::uint64_t>(repeat));
            std::vector<std::uint64_t> discard_bits(runtime.batch_words, 0);
            std::vector<std::uint64_t> logical_bits(runtime.batch_words, 0);
            std::vector<std::uint64_t> scratch(runtime.batch_words, 0);
            std::uint64_t exogenous_seed = 0x7eed0000ULL + static_cast<std::uint64_t>(repeat);
            std::uint64_t offset = 0;
            while (offset < shots) {
                const int block = static_cast<int>(std::min<std::uint64_t>(
                    static_cast<std::uint64_t>(block_shots),
                    shots - offset));

                const auto presample_start = Clock::now();
                const auto samples = symft::presample_exogenous(program, block, exogenous_seed);
                const auto presample_stop = Clock::now();
                exogenous_seed = samples.next_rng_state;

                symft::reset_batch_executor(runtime, program, block);
                symft::execute_batch_in_place(runtime, program, samples);
                const auto execute_stop = Clock::now();
                accumulate_block_counts(
                    counts,
                    runtime,
                    block,
                    parsed.detectors,
                    logical_records,
                    discard_bits,
                    logical_bits,
                    scratch);
                const auto accumulate_stop = Clock::now();

                presample_total += seconds_between(presample_start, presample_stop);
                execute_total += seconds_between(presample_stop, execute_stop);
                accumulate_total += seconds_between(execute_stop, accumulate_stop);
                offset += static_cast<std::uint64_t>(block);
            }

            total_counts.shots += counts.shots;
            total_counts.discarded += counts.discarded;
            total_counts.accepted += counts.accepted;
            total_counts.logical_errors += counts.logical_errors;
            parse_total += seconds_between(parse_start, parse_stop);
            plan_total += seconds_between(plan_start, plan_stop);
        }

        const double inv_repeats = 1.0 / static_cast<double>(repeats);
        const double parse_s = parse_total * inv_repeats;
        const double plan_s = plan_total * inv_repeats;
        const double presample_s = presample_total * inv_repeats;
        const double execute_s = execute_total * inv_repeats;
        const double accumulate_s = accumulate_total * inv_repeats;
        const double sample_s = presample_s + execute_s + accumulate_s;
        const double shots_per_s = sample_s > 0.0 ? static_cast<double>(shots) / sample_s : 0.0;
        const double discard_rate = total_counts.shots == 0 ? std::numeric_limits<double>::quiet_NaN()
                                                            : static_cast<double>(total_counts.discarded) / static_cast<double>(total_counts.shots);
        const double logical_rate = total_counts.accepted == 0 ? std::numeric_limits<double>::quiet_NaN()
                                                               : static_cast<double>(total_counts.logical_errors) / static_cast<double>(total_counts.accepted);

        std::cout << "file " << path << "\n";
        std::cout << "qubits " << n << "\n";
        std::cout << "records " << records << "\n";
        std::cout << "detectors " << detectors_count << "\n";
        std::cout << "observable_includes " << observable_includes_count << "\n";
        std::cout << "observable " << observable << "\n";
        std::cout << "max_active_qubits " << max_k << "\n";
        std::cout << "simd_backend " << symft::active_simd_backend() << "\n";
        std::cout << "shots " << shots << "\n";
        std::cout << "sampled_shots " << total_counts.shots << "\n";
        std::cout << "block_shots " << block_shots << "\n";
        std::cout << "repeats " << repeats << "\n";
        std::cout << "sampler batch\n";
        std::cout << "active_layout basis_major_shots_contiguous\n";
        std::cout << "exogenous_mode presampled_streaming\n";
        std::cout << "rng_streams split_exogenous_branch\n";
        std::cout << "parse_s_avg " << parse_s << "\n";
        std::cout << "plan_s_avg " << plan_s << "\n";
        std::cout << "presample_s_avg " << presample_s << "\n";
        std::cout << "execute_s_avg " << execute_s << "\n";
        std::cout << "accumulate_s_avg " << accumulate_s << "\n";
        std::cout << "sample_s_avg " << sample_s << "\n";
        std::cout << "sample_shots_per_s " << shots_per_s << "\n";
        std::cout << "discarded " << total_counts.discarded << "\n";
        std::cout << "accepted " << total_counts.accepted << "\n";
        std::cout << "logical_errors " << total_counts.logical_errors << "\n";
        std::cout << "discard_rate " << discard_rate << "\n";
        std::cout << "logical_error_rate " << logical_rate << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "symft_rate_bench: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
