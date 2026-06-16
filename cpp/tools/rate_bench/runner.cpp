#include "rate_bench/rate_bench.hpp"

#include <algorithm>
#include <thread>

namespace symft_rate_bench {

BenchResult run_single_sampler(const Options& options) {
    BenchResult result;
    result.path = options.path;
    result.sampler = options.postselect_detectors ? "single_postselected" : "single";
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
        const auto postselection_detectors_by_record = detectors_by_max_record(parsed.detectors, program.nrecords);
        const auto instruction_records_by_index = instruction_records(program);
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
                bool discarded = false;
                if (options.postselect_detectors) {
                    symft::assign_presampled_exogenous_in_place(runtime, samples, shot);
                    for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
                        symft::execute_instruction_in_place(runtime, program.instructions[idx]);
                        const int record = instruction_records_by_index[idx];
                        if (record <= 0 || record >= static_cast<int>(postselection_detectors_by_record.size())) {
                            continue;
                        }
                        if (any_detector_fires(
                                runtime.measurement_words,
                                postselection_detectors_by_record[static_cast<std::size_t>(record)])) {
                            discarded = true;
                            break;
                        }
                    }
                } else {
                    symft::execute_in_place(runtime, program, samples, shot);
                }
                const auto execute_stop = Clock::now();
                if (discarded) {
                    ++counts.discarded;
                } else {
                    accumulate_single_counts(counts, runtime.measurement_words, parsed.detectors, logical_records);
                }
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

} // namespace symft_rate_bench
