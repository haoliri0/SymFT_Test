#include "batch_internal.hpp"

#include <algorithm>
#include <memory>
#include <utility>

namespace symft {

BatchThreadPool::BatchThreadPool(int threads) : thread_count_(normalized_batch_threads(threads)) {
    workers_.reserve(static_cast<std::size_t>(std::max(0, thread_count_ - 1)));
    for (int worker = 1; worker < thread_count_; ++worker) {
        workers_.emplace_back([this, worker] { worker_loop(worker); });
    }
}

BatchThreadPool::~BatchThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        ++generation_;
    }
    task_cv_.notify_all();
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

int BatchThreadPool::size() const {
    return thread_count_;
}

void BatchThreadPool::worker_loop(int worker_id) {
    int seen_generation = 0;
    while (true) {
        std::function<void(int)> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            task_cv_.wait(lock, [&] { return stop_ || generation_ != seen_generation; });
            if (stop_) {
                return;
            }
            seen_generation = generation_;
            task = task_;
        }
        try {
            task(worker_id);
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!error_) {
                error_ = std::current_exception();
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            --pending_;
            if (pending_ == 0) {
                completed_generation_ = seen_generation;
            }
        }
        done_cv_.notify_one();
    }
}

void BatchThreadPool::parallel_for(
    int workers,
    std::size_t items,
    const std::function<void(int, std::size_t, std::size_t)>& fn) {
    const int active_workers = std::max(1, std::min(normalized_batch_threads(workers), thread_count_));
    if (active_workers <= 1 || items == 0) {
        fn(0, std::size_t{0}, items);
        return;
    }
    auto run_worker = [=, &fn](int worker) {
        const std::size_t first = items * static_cast<std::size_t>(worker) / static_cast<std::size_t>(active_workers);
        const std::size_t last = items * static_cast<std::size_t>(worker + 1) / static_cast<std::size_t>(active_workers);
        fn(worker, first, last);
    };
    int task_generation = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        error_ = nullptr;
        task_ = run_worker;
        pending_ = active_workers - 1;
        task_generation = ++generation_;
    }
    task_cv_.notify_all();
    try {
        run_worker(0);
    } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!error_) {
            error_ = std::current_exception();
        }
    }
    {
        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [&] { return completed_generation_ == task_generation || pending_ == 0; });
        task_ = nullptr;
        if (error_) {
            std::rethrow_exception(error_);
        }
    }
}

void execute_batch_instruction(BatchFactoredExecutorState& runtime, const ApplyPrecomputedActivePauliRotation& instruction) {
    eval_symbolic_bool_batch(runtime.eval_scratch, instruction.sign_plan, runtime);
    rotate_pauli_batch(runtime, instruction.rotation_kernel, runtime.eval_scratch);
}

void execute_batch_instruction(BatchFactoredExecutorState& runtime, const ApplyActiveBasisChange& instruction) {
    apply_active_basis_change_batch(runtime, instruction.kind, instruction.qubit);
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

void execute_batch_instruction_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstruction& instruction) {
    execute_batch_instruction(runtime, instruction);
}

BatchFactoredExecutorState::BatchFactoredExecutorState(
    const FactoredInstructionProgram& program,
    int batches_,
    std::uint64_t seed,
    int threads_)
    : batches(batches_ > 0 ? batches_ : default_batch_count(program.max_k)),
      rng_state(seed),
      threads(normalized_batch_threads(threads_)) {
    if (threads > 1) {
        thread_pool = std::make_shared<BatchThreadPool>(threads);
    }
    reset_batch_executor(*this, program, batches);
}

void set_batch_executor_threads(BatchFactoredExecutorState& runtime, int threads) {
    const int normalized = normalized_batch_threads(threads);
    if (runtime.threads == normalized && ((normalized <= 1 && !runtime.thread_pool) ||
                                          (runtime.thread_pool && runtime.thread_pool->size() == normalized))) {
        return;
    }
    runtime.threads = normalized;
    if (normalized <= 1) {
        runtime.thread_pool.reset();
    } else {
        runtime.thread_pool = std::make_shared<BatchThreadPool>(normalized);
    }
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

std::vector<std::vector<std::uint64_t>> sample_measurements_batch(
    const FactoredInstructionProgram& program,
    int shots,
    int batches,
    std::uint64_t seed,
    int threads) {
    if (shots < 0) {
        fail("shot count must be nonnegative");
    }
    BatchFactoredExecutorState runtime(program, batches, seed, threads);
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
