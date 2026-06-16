#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

namespace symft {

inline int normalized_batch_thread_count(int threads) {
    return std::max(1, threads);
}

inline int batch_parallel_workers(int requested_threads, std::size_t outer_items, int active_shots) {
    const int requested = normalized_batch_thread_count(requested_threads);
    if (requested <= 1 || outer_items <= 1 || active_shots <= 0) {
        return 1;
    }

    constexpr std::size_t kMinOuterItemsPerThread = 4;
    constexpr std::size_t kMinShotOpsPerThread = 32768;
    const int by_outer = static_cast<int>(std::max<std::size_t>(1, outer_items / kMinOuterItemsPerThread));
    const std::size_t shot_ops =
        outer_items > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(active_shots)
            ? std::numeric_limits<std::size_t>::max()
            : outer_items * static_cast<std::size_t>(active_shots);
    const int by_work = static_cast<int>(std::max<std::size_t>(1, shot_ops / kMinShotOpsPerThread));
    return std::max(1, std::min({requested, by_outer, by_work}));
}

template <typename Func>
void batch_parallel_for_outer(int requested_threads, std::size_t outer_items, int active_shots, Func&& func) {
    const int workers = batch_parallel_workers(requested_threads, outer_items, active_shots);
    if (workers <= 1) {
        func(0, outer_items, 0);
        return;
    }

    std::exception_ptr first_exception;
    std::mutex exception_mutex;
    auto run_worker = [&](int worker) {
        const std::size_t begin = outer_items * static_cast<std::size_t>(worker) / static_cast<std::size_t>(workers);
        const std::size_t end = outer_items * static_cast<std::size_t>(worker + 1) / static_cast<std::size_t>(workers);
        try {
            func(begin, end, worker);
        } catch (...) {
            std::lock_guard<std::mutex> lock(exception_mutex);
            if (!first_exception) {
                first_exception = std::current_exception();
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(workers - 1));
    for (int worker = 1; worker < workers; ++worker) {
        threads.emplace_back(run_worker, worker);
    }
    run_worker(0);
    for (auto& thread : threads) {
        thread.join();
    }
    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
}

} // namespace symft
