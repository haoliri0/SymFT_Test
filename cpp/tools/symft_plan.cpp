#include "frontend/stim.hpp"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

double seconds_between(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double>(stop - start).count();
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: symft_plan <circuit.stim>\n";
        return 2;
    }
    try {
        const std::string path = argv[1];
        const auto parse_start = Clock::now();
        const auto parsed = symft::parse_stim_file(path);
        const auto parse_stop = Clock::now();

        symft::PendingOptimizationStats optimization;
        {
            symft::PendingFactoredState pending_diagnostics(parsed.state);
            std::vector<int> detector_prefixes;
            detector_prefixes.reserve(parsed.detectors.size());
            for (const auto& detector : parsed.detectors) {
                detector_prefixes.push_back(detector.after_pending_operation);
            }
            optimization =
                symft::optimize_pending_operations(pending_diagnostics, detector_prefixes);
        }

        const auto plan_start = Clock::now();
        const auto plan = symft::plan_stim_factored_program(parsed);
        const auto plan_stop = Clock::now();

        std::cout << "qubits " << parsed.state.n << "\n";
        std::cout << "records " << plan.nrecords << "\n";
        std::cout << "detectors " << plan.ndetectors << "\n";
        std::cout << "instructions " << plan.instructions.size() << "\n";
        std::cout << "max_active_qubits " << plan.max_k << "\n";
        std::cout << "pending_operations_before " << optimization.input_operations << "\n";
        std::cout << "pending_operations_after " << optimization.output_operations << "\n";
        std::cout << "fused_rotations " << optimization.fused_rotations << "\n";
        std::cout << "cancelled_rotations " << optimization.cancelled_rotations << "\n";
        std::cout << "measurement_left_swaps " << optimization.measurement_left_swaps << "\n";
        std::cout << "parse_seconds " << seconds_between(parse_start, parse_stop) << "\n";
        std::cout << "plan_seconds " << seconds_between(plan_start, plan_stop) << "\n";
#if defined(__unix__) || defined(__APPLE__)
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
#if defined(__APPLE__)
            std::cout << "peak_rss_kib " << usage.ru_maxrss / 1024 << "\n";
#else
            std::cout << "peak_rss_kib " << usage.ru_maxrss << "\n";
#endif
        }
#endif
    } catch (const std::exception& ex) {
        std::cerr << "symft_plan: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
