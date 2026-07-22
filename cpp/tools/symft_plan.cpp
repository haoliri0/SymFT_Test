#include "frontend/stim.hpp"

#include <chrono>
#include <iostream>
#include <string>

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
        const auto plan = symft::plan_stim_factored_program(parsed);
        const auto plan_stop = Clock::now();

        std::cout << "qubits " << parsed.state.n << "\n";
        std::cout << "records " << plan.nrecords << "\n";
        std::cout << "detectors " << plan.ndetectors << "\n";
        std::cout << "instructions " << plan.instructions.size() << "\n";
        std::cout << "max_active_qubits " << plan.max_k << "\n";
        std::cout << "parse_seconds " << seconds_between(parse_start, parse_stop) << "\n";
        std::cout << "plan_seconds " << seconds_between(parse_stop, plan_stop) << "\n";
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
