#include "rate_bench/rate_bench.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        const symft_rate_bench::Options options = symft_rate_bench::parse_options(argc, argv);
        if (options.sampler == symft_rate_bench::SamplerMode::Single) {
            symft_rate_bench::print_result(symft_rate_bench::run_single_sampler(options));
        } else if (options.sampler == symft_rate_bench::SamplerMode::Batch) {
            symft_rate_bench::print_result(symft_rate_bench::run_batch_sampler(options));
        } else {
            symft_rate_bench::print_result(symft_rate_bench::run_single_sampler(options));
            std::cout << "\n";
            symft_rate_bench::print_result(symft_rate_bench::run_batch_sampler(options));
        }
    } catch (const std::exception& ex) {
        std::cerr << "symft_rate_bench: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
