#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>

namespace symft::simd {

using Complex = std::complex<double>;

struct KernelTable {
    const char* name;
    void (*mul_assign)(Complex* alpha, const Complex* coeff, double c, std::size_t n);
    double (*norm_sum)(const Complex* alpha, const std::size_t* indices, std::size_t n);
    void (*rotate_uniform_imag_pairs)(
        Complex* alpha,
        const std::size_t* left_indices,
        const std::size_t* right_indices,
        std::size_t npairs,
        double c,
        double q);
    void (*rotate_real_pair_flip)(
        Complex* alpha,
        const std::size_t* left_indices,
        const std::size_t* right_indices,
        std::size_t npairs,
        double c,
        double base_coeff,
        std::uint64_t zmask);
};

const KernelTable& scalar_table();
const KernelTable& dispatch_table();
const char* dispatch_name();

} // namespace symft::simd
