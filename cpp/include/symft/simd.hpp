#pragma once

#include <complex>
#include <cstddef>

namespace symft::simd {

using Complex = std::complex<double>;

struct KernelTable {
    const char* name;
    void (*mul_assign)(Complex* alpha, const Complex* coeff, double c, std::size_t n);
    double (*norm_sum)(const Complex* alpha, const std::size_t* indices, std::size_t n);
};

const KernelTable& scalar_table();
const KernelTable& dispatch_table();
const char* dispatch_name();

} // namespace symft::simd
