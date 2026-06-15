#include "symft/simd.hpp"

#include <cmath>

namespace symft::simd {
namespace {

void scalar_mul_assign(Complex* alpha, const Complex* coeff, double c, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        alpha[i] *= Complex(c, 0.0) + coeff[i];
    }
}

double scalar_norm_sum(const Complex* alpha, const std::size_t* indices, std::size_t n) {
    double out = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const Complex a = alpha[indices[i]];
        out += std::norm(a);
    }
    return out;
}

const KernelTable table = {
    "scalar",
    scalar_mul_assign,
    scalar_norm_sum,
};

} // namespace

const KernelTable& scalar_table() {
    return table;
}

} // namespace symft::simd
