#include "symft/simd.hpp"

#include <cmath>

namespace symft::simd {
namespace {

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

void scalar_rotate_uniform_imag_pairs(
    Complex* alpha,
    const std::size_t* left_indices,
    const std::size_t* right_indices,
    std::size_t npairs,
    double c,
    double q) {
    auto* raw = reinterpret_cast<double*>(alpha);
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        double* a0 = raw + 2 * left_indices[idx];
        double* a1 = raw + 2 * right_indices[idx];
        const double r0 = a0[0];
        const double i0 = a0[1];
        const double r1 = a1[0];
        const double i1 = a1[1];
        a0[0] = c * r0 - q * i1;
        a0[1] = c * i0 + q * r1;
        a1[0] = c * r1 - q * i0;
        a1[1] = c * i1 + q * r0;
    }
}

void scalar_rotate_real_pair_flip(
    Complex* alpha,
    const std::size_t* left_indices,
    const std::size_t* right_indices,
    std::size_t npairs,
    double c,
    double base_coeff,
    std::uint64_t zmask) {
    auto* raw = reinterpret_cast<double*>(alpha);
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        const std::size_t i0 = left_indices[idx];
        const std::size_t i1 = right_indices[idx];
        const double q = (popcount64(static_cast<std::uint64_t>(i0) & zmask) & 1) != 0 ? -base_coeff : base_coeff;
        double* a0 = raw + 2 * i0;
        double* a1 = raw + 2 * i1;
        const double r0 = a0[0];
        const double i0v = a0[1];
        const double r1 = a1[0];
        const double i1v = a1[1];
        a0[0] = c * r0 - q * r1;
        a0[1] = c * i0v - q * i1v;
        a1[0] = c * r1 + q * r0;
        a1[1] = c * i1v + q * i0v;
    }
}

const KernelTable table = {
    "scalar",
    scalar_mul_assign,
    scalar_norm_sum,
    scalar_rotate_uniform_imag_pairs,
    scalar_rotate_real_pair_flip,
};

} // namespace

const KernelTable& scalar_table() {
    return table;
}

} // namespace symft::simd
