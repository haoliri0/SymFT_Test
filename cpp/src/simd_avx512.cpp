#include "symft/simd.hpp"

#include <immintrin.h>

namespace symft::simd {
namespace {

void avx512_mul_assign(Complex* alpha, const Complex* coeff, double c, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        alpha[i] *= Complex(c, 0.0) + coeff[i];
    }
}

double avx512_norm_sum(const Complex* alpha, const std::size_t* indices, std::size_t n) {
    double out = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        out += std::norm(alpha[indices[i]]);
    }
    return out;
}

const KernelTable table = {
    "avx512",
    avx512_mul_assign,
    avx512_norm_sum,
};

} // namespace

const KernelTable& avx512_table() {
    return table;
}

} // namespace symft::simd
