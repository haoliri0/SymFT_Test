#include "symft/simd.hpp"

#include <immintrin.h>

namespace symft::simd {
namespace {

void avx2_mul_assign(Complex* alpha, const Complex* coeff, double c, std::size_t n) {
    const auto* cr = reinterpret_cast<const double*>(coeff);
    auto* ar = reinterpret_cast<double*>(alpha);
    const __m256d vc = _mm256_set_pd(0.0, c, 0.0, c);
    const __m256d signs = _mm256_set_pd(1.0, -1.0, 1.0, -1.0);
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        const __m256d a = _mm256_loadu_pd(ar + 2 * i);
        const __m256d k = _mm256_loadu_pd(cr + 2 * i);
        const __m256d factor = _mm256_add_pd(k, vc);
        const __m256d real = _mm256_movedup_pd(a);
        const __m256d imag = _mm256_permute_pd(a, 0b1111);
        const __m256d factor_swapped = _mm256_permute_pd(factor, 0b0101);
        const __m256d out = _mm256_add_pd(_mm256_mul_pd(real, factor), _mm256_mul_pd(_mm256_mul_pd(imag, factor_swapped), signs));
        _mm256_storeu_pd(ar + 2 * i, out);
    }
    for (; i < n; ++i) {
        alpha[i] *= Complex(c, 0.0) + coeff[i];
    }
}

double avx2_norm_sum(const Complex* alpha, const std::size_t* indices, std::size_t n) {
    double out = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        out += std::norm(alpha[indices[i]]);
    }
    return out;
}

const KernelTable table = {
    "avx2",
    avx2_mul_assign,
    avx2_norm_sum,
};

} // namespace

const KernelTable& avx2_table() {
    return table;
}

} // namespace symft::simd
