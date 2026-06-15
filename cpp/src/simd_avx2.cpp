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

void avx2_rotate_uniform_imag_pairs(
    Complex* alpha,
    const std::size_t* left_indices,
    const std::size_t* right_indices,
    std::size_t npairs,
    double c,
    double q) {
    auto* raw = reinterpret_cast<double*>(alpha);
    const __m256d vc = _mm256_set1_pd(c);
    const __m256d vq = _mm256_setr_pd(-q, q, -q, q);
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        double* a0 = raw + 2 * left_indices[idx];
        double* a1 = raw + 2 * right_indices[idx];
        const __m128d lo = _mm_loadu_pd(a0);
        const __m128d hi = _mm_loadu_pd(a1);
        const __m256d v = _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), hi, 1);
        const __m256d cross = _mm256_permute4x64_pd(v, 0x1b);
        const __m256d result = _mm256_fmadd_pd(vq, cross, _mm256_mul_pd(vc, v));
        _mm_storeu_pd(a0, _mm256_castpd256_pd128(result));
        _mm_storeu_pd(a1, _mm256_extractf128_pd(result, 1));
    }
}

void avx2_rotate_real_pair_flip(
    Complex* alpha,
    const std::size_t* left_indices,
    const std::size_t* right_indices,
    std::size_t npairs,
    double c,
    double base_coeff,
    std::uint64_t zmask) {
    auto* raw = reinterpret_cast<double*>(alpha);
    const __m256d vc = _mm256_set1_pd(c);
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        const std::size_t i0 = left_indices[idx];
        const std::size_t i1 = right_indices[idx];
        const double q = (__builtin_popcountll(static_cast<std::uint64_t>(i0) & zmask) & 1) != 0 ? -base_coeff : base_coeff;
        double* a0 = raw + 2 * i0;
        double* a1 = raw + 2 * i1;
        const __m128d lo = _mm_loadu_pd(a0);
        const __m128d hi = _mm_loadu_pd(a1);
        const __m256d v = _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), hi, 1);
        const __m256d cross = _mm256_permute4x64_pd(v, 0x4e);
        const __m256d vq = _mm256_setr_pd(-q, -q, q, q);
        const __m256d result = _mm256_fmadd_pd(vq, cross, _mm256_mul_pd(vc, v));
        _mm_storeu_pd(a0, _mm256_castpd256_pd128(result));
        _mm_storeu_pd(a1, _mm256_extractf128_pd(result, 1));
    }
}

const KernelTable table = {
    "avx2",
    avx2_mul_assign,
    avx2_norm_sum,
    avx2_rotate_uniform_imag_pairs,
    avx2_rotate_real_pair_flip,
};

} // namespace

const KernelTable& avx2_table() {
    return table;
}

} // namespace symft::simd
