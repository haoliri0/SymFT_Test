#include "symft/simd/simd.hpp"

#include <immintrin.h>

namespace symft::simd {
namespace {

#if defined(__clang__)
#define SYMFT_SIMD_LOOP _Pragma("clang loop vectorize(enable) interleave(enable)")
#elif defined(__GNUC__)
#define SYMFT_SIMD_LOOP _Pragma("GCC ivdep")
#else
#define SYMFT_SIMD_LOOP
#endif

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

void avx512_mul_assign(Complex* alpha, const Complex* coeff, double c, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        alpha[i] *= Complex(c, 0.0) + coeff[i];
    }
}

double avx512_norm_sum(const Complex* alpha, const std::size_t* indices, std::size_t n) {
    const auto* raw = reinterpret_cast<const double*>(alpha);
    double out = 0.0;
    SYMFT_SIMD_LOOP
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t src = 2 * indices[i];
        const double r = raw[src];
        const double im = raw[src + 1];
        out += r * r + im * im;
    }
    return out;
}

void avx512_rotate_uniform_imag_pairs(
    Complex* alpha,
    const std::size_t* left_indices,
    const std::size_t* right_indices,
    std::size_t npairs,
    double c,
    double q) {
    auto* raw = reinterpret_cast<double*>(alpha);
    SYMFT_SIMD_LOOP
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

void avx512_rotate_real_pair_flip(
    Complex* alpha,
    const std::size_t* left_indices,
    const std::size_t* right_indices,
    std::size_t npairs,
    double c,
    double base_coeff,
    std::uint64_t zmask) {
    auto* raw = reinterpret_cast<double*>(alpha);
    SYMFT_SIMD_LOOP
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

void avx512_mul_assign_soa(double* re, double* im, const Complex* coeff, double c, std::size_t n) {
    const auto* cr = reinterpret_cast<const double*>(coeff);
    SYMFT_SIMD_LOOP
    for (std::size_t i = 0; i < n; ++i) {
        const double fr = c + cr[2 * i];
        const double fi = cr[2 * i + 1];
        const double r = re[i];
        const double v = im[i];
        re[i] = fr * r - fi * v;
        im[i] = fr * v + fi * r;
    }
}

double avx512_norm_sum_soa(const double* re, const double* im, const std::size_t* indices, std::size_t n) {
    double out = 0.0;
    SYMFT_SIMD_LOOP
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t src = indices[i];
        const double r = re[src];
        const double v = im[src];
        out += r * r + v * v;
    }
    return out;
}

void avx512_rotate_uniform_imag_pairs_soa(
    double* re,
    double* im,
    std::size_t dim,
    std::uint64_t xmask,
    unsigned pair_bit,
    double c,
    double q) {
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t step = selector << 1;
    const __m512d vc = _mm512_set1_pd(c);
    const __m512d vq = _mm512_set1_pd(q);
    for (std::size_t block = 0; block < dim; block += step) {
        const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
        std::size_t offset = 0;
        for (; offset + 8 <= selector; offset += 8) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = right_base + offset;
            const __m512d r0 = _mm512_loadu_pd(re + i0);
            const __m512d im0 = _mm512_loadu_pd(im + i0);
            const __m512d r1 = _mm512_loadu_pd(re + i1);
            const __m512d im1 = _mm512_loadu_pd(im + i1);
            _mm512_storeu_pd(re + i0, _mm512_fnmadd_pd(vq, im1, _mm512_mul_pd(vc, r0)));
            _mm512_storeu_pd(im + i0, _mm512_fmadd_pd(vq, r1, _mm512_mul_pd(vc, im0)));
            _mm512_storeu_pd(re + i1, _mm512_fnmadd_pd(vq, im0, _mm512_mul_pd(vc, r1)));
            _mm512_storeu_pd(im + i1, _mm512_fmadd_pd(vq, r0, _mm512_mul_pd(vc, im1)));
        }
        for (; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = right_base + offset;
            const double r0 = re[i0];
            const double im0 = im[i0];
            const double r1 = re[i1];
            const double im1 = im[i1];
            re[i0] = c * r0 - q * im1;
            im[i0] = c * im0 + q * r1;
            re[i1] = c * r1 - q * im0;
            im[i1] = c * im1 + q * r0;
        }
    }
}

void avx512_rotate_real_pair_flip_soa(
    double* re,
    double* im,
    std::size_t dim,
    std::uint64_t xmask,
    unsigned pair_bit,
    const double* phase_signs,
    double c,
    double base_coeff) {
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t step = selector << 1;
    const __m512d vc = _mm512_set1_pd(c);
    const __m512d vbase = _mm512_set1_pd(base_coeff);
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
        std::size_t offset = 0;
        for (; offset + 8 <= selector; offset += 8) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = right_base + offset;
            const __m512d q = _mm512_mul_pd(_mm512_loadu_pd(phase_signs + pair_idx), vbase);
            const __m512d r0 = _mm512_loadu_pd(re + i0);
            const __m512d im0 = _mm512_loadu_pd(im + i0);
            const __m512d r1 = _mm512_loadu_pd(re + i1);
            const __m512d im1 = _mm512_loadu_pd(im + i1);
            _mm512_storeu_pd(re + i0, _mm512_fnmadd_pd(q, r1, _mm512_mul_pd(vc, r0)));
            _mm512_storeu_pd(im + i0, _mm512_fnmadd_pd(q, im1, _mm512_mul_pd(vc, im0)));
            _mm512_storeu_pd(re + i1, _mm512_fmadd_pd(q, r0, _mm512_mul_pd(vc, r1)));
            _mm512_storeu_pd(im + i1, _mm512_fmadd_pd(q, im0, _mm512_mul_pd(vc, im1)));
            pair_idx += 8;
        }
        for (; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = right_base + offset;
            const double q = phase_signs[pair_idx++] * base_coeff;
            const double r0 = re[i0];
            const double im0 = im[i0];
            const double r1 = re[i1];
            const double im1 = im[i1];
            re[i0] = c * r0 - q * r1;
            im[i0] = c * im0 - q * im1;
            re[i1] = c * r1 + q * r0;
            im[i1] = c * im1 + q * im0;
        }
    }
}

void avx512_rotate_general_pairs_soa(
    double* re,
    double* im,
    std::size_t dim,
    std::uint64_t xmask,
    unsigned pair_bit,
    const Complex* left_coeff,
    const Complex* right_coeff,
    double c) {
    const auto* left = reinterpret_cast<const double*>(left_coeff);
    const auto* right = reinterpret_cast<const double*>(right_coeff);
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t step = selector << 1;
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
        SYMFT_SIMD_LOOP
        for (std::size_t offset = 0; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = right_base + offset;
            const double r0 = re[i0];
            const double im0 = im[i0];
            const double r1 = re[i1];
            const double im1 = im[i1];
            const double lr = left[2 * pair_idx];
            const double li = left[2 * pair_idx + 1];
            const double rr = right[2 * pair_idx];
            const double ri = right[2 * pair_idx + 1];
            re[i0] = c * r0 + rr * r1 - ri * im1;
            im[i0] = c * im0 + rr * im1 + ri * r1;
            re[i1] = c * r1 + lr * r0 - li * im0;
            im[i1] = c * im1 + lr * im0 + li * r0;
            ++pair_idx;
        }
    }
}

const KernelTable table = {
    "avx512",
    avx512_mul_assign,
    avx512_norm_sum,
    avx512_rotate_uniform_imag_pairs,
    avx512_rotate_real_pair_flip,
    avx512_mul_assign_soa,
    avx512_norm_sum_soa,
    avx512_rotate_uniform_imag_pairs_soa,
    avx512_rotate_real_pair_flip_soa,
    avx512_rotate_general_pairs_soa,
};

} // namespace

const KernelTable& avx512_table() {
    return table;
}

} // namespace symft::simd
