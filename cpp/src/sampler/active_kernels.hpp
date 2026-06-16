#pragma once

#include "../core/symft_internal.hpp"

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace symft::detail {

#if defined(__clang__)
#define SYMFT_SINGLE_SIMD_LOOP _Pragma("clang loop vectorize(enable) interleave(enable)")
#elif defined(__GNUC__)
#define SYMFT_SINGLE_SIMD_LOOP _Pragma("GCC ivdep")
#else
#define SYMFT_SINGLE_SIMD_LOOP
#endif

#if defined(__AVX2__) && defined(__FMA__)
#define SYMFT_INLINE_AVX2 1
#else
#define SYMFT_INLINE_AVX2 0
#endif


inline void rotate_uniform_imag_pairs_scalar(
    Complex* alpha,
    const std::size_t* left_indices,
    const std::size_t* right_indices,
    std::size_t npairs,
    double c,
    double q) {
    auto* raw = reinterpret_cast<double*>(alpha);
    SYMFT_SINGLE_SIMD_LOOP
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

inline void rotate_real_pair_flip_scalar(
    Complex* alpha,
    const std::size_t* left_indices,
    const std::size_t* right_indices,
    std::size_t npairs,
    double c,
    double base_coeff,
    std::uint64_t zmask) {
    auto* raw = reinterpret_cast<double*>(alpha);
    SYMFT_SINGLE_SIMD_LOOP
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        const std::size_t i0 = left_indices[idx];
        const std::size_t i1 = right_indices[idx];
        const double q = is_odd_popcount(static_cast<std::uint64_t>(i0) & zmask) ? -base_coeff : base_coeff;
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

#if SYMFT_INLINE_AVX2
inline __m256d load_complex_re4_inline(const double* coeff) {
    const __m256d c0 = _mm256_loadu_pd(coeff);
    const __m256d c1 = _mm256_loadu_pd(coeff + 4);
    return _mm256_permute4x64_pd(_mm256_unpacklo_pd(c0, c1), 0xd8);
}

inline __m256d load_complex_im4_inline(const double* coeff) {
    const __m256d c0 = _mm256_loadu_pd(coeff);
    const __m256d c1 = _mm256_loadu_pd(coeff + 4);
    return _mm256_permute4x64_pd(_mm256_unpackhi_pd(c0, c1), 0xd8);
}
#endif

inline void mul_assign_soa_inline(double* re, double* im, const Complex* coefficients, double c, std::size_t dim) {
    const auto* coeff = reinterpret_cast<const double*>(coefficients);
    std::size_t basis = 0;
#if SYMFT_INLINE_AVX2
    const __m256d vc = _mm256_set1_pd(c);
    for (; basis + 4 <= dim; basis += 4) {
        const __m256d r = _mm256_loadu_pd(re + basis);
        const __m256d i = _mm256_loadu_pd(im + basis);
        const __m256d fr = _mm256_add_pd(vc, load_complex_re4_inline(coeff + 2 * basis));
        const __m256d fi = load_complex_im4_inline(coeff + 2 * basis);
        _mm256_storeu_pd(re + basis, _mm256_fnmadd_pd(fi, i, _mm256_mul_pd(fr, r)));
        _mm256_storeu_pd(im + basis, _mm256_fmadd_pd(fi, r, _mm256_mul_pd(fr, i)));
    }
#endif
    SYMFT_SINGLE_SIMD_LOOP
    for (; basis < dim; ++basis) {
        const double fr = c + coeff[2 * basis];
        const double fi = coeff[2 * basis + 1];
        const double r = re[basis];
        const double i = im[basis];
        re[basis] = fr * r - fi * i;
        im[basis] = fr * i + fi * r;
    }
}

inline double norm_sum_soa_inline(const double* re, const double* im, const std::size_t* sources, std::size_t n) {
    double probability = 0.0;
    SYMFT_SINGLE_SIMD_LOOP
    for (std::size_t idx = 0; idx < n; ++idx) {
        const std::size_t source = sources[idx];
        const double r = re[source];
        const double i = im[source];
        probability += r * r + i * i;
    }
    return probability;
}

inline void rotate_uniform_imag_pairs_soa_inline(
    double* re,
    double* im,
    std::size_t dim,
    std::uint64_t xmask,
    unsigned pair_bit,
    double c,
    double q) {
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t step = selector << 1;
#if SYMFT_INLINE_AVX2
    const __m256d vc = _mm256_set1_pd(c);
    const __m256d vq = _mm256_set1_pd(q);
    if (selector == 1 && xmask == 1 && dim >= 4) {
        std::size_t basis = 0;
        for (; basis + 4 <= dim; basis += 4) {
            const __m256d r = _mm256_loadu_pd(re + basis);
            const __m256d v = _mm256_loadu_pd(im + basis);
            const __m256d swapped_r = _mm256_permute_pd(r, 0b0101);
            const __m256d swapped_v = _mm256_permute_pd(v, 0b0101);
            _mm256_storeu_pd(re + basis, _mm256_fnmadd_pd(vq, swapped_v, _mm256_mul_pd(vc, r)));
            _mm256_storeu_pd(im + basis, _mm256_fmadd_pd(vq, swapped_r, _mm256_mul_pd(vc, v)));
        }
        for (; basis < dim; basis += 2) {
            const double r0 = re[basis];
            const double im0 = im[basis];
            const double r1 = re[basis + 1];
            const double im1 = im[basis + 1];
            re[basis] = c * r0 - q * im1;
            im[basis] = c * im0 + q * r1;
            re[basis + 1] = c * r1 - q * im0;
            im[basis + 1] = c * im1 + q * r0;
        }
        return;
    }
#endif
    for (std::size_t block = 0; block < dim; block += step) {
        const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
        std::size_t offset = 0;
#if SYMFT_INLINE_AVX2
        for (; offset + 4 <= selector; offset += 4) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = right_base + offset;
            const __m256d r0 = _mm256_loadu_pd(re + i0);
            const __m256d im0 = _mm256_loadu_pd(im + i0);
            const __m256d r1 = _mm256_loadu_pd(re + i1);
            const __m256d im1 = _mm256_loadu_pd(im + i1);
            _mm256_storeu_pd(re + i0, _mm256_fnmadd_pd(vq, im1, _mm256_mul_pd(vc, r0)));
            _mm256_storeu_pd(im + i0, _mm256_fmadd_pd(vq, r1, _mm256_mul_pd(vc, im0)));
            _mm256_storeu_pd(re + i1, _mm256_fnmadd_pd(vq, im0, _mm256_mul_pd(vc, r1)));
            _mm256_storeu_pd(im + i1, _mm256_fmadd_pd(vq, r0, _mm256_mul_pd(vc, im1)));
        }
#endif
        SYMFT_SINGLE_SIMD_LOOP
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

inline void rotate_real_pair_flip_soa_inline(
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
#if SYMFT_INLINE_AVX2
    const __m256d vc = _mm256_set1_pd(c);
    const __m256d vbase = _mm256_set1_pd(base_coeff);
    if (selector == 1 && xmask == 1 && dim >= 4) {
        std::size_t basis = 0;
        for (; basis + 4 <= dim; basis += 4) {
            const double q0 = phase_signs[basis >> 1] * base_coeff;
            const double q1 = phase_signs[(basis >> 1) + 1] * base_coeff;
            const __m256d signed_q = _mm256_set_pd(q1, -q1, q0, -q0);
            const __m256d r = _mm256_loadu_pd(re + basis);
            const __m256d v = _mm256_loadu_pd(im + basis);
            const __m256d swapped_r = _mm256_permute_pd(r, 0b0101);
            const __m256d swapped_v = _mm256_permute_pd(v, 0b0101);
            _mm256_storeu_pd(re + basis, _mm256_fmadd_pd(signed_q, swapped_r, _mm256_mul_pd(vc, r)));
            _mm256_storeu_pd(im + basis, _mm256_fmadd_pd(signed_q, swapped_v, _mm256_mul_pd(vc, v)));
        }
        for (; basis < dim; basis += 2) {
            const double q = phase_signs[basis >> 1] * base_coeff;
            const double r0 = re[basis];
            const double im0 = im[basis];
            const double r1 = re[basis + 1];
            const double im1 = im[basis + 1];
            re[basis] = c * r0 - q * r1;
            im[basis] = c * im0 - q * im1;
            re[basis + 1] = c * r1 + q * r0;
            im[basis + 1] = c * im1 + q * im0;
        }
        return;
    }
#endif
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
        std::size_t offset = 0;
#if SYMFT_INLINE_AVX2
        for (; offset + 4 <= selector; offset += 4) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = right_base + offset;
            const __m256d q = _mm256_mul_pd(_mm256_loadu_pd(phase_signs + pair_idx), vbase);
            const __m256d r0 = _mm256_loadu_pd(re + i0);
            const __m256d im0 = _mm256_loadu_pd(im + i0);
            const __m256d r1 = _mm256_loadu_pd(re + i1);
            const __m256d im1 = _mm256_loadu_pd(im + i1);
            _mm256_storeu_pd(re + i0, _mm256_fnmadd_pd(q, r1, _mm256_mul_pd(vc, r0)));
            _mm256_storeu_pd(im + i0, _mm256_fnmadd_pd(q, im1, _mm256_mul_pd(vc, im0)));
            _mm256_storeu_pd(re + i1, _mm256_fmadd_pd(q, r0, _mm256_mul_pd(vc, r1)));
            _mm256_storeu_pd(im + i1, _mm256_fmadd_pd(q, im0, _mm256_mul_pd(vc, im1)));
            pair_idx += 4;
        }
#endif
        SYMFT_SINGLE_SIMD_LOOP
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

inline void rotate_general_pairs_soa_inline(
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
#if SYMFT_INLINE_AVX2
    const __m256d vc = _mm256_set1_pd(c);
    if (selector == 1 && xmask == 1 && dim >= 4) {
        std::size_t basis = 0;
        for (; basis + 4 <= dim; basis += 4) {
            const std::size_t pair_idx = basis >> 1;
            const __m256d coeff_r =
                _mm256_set_pd(left[2 * (pair_idx + 1)], right[2 * (pair_idx + 1)], left[2 * pair_idx], right[2 * pair_idx]);
            const __m256d coeff_i = _mm256_set_pd(
                left[2 * (pair_idx + 1) + 1],
                right[2 * (pair_idx + 1) + 1],
                left[2 * pair_idx + 1],
                right[2 * pair_idx + 1]);
            const __m256d r = _mm256_loadu_pd(re + basis);
            const __m256d v = _mm256_loadu_pd(im + basis);
            const __m256d swapped_r = _mm256_permute_pd(r, 0b0101);
            const __m256d swapped_v = _mm256_permute_pd(v, 0b0101);
            __m256d out_r = _mm256_fmadd_pd(vc, r, _mm256_mul_pd(coeff_r, swapped_r));
            out_r = _mm256_fnmadd_pd(coeff_i, swapped_v, out_r);
            __m256d out_i = _mm256_fmadd_pd(vc, v, _mm256_mul_pd(coeff_r, swapped_v));
            out_i = _mm256_fmadd_pd(coeff_i, swapped_r, out_i);
            _mm256_storeu_pd(re + basis, out_r);
            _mm256_storeu_pd(im + basis, out_i);
        }
        for (; basis < dim; basis += 2) {
            const std::size_t pair_idx = basis >> 1;
            const double r0 = re[basis];
            const double im0 = im[basis];
            const double r1 = re[basis + 1];
            const double im1 = im[basis + 1];
            const double lr = left[2 * pair_idx];
            const double li = left[2 * pair_idx + 1];
            const double rr = right[2 * pair_idx];
            const double ri = right[2 * pair_idx + 1];
            re[basis] = c * r0 + rr * r1 - ri * im1;
            im[basis] = c * im0 + rr * im1 + ri * r1;
            re[basis + 1] = c * r1 + lr * r0 - li * im0;
            im[basis + 1] = c * im1 + lr * im0 + li * r0;
        }
        return;
    }
#endif
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
        std::size_t offset = 0;
#if SYMFT_INLINE_AVX2
        for (; offset + 4 <= selector; offset += 4) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = right_base + offset;
            const __m256d r0 = _mm256_loadu_pd(re + i0);
            const __m256d im0 = _mm256_loadu_pd(im + i0);
            const __m256d r1 = _mm256_loadu_pd(re + i1);
            const __m256d im1 = _mm256_loadu_pd(im + i1);
            const __m256d lr = load_complex_re4_inline(left + 2 * pair_idx);
            const __m256d li = load_complex_im4_inline(left + 2 * pair_idx);
            const __m256d rr = load_complex_re4_inline(right + 2 * pair_idx);
            const __m256d ri = load_complex_im4_inline(right + 2 * pair_idx);
            __m256d out_r0 = _mm256_fmadd_pd(vc, r0, _mm256_mul_pd(rr, r1));
            out_r0 = _mm256_fnmadd_pd(ri, im1, out_r0);
            __m256d out_i0 = _mm256_fmadd_pd(vc, im0, _mm256_mul_pd(rr, im1));
            out_i0 = _mm256_fmadd_pd(ri, r1, out_i0);
            __m256d out_r1 = _mm256_fmadd_pd(vc, r1, _mm256_mul_pd(lr, r0));
            out_r1 = _mm256_fnmadd_pd(li, im0, out_r1);
            __m256d out_i1 = _mm256_fmadd_pd(vc, im1, _mm256_mul_pd(lr, im0));
            out_i1 = _mm256_fmadd_pd(li, r0, out_i1);
            _mm256_storeu_pd(re + i0, out_r0);
            _mm256_storeu_pd(im + i0, out_i0);
            _mm256_storeu_pd(re + i1, out_r1);
            _mm256_storeu_pd(im + i1, out_i1);
            pair_idx += 4;
        }
#endif
        SYMFT_SINGLE_SIMD_LOOP
        for (; offset < selector; ++offset) {
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


} // namespace symft::detail

#undef SYMFT_INLINE_AVX2
