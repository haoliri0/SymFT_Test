#include "simd/simd.hpp"

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
    SYMFT_SIMD_LOOP
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
    SYMFT_SIMD_LOOP
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

__m256d load_complex_re4(const double* coeff) {
    const __m256d c0 = _mm256_loadu_pd(coeff);
    const __m256d c1 = _mm256_loadu_pd(coeff + 4);
    return _mm256_permute4x64_pd(_mm256_unpacklo_pd(c0, c1), 0xd8);
}

__m256d load_complex_im4(const double* coeff) {
    const __m256d c0 = _mm256_loadu_pd(coeff);
    const __m256d c1 = _mm256_loadu_pd(coeff + 4);
    return _mm256_permute4x64_pd(_mm256_unpackhi_pd(c0, c1), 0xd8);
}

__m256i pair_indices4(std::size_t i0, std::uint64_t xmask) {
    const auto mask = static_cast<std::size_t>(xmask);
    return _mm256_set_epi64x(
        static_cast<long long>((i0 + 3) ^ mask),
        static_cast<long long>((i0 + 2) ^ mask),
        static_cast<long long>((i0 + 1) ^ mask),
        static_cast<long long>(i0 ^ mask));
}

__m256d gather_pair4(const double* values, std::size_t i0, std::uint64_t xmask) {
    return _mm256_i64gather_pd(values, pair_indices4(i0, xmask), 8);
}

void scatter_pair4(double* values, std::size_t i0, std::uint64_t xmask, __m256d lanes) {
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, lanes);
    const auto mask = static_cast<std::size_t>(xmask);
    values[i0 ^ mask] = tmp[0];
    values[(i0 + 1) ^ mask] = tmp[1];
    values[(i0 + 2) ^ mask] = tmp[2];
    values[(i0 + 3) ^ mask] = tmp[3];
}

void avx2_mul_assign_soa(double* re, double* im, const Complex* coeff, double c, std::size_t n) {
    const auto* cr = reinterpret_cast<const double*>(coeff);
    const __m256d vc = _mm256_set1_pd(c);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d r = _mm256_loadu_pd(re + i);
        const __m256d v = _mm256_loadu_pd(im + i);
        const __m256d cre = _mm256_add_pd(vc, load_complex_re4(cr + 2 * i));
        const __m256d cim = load_complex_im4(cr + 2 * i);
        _mm256_storeu_pd(re + i, _mm256_fnmadd_pd(cim, v, _mm256_mul_pd(cre, r)));
        _mm256_storeu_pd(im + i, _mm256_fmadd_pd(cim, r, _mm256_mul_pd(cre, v)));
    }
    for (; i < n; ++i) {
        const double fr = c + cr[2 * i];
        const double fi = cr[2 * i + 1];
        const double r = re[i];
        const double v = im[i];
        re[i] = fr * r - fi * v;
        im[i] = fr * v + fi * r;
    }
}

double avx2_norm_sum_soa(const double* re, const double* im, const std::size_t* indices, std::size_t n) {
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

void avx2_rotate_uniform_imag_pairs_soa(
    double* re,
    double* im,
    std::size_t dim,
    std::uint64_t xmask,
    unsigned pair_bit,
    double c,
    double q) {
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t step = selector << 1;
    const bool contiguous_partner = (static_cast<std::size_t>(xmask) & (selector - 1)) == 0;
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
    for (std::size_t block = 0; block < dim; block += step) {
        std::size_t offset = 0;
        if (contiguous_partner) {
            const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
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
        }
        if (!contiguous_partner) {
            for (; offset + 4 <= selector; offset += 4) {
                const std::size_t i0 = block + offset;
                const __m256d r0 = _mm256_loadu_pd(re + i0);
                const __m256d im0 = _mm256_loadu_pd(im + i0);
                const __m256d r1 = gather_pair4(re, i0, xmask);
                const __m256d im1 = gather_pair4(im, i0, xmask);
                _mm256_storeu_pd(re + i0, _mm256_fnmadd_pd(vq, im1, _mm256_mul_pd(vc, r0)));
                _mm256_storeu_pd(im + i0, _mm256_fmadd_pd(vq, r1, _mm256_mul_pd(vc, im0)));
                scatter_pair4(re, i0, xmask, _mm256_fnmadd_pd(vq, im0, _mm256_mul_pd(vc, r1)));
                scatter_pair4(im, i0, xmask, _mm256_fmadd_pd(vq, r0, _mm256_mul_pd(vc, im1)));
            }
        }
        for (; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = i0 ^ static_cast<std::size_t>(xmask);
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

void avx2_rotate_real_pair_flip_soa(
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
    const bool contiguous_partner = (static_cast<std::size_t>(xmask) & (selector - 1)) == 0;
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
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        std::size_t offset = 0;
        if (contiguous_partner) {
            const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
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
        }
        if (!contiguous_partner) {
            for (; offset + 4 <= selector; offset += 4) {
                const std::size_t i0 = block + offset;
                const __m256d q = _mm256_mul_pd(_mm256_loadu_pd(phase_signs + pair_idx), vbase);
                const __m256d r0 = _mm256_loadu_pd(re + i0);
                const __m256d im0 = _mm256_loadu_pd(im + i0);
                const __m256d r1 = gather_pair4(re, i0, xmask);
                const __m256d im1 = gather_pair4(im, i0, xmask);
                _mm256_storeu_pd(re + i0, _mm256_fnmadd_pd(q, r1, _mm256_mul_pd(vc, r0)));
                _mm256_storeu_pd(im + i0, _mm256_fnmadd_pd(q, im1, _mm256_mul_pd(vc, im0)));
                scatter_pair4(re, i0, xmask, _mm256_fmadd_pd(q, r0, _mm256_mul_pd(vc, r1)));
                scatter_pair4(im, i0, xmask, _mm256_fmadd_pd(q, im0, _mm256_mul_pd(vc, im1)));
                pair_idx += 4;
            }
        }
        for (; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = i0 ^ static_cast<std::size_t>(xmask);
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

void avx2_rotate_general_pairs_soa(
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
    const bool contiguous_partner = (static_cast<std::size_t>(xmask) & (selector - 1)) == 0;
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
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        std::size_t offset = 0;
        if (contiguous_partner) {
            const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
            for (; offset + 4 <= selector; offset += 4) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = right_base + offset;
                const __m256d r0 = _mm256_loadu_pd(re + i0);
                const __m256d im0 = _mm256_loadu_pd(im + i0);
                const __m256d r1 = _mm256_loadu_pd(re + i1);
                const __m256d im1 = _mm256_loadu_pd(im + i1);
                const __m256d lr = load_complex_re4(left + 2 * pair_idx);
                const __m256d li = load_complex_im4(left + 2 * pair_idx);
                const __m256d rr = load_complex_re4(right + 2 * pair_idx);
                const __m256d ri = load_complex_im4(right + 2 * pair_idx);
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
        }
        if (!contiguous_partner) {
            for (; offset + 4 <= selector; offset += 4) {
                const std::size_t i0 = block + offset;
                const __m256d r0 = _mm256_loadu_pd(re + i0);
                const __m256d im0 = _mm256_loadu_pd(im + i0);
                const __m256d r1 = gather_pair4(re, i0, xmask);
                const __m256d im1 = gather_pair4(im, i0, xmask);
                const __m256d lr = load_complex_re4(left + 2 * pair_idx);
                const __m256d li = load_complex_im4(left + 2 * pair_idx);
                const __m256d rr = load_complex_re4(right + 2 * pair_idx);
                const __m256d ri = load_complex_im4(right + 2 * pair_idx);
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
                scatter_pair4(re, i0, xmask, out_r1);
                scatter_pair4(im, i0, xmask, out_i1);
                pair_idx += 4;
            }
        }
        for (; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = i0 ^ static_cast<std::size_t>(xmask);
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
    "avx2",
    avx2_mul_assign,
    avx2_norm_sum,
    avx2_rotate_uniform_imag_pairs,
    avx2_rotate_real_pair_flip,
    avx2_mul_assign_soa,
    avx2_norm_sum_soa,
    avx2_rotate_uniform_imag_pairs_soa,
    avx2_rotate_real_pair_flip_soa,
    avx2_rotate_general_pairs_soa,
};

} // namespace

const KernelTable& avx2_table() {
    return table;
}

} // namespace symft::simd
