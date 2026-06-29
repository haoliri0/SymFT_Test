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

__m512i pair_indices8(std::size_t i0, std::uint64_t xmask) {
    const auto mask = static_cast<std::size_t>(xmask);
    return _mm512_set_epi64(
        static_cast<long long>((i0 + 7) ^ mask),
        static_cast<long long>((i0 + 6) ^ mask),
        static_cast<long long>((i0 + 5) ^ mask),
        static_cast<long long>((i0 + 4) ^ mask),
        static_cast<long long>((i0 + 3) ^ mask),
        static_cast<long long>((i0 + 2) ^ mask),
        static_cast<long long>((i0 + 1) ^ mask),
        static_cast<long long>(i0 ^ mask));
}

__m512d gather_pair8(const double* values, std::size_t i0, std::uint64_t xmask) {
    return _mm512_i64gather_pd(pair_indices8(i0, xmask), values, 8);
}

void scatter_pair8(double* values, std::size_t i0, std::uint64_t xmask, __m512d lanes) {
    _mm512_i64scatter_pd(values, pair_indices8(i0, xmask), lanes, 8);
}

__m512i complex_indices8(std::size_t pair_idx, int component) {
    const std::size_t base = pair_idx << 1;
    return _mm512_set_epi64(
        static_cast<long long>(base + 14 + component),
        static_cast<long long>(base + 12 + component),
        static_cast<long long>(base + 10 + component),
        static_cast<long long>(base + 8 + component),
        static_cast<long long>(base + 6 + component),
        static_cast<long long>(base + 4 + component),
        static_cast<long long>(base + 2 + component),
        static_cast<long long>(base + component));
}

__m512d load_complex_re8(const double* coeff, std::size_t pair_idx) {
    return _mm512_i64gather_pd(complex_indices8(pair_idx, 0), coeff, 8);
}

__m512d load_complex_im8(const double* coeff, std::size_t pair_idx) {
    return _mm512_i64gather_pd(complex_indices8(pair_idx, 1), coeff, 8);
}

std::size_t xor_contiguous_run(std::size_t lower_mask, std::size_t selector) {
    return lower_mask == 0 ? selector : (lower_mask & (~lower_mask + std::size_t{1}));
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
    const std::size_t lower_mask = static_cast<std::size_t>(xmask) & (selector - 1);
    const bool contiguous_partner = lower_mask == 0;
    const std::size_t xor_run = xor_contiguous_run(lower_mask, selector);
    const __m512d vc = _mm512_set1_pd(c);
    const __m512d vq = _mm512_set1_pd(q);
    for (std::size_t block = 0; block < dim; block += step) {
        std::size_t offset = 0;
        if (contiguous_partner) {
            const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
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
        }
        if (!contiguous_partner) {
            if (xor_run >= 8) {
                for (std::size_t segment = 0; segment < selector; segment += xor_run) {
                    const std::size_t right_base = block + selector + (segment ^ lower_mask);
                    for (std::size_t local = 0; local + 8 <= xor_run; local += 8) {
                        const std::size_t i0 = block + segment + local;
                        const std::size_t i1 = right_base + local;
                        const __m512d r0 = _mm512_loadu_pd(re + i0);
                        const __m512d im0 = _mm512_loadu_pd(im + i0);
                        const __m512d r1 = _mm512_loadu_pd(re + i1);
                        const __m512d im1 = _mm512_loadu_pd(im + i1);
                        _mm512_storeu_pd(re + i0, _mm512_fnmadd_pd(vq, im1, _mm512_mul_pd(vc, r0)));
                        _mm512_storeu_pd(im + i0, _mm512_fmadd_pd(vq, r1, _mm512_mul_pd(vc, im0)));
                        _mm512_storeu_pd(re + i1, _mm512_fnmadd_pd(vq, im0, _mm512_mul_pd(vc, r1)));
                        _mm512_storeu_pd(im + i1, _mm512_fmadd_pd(vq, r0, _mm512_mul_pd(vc, im1)));
                    }
                }
                offset = selector;
            } else {
                for (; offset + 8 <= selector; offset += 8) {
                    const std::size_t i0 = block + offset;
                    const __m512d r0 = _mm512_loadu_pd(re + i0);
                    const __m512d im0 = _mm512_loadu_pd(im + i0);
                    const __m512d r1 = gather_pair8(re, i0, xmask);
                    const __m512d im1 = gather_pair8(im, i0, xmask);
                    _mm512_storeu_pd(re + i0, _mm512_fnmadd_pd(vq, im1, _mm512_mul_pd(vc, r0)));
                    _mm512_storeu_pd(im + i0, _mm512_fmadd_pd(vq, r1, _mm512_mul_pd(vc, im0)));
                    scatter_pair8(re, i0, xmask, _mm512_fnmadd_pd(vq, im0, _mm512_mul_pd(vc, r1)));
                    scatter_pair8(im, i0, xmask, _mm512_fmadd_pd(vq, r0, _mm512_mul_pd(vc, im1)));
                }
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
    const std::size_t lower_mask = static_cast<std::size_t>(xmask) & (selector - 1);
    const bool contiguous_partner = lower_mask == 0;
    const std::size_t xor_run = xor_contiguous_run(lower_mask, selector);
    const __m512d vc = _mm512_set1_pd(c);
    const __m512d vbase = _mm512_set1_pd(base_coeff);
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        std::size_t offset = 0;
        if (contiguous_partner) {
            const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
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
        }
        if (!contiguous_partner) {
            if (xor_run >= 8) {
                for (std::size_t segment = 0; segment < selector; segment += xor_run) {
                    const std::size_t right_base = block + selector + (segment ^ lower_mask);
                    for (std::size_t local = 0; local + 8 <= xor_run; local += 8) {
                        const std::size_t i0 = block + segment + local;
                        const std::size_t i1 = right_base + local;
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
                }
                offset = selector;
            } else {
                for (; offset + 8 <= selector; offset += 8) {
                    const std::size_t i0 = block + offset;
                    const __m512d q = _mm512_mul_pd(_mm512_loadu_pd(phase_signs + pair_idx), vbase);
                    const __m512d r0 = _mm512_loadu_pd(re + i0);
                    const __m512d im0 = _mm512_loadu_pd(im + i0);
                    const __m512d r1 = gather_pair8(re, i0, xmask);
                    const __m512d im1 = gather_pair8(im, i0, xmask);
                    _mm512_storeu_pd(re + i0, _mm512_fnmadd_pd(q, r1, _mm512_mul_pd(vc, r0)));
                    _mm512_storeu_pd(im + i0, _mm512_fnmadd_pd(q, im1, _mm512_mul_pd(vc, im0)));
                    scatter_pair8(re, i0, xmask, _mm512_fmadd_pd(q, r0, _mm512_mul_pd(vc, r1)));
                    scatter_pair8(im, i0, xmask, _mm512_fmadd_pd(q, im0, _mm512_mul_pd(vc, im1)));
                    pair_idx += 8;
                }
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
    const std::size_t lower_mask = static_cast<std::size_t>(xmask) & (selector - 1);
    const bool contiguous_partner = lower_mask == 0;
    const std::size_t xor_run = xor_contiguous_run(lower_mask, selector);
    const __m512d vc = _mm512_set1_pd(c);
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        std::size_t offset = 0;
        if (contiguous_partner) {
            const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
            for (; offset + 8 <= selector; offset += 8) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = right_base + offset;
                const __m512d r0 = _mm512_loadu_pd(re + i0);
                const __m512d im0 = _mm512_loadu_pd(im + i0);
                const __m512d r1 = _mm512_loadu_pd(re + i1);
                const __m512d im1 = _mm512_loadu_pd(im + i1);
                const __m512d lr = load_complex_re8(left, pair_idx);
                const __m512d li = load_complex_im8(left, pair_idx);
                const __m512d rr = load_complex_re8(right, pair_idx);
                const __m512d ri = load_complex_im8(right, pair_idx);
                __m512d out_r0 = _mm512_fmadd_pd(vc, r0, _mm512_mul_pd(rr, r1));
                out_r0 = _mm512_fnmadd_pd(ri, im1, out_r0);
                __m512d out_i0 = _mm512_fmadd_pd(vc, im0, _mm512_mul_pd(rr, im1));
                out_i0 = _mm512_fmadd_pd(ri, r1, out_i0);
                __m512d out_r1 = _mm512_fmadd_pd(vc, r1, _mm512_mul_pd(lr, r0));
                out_r1 = _mm512_fnmadd_pd(li, im0, out_r1);
                __m512d out_i1 = _mm512_fmadd_pd(vc, im1, _mm512_mul_pd(lr, im0));
                out_i1 = _mm512_fmadd_pd(li, r0, out_i1);
                _mm512_storeu_pd(re + i0, out_r0);
                _mm512_storeu_pd(im + i0, out_i0);
                _mm512_storeu_pd(re + i1, out_r1);
                _mm512_storeu_pd(im + i1, out_i1);
                pair_idx += 8;
            }
        }
        if (!contiguous_partner) {
            if (xor_run >= 8) {
                for (std::size_t segment = 0; segment < selector; segment += xor_run) {
                    const std::size_t right_base = block + selector + (segment ^ lower_mask);
                    for (std::size_t local = 0; local + 8 <= xor_run; local += 8) {
                        const std::size_t i0 = block + segment + local;
                        const std::size_t i1 = right_base + local;
                        const __m512d r0 = _mm512_loadu_pd(re + i0);
                        const __m512d im0 = _mm512_loadu_pd(im + i0);
                        const __m512d r1 = _mm512_loadu_pd(re + i1);
                        const __m512d im1 = _mm512_loadu_pd(im + i1);
                        const __m512d lr = load_complex_re8(left, pair_idx);
                        const __m512d li = load_complex_im8(left, pair_idx);
                        const __m512d rr = load_complex_re8(right, pair_idx);
                        const __m512d ri = load_complex_im8(right, pair_idx);
                        __m512d out_r0 = _mm512_fmadd_pd(vc, r0, _mm512_mul_pd(rr, r1));
                        out_r0 = _mm512_fnmadd_pd(ri, im1, out_r0);
                        __m512d out_i0 = _mm512_fmadd_pd(vc, im0, _mm512_mul_pd(rr, im1));
                        out_i0 = _mm512_fmadd_pd(ri, r1, out_i0);
                        __m512d out_r1 = _mm512_fmadd_pd(vc, r1, _mm512_mul_pd(lr, r0));
                        out_r1 = _mm512_fnmadd_pd(li, im0, out_r1);
                        __m512d out_i1 = _mm512_fmadd_pd(vc, im1, _mm512_mul_pd(lr, im0));
                        out_i1 = _mm512_fmadd_pd(li, r0, out_i1);
                        _mm512_storeu_pd(re + i0, out_r0);
                        _mm512_storeu_pd(im + i0, out_i0);
                        _mm512_storeu_pd(re + i1, out_r1);
                        _mm512_storeu_pd(im + i1, out_i1);
                        pair_idx += 8;
                    }
                }
                offset = selector;
            } else {
                for (; offset + 8 <= selector; offset += 8) {
                    const std::size_t i0 = block + offset;
                    const __m512d r0 = _mm512_loadu_pd(re + i0);
                    const __m512d im0 = _mm512_loadu_pd(im + i0);
                    const __m512d r1 = gather_pair8(re, i0, xmask);
                    const __m512d im1 = gather_pair8(im, i0, xmask);
                    const __m512d lr = load_complex_re8(left, pair_idx);
                    const __m512d li = load_complex_im8(left, pair_idx);
                    const __m512d rr = load_complex_re8(right, pair_idx);
                    const __m512d ri = load_complex_im8(right, pair_idx);
                    __m512d out_r0 = _mm512_fmadd_pd(vc, r0, _mm512_mul_pd(rr, r1));
                    out_r0 = _mm512_fnmadd_pd(ri, im1, out_r0);
                    __m512d out_i0 = _mm512_fmadd_pd(vc, im0, _mm512_mul_pd(rr, im1));
                    out_i0 = _mm512_fmadd_pd(ri, r1, out_i0);
                    __m512d out_r1 = _mm512_fmadd_pd(vc, r1, _mm512_mul_pd(lr, r0));
                    out_r1 = _mm512_fnmadd_pd(li, im0, out_r1);
                    __m512d out_i1 = _mm512_fmadd_pd(vc, im1, _mm512_mul_pd(lr, im0));
                    out_i1 = _mm512_fmadd_pd(li, r0, out_i1);
                    _mm512_storeu_pd(re + i0, out_r0);
                    _mm512_storeu_pd(im + i0, out_i0);
                    scatter_pair8(re, i0, xmask, out_r1);
                    scatter_pair8(im, i0, xmask, out_i1);
                    pair_idx += 8;
                }
            }
        }
        SYMFT_SIMD_LOOP
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
    "avx512",
    avx512_mul_assign,
    avx512_norm_sum,
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
