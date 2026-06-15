#include "symft/batch_simd.hpp"

#include <immintrin.h>

namespace symft::batch_simd {
namespace {

constexpr double kInvSqrt2 = 0.70710678118654752440;

bool batch_bit(const std::uint64_t* bits, int shot) {
    return (bits[static_cast<std::size_t>(shot >> 6)] & (std::uint64_t{1} << (shot & 63))) != 0;
}

bool is_odd_popcount(std::uint64_t value) {
    return (__builtin_popcountll(value) & 1) != 0;
}

std::uint64_t insert_zero_bit(std::uint64_t packed, unsigned bit) {
    const std::uint64_t low_mask = (std::uint64_t{1} << bit) - std::uint64_t{1};
    const std::uint64_t low = packed & low_mask;
    const std::uint64_t high = packed & ~low_mask;
    return low | (high << 1);
}

__mmask8 branch_mask8(const std::uint64_t* bits, int shot) {
    unsigned mask = 0;
    for (int lane = 0; lane < 8; ++lane) {
        if (batch_bit(bits, shot + lane)) {
            mask |= 1u << lane;
        }
    }
    return static_cast<__mmask8>(mask);
}

void zero_prob(double* prob_true, int active_shots) {
    int shot = 0;
    const __m512d zero = _mm512_setzero_pd();
    for (; shot + 8 <= active_shots; shot += 8) {
        _mm512_storeu_pd(prob_true + shot, zero);
    }
    for (; shot < active_shots; ++shot) {
        prob_true[shot] = 0.0;
    }
}

void avx512_rotate_uniform_imag_pairs(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* left_indices,
    const std::size_t* right_indices,
    std::size_t npairs,
    double c,
    const double* q_by_shot) {
    const __m512d vc = _mm512_set1_pd(c);
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        double* r0p = re + left_indices[idx] * leading_shots;
        double* i0p = im + left_indices[idx] * leading_shots;
        double* r1p = re + right_indices[idx] * leading_shots;
        double* i1p = im + right_indices[idx] * leading_shots;
        int shot = 0;
        for (; shot + 8 <= active_shots; shot += 8) {
            const __m512d q = _mm512_loadu_pd(q_by_shot + shot);
            const __m512d r0 = _mm512_loadu_pd(r0p + shot);
            const __m512d i0 = _mm512_loadu_pd(i0p + shot);
            const __m512d r1 = _mm512_loadu_pd(r1p + shot);
            const __m512d i1 = _mm512_loadu_pd(i1p + shot);
            _mm512_storeu_pd(r0p + shot, _mm512_fmsub_pd(vc, r0, _mm512_mul_pd(q, i1)));
            _mm512_storeu_pd(i0p + shot, _mm512_fmadd_pd(q, r1, _mm512_mul_pd(vc, i0)));
            _mm512_storeu_pd(r1p + shot, _mm512_fmsub_pd(vc, r1, _mm512_mul_pd(q, i0)));
            _mm512_storeu_pd(i1p + shot, _mm512_fmadd_pd(q, r0, _mm512_mul_pd(vc, i1)));
        }
        for (; shot < active_shots; ++shot) {
            const double q = q_by_shot[shot];
            const double r0 = r0p[shot];
            const double i0 = i0p[shot];
            const double r1 = r1p[shot];
            const double i1 = i1p[shot];
            r0p[shot] = c * r0 - q * i1;
            i0p[shot] = c * i0 + q * r1;
            r1p[shot] = c * r1 - q * i0;
            i1p[shot] = c * i1 + q * r0;
        }
    }
}

void avx512_rotate_uniform_imag_xmask(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    double c,
    const double* q_by_shot) {
    const __m512d vc = _mm512_set1_pd(c);
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        const std::uint64_t left = insert_zero_bit(static_cast<std::uint64_t>(idx), pair_bit);
        const std::uint64_t right = left ^ xmask;
        double* r0p = re + static_cast<std::size_t>(left) * leading_shots;
        double* i0p = im + static_cast<std::size_t>(left) * leading_shots;
        double* r1p = re + static_cast<std::size_t>(right) * leading_shots;
        double* i1p = im + static_cast<std::size_t>(right) * leading_shots;
        int shot = 0;
        for (; shot + 8 <= active_shots; shot += 8) {
            const __m512d q = _mm512_loadu_pd(q_by_shot + shot);
            const __m512d r0 = _mm512_loadu_pd(r0p + shot);
            const __m512d i0 = _mm512_loadu_pd(i0p + shot);
            const __m512d r1 = _mm512_loadu_pd(r1p + shot);
            const __m512d i1 = _mm512_loadu_pd(i1p + shot);
            _mm512_storeu_pd(r0p + shot, _mm512_fmsub_pd(vc, r0, _mm512_mul_pd(q, i1)));
            _mm512_storeu_pd(i0p + shot, _mm512_fmadd_pd(q, r1, _mm512_mul_pd(vc, i0)));
            _mm512_storeu_pd(r1p + shot, _mm512_fmsub_pd(vc, r1, _mm512_mul_pd(q, i0)));
            _mm512_storeu_pd(i1p + shot, _mm512_fmadd_pd(q, r0, _mm512_mul_pd(vc, i1)));
        }
        for (; shot < active_shots; ++shot) {
            const double q = q_by_shot[shot];
            const double r0 = r0p[shot];
            const double i0 = i0p[shot];
            const double r1 = r1p[shot];
            const double i1 = i1p[shot];
            r0p[shot] = c * r0 - q * i1;
            i0p[shot] = c * i0 + q * r1;
            r1p[shot] = c * r1 - q * i0;
            i1p[shot] = c * i1 + q * r0;
        }
    }
}

void avx512_rotate_real_pair_flip(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* left_indices,
    const std::size_t* right_indices,
    std::size_t npairs,
    double c,
    const double* q_by_shot,
    std::uint64_t zmask) {
    const __m512d vc = _mm512_set1_pd(c);
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        const std::size_t i0 = left_indices[idx];
        const double phase_sign = is_odd_popcount(static_cast<std::uint64_t>(i0) & zmask) ? -1.0 : 1.0;
        const __m512d vphase = _mm512_set1_pd(phase_sign);
        double* r0p = re + i0 * leading_shots;
        double* i0p = im + i0 * leading_shots;
        double* r1p = re + right_indices[idx] * leading_shots;
        double* i1p = im + right_indices[idx] * leading_shots;
        int shot = 0;
        for (; shot + 8 <= active_shots; shot += 8) {
            const __m512d q = _mm512_mul_pd(vphase, _mm512_loadu_pd(q_by_shot + shot));
            const __m512d r0 = _mm512_loadu_pd(r0p + shot);
            const __m512d i0v = _mm512_loadu_pd(i0p + shot);
            const __m512d r1 = _mm512_loadu_pd(r1p + shot);
            const __m512d i1 = _mm512_loadu_pd(i1p + shot);
            _mm512_storeu_pd(r0p + shot, _mm512_fmsub_pd(vc, r0, _mm512_mul_pd(q, r1)));
            _mm512_storeu_pd(i0p + shot, _mm512_fmsub_pd(vc, i0v, _mm512_mul_pd(q, i1)));
            _mm512_storeu_pd(r1p + shot, _mm512_fmadd_pd(q, r0, _mm512_mul_pd(vc, r1)));
            _mm512_storeu_pd(i1p + shot, _mm512_fmadd_pd(q, i0v, _mm512_mul_pd(vc, i1)));
        }
        for (; shot < active_shots; ++shot) {
            const double q = phase_sign * q_by_shot[shot];
            const double r0 = r0p[shot];
            const double i0v = i0p[shot];
            const double r1 = r1p[shot];
            const double i1 = i1p[shot];
            r0p[shot] = c * r0 - q * r1;
            i0p[shot] = c * i0v - q * i1;
            r1p[shot] = c * r1 + q * r0;
            i1p[shot] = c * i1 + q * i0v;
        }
    }
}

void avx512_rotate_real_pair_flip_xmask(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    double c,
    const double* q_by_shot,
    std::uint64_t zmask) {
    const __m512d vc = _mm512_set1_pd(c);
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        const std::uint64_t left = insert_zero_bit(static_cast<std::uint64_t>(idx), pair_bit);
        const std::uint64_t right = left ^ xmask;
        const double phase_sign = is_odd_popcount(left & zmask) ? -1.0 : 1.0;
        const __m512d vphase = _mm512_set1_pd(phase_sign);
        double* r0p = re + static_cast<std::size_t>(left) * leading_shots;
        double* i0p = im + static_cast<std::size_t>(left) * leading_shots;
        double* r1p = re + static_cast<std::size_t>(right) * leading_shots;
        double* i1p = im + static_cast<std::size_t>(right) * leading_shots;
        int shot = 0;
        for (; shot + 8 <= active_shots; shot += 8) {
            const __m512d q = _mm512_mul_pd(vphase, _mm512_loadu_pd(q_by_shot + shot));
            const __m512d r0 = _mm512_loadu_pd(r0p + shot);
            const __m512d i0v = _mm512_loadu_pd(i0p + shot);
            const __m512d r1 = _mm512_loadu_pd(r1p + shot);
            const __m512d i1 = _mm512_loadu_pd(i1p + shot);
            _mm512_storeu_pd(r0p + shot, _mm512_fmsub_pd(vc, r0, _mm512_mul_pd(q, r1)));
            _mm512_storeu_pd(i0p + shot, _mm512_fmsub_pd(vc, i0v, _mm512_mul_pd(q, i1)));
            _mm512_storeu_pd(r1p + shot, _mm512_fmadd_pd(q, r0, _mm512_mul_pd(vc, r1)));
            _mm512_storeu_pd(i1p + shot, _mm512_fmadd_pd(q, i0v, _mm512_mul_pd(vc, i1)));
        }
        for (; shot < active_shots; ++shot) {
            const double q = phase_sign * q_by_shot[shot];
            const double r0 = r0p[shot];
            const double i0v = i0p[shot];
            const double r1 = r1p[shot];
            const double i1 = i1p[shot];
            r0p[shot] = c * r0 - q * r1;
            i0p[shot] = c * i0v - q * i1;
            r1p[shot] = c * r1 + q * r0;
            i1p[shot] = c * i1 + q * i0v;
        }
    }
}

void avx512_rotate_diagonal(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::size_t dim,
    double c,
    const Complex* minus_coefficients,
    const Complex* plus_coefficients,
    const std::uint64_t* sign_bits) {
    for (std::size_t basis = 0; basis < dim; ++basis) {
        double* rp = re + basis * leading_shots;
        double* ip = im + basis * leading_shots;
        const Complex plus = plus_coefficients[basis];
        const Complex minus = minus_coefficients[basis];
        const __m512d fr_plus = _mm512_set1_pd(c + plus.real());
        const __m512d fi_plus = _mm512_set1_pd(plus.imag());
        const __m512d fr_minus = _mm512_set1_pd(c + minus.real());
        const __m512d fi_minus = _mm512_set1_pd(minus.imag());
        int shot = 0;
        for (; shot + 8 <= active_shots; shot += 8) {
            const __mmask8 mask = branch_mask8(sign_bits, shot);
            const __m512d fr = _mm512_mask_blend_pd(mask, fr_minus, fr_plus);
            const __m512d fi = _mm512_mask_blend_pd(mask, fi_minus, fi_plus);
            const __m512d r = _mm512_loadu_pd(rp + shot);
            const __m512d i = _mm512_loadu_pd(ip + shot);
            _mm512_storeu_pd(rp + shot, _mm512_fmsub_pd(fr, r, _mm512_mul_pd(fi, i)));
            _mm512_storeu_pd(ip + shot, _mm512_fmadd_pd(fi, r, _mm512_mul_pd(fr, i)));
        }
        for (; shot < active_shots; ++shot) {
            const Complex coeff = batch_bit(sign_bits, shot) ? plus : minus;
            const double fr = c + coeff.real();
            const double fi = coeff.imag();
            const double r = rp[shot];
            const double i = ip[shot];
            rp[shot] = fr * r - fi * i;
            ip[shot] = fr * i + fi * r;
        }
    }
}

void avx512_promote_first_dormant_rotation(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::size_t dim,
    double c,
    const double* q_by_shot) {
    const __m512d vc = _mm512_set1_pd(c);
    const __m512d zero = _mm512_setzero_pd();
    for (std::size_t basis = 0; basis < dim; ++basis) {
        double* r0p = re + basis * leading_shots;
        double* i0p = im + basis * leading_shots;
        double* r1p = re + (dim + basis) * leading_shots;
        double* i1p = im + (dim + basis) * leading_shots;
        int shot = 0;
        for (; shot + 8 <= active_shots; shot += 8) {
            const __m512d q = _mm512_loadu_pd(q_by_shot + shot);
            const __m512d r = _mm512_loadu_pd(r0p + shot);
            const __m512d i = _mm512_loadu_pd(i0p + shot);
            _mm512_storeu_pd(r0p + shot, _mm512_mul_pd(vc, r));
            _mm512_storeu_pd(i0p + shot, _mm512_mul_pd(vc, i));
            _mm512_storeu_pd(r1p + shot, _mm512_sub_pd(zero, _mm512_mul_pd(q, i)));
            _mm512_storeu_pd(i1p + shot, _mm512_mul_pd(q, r));
        }
        for (; shot < active_shots; ++shot) {
            const double q = q_by_shot[shot];
            const double r = r0p[shot];
            const double i = i0p[shot];
            r0p[shot] = c * r;
            i0p[shot] = c * i;
            r1p[shot] = -q * i;
            i1p[shot] = q * r;
        }
    }
}

void avx512_last_z_measure_true_prob(
    const double* re,
    const double* im,
    std::size_t leading_shots,
    int active_shots,
    std::size_t dim,
    double* prob_true) {
    zero_prob(prob_true, active_shots);
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const double* rp = re + (dim + basis) * leading_shots;
        const double* ip = im + (dim + basis) * leading_shots;
        int shot = 0;
        for (; shot + 8 <= active_shots; shot += 8) {
            const __m512d r = _mm512_loadu_pd(rp + shot);
            const __m512d i = _mm512_loadu_pd(ip + shot);
            const __m512d old = _mm512_loadu_pd(prob_true + shot);
            const __m512d norm = _mm512_fmadd_pd(i, i, _mm512_mul_pd(r, r));
            _mm512_storeu_pd(prob_true + shot, _mm512_add_pd(old, norm));
        }
        for (; shot < active_shots; ++shot) {
            const double r = rp[shot];
            const double i = ip[shot];
            prob_true[shot] += r * r + i * i;
        }
    }
}

void avx512_last_z_project(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::size_t dim,
    const std::uint64_t* branch_bits,
    const double* invnorms) {
    for (std::size_t basis = 0; basis < dim; ++basis) {
        double* dst_r = re + basis * leading_shots;
        double* dst_i = im + basis * leading_shots;
        const double* false_r = re + basis * leading_shots;
        const double* false_i = im + basis * leading_shots;
        const double* true_r = re + (dim + basis) * leading_shots;
        const double* true_i = im + (dim + basis) * leading_shots;
        int shot = 0;
        for (; shot + 8 <= active_shots; shot += 8) {
            const __mmask8 mask = branch_mask8(branch_bits, shot);
            const __m512d n = _mm512_loadu_pd(invnorms + shot);
            const __m512d sr = _mm512_mask_blend_pd(mask, _mm512_loadu_pd(false_r + shot), _mm512_loadu_pd(true_r + shot));
            const __m512d si = _mm512_mask_blend_pd(mask, _mm512_loadu_pd(false_i + shot), _mm512_loadu_pd(true_i + shot));
            _mm512_storeu_pd(dst_r + shot, _mm512_mul_pd(sr, n));
            _mm512_storeu_pd(dst_i + shot, _mm512_mul_pd(si, n));
        }
        for (; shot < active_shots; ++shot) {
            const bool branch = batch_bit(branch_bits, shot);
            const double n = invnorms[shot];
            dst_r[shot] = (branch ? true_r[shot] : false_r[shot]) * n;
            dst_i[shot] = (branch ? true_i[shot] : false_i[shot]) * n;
        }
    }
}

void avx512_diagonal_measure_true_prob(
    const double* re,
    const double* im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source_true,
    std::size_t out_dim,
    double* prob_true) {
    zero_prob(prob_true, active_shots);
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const double* rp = re + source_true[idx] * leading_shots;
        const double* ip = im + source_true[idx] * leading_shots;
        int shot = 0;
        for (; shot + 8 <= active_shots; shot += 8) {
            const __m512d r = _mm512_loadu_pd(rp + shot);
            const __m512d i = _mm512_loadu_pd(ip + shot);
            const __m512d old = _mm512_loadu_pd(prob_true + shot);
            const __m512d norm = _mm512_fmadd_pd(i, i, _mm512_mul_pd(r, r));
            _mm512_storeu_pd(prob_true + shot, _mm512_add_pd(old, norm));
        }
        for (; shot < active_shots; ++shot) {
            const double r = rp[shot];
            const double i = ip[shot];
            prob_true[shot] += r * r + i * i;
        }
    }
}

void avx512_diagonal_project(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source_false,
    const std::size_t* source_true,
    std::size_t out_dim,
    const std::uint64_t* branch_bits,
    const double* invnorms) {
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        double* dst_r = re + idx * leading_shots;
        double* dst_i = im + idx * leading_shots;
        const double* false_r = re + source_false[idx] * leading_shots;
        const double* false_i = im + source_false[idx] * leading_shots;
        const double* true_r = re + source_true[idx] * leading_shots;
        const double* true_i = im + source_true[idx] * leading_shots;
        int shot = 0;
        for (; shot + 8 <= active_shots; shot += 8) {
            const __mmask8 mask = branch_mask8(branch_bits, shot);
            const __m512d n = _mm512_loadu_pd(invnorms + shot);
            const __m512d sr = _mm512_mask_blend_pd(mask, _mm512_loadu_pd(false_r + shot), _mm512_loadu_pd(true_r + shot));
            const __m512d si = _mm512_mask_blend_pd(mask, _mm512_loadu_pd(false_i + shot), _mm512_loadu_pd(true_i + shot));
            _mm512_storeu_pd(dst_r + shot, _mm512_mul_pd(sr, n));
            _mm512_storeu_pd(dst_i + shot, _mm512_mul_pd(si, n));
        }
        for (; shot < active_shots; ++shot) {
            const bool branch = batch_bit(branch_bits, shot);
            const double n = invnorms[shot];
            dst_r[shot] = (branch ? true_r[shot] : false_r[shot]) * n;
            dst_i[shot] = (branch ? true_i[shot] : false_i[shot]) * n;
        }
    }
}

void avx512_nondiagonal_measure_true_prob(
    const double* re,
    const double* im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source0_false,
    const std::size_t* source1_false,
    const Complex* coeff1_false,
    std::size_t out_dim,
    double* prob_true) {
    zero_prob(prob_true, active_shots);
    const __m512d vinv = _mm512_set1_pd(kInvSqrt2);
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const double* r0p = re + source0_false[idx] * leading_shots;
        const double* i0p = im + source0_false[idx] * leading_shots;
        const double* r1p = re + source1_false[idx] * leading_shots;
        const double* i1p = im + source1_false[idx] * leading_shots;
        const __m512d c1r = _mm512_set1_pd(coeff1_false[idx].real());
        const __m512d c1i = _mm512_set1_pd(coeff1_false[idx].imag());
        int shot = 0;
        for (; shot + 8 <= active_shots; shot += 8) {
            const __m512d r0 = _mm512_loadu_pd(r0p + shot);
            const __m512d i0 = _mm512_loadu_pd(i0p + shot);
            const __m512d r1 = _mm512_loadu_pd(r1p + shot);
            const __m512d i1 = _mm512_loadu_pd(i1p + shot);
            const __m512d prod_r = _mm512_fmsub_pd(c1r, r1, _mm512_mul_pd(c1i, i1));
            const __m512d prod_i = _mm512_fmadd_pd(c1i, r1, _mm512_mul_pd(c1r, i1));
            const __m512d amp_r = _mm512_sub_pd(_mm512_mul_pd(vinv, r0), prod_r);
            const __m512d amp_i = _mm512_sub_pd(_mm512_mul_pd(vinv, i0), prod_i);
            const __m512d old = _mm512_loadu_pd(prob_true + shot);
            const __m512d norm = _mm512_fmadd_pd(amp_i, amp_i, _mm512_mul_pd(amp_r, amp_r));
            _mm512_storeu_pd(prob_true + shot, _mm512_add_pd(old, norm));
        }
        for (; shot < active_shots; ++shot) {
            const double c1rs = coeff1_false[idx].real();
            const double c1is = coeff1_false[idx].imag();
            const double prod_r = c1rs * r1p[shot] - c1is * i1p[shot];
            const double prod_i = c1rs * i1p[shot] + c1is * r1p[shot];
            const double amp_r = kInvSqrt2 * r0p[shot] - prod_r;
            const double amp_i = kInvSqrt2 * i0p[shot] - prod_i;
            prob_true[shot] += amp_r * amp_r + amp_i * amp_i;
        }
    }
}

void avx512_nondiagonal_project(
    const double* re,
    const double* im,
    double* scratch_re,
    double* scratch_im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source0_false,
    const std::size_t* source1_false,
    const Complex* coeff1_false,
    std::size_t out_dim,
    const std::uint64_t* branch_bits,
    const double* invnorms) {
    const __m512d vinv = _mm512_set1_pd(kInvSqrt2);
    const __m512d vpos = _mm512_set1_pd(1.0);
    const __m512d vneg = _mm512_set1_pd(-1.0);
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const double* r0p = re + source0_false[idx] * leading_shots;
        const double* i0p = im + source0_false[idx] * leading_shots;
        const double* r1p = re + source1_false[idx] * leading_shots;
        const double* i1p = im + source1_false[idx] * leading_shots;
        double* dst_r = scratch_re + idx * leading_shots;
        double* dst_i = scratch_im + idx * leading_shots;
        const __m512d c1r = _mm512_set1_pd(coeff1_false[idx].real());
        const __m512d c1i = _mm512_set1_pd(coeff1_false[idx].imag());
        int shot = 0;
        for (; shot + 8 <= active_shots; shot += 8) {
            const __m512d sign = _mm512_mask_blend_pd(branch_mask8(branch_bits, shot), vpos, vneg);
            const __m512d n = _mm512_loadu_pd(invnorms + shot);
            const __m512d r0 = _mm512_loadu_pd(r0p + shot);
            const __m512d i0 = _mm512_loadu_pd(i0p + shot);
            const __m512d r1 = _mm512_loadu_pd(r1p + shot);
            const __m512d i1 = _mm512_loadu_pd(i1p + shot);
            const __m512d prod_r = _mm512_fmsub_pd(c1r, r1, _mm512_mul_pd(c1i, i1));
            const __m512d prod_i = _mm512_fmadd_pd(c1i, r1, _mm512_mul_pd(c1r, i1));
            const __m512d amp_r = _mm512_fmadd_pd(sign, prod_r, _mm512_mul_pd(vinv, r0));
            const __m512d amp_i = _mm512_fmadd_pd(sign, prod_i, _mm512_mul_pd(vinv, i0));
            _mm512_storeu_pd(dst_r + shot, _mm512_mul_pd(amp_r, n));
            _mm512_storeu_pd(dst_i + shot, _mm512_mul_pd(amp_i, n));
        }
        for (; shot < active_shots; ++shot) {
            const double c1rs = coeff1_false[idx].real();
            const double c1is = coeff1_false[idx].imag();
            const double branch_sign = batch_bit(branch_bits, shot) ? -1.0 : 1.0;
            const double prod_r = c1rs * r1p[shot] - c1is * i1p[shot];
            const double prod_i = c1rs * i1p[shot] + c1is * r1p[shot];
            const double n = invnorms[shot];
            dst_r[shot] = (kInvSqrt2 * r0p[shot] + branch_sign * prod_r) * n;
            dst_i[shot] = (kInvSqrt2 * i0p[shot] + branch_sign * prod_i) * n;
        }
    }
}

const KernelTable table = {
    "avx512",
    avx512_rotate_uniform_imag_pairs,
    avx512_rotate_uniform_imag_xmask,
    avx512_rotate_real_pair_flip,
    avx512_rotate_real_pair_flip_xmask,
    avx512_rotate_diagonal,
    avx512_promote_first_dormant_rotation,
    avx512_last_z_measure_true_prob,
    avx512_last_z_project,
    avx512_diagonal_measure_true_prob,
    avx512_diagonal_project,
    avx512_nondiagonal_measure_true_prob,
    avx512_nondiagonal_project,
};

} // namespace

const KernelTable& avx512_table() {
    return table;
}

} // namespace symft::batch_simd
