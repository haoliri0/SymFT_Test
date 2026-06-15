#include "symft/batch_interleaved.hpp"

#include <algorithm>
#include <immintrin.h>

namespace symft::batch_interleaved_avx2 {
namespace {

constexpr double kInvSqrt2 = 0.70710678118654752440;

std::uint64_t insert_zero_bit(std::uint64_t packed, unsigned bit) {
    const std::uint64_t low_mask = (std::uint64_t{1} << bit) - std::uint64_t{1};
    const std::uint64_t low = packed & low_mask;
    const std::uint64_t high = packed & ~low_mask;
    return low | (high << 1);
}

bool is_odd_popcount(std::uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return (__builtin_popcountll(value) & 1) != 0;
#else
    bool parity = false;
    while (value != 0) {
        parity = !parity;
        value &= value - 1;
    }
    return parity;
#endif
}

} // namespace

void rotate_uniform_imag_xmask(
    double* active,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    double c,
    const double* q_by_shot) {
    const __m256d vc = _mm256_set1_pd(c);
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        const std::uint64_t left = insert_zero_bit(static_cast<std::uint64_t>(idx), pair_bit);
        const std::uint64_t right = left ^ xmask;
        double* a0 = active + 2 * static_cast<std::size_t>(left) * leading_shots;
        double* a1 = active + 2 * static_cast<std::size_t>(right) * leading_shots;
        int shot = 0;
        for (; shot + 1 < active_shots; shot += 2) {
            const double q0 = q_by_shot[shot];
            const double q1 = q_by_shot[shot + 1];
            const __m256d vq = _mm256_setr_pd(-q0, q0, -q1, q1);
            const __m256d v0 = _mm256_loadu_pd(a0 + 2 * shot);
            const __m256d v1 = _mm256_loadu_pd(a1 + 2 * shot);
            const __m256d x1 = _mm256_permute_pd(v1, 0x5);
            const __m256d x0 = _mm256_permute_pd(v0, 0x5);
            const __m256d out0 = _mm256_fmadd_pd(vq, x1, _mm256_mul_pd(vc, v0));
            const __m256d out1 = _mm256_fmadd_pd(vq, x0, _mm256_mul_pd(vc, v1));
            _mm256_storeu_pd(a0 + 2 * shot, out0);
            _mm256_storeu_pd(a1 + 2 * shot, out1);
        }
        for (; shot < active_shots; ++shot) {
            double* p0 = a0 + 2 * shot;
            double* p1 = a1 + 2 * shot;
            const double q = q_by_shot[shot];
            const double r0 = p0[0];
            const double i0 = p0[1];
            const double r1 = p1[0];
            const double i1 = p1[1];
            p0[0] = c * r0 - q * i1;
            p0[1] = c * i0 + q * r1;
            p1[0] = c * r1 - q * i0;
            p1[1] = c * i1 + q * r0;
        }
    }
}

void rotate_real_pair_flip_xmask(
    double* active,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    double c,
    const double* q_by_shot,
    std::uint64_t zmask) {
    const __m256d vc = _mm256_set1_pd(c);
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        const std::uint64_t left = insert_zero_bit(static_cast<std::uint64_t>(idx), pair_bit);
        const std::uint64_t right = left ^ xmask;
        double* a0 = active + 2 * static_cast<std::size_t>(left) * leading_shots;
        double* a1 = active + 2 * static_cast<std::size_t>(right) * leading_shots;
        const double phase_sign = is_odd_popcount(left & zmask) ? -1.0 : 1.0;
        int shot = 0;
        for (; shot + 1 < active_shots; shot += 2) {
            const double q0 = phase_sign * q_by_shot[shot];
            const double q1 = phase_sign * q_by_shot[shot + 1];
            const __m256d vq = _mm256_setr_pd(q0, q0, q1, q1);
            const __m256d v0 = _mm256_loadu_pd(a0 + 2 * shot);
            const __m256d v1 = _mm256_loadu_pd(a1 + 2 * shot);
            const __m256d out0 = _mm256_fmsub_pd(vc, v0, _mm256_mul_pd(vq, v1));
            const __m256d out1 = _mm256_fmadd_pd(vq, v0, _mm256_mul_pd(vc, v1));
            _mm256_storeu_pd(a0 + 2 * shot, out0);
            _mm256_storeu_pd(a1 + 2 * shot, out1);
        }
        for (; shot < active_shots; ++shot) {
            double* p0 = a0 + 2 * shot;
            double* p1 = a1 + 2 * shot;
            const double q = phase_sign * q_by_shot[shot];
            const double r0 = p0[0];
            const double i0 = p0[1];
            const double r1 = p1[0];
            const double i1 = p1[1];
            p0[0] = c * r0 - q * r1;
            p0[1] = c * i0 - q * i1;
            p1[0] = c * r1 + q * r0;
            p1[1] = c * i1 + q * i0;
        }
    }
}

void diagonal_measure_true_prob(
    const double* active,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source_true,
    std::size_t out_dim,
    double* prob_true) {
    std::fill(prob_true, prob_true + active_shots, 0.0);
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const double* src = active + 2 * source_true[idx] * leading_shots;
        int shot = 0;
        for (; shot + 1 < active_shots; shot += 2) {
            const __m256d v = _mm256_loadu_pd(src + 2 * shot);
            const __m256d sq = _mm256_mul_pd(v, v);
            const __m256d sum_dup = _mm256_add_pd(sq, _mm256_permute_pd(sq, 0x5));
            const __m128d lo = _mm256_castpd256_pd128(sum_dup);
            const __m128d hi = _mm256_extractf128_pd(sum_dup, 1);
            const __m128d sums = _mm_unpacklo_pd(lo, hi);
            const __m128d old_prob = _mm_loadu_pd(prob_true + shot);
            _mm_storeu_pd(prob_true + shot, _mm_add_pd(old_prob, sums));
        }
        for (; shot < active_shots; ++shot) {
            const double* amp = src + 2 * shot;
            prob_true[shot] += amp[0] * amp[0] + amp[1] * amp[1];
        }
    }
}

void diagonal_project(
    const double* active,
    double* scratch,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source_false,
    const std::size_t* source_true,
    std::size_t out_dim,
    const std::uint64_t* branch_bits,
    const double* invnorms) {
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const double* sf = active + 2 * source_false[idx] * leading_shots;
        const double* st = active + 2 * source_true[idx] * leading_shots;
        double* dst = scratch + 2 * idx * leading_shots;
        int shot = 0;
        for (; shot + 1 < active_shots; shot += 2) {
            const std::uint64_t word0 = branch_bits[shot >> 6];
            const std::uint64_t word1 = branch_bits[(shot + 1) >> 6];
            const std::uint64_t bit0 = (word0 >> (shot & 63)) & 1U;
            const std::uint64_t bit1 = (word1 >> ((shot + 1) & 63)) & 1U;
            const long long mask0 = bit0 ? -1LL : 0LL;
            const long long mask1 = bit1 ? -1LL : 0LL;
            const __m256d branch_mask =
                _mm256_castsi256_pd(_mm256_setr_epi64x(mask0, mask0, mask1, mask1));
            const __m256d vf = _mm256_loadu_pd(sf + 2 * shot);
            const __m256d vt = _mm256_loadu_pd(st + 2 * shot);
            const __m256d selected = _mm256_blendv_pd(vf, vt, branch_mask);
            const __m256d norms =
                _mm256_setr_pd(invnorms[shot], invnorms[shot], invnorms[shot + 1], invnorms[shot + 1]);
            _mm256_storeu_pd(dst + 2 * shot, _mm256_mul_pd(selected, norms));
        }
        for (; shot < active_shots; ++shot) {
            const std::uint64_t word = branch_bits[shot >> 6];
            const bool branch = ((word >> (shot & 63)) & 1U) != 0;
            const double* src = branch ? st : sf;
            const double norm = invnorms[shot];
            dst[2 * shot] = src[2 * shot] * norm;
            dst[2 * shot + 1] = src[2 * shot + 1] * norm;
        }
    }
}

void nondiagonal_measure_true_prob(
    const double* active,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source0,
    const std::size_t* source1,
    const Complex* coeff1_false,
    std::size_t out_dim,
    double* prob_true) {
    const __m256d vinvsqrt2 = _mm256_set1_pd(kInvSqrt2);
    std::fill(prob_true, prob_true + active_shots, 0.0);
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const double* s0 = active + 2 * source0[idx] * leading_shots;
        const double* s1 = active + 2 * source1[idx] * leading_shots;
        const double cr = coeff1_false[idx].real();
        const double ci = coeff1_false[idx].imag();
        const __m256d vcr = _mm256_set1_pd(cr);
        const __m256d vci = _mm256_setr_pd(-ci, ci, -ci, ci);
        int shot = 0;
        for (; shot + 1 < active_shots; shot += 2) {
            const __m256d v0 = _mm256_loadu_pd(s0 + 2 * shot);
            const __m256d v1 = _mm256_loadu_pd(s1 + 2 * shot);
            const __m256d v1_swap = _mm256_permute_pd(v1, 0x5);
            const __m256d prod = _mm256_fmadd_pd(vci, v1_swap, _mm256_mul_pd(vcr, v1));
            const __m256d amp = _mm256_sub_pd(_mm256_mul_pd(vinvsqrt2, v0), prod);
            const __m256d sq = _mm256_mul_pd(amp, amp);
            const __m256d sum_dup = _mm256_add_pd(sq, _mm256_permute_pd(sq, 0x5));
            const __m128d lo = _mm256_castpd256_pd128(sum_dup);
            const __m128d hi = _mm256_extractf128_pd(sum_dup, 1);
            const __m128d sums = _mm_unpacklo_pd(lo, hi);
            const __m128d old_prob = _mm_loadu_pd(prob_true + shot);
            _mm_storeu_pd(prob_true + shot, _mm_add_pd(old_prob, sums));
        }
        for (; shot < active_shots; ++shot) {
            const double r0 = s0[2 * shot];
            const double i0 = s0[2 * shot + 1];
            const double r1 = s1[2 * shot];
            const double i1 = s1[2 * shot + 1];
            const double prod_r = cr * r1 - ci * i1;
            const double prod_i = cr * i1 + ci * r1;
            const double amp_r = kInvSqrt2 * r0 - prod_r;
            const double amp_i = kInvSqrt2 * i0 - prod_i;
            prob_true[shot] += amp_r * amp_r + amp_i * amp_i;
        }
    }
}

void nondiagonal_project(
    const double* active,
    double* scratch,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source0,
    const std::size_t* source1,
    const Complex* coeff1_false,
    std::size_t out_dim,
    const std::uint64_t* branch_bits,
    const double* invnorms) {
    const __m256d vinvsqrt2 = _mm256_set1_pd(kInvSqrt2);
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const double* s0 = active + 2 * source0[idx] * leading_shots;
        const double* s1 = active + 2 * source1[idx] * leading_shots;
        double* dst = scratch + 2 * idx * leading_shots;
        const double cr = coeff1_false[idx].real();
        const double ci = coeff1_false[idx].imag();
        const __m256d vcr = _mm256_set1_pd(cr);
        const __m256d vci = _mm256_setr_pd(-ci, ci, -ci, ci);
        int shot = 0;
        for (; shot + 1 < active_shots; shot += 2) {
            const std::uint64_t word0 = branch_bits[shot >> 6];
            const std::uint64_t word1 = branch_bits[(shot + 1) >> 6];
            const double sign0 = ((word0 >> (shot & 63)) & 1U) != 0 ? -1.0 : 1.0;
            const double sign1 = ((word1 >> ((shot + 1) & 63)) & 1U) != 0 ? -1.0 : 1.0;
            const __m256d signs = _mm256_setr_pd(sign0, sign0, sign1, sign1);
            const __m256d norms =
                _mm256_setr_pd(invnorms[shot], invnorms[shot], invnorms[shot + 1], invnorms[shot + 1]);
            const __m256d v0 = _mm256_loadu_pd(s0 + 2 * shot);
            const __m256d v1 = _mm256_loadu_pd(s1 + 2 * shot);
            const __m256d v1_swap = _mm256_permute_pd(v1, 0x5);
            const __m256d prod = _mm256_fmadd_pd(vci, v1_swap, _mm256_mul_pd(vcr, v1));
            const __m256d amp = _mm256_fmadd_pd(signs, prod, _mm256_mul_pd(vinvsqrt2, v0));
            _mm256_storeu_pd(dst + 2 * shot, _mm256_mul_pd(amp, norms));
        }
        for (; shot < active_shots; ++shot) {
            const std::uint64_t word = branch_bits[shot >> 6];
            const double sign = ((word >> (shot & 63)) & 1U) != 0 ? -1.0 : 1.0;
            const double norm = invnorms[shot];
            const double r0 = s0[2 * shot];
            const double i0 = s0[2 * shot + 1];
            const double r1 = s1[2 * shot];
            const double i1 = s1[2 * shot + 1];
            const double prod_r = cr * r1 - ci * i1;
            const double prod_i = cr * i1 + ci * r1;
            dst[2 * shot] = (kInvSqrt2 * r0 + sign * prod_r) * norm;
            dst[2 * shot + 1] = (kInvSqrt2 * i0 + sign * prod_i) * norm;
        }
    }
}

void promote_first_dormant_rotation(
    double* active,
    std::size_t leading_shots,
    int active_shots,
    std::size_t dim,
    double c,
    const double* q_by_shot) {
    const __m256d vc = _mm256_set1_pd(c);
    for (std::size_t idx = 0; idx < dim; ++idx) {
        double* src = active + 2 * idx * leading_shots;
        double* dst = active + 2 * (dim + idx) * leading_shots;
        int shot = 0;
        for (; shot + 1 < active_shots; shot += 2) {
            const double q0 = q_by_shot[shot];
            const double q1 = q_by_shot[shot + 1];
            const __m256d vq = _mm256_setr_pd(-q0, q0, -q1, q1);
            const __m256d v = _mm256_loadu_pd(src + 2 * shot);
            const __m256d x = _mm256_permute_pd(v, 0x5);
            _mm256_storeu_pd(dst + 2 * shot, _mm256_mul_pd(vq, x));
            _mm256_storeu_pd(src + 2 * shot, _mm256_mul_pd(vc, v));
        }
        for (; shot < active_shots; ++shot) {
            double* p = src + 2 * shot;
            double* out = dst + 2 * shot;
            const double q = q_by_shot[shot];
            const double r = p[0];
            const double i = p[1];
            out[0] = -q * i;
            out[1] = q * r;
            p[0] = c * r;
            p[1] = c * i;
        }
    }
}

} // namespace symft::batch_interleaved_avx2
