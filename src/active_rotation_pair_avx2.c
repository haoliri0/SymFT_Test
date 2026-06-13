#include <immintrin.h>
#include <stdint.h>

static inline uint64_t insert_zero_bit_u64(uint64_t packed, uint32_t bit) {
    uint64_t low_mask = ((uint64_t)1 << bit) - (uint64_t)1;
    uint64_t low = packed & low_mask;
    uint64_t high = packed & ~low_mask;
    return low | (high << 1);
}

void symft_batch_rotate_uniform_imag_pairs_colmajor_avx2(
    double *active,
    const int64_t leading_shots,
    const int64_t active_shots,
    const int64_t *left_indices,
    const int64_t *right_indices,
    const int64_t npairs,
    const double c,
    const double *q_by_shot
) {
    const __m256d vc = _mm256_set1_pd(c);
    for (int64_t idx = 0; idx < npairs; ++idx) {
        /* Julia stores active[shot, basis] column-major, so shots are contiguous. */
        double *a0 = active + 2 * (left_indices[idx] - 1) * leading_shots;
        double *a1 = active + 2 * (right_indices[idx] - 1) * leading_shots;
        int64_t shot = 0;
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
            double *p0 = a0 + 2 * shot;
            double *p1 = a1 + 2 * shot;
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

void symft_batch_rotate_uniform_imag_xmask_colmajor_avx2(
    double *active,
    const int64_t leading_shots,
    const int64_t active_shots,
    const uint64_t xmask,
    const uint32_t pair_bit,
    const int64_t npairs,
    const double c,
    const double *q_by_shot
) {
    const __m256d vc = _mm256_set1_pd(c);
    for (int64_t idx = 0; idx < npairs; ++idx) {
        const uint64_t left = insert_zero_bit_u64((uint64_t)idx, pair_bit);
        const uint64_t right = left ^ xmask;
        double *a0 = active + 2 * left * leading_shots;
        double *a1 = active + 2 * right * leading_shots;
        int64_t shot = 0;
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
            double *p0 = a0 + 2 * shot;
            double *p1 = a1 + 2 * shot;
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

void symft_batch_rotate_real_pair_flip_colmajor_avx2(
    double *active,
    const int64_t leading_shots,
    const int64_t active_shots,
    const int64_t *left_indices,
    const int64_t *right_indices,
    const int64_t npairs,
    const double c,
    const double *q_by_shot,
    const uint64_t zmask
) {
    const __m256d vc = _mm256_set1_pd(c);
    for (int64_t idx = 0; idx < npairs; ++idx) {
        /* Julia stores active[shot, basis] column-major, so shots are contiguous. */
        double *a0 = active + 2 * (left_indices[idx] - 1) * leading_shots;
        double *a1 = active + 2 * (right_indices[idx] - 1) * leading_shots;
        const double phase_sign = (__builtin_popcountll(((uint64_t)(left_indices[idx] - 1)) & zmask) & 1) ? -1.0 : 1.0;
        int64_t shot = 0;
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
            double *p0 = a0 + 2 * shot;
            double *p1 = a1 + 2 * shot;
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

void symft_batch_rotate_real_pair_flip_xmask_colmajor_avx2(
    double *active,
    const int64_t leading_shots,
    const int64_t active_shots,
    const uint64_t xmask,
    const uint32_t pair_bit,
    const int64_t npairs,
    const double c,
    const double *q_by_shot,
    const uint64_t zmask
) {
    const __m256d vc = _mm256_set1_pd(c);
    for (int64_t idx = 0; idx < npairs; ++idx) {
        const uint64_t left = insert_zero_bit_u64((uint64_t)idx, pair_bit);
        const uint64_t right = left ^ xmask;
        double *a0 = active + 2 * left * leading_shots;
        double *a1 = active + 2 * right * leading_shots;
        const double phase_sign = (__builtin_popcountll(left & zmask) & 1) ? -1.0 : 1.0;
        int64_t shot = 0;
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
            double *p0 = a0 + 2 * shot;
            double *p1 = a1 + 2 * shot;
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

void symft_batch_diagonal_measure_true_prob_colmajor_avx2(
    const double *active,
    const int64_t leading_shots,
    const int64_t active_shots,
    const int64_t *source_true,
    const int64_t out_dim,
    double *prob_true
) {
    int64_t shot = 0;
    for (; shot + 1 < active_shots; shot += 2) {
        _mm_storeu_pd(prob_true + shot, _mm_setzero_pd());
    }
    for (; shot < active_shots; ++shot) {
        prob_true[shot] = 0.0;
    }

    for (int64_t idx = 0; idx < out_dim; ++idx) {
        const double *src = active + 2 * (source_true[idx] - 1) * leading_shots;
        shot = 0;
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
            const double *amp = src + 2 * shot;
            prob_true[shot] += amp[0] * amp[0] + amp[1] * amp[1];
        }
    }
}

void symft_batch_project_diagonal_measure_colmajor_avx2(
    const double *active,
    double *scratch,
    const int64_t leading_shots,
    const int64_t active_shots,
    const int64_t *source_false,
    const int64_t *source_true,
    const int64_t out_dim,
    const uint64_t *branch_bits,
    const double *invnorms
) {
    for (int64_t idx = 0; idx < out_dim; ++idx) {
        const double *sf = active + 2 * (source_false[idx] - 1) * leading_shots;
        const double *st = active + 2 * (source_true[idx] - 1) * leading_shots;
        double *dst = scratch + 2 * idx * leading_shots;
        int64_t shot = 0;
        for (; shot + 1 < active_shots; shot += 2) {
            const uint64_t word0 = branch_bits[shot >> 6];
            const uint64_t word1 = branch_bits[(shot + 1) >> 6];
            const uint64_t bit0 = (word0 >> (shot & 63)) & 1;
            const uint64_t bit1 = (word1 >> ((shot + 1) & 63)) & 1;
            const int64_t mask0 = bit0 ? -1 : 0;
            const int64_t mask1 = bit1 ? -1 : 0;
            const __m256d branch_mask = _mm256_castsi256_pd(
                _mm256_setr_epi64x(mask0, mask0, mask1, mask1)
            );
            const __m256d vf = _mm256_loadu_pd(sf + 2 * shot);
            const __m256d vt = _mm256_loadu_pd(st + 2 * shot);
            const __m256d selected = _mm256_blendv_pd(vf, vt, branch_mask);
            const double n0 = invnorms[shot];
            const double n1 = invnorms[shot + 1];
            const __m256d norms = _mm256_setr_pd(n0, n0, n1, n1);
            _mm256_storeu_pd(dst + 2 * shot, _mm256_mul_pd(selected, norms));
        }
        for (; shot < active_shots; ++shot) {
            const uint64_t word = branch_bits[shot >> 6];
            const uint64_t branch = (word >> (shot & 63)) & 1;
            const double *src = branch ? st : sf;
            const double norm = invnorms[shot];
            dst[2 * shot] = src[2 * shot] * norm;
            dst[2 * shot + 1] = src[2 * shot + 1] * norm;
        }
    }
}

void symft_batch_nondiagonal_measure_true_prob_colmajor_avx2(
    const double *active,
    const int64_t leading_shots,
    const int64_t active_shots,
    const int64_t *source0,
    const int64_t *source1,
    const double *coeff1_false,
    const int64_t out_dim,
    double *prob_true
) {
    const __m256d vinvsqrt2 = _mm256_set1_pd(0.70710678118654752440);
    int64_t shot = 0;
    for (; shot + 1 < active_shots; shot += 2) {
        _mm_storeu_pd(prob_true + shot, _mm_setzero_pd());
    }
    for (; shot < active_shots; ++shot) {
        prob_true[shot] = 0.0;
    }

    for (int64_t idx = 0; idx < out_dim; ++idx) {
        const double *s0 = active + 2 * (source0[idx] - 1) * leading_shots;
        const double *s1 = active + 2 * (source1[idx] - 1) * leading_shots;
        const double cr = coeff1_false[2 * idx];
        const double ci = coeff1_false[2 * idx + 1];
        const __m256d vcr = _mm256_set1_pd(cr);
        const __m256d vci = _mm256_setr_pd(-ci, ci, -ci, ci);
        shot = 0;
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
            const double amp_r = 0.70710678118654752440 * r0 - prod_r;
            const double amp_i = 0.70710678118654752440 * i0 - prod_i;
            prob_true[shot] += amp_r * amp_r + amp_i * amp_i;
        }
    }
}

void symft_batch_project_nondiagonal_measure_colmajor_avx2(
    const double *active,
    double *scratch,
    const int64_t leading_shots,
    const int64_t active_shots,
    const int64_t *source0,
    const int64_t *source1,
    const double *coeff1_false,
    const int64_t out_dim,
    const uint64_t *branch_bits,
    const double *invnorms
) {
    const __m256d vinvsqrt2 = _mm256_set1_pd(0.70710678118654752440);
    for (int64_t idx = 0; idx < out_dim; ++idx) {
        const double *s0 = active + 2 * (source0[idx] - 1) * leading_shots;
        const double *s1 = active + 2 * (source1[idx] - 1) * leading_shots;
        double *dst = scratch + 2 * idx * leading_shots;
        const double cr = coeff1_false[2 * idx];
        const double ci = coeff1_false[2 * idx + 1];
        const __m256d vcr = _mm256_set1_pd(cr);
        const __m256d vci = _mm256_setr_pd(-ci, ci, -ci, ci);
        int64_t shot = 0;
        for (; shot + 1 < active_shots; shot += 2) {
            const uint64_t word0 = branch_bits[shot >> 6];
            const uint64_t word1 = branch_bits[(shot + 1) >> 6];
            const uint64_t bit0 = (word0 >> (shot & 63)) & 1;
            const uint64_t bit1 = (word1 >> ((shot + 1) & 63)) & 1;
            const double sign0 = bit0 ? -1.0 : 1.0;
            const double sign1 = bit1 ? -1.0 : 1.0;
            const __m256d signs = _mm256_setr_pd(sign0, sign0, sign1, sign1);
            const __m256d norms = _mm256_setr_pd(
                invnorms[shot],
                invnorms[shot],
                invnorms[shot + 1],
                invnorms[shot + 1]
            );
            const __m256d v0 = _mm256_loadu_pd(s0 + 2 * shot);
            const __m256d v1 = _mm256_loadu_pd(s1 + 2 * shot);
            const __m256d v1_swap = _mm256_permute_pd(v1, 0x5);
            const __m256d prod = _mm256_fmadd_pd(vci, v1_swap, _mm256_mul_pd(vcr, v1));
            const __m256d amp = _mm256_fmadd_pd(signs, prod, _mm256_mul_pd(vinvsqrt2, v0));
            _mm256_storeu_pd(dst + 2 * shot, _mm256_mul_pd(amp, norms));
        }
        for (; shot < active_shots; ++shot) {
            const uint64_t word = branch_bits[shot >> 6];
            const uint64_t branch = (word >> (shot & 63)) & 1;
            const double sign = branch ? -1.0 : 1.0;
            const double norm = invnorms[shot];
            const double r0 = s0[2 * shot];
            const double i0 = s0[2 * shot + 1];
            const double r1 = s1[2 * shot];
            const double i1 = s1[2 * shot + 1];
            const double prod_r = cr * r1 - ci * i1;
            const double prod_i = cr * i1 + ci * r1;
            dst[2 * shot] = (0.70710678118654752440 * r0 + sign * prod_r) * norm;
            dst[2 * shot + 1] = (0.70710678118654752440 * i0 + sign * prod_i) * norm;
        }
    }
}

void symft_batch_promote_first_dormant_rotation_colmajor_avx2(
    double *active,
    const int64_t leading_shots,
    const int64_t active_shots,
    const int64_t dim,
    const double c,
    const double *q_by_shot
) {
    const __m256d vc = _mm256_set1_pd(c);
    for (int64_t idx = 0; idx < dim; ++idx) {
        double *src = active + 2 * idx * leading_shots;
        double *dst = active + 2 * (dim + idx) * leading_shots;
        int64_t shot = 0;
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
            double *p = src + 2 * shot;
            double *out = dst + 2 * shot;
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
