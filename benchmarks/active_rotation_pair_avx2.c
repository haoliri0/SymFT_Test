#include <immintrin.h>
#include <stdint.h>

static inline uint64_t insert_zero_bit_u64(uint64_t packed, uint32_t bit) {
    uint64_t low_mask = ((uint64_t)1 << bit) - (uint64_t)1;
    uint64_t low = packed & low_mask;
    uint64_t high = packed & ~low_mask;
    return low | (high << 1);
}

void symft_rotate_uniform_imag_pairs_scalar(
    double *alpha,
    const int64_t *left_indices,
    const int64_t *right_indices,
    int64_t npairs,
    double c,
    double q
) {
    for (int64_t idx = 0; idx < npairs; ++idx) {
        double *a0 = alpha + 2 * (left_indices[idx] - 1);
        double *a1 = alpha + 2 * (right_indices[idx] - 1);
        double r0 = a0[0];
        double i0 = a0[1];
        double r1 = a1[0];
        double i1 = a1[1];
        a0[0] = c * r0 - q * i1;
        a0[1] = c * i0 + q * r1;
        a1[0] = c * r1 - q * i0;
        a1[1] = c * i1 + q * r0;
    }
}

void symft_rotate_uniform_imag_xmask_scalar(
    double *alpha,
    uint32_t k,
    uint64_t xmask,
    uint32_t pair_bit,
    double c,
    double q
) {
    uint64_t npairs = (uint64_t)1 << (k - 1);
    for (uint64_t packed = 0; packed < npairs; ++packed) {
        uint64_t left = insert_zero_bit_u64(packed, pair_bit);
        uint64_t right = left ^ xmask;
        double *a0 = alpha + 2 * left;
        double *a1 = alpha + 2 * right;
        double r0 = a0[0];
        double i0 = a0[1];
        double r1 = a1[0];
        double i1 = a1[1];
        a0[0] = c * r0 - q * i1;
        a0[1] = c * i0 + q * r1;
        a1[0] = c * r1 - q * i0;
        a1[1] = c * i1 + q * r0;
    }
}

void symft_rotate_uniform_imag_pairs_avx2(
    double *alpha,
    const int64_t *left_indices,
    const int64_t *right_indices,
    int64_t npairs,
    double c,
    double q
) {
    const __m256d vc = _mm256_set1_pd(c);
    const __m256d vq = _mm256_setr_pd(-q, q, -q, q);
    double out[4];
    for (int64_t idx = 0; idx < npairs; ++idx) {
        double *a0 = alpha + 2 * (left_indices[idx] - 1);
        double *a1 = alpha + 2 * (right_indices[idx] - 1);
        const __m256d v = _mm256_setr_pd(a0[0], a0[1], a1[0], a1[1]);
        const __m256d cross = _mm256_permute4x64_pd(v, 0x1b);
        const __m256d result = _mm256_add_pd(_mm256_mul_pd(vc, v), _mm256_mul_pd(vq, cross));
        _mm256_storeu_pd(out, result);
        a0[0] = out[0];
        a0[1] = out[1];
        a1[0] = out[2];
        a1[1] = out[3];
    }
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
            const __m256d out0 = _mm256_add_pd(_mm256_mul_pd(vc, v0), _mm256_mul_pd(vq, x1));
            const __m256d out1 = _mm256_add_pd(_mm256_mul_pd(vc, v1), _mm256_mul_pd(vq, x0));
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
