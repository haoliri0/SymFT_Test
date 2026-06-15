#include "symft/batch_simd.hpp"

#include <algorithm>
#include <cmath>

namespace symft::batch_simd {
namespace {

constexpr double kInvSqrt2 = 0.70710678118654752440;

#if defined(__clang__)
#define SYMFT_BATCH_SIMD_LOOP _Pragma("clang loop vectorize(enable) interleave(enable)")
#elif defined(__GNUC__)
#define SYMFT_BATCH_SIMD_LOOP _Pragma("GCC ivdep")
#else
#define SYMFT_BATCH_SIMD_LOOP
#endif

bool batch_bit(const std::uint64_t* bits, int shot) {
    return (bits[static_cast<std::size_t>(shot >> 6)] & (std::uint64_t{1} << (shot & 63))) != 0;
}

std::size_t insert_zero_bit(std::size_t packed, unsigned bit) {
    const std::size_t low_mask = (std::size_t{1} << bit) - std::size_t{1};
    const std::size_t low = packed & low_mask;
    const std::size_t high = packed & ~low_mask;
    return low | (high << 1);
}

void scalar_rotate_uniform_imag_pairs(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* left_indices,
    const std::size_t* right_indices,
    std::size_t npairs,
    double c,
    const double* q_by_shot) {
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        double* r0p = re + left_indices[idx] * leading_shots;
        double* i0p = im + left_indices[idx] * leading_shots;
        double* r1p = re + right_indices[idx] * leading_shots;
        double* i1p = im + right_indices[idx] * leading_shots;
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < active_shots; ++shot) {
            const double q = q_by_shot[static_cast<std::size_t>(shot)];
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

void scalar_rotate_uniform_imag_xmask(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    double c,
    const double* q_by_shot) {
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        const std::size_t i0 = insert_zero_bit(idx, pair_bit);
        const std::size_t i1 = i0 ^ static_cast<std::size_t>(xmask);
        double* r0p = re + i0 * leading_shots;
        double* i0p = im + i0 * leading_shots;
        double* r1p = re + i1 * leading_shots;
        double* i1p = im + i1 * leading_shots;
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < active_shots; ++shot) {
            const double q = q_by_shot[static_cast<std::size_t>(shot)];
            const double r0 = r0p[shot];
            const double i0v = i0p[shot];
            const double r1 = r1p[shot];
            const double i1v = i1p[shot];
            r0p[shot] = c * r0 - q * i1v;
            i0p[shot] = c * i0v + q * r1;
            r1p[shot] = c * r1 - q * i0v;
            i1p[shot] = c * i1v + q * r0;
        }
    }
}

void scalar_rotate_real_pair_flip(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* left_indices,
    const std::size_t* right_indices,
    const double* basis_phase_signs,
    std::size_t npairs,
    double c,
    const double* q_by_shot) {
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        const std::size_t i0 = left_indices[idx];
        const double basis_phase_sign = basis_phase_signs[idx];
        double* r0p = re + i0 * leading_shots;
        double* i0p = im + i0 * leading_shots;
        double* r1p = re + right_indices[idx] * leading_shots;
        double* i1p = im + right_indices[idx] * leading_shots;
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < active_shots; ++shot) {
            const double q = basis_phase_sign * q_by_shot[static_cast<std::size_t>(shot)];
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

void scalar_rotate_real_pair_flip_xmask(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    const double* basis_phase_signs,
    std::size_t npairs,
    double c,
    const double* q_by_shot) {
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        const std::size_t i0 = insert_zero_bit(idx, pair_bit);
        const std::size_t i1 = i0 ^ static_cast<std::size_t>(xmask);
        const double basis_phase_sign = basis_phase_signs[idx];
        double* r0p = re + i0 * leading_shots;
        double* i0p = im + i0 * leading_shots;
        double* r1p = re + i1 * leading_shots;
        double* i1p = im + i1 * leading_shots;
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < active_shots; ++shot) {
            const double q = basis_phase_sign * q_by_shot[static_cast<std::size_t>(shot)];
            const double r0 = r0p[shot];
            const double i0v = i0p[shot];
            const double r1 = r1p[shot];
            const double i1v = i1p[shot];
            r0p[shot] = c * r0 - q * r1;
            i0p[shot] = c * i0v - q * i1v;
            r1p[shot] = c * r1 + q * r0;
            i1p[shot] = c * i1v + q * i0v;
        }
    }
}

void scalar_promote_first_dormant_rotation(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::size_t dim,
    double c,
    const double* q_by_shot) {
    for (std::size_t basis = 0; basis < dim; ++basis) {
        double* r0p = re + basis * leading_shots;
        double* i0p = im + basis * leading_shots;
        double* r1p = re + (dim + basis) * leading_shots;
        double* i1p = im + (dim + basis) * leading_shots;
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < active_shots; ++shot) {
            const double q = q_by_shot[static_cast<std::size_t>(shot)];
            const double r = r0p[shot];
            const double i = i0p[shot];
            r0p[shot] = c * r;
            i0p[shot] = c * i;
            r1p[shot] = -q * i;
            i1p[shot] = q * r;
        }
    }
}

void scalar_last_z_measure_true_prob(
    const double* re,
    const double* im,
    std::size_t leading_shots,
    int active_shots,
    std::size_t dim,
    double* prob_true) {
    std::fill(prob_true, prob_true + active_shots, 0.0);
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const double* rp = re + (dim + basis) * leading_shots;
        const double* ip = im + (dim + basis) * leading_shots;
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < active_shots; ++shot) {
            const double r = rp[shot];
            const double i = ip[shot];
            prob_true[shot] += r * r + i * i;
        }
    }
}

void scalar_last_z_project(
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
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < active_shots; ++shot) {
            const bool branch = batch_bit(branch_bits, shot);
            const double n = invnorms[shot];
            dst_r[shot] = (branch ? true_r[shot] : false_r[shot]) * n;
            dst_i[shot] = (branch ? true_i[shot] : false_i[shot]) * n;
        }
    }
}

void scalar_diagonal_measure_true_prob(
    const double* re,
    const double* im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source_true,
    std::size_t out_dim,
    double* prob_true) {
    std::fill(prob_true, prob_true + active_shots, 0.0);
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const double* rp = re + source_true[idx] * leading_shots;
        const double* ip = im + source_true[idx] * leading_shots;
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < active_shots; ++shot) {
            const double r = rp[shot];
            const double i = ip[shot];
            prob_true[shot] += r * r + i * i;
        }
    }
}

void scalar_diagonal_project(
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
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < active_shots; ++shot) {
            const bool branch = batch_bit(branch_bits, shot);
            const double n = invnorms[shot];
            dst_r[shot] = (branch ? true_r[shot] : false_r[shot]) * n;
            dst_i[shot] = (branch ? true_i[shot] : false_i[shot]) * n;
        }
    }
}

void scalar_nondiagonal_measure_true_prob(
    const double* re,
    const double* im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source0_false,
    const std::size_t* source1_false,
    const double* coeff1_false_real,
    const double* coeff1_false_imag,
    std::size_t out_dim,
    double* prob_true) {
    std::fill(prob_true, prob_true + active_shots, 0.0);
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const double* r0p = re + source0_false[idx] * leading_shots;
        const double* i0p = im + source0_false[idx] * leading_shots;
        const double* r1p = re + source1_false[idx] * leading_shots;
        const double* i1p = im + source1_false[idx] * leading_shots;
        const double c1r = coeff1_false_real[idx];
        const double c1i = coeff1_false_imag[idx];
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < active_shots; ++shot) {
            const double prod_r = c1r * r1p[shot] - c1i * i1p[shot];
            const double prod_i = c1r * i1p[shot] + c1i * r1p[shot];
            const double amp_r = kInvSqrt2 * r0p[shot] - prod_r;
            const double amp_i = kInvSqrt2 * i0p[shot] - prod_i;
            prob_true[shot] += amp_r * amp_r + amp_i * amp_i;
        }
    }
}

void scalar_nondiagonal_project(
    const double* re,
    const double* im,
    double* scratch_re,
    double* scratch_im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source0_false,
    const std::size_t* source1_false,
    const double* coeff1_false_real,
    const double* coeff1_false_imag,
    std::size_t out_dim,
    const std::uint64_t* branch_bits,
    const double* invnorms) {
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const double* r0p = re + source0_false[idx] * leading_shots;
        const double* i0p = im + source0_false[idx] * leading_shots;
        const double* r1p = re + source1_false[idx] * leading_shots;
        const double* i1p = im + source1_false[idx] * leading_shots;
        double* dst_r = scratch_re + idx * leading_shots;
        double* dst_i = scratch_im + idx * leading_shots;
        const double c1r = coeff1_false_real[idx];
        const double c1i = coeff1_false_imag[idx];
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < active_shots; ++shot) {
            const double branch_sign = batch_bit(branch_bits, shot) ? -1.0 : 1.0;
            const double prod_r = c1r * r1p[shot] - c1i * i1p[shot];
            const double prod_i = c1r * i1p[shot] + c1i * r1p[shot];
            const double n = invnorms[shot];
            dst_r[shot] = (kInvSqrt2 * r0p[shot] + branch_sign * prod_r) * n;
            dst_i[shot] = (kInvSqrt2 * i0p[shot] + branch_sign * prod_i) * n;
        }
    }
}

const KernelTable table = {
#if defined(SYMFT_CPP_NATIVE_BUILD)
    "autovec-native",
#else
    "autovec",
#endif
    scalar_rotate_uniform_imag_pairs,
    scalar_rotate_uniform_imag_xmask,
    scalar_rotate_real_pair_flip,
    scalar_rotate_real_pair_flip_xmask,
    scalar_promote_first_dormant_rotation,
    scalar_last_z_measure_true_prob,
    scalar_last_z_project,
    scalar_diagonal_measure_true_prob,
    scalar_diagonal_project,
    scalar_nondiagonal_measure_true_prob,
    scalar_nondiagonal_project,
};

} // namespace

const KernelTable& scalar_table() {
    return table;
}

} // namespace symft::batch_simd

#undef SYMFT_BATCH_SIMD_LOOP
