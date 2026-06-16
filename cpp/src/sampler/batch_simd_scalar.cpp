#include "simd/batch_simd.hpp"

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

std::uint64_t low_bits_mask(int nbits) {
    if (nbits <= 0) {
        return 0;
    }
    if (nbits >= 64) {
        return ~std::uint64_t{0};
    }
    return (std::uint64_t{1} << nbits) - 1;
}

std::uint64_t live_word_mask(int active_shots, std::size_t word) {
    return low_bits_mask(active_shots - static_cast<int>(word << 6));
}

void rotate_uniform_imag_pair_const(
    double* r0p,
    double* i0p,
    double* r1p,
    double* i1p,
    int active_shots,
    double c,
    double q) {
    SYMFT_BATCH_SIMD_LOOP
    for (int shot = 0; shot < active_shots; ++shot) {
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

void rotate_real_pair_flip_const(
    double* r0p,
    double* i0p,
    double* r1p,
    double* i1p,
    int active_shots,
    double c,
    double q) {
    SYMFT_BATCH_SIMD_LOOP
    for (int shot = 0; shot < active_shots; ++shot) {
        const double r0 = r0p[shot];
        const double i0 = i0p[shot];
        const double r1 = r1p[shot];
        const double i1 = i1p[shot];
        r0p[shot] = c * r0 - q * r1;
        i0p[shot] = c * i0 - q * i1;
        r1p[shot] = c * r1 + q * r0;
        i1p[shot] = c * i1 + q * i0;
    }
}

void rotate_general_pair_const(
    double* r0p,
    double* i0p,
    double* r1p,
    double* i1p,
    int first_shot,
    int last_shot,
    double c,
    Complex left_coeff,
    Complex right_coeff) {
    const double lr = left_coeff.real();
    const double li = left_coeff.imag();
    const double rr = right_coeff.real();
    const double ri = right_coeff.imag();
    SYMFT_BATCH_SIMD_LOOP
    for (int shot = first_shot; shot < last_shot; ++shot) {
        const double r0 = r0p[shot];
        const double im0 = i0p[shot];
        const double r1 = r1p[shot];
        const double im1 = i1p[shot];
        r0p[shot] = c * r0 + rr * r1 - ri * im1;
        i0p[shot] = c * im0 + rr * im1 + ri * r1;
        r1p[shot] = c * r1 + lr * r0 - li * im0;
        i1p[shot] = c * im1 + lr * im0 + li * r0;
    }
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

void scalar_rotate_uniform_imag_pairs_const(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* left_indices,
    const std::size_t* right_indices,
    std::size_t npairs,
    double c,
    double q) {
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        rotate_uniform_imag_pair_const(
            re + left_indices[idx] * leading_shots,
            im + left_indices[idx] * leading_shots,
            re + right_indices[idx] * leading_shots,
            im + right_indices[idx] * leading_shots,
            active_shots,
            c,
            q);
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
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t dim = npairs << 1;
    const std::size_t step = selector << 1;
    for (std::size_t block = 0; block < dim; block += step) {
        const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
        for (std::size_t offset = 0; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = right_base + offset;
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
}

void scalar_rotate_uniform_imag_xmask_const(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    double c,
    double q) {
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t dim = npairs << 1;
    const std::size_t step = selector << 1;
    for (std::size_t block = 0; block < dim; block += step) {
        const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
        for (std::size_t offset = 0; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = right_base + offset;
            rotate_uniform_imag_pair_const(
                re + i0 * leading_shots,
                im + i0 * leading_shots,
                re + i1 * leading_shots,
                im + i1 * leading_shots,
                active_shots,
                c,
                q);
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

void scalar_rotate_real_pair_flip_pairs_const(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* left_indices,
    const std::size_t* right_indices,
    const double* basis_phase_signs,
    std::size_t npairs,
    double c,
    double q) {
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        rotate_real_pair_flip_const(
            re + left_indices[idx] * leading_shots,
            im + left_indices[idx] * leading_shots,
            re + right_indices[idx] * leading_shots,
            im + right_indices[idx] * leading_shots,
            active_shots,
            c,
            basis_phase_signs[idx] * q);
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
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t dim = npairs << 1;
    const std::size_t step = selector << 1;
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
        for (std::size_t offset = 0; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = right_base + offset;
            const double basis_phase_sign = basis_phase_signs[pair_idx++];
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
}

void scalar_rotate_real_pair_flip_xmask_const(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    const double* basis_phase_signs,
    std::size_t npairs,
    double c,
    double q) {
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t dim = npairs << 1;
    const std::size_t step = selector << 1;
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
        for (std::size_t offset = 0; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = right_base + offset;
            rotate_real_pair_flip_const(
                re + i0 * leading_shots,
                im + i0 * leading_shots,
                re + i1 * leading_shots,
                im + i1 * leading_shots,
                active_shots,
                c,
                basis_phase_signs[pair_idx++] * q);
        }
    }
}

void scalar_rotate_diagonal_const(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::size_t dim,
    const Complex* coeff,
    double c) {
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const double fr = c + coeff[basis].real();
        const double fi = coeff[basis].imag();
        double* rp = re + basis * leading_shots;
        double* ip = im + basis * leading_shots;
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < active_shots; ++shot) {
            const double r = rp[shot];
            const double v = ip[shot];
            rp[shot] = fr * r - fi * v;
            ip[shot] = fr * v + fi * r;
        }
    }
}

void rotate_diagonal_range_const(
    double* rp,
    double* ip,
    int first_shot,
    int last_shot,
    double fr,
    double fi) {
    SYMFT_BATCH_SIMD_LOOP
    for (int shot = first_shot; shot < last_shot; ++shot) {
        const double r = rp[shot];
        const double v = ip[shot];
        rp[shot] = fr * r - fi * v;
        ip[shot] = fr * v + fi * r;
    }
}

void scalar_rotate_diagonal_mixed(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::size_t dim,
    const Complex* minus_coeff,
    const Complex* plus_coeff,
    const std::uint64_t* sign_bits,
    double c) {
    const std::size_t nwords = static_cast<std::size_t>((active_shots + 63) >> 6);
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const double minus_fr = c + minus_coeff[basis].real();
        const double minus_fi = minus_coeff[basis].imag();
        const double plus_fr = c + plus_coeff[basis].real();
        const double plus_fi = plus_coeff[basis].imag();
        double* rp = re + basis * leading_shots;
        double* ip = im + basis * leading_shots;
        for (std::size_t word = 0; word < nwords; ++word) {
            const int first = static_cast<int>(word << 6);
            const int last = std::min(active_shots, first + 64);
            const std::uint64_t live = live_word_mask(active_shots, word);
            const std::uint64_t bits = sign_bits[word] & live;
            if (bits == 0) {
                rotate_diagonal_range_const(rp, ip, first, last, minus_fr, minus_fi);
            } else if (bits == live) {
                rotate_diagonal_range_const(rp, ip, first, last, plus_fr, plus_fi);
            } else {
                SYMFT_BATCH_SIMD_LOOP
                for (int shot = first; shot < last; ++shot) {
                    const bool sign = ((bits >> (shot - first)) & 1ULL) != 0;
                    const double fr = sign ? plus_fr : minus_fr;
                    const double fi = sign ? plus_fi : minus_fi;
                    const double r = rp[shot];
                    const double v = ip[shot];
                    rp[shot] = fr * r - fi * v;
                    ip[shot] = fr * v + fi * r;
                }
            }
        }
    }
}

void scalar_rotate_general_xmask_const(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    const Complex* left_coeff,
    const Complex* right_coeff,
    double c) {
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t dim = npairs << 1;
    const std::size_t step = selector << 1;
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
        for (std::size_t offset = 0; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = right_base + offset;
            rotate_general_pair_const(
                re + i0 * leading_shots,
                im + i0 * leading_shots,
                re + i1 * leading_shots,
                im + i1 * leading_shots,
                0,
                active_shots,
                c,
                left_coeff[pair_idx],
                right_coeff[pair_idx]);
            ++pair_idx;
        }
    }
}

void scalar_rotate_general_pairs_const(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* left_indices,
    const std::size_t* right_indices,
    std::size_t npairs,
    const Complex* left_coeff,
    const Complex* right_coeff,
    double c) {
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        rotate_general_pair_const(
            re + left_indices[idx] * leading_shots,
            im + left_indices[idx] * leading_shots,
            re + right_indices[idx] * leading_shots,
            im + right_indices[idx] * leading_shots,
            0,
            active_shots,
            c,
            left_coeff[idx],
            right_coeff[idx]);
    }
}

void scalar_rotate_general_xmask_mixed(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    const Complex* left_minus_coeff,
    const Complex* right_minus_coeff,
    const Complex* left_plus_coeff,
    const Complex* right_plus_coeff,
    const std::uint64_t* sign_bits,
    double c) {
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t dim = npairs << 1;
    const std::size_t step = selector << 1;
    const std::size_t nwords = static_cast<std::size_t>((active_shots + 63) >> 6);
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
        for (std::size_t offset = 0; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = right_base + offset;
            double* r0p = re + i0 * leading_shots;
            double* i0p = im + i0 * leading_shots;
            double* r1p = re + i1 * leading_shots;
            double* i1p = im + i1 * leading_shots;
            for (std::size_t word = 0; word < nwords; ++word) {
                const int first = static_cast<int>(word << 6);
                const int last = std::min(active_shots, first + 64);
                const std::uint64_t live = live_word_mask(active_shots, word);
                const std::uint64_t bits = sign_bits[word] & live;
                if (bits == 0) {
                    rotate_general_pair_const(
                        r0p, i0p, r1p, i1p, first, last, c, left_minus_coeff[pair_idx], right_minus_coeff[pair_idx]);
                } else if (bits == live) {
                    rotate_general_pair_const(
                        r0p, i0p, r1p, i1p, first, last, c, left_plus_coeff[pair_idx], right_plus_coeff[pair_idx]);
                } else {
                    const double lmr = left_minus_coeff[pair_idx].real();
                    const double lmi = left_minus_coeff[pair_idx].imag();
                    const double rmr = right_minus_coeff[pair_idx].real();
                    const double rmi = right_minus_coeff[pair_idx].imag();
                    const double lpr = left_plus_coeff[pair_idx].real();
                    const double lpi = left_plus_coeff[pair_idx].imag();
                    const double rpr = right_plus_coeff[pair_idx].real();
                    const double rpi = right_plus_coeff[pair_idx].imag();
                    SYMFT_BATCH_SIMD_LOOP
                    for (int shot = first; shot < last; ++shot) {
                        const bool sign = ((bits >> (shot - first)) & 1ULL) != 0;
                        const double lr = sign ? lpr : lmr;
                        const double li = sign ? lpi : lmi;
                        const double rr = sign ? rpr : rmr;
                        const double ri = sign ? rpi : rmi;
                        const double r0 = r0p[shot];
                        const double im0 = i0p[shot];
                        const double r1 = r1p[shot];
                        const double im1 = i1p[shot];
                        r0p[shot] = c * r0 + rr * r1 - ri * im1;
                        i0p[shot] = c * im0 + rr * im1 + ri * r1;
                        r1p[shot] = c * r1 + lr * r0 - li * im0;
                        i1p[shot] = c * im1 + lr * im0 + li * r0;
                    }
                }
            }
            ++pair_idx;
        }
    }
}

void scalar_rotate_general_pairs_mixed(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* left_indices,
    const std::size_t* right_indices,
    std::size_t npairs,
    const Complex* left_minus_coeff,
    const Complex* right_minus_coeff,
    const Complex* left_plus_coeff,
    const Complex* right_plus_coeff,
    const std::uint64_t* sign_bits,
    double c) {
    const std::size_t nwords = static_cast<std::size_t>((active_shots + 63) >> 6);
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        double* r0p = re + left_indices[idx] * leading_shots;
        double* i0p = im + left_indices[idx] * leading_shots;
        double* r1p = re + right_indices[idx] * leading_shots;
        double* i1p = im + right_indices[idx] * leading_shots;
        for (std::size_t word = 0; word < nwords; ++word) {
            const int first = static_cast<int>(word << 6);
            const int last = std::min(active_shots, first + 64);
            const std::uint64_t live = live_word_mask(active_shots, word);
            const std::uint64_t bits = sign_bits[word] & live;
            if (bits == 0) {
                rotate_general_pair_const(
                    r0p, i0p, r1p, i1p, first, last, c, left_minus_coeff[idx], right_minus_coeff[idx]);
            } else if (bits == live) {
                rotate_general_pair_const(
                    r0p, i0p, r1p, i1p, first, last, c, left_plus_coeff[idx], right_plus_coeff[idx]);
            } else {
                const double lmr = left_minus_coeff[idx].real();
                const double lmi = left_minus_coeff[idx].imag();
                const double rmr = right_minus_coeff[idx].real();
                const double rmi = right_minus_coeff[idx].imag();
                const double lpr = left_plus_coeff[idx].real();
                const double lpi = left_plus_coeff[idx].imag();
                const double rpr = right_plus_coeff[idx].real();
                const double rpi = right_plus_coeff[idx].imag();
                SYMFT_BATCH_SIMD_LOOP
                for (int shot = first; shot < last; ++shot) {
                    const bool sign = ((bits >> (shot - first)) & 1ULL) != 0;
                    const double lr = sign ? lpr : lmr;
                    const double li = sign ? lpi : lmi;
                    const double rr = sign ? rpr : rmr;
                    const double ri = sign ? rpi : rmi;
                    const double r0 = r0p[shot];
                    const double im0 = i0p[shot];
                    const double r1 = r1p[shot];
                    const double im1 = i1p[shot];
                    r0p[shot] = c * r0 + rr * r1 - ri * im1;
                    i0p[shot] = c * im0 + rr * im1 + ri * r1;
                    r1p[shot] = c * r1 + lr * r0 - li * im0;
                    i1p[shot] = c * im1 + lr * im0 + li * r0;
                }
            }
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

void scalar_promote_first_dormant_rotation_range(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::size_t dim,
    std::size_t first_basis,
    std::size_t last_basis,
    double c,
    const double* q_by_shot) {
    for (std::size_t basis = first_basis; basis < last_basis; ++basis) {
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

void scalar_last_z_measure_true_prob_range(
    const double* re,
    const double* im,
    std::size_t leading_shots,
    int active_shots,
    std::size_t dim,
    std::size_t first_basis,
    std::size_t last_basis,
    double* prob_true) {
    for (std::size_t basis = first_basis; basis < last_basis; ++basis) {
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

void scalar_last_z_project_range(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::size_t dim,
    std::size_t first_basis,
    std::size_t last_basis,
    const std::uint64_t* branch_bits,
    const double* invnorms) {
    for (std::size_t basis = first_basis; basis < last_basis; ++basis) {
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

void scalar_diagonal_measure_true_prob_range(
    const double* re,
    const double* im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source_true,
    std::size_t first_idx,
    std::size_t last_idx,
    double* prob_true) {
    for (std::size_t idx = first_idx; idx < last_idx; ++idx) {
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

void scalar_diagonal_project_range(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source_false,
    const std::size_t* source_true,
    std::size_t first_idx,
    std::size_t last_idx,
    const std::uint64_t* branch_bits,
    const double* invnorms) {
    for (std::size_t idx = first_idx; idx < last_idx; ++idx) {
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

void scalar_nondiagonal_measure_true_prob_range(
    const double* re,
    const double* im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source0_false,
    const std::size_t* source1_false,
    const double* coeff1_false_real,
    const double* coeff1_false_imag,
    std::size_t first_idx,
    std::size_t last_idx,
    double* prob_true) {
    for (std::size_t idx = first_idx; idx < last_idx; ++idx) {
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

void scalar_nondiagonal_project_range(
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
    std::size_t first_idx,
    std::size_t last_idx,
    const std::uint64_t* branch_bits,
    const double* invnorms) {
    for (std::size_t idx = first_idx; idx < last_idx; ++idx) {
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
    scalar_rotate_uniform_imag_xmask_const,
    scalar_rotate_real_pair_flip,
    scalar_rotate_real_pair_flip_xmask,
    scalar_rotate_real_pair_flip_xmask_const,
    scalar_rotate_diagonal_const,
    scalar_rotate_diagonal_mixed,
    scalar_rotate_general_xmask_const,
    scalar_rotate_general_xmask_mixed,
    scalar_promote_first_dormant_rotation,
    scalar_last_z_measure_true_prob,
    scalar_last_z_project,
    scalar_diagonal_measure_true_prob,
    scalar_diagonal_project,
    scalar_nondiagonal_measure_true_prob,
    scalar_nondiagonal_project,
    scalar_rotate_uniform_imag_pairs_const,
    scalar_rotate_real_pair_flip_pairs_const,
    scalar_rotate_general_pairs_const,
    scalar_rotate_general_pairs_mixed,
    scalar_promote_first_dormant_rotation_range,
    scalar_last_z_measure_true_prob_range,
    scalar_last_z_project_range,
    scalar_diagonal_measure_true_prob_range,
    scalar_diagonal_project_range,
    scalar_nondiagonal_measure_true_prob_range,
    scalar_nondiagonal_project_range,
};

} // namespace

const KernelTable& scalar_table() {
    return table;
}

} // namespace symft::batch_simd

#undef SYMFT_BATCH_SIMD_LOOP
