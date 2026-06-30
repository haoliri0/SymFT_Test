#include "batch_internal.hpp"
#include "sampler/active_kernels.hpp"
#include "simd/simd.hpp"

namespace symft {

BatchSignMode batch_sign_mode(const BatchFactoredExecutorState& runtime, const std::vector<std::uint64_t>& sign_bits) {
    bool saw_zero = false;
    bool saw_one = false;
    const std::size_t nwords = runtime_batch_word_count(runtime);
    for (std::size_t word = 0; word < nwords; ++word) {
        const std::uint64_t live = batch_live_word_mask(runtime, word);
        const std::uint64_t bits = (word < sign_bits.size() ? sign_bits[word] : 0) & live;
        saw_one = saw_one || bits != 0;
        saw_zero = saw_zero || bits != live;
        if (saw_one && saw_zero) {
            return BatchSignMode::Mixed;
        }
    }
    return saw_one ? BatchSignMode::AllPlus : BatchSignMode::AllMinus;
}

bool batch_bit_at(const std::vector<std::uint64_t>& bits, int shot) {
    const std::size_t word = batch_shot_word(shot);
    return word < bits.size() && ((bits[word] & batch_shot_mask(shot)) != 0);
}

double* active_re_for_shot(BatchFactoredExecutorState& runtime, int shot) {
    return runtime.active_re.data() + static_cast<std::size_t>(shot) * static_cast<std::size_t>(runtime.active_pitch);
}

double* active_im_for_shot(BatchFactoredExecutorState& runtime, int shot) {
    return runtime.active_im.data() + static_cast<std::size_t>(shot) * static_cast<std::size_t>(runtime.active_pitch);
}

bool sign_for_shot(BatchSignMode mode, const std::vector<std::uint64_t>& sign_bits, int shot) {
    return mode == BatchSignMode::Mixed ? batch_bit_at(sign_bits, shot) : mode == BatchSignMode::AllPlus;
}

constexpr int kShotMajorRotationTile = 4;
constexpr std::size_t kShotMajorTiledRotationMaxDim = 2048;

struct ShotMajorRotationTile {
    int count = 0;
    double* re[kShotMajorRotationTile]{};
    double* im[kShotMajorRotationTile]{};
    bool sign[kShotMajorRotationTile]{};
};

ShotMajorRotationTile make_rotation_tile(
    BatchFactoredExecutorState& runtime,
    const std::vector<std::uint64_t>& sign_bits,
    BatchSignMode mode,
    int first_shot,
    int end_shot) {
    ShotMajorRotationTile tile;
    tile.count = std::min(kShotMajorRotationTile, end_shot - first_shot);
    for (int lane = 0; lane < tile.count; ++lane) {
        const int shot = first_shot + lane;
        tile.re[lane] = active_re_for_shot(runtime, shot);
        tile.im[lane] = active_im_for_shot(runtime, shot);
        tile.sign[lane] = sign_for_shot(mode, sign_bits, shot);
    }
    return tile;
}

#if SYMFT_INLINE_AVX2
template <std::size_t LaneMask>
__m256d permute_lanes_xor2_tile(__m256d lanes) {
    if constexpr (LaneMask == 0) {
        return lanes;
    } else if constexpr (LaneMask == 1) {
        return _mm256_permute_pd(lanes, 0b0101);
    } else if constexpr (LaneMask == 2) {
        return _mm256_permute4x64_pd(lanes, 0x4e);
    } else {
        return _mm256_permute4x64_pd(lanes, 0x1b);
    }
}

template <std::size_t LaneMask>
void rotate_uniform_imag_tiled_batch_avx2(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits,
    BatchSignMode mode,
    std::size_t dim,
    int tiled_shots,
    std::size_t lower_mask) {
    const double c = kernel.cos_theta;
    const std::size_t selector = std::size_t{1} << kernel.pair_bit;
    const std::size_t step = selector << 1;
    const std::size_t chunk_mask = lower_mask & ~std::size_t{3};
    const __m256d vc = _mm256_set1_pd(c);
    if (mode != BatchSignMode::Mixed) {
        const double q = mode == BatchSignMode::AllPlus
                             ? kernel.pair_left_plus_coefficients.front().imag()
                             : kernel.pair_left_minus_coefficients.front().imag();
        const __m256d vq = _mm256_set1_pd(q);
        for (int first_shot = 0; first_shot < tiled_shots; first_shot += kShotMajorRotationTile) {
            const ShotMajorRotationTile tile = make_rotation_tile(runtime, sign_bits, mode, first_shot, tiled_shots);
            for (std::size_t block = 0; block < dim; block += step) {
                for (std::size_t offset = 0; offset < selector; offset += 4) {
                    const std::size_t i0 = block + offset;
                    const std::size_t i1 = block + selector + (offset ^ chunk_mask);
                    for (int lane = 0; lane < tile.count; ++lane) {
                        const __m256d r0 = _mm256_loadu_pd(tile.re[lane] + i0);
                        const __m256d im0 = _mm256_loadu_pd(tile.im[lane] + i0);
                        const __m256d r1 = permute_lanes_xor2_tile<LaneMask>(_mm256_loadu_pd(tile.re[lane] + i1));
                        const __m256d im1 = permute_lanes_xor2_tile<LaneMask>(_mm256_loadu_pd(tile.im[lane] + i1));
                        _mm256_storeu_pd(tile.re[lane] + i0, _mm256_fnmadd_pd(vq, im1, _mm256_mul_pd(vc, r0)));
                        _mm256_storeu_pd(tile.im[lane] + i0, _mm256_fmadd_pd(vq, r1, _mm256_mul_pd(vc, im0)));
                        const __m256d out_r1 = _mm256_fnmadd_pd(vq, im0, _mm256_mul_pd(vc, r1));
                        const __m256d out_im1 = _mm256_fmadd_pd(vq, r0, _mm256_mul_pd(vc, im1));
                        _mm256_storeu_pd(tile.re[lane] + i1, permute_lanes_xor2_tile<LaneMask>(out_r1));
                        _mm256_storeu_pd(tile.im[lane] + i1, permute_lanes_xor2_tile<LaneMask>(out_im1));
                    }
                }
            }
        }
        return;
    }

    for (int first_shot = 0; first_shot < tiled_shots; first_shot += kShotMajorRotationTile) {
        const ShotMajorRotationTile tile = make_rotation_tile(runtime, sign_bits, mode, first_shot, tiled_shots);
        double q[kShotMajorRotationTile]{};
        for (int lane = 0; lane < tile.count; ++lane) {
            const Complex coefficient = tile.sign[lane] ? kernel.pair_left_plus_coefficients.front()
                                                        : kernel.pair_left_minus_coefficients.front();
            q[lane] = coefficient.imag();
        }
        for (std::size_t block = 0; block < dim; block += step) {
            for (std::size_t offset = 0; offset < selector; offset += 4) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = block + selector + (offset ^ chunk_mask);
                for (int lane = 0; lane < tile.count; ++lane) {
                    const __m256d vq = _mm256_set1_pd(q[lane]);
                    const __m256d r0 = _mm256_loadu_pd(tile.re[lane] + i0);
                    const __m256d im0 = _mm256_loadu_pd(tile.im[lane] + i0);
                    const __m256d r1 = permute_lanes_xor2_tile<LaneMask>(_mm256_loadu_pd(tile.re[lane] + i1));
                    const __m256d im1 = permute_lanes_xor2_tile<LaneMask>(_mm256_loadu_pd(tile.im[lane] + i1));
                    _mm256_storeu_pd(tile.re[lane] + i0, _mm256_fnmadd_pd(vq, im1, _mm256_mul_pd(vc, r0)));
                    _mm256_storeu_pd(tile.im[lane] + i0, _mm256_fmadd_pd(vq, r1, _mm256_mul_pd(vc, im0)));
                    const __m256d out_r1 = _mm256_fnmadd_pd(vq, im0, _mm256_mul_pd(vc, r1));
                    const __m256d out_im1 = _mm256_fmadd_pd(vq, r0, _mm256_mul_pd(vc, im1));
                    _mm256_storeu_pd(tile.re[lane] + i1, permute_lanes_xor2_tile<LaneMask>(out_r1));
                    _mm256_storeu_pd(tile.im[lane] + i1, permute_lanes_xor2_tile<LaneMask>(out_im1));
                }
            }
        }
    }
}

template <std::size_t LaneMask>
void rotate_real_pair_flip_tiled_batch_avx2(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits,
    BatchSignMode mode,
    std::size_t dim,
    int tiled_shots,
    std::size_t lower_mask) {
    const double c = kernel.cos_theta;
    const std::size_t selector = std::size_t{1} << kernel.pair_bit;
    const std::size_t step = selector << 1;
    const std::size_t chunk_mask = lower_mask & ~std::size_t{3};
    const __m256d vc = _mm256_set1_pd(c);
    if (mode != BatchSignMode::Mixed) {
        const double base_coeff = mode == BatchSignMode::AllPlus
                                      ? kernel.pair_left_plus_coefficients.front().real()
                                      : kernel.pair_left_minus_coefficients.front().real();
        const __m256d vbase = _mm256_set1_pd(base_coeff);
        for (int first_shot = 0; first_shot < tiled_shots; first_shot += kShotMajorRotationTile) {
            const ShotMajorRotationTile tile = make_rotation_tile(runtime, sign_bits, mode, first_shot, tiled_shots);
            std::size_t pair_idx = 0;
            for (std::size_t block = 0; block < dim; block += step) {
                for (std::size_t offset = 0; offset < selector; offset += 4) {
                    const std::size_t i0 = block + offset;
                    const std::size_t i1 = block + selector + (offset ^ chunk_mask);
                    const __m256d q = _mm256_mul_pd(_mm256_loadu_pd(kernel.real_pair_flip_basis_phase_signs.data() + pair_idx), vbase);
                    for (int lane = 0; lane < tile.count; ++lane) {
                        const __m256d r0 = _mm256_loadu_pd(tile.re[lane] + i0);
                        const __m256d im0 = _mm256_loadu_pd(tile.im[lane] + i0);
                        const __m256d r1 = permute_lanes_xor2_tile<LaneMask>(_mm256_loadu_pd(tile.re[lane] + i1));
                        const __m256d im1 = permute_lanes_xor2_tile<LaneMask>(_mm256_loadu_pd(tile.im[lane] + i1));
                        _mm256_storeu_pd(tile.re[lane] + i0, _mm256_fnmadd_pd(q, r1, _mm256_mul_pd(vc, r0)));
                        _mm256_storeu_pd(tile.im[lane] + i0, _mm256_fnmadd_pd(q, im1, _mm256_mul_pd(vc, im0)));
                        const __m256d out_r1 = _mm256_fmadd_pd(q, r0, _mm256_mul_pd(vc, r1));
                        const __m256d out_im1 = _mm256_fmadd_pd(q, im0, _mm256_mul_pd(vc, im1));
                        _mm256_storeu_pd(tile.re[lane] + i1, permute_lanes_xor2_tile<LaneMask>(out_r1));
                        _mm256_storeu_pd(tile.im[lane] + i1, permute_lanes_xor2_tile<LaneMask>(out_im1));
                    }
                    pair_idx += 4;
                }
            }
        }
        return;
    }

    for (int first_shot = 0; first_shot < tiled_shots; first_shot += kShotMajorRotationTile) {
        const ShotMajorRotationTile tile = make_rotation_tile(runtime, sign_bits, mode, first_shot, tiled_shots);
        double base_coeff[kShotMajorRotationTile]{};
        for (int lane = 0; lane < tile.count; ++lane) {
            const Complex coefficient = tile.sign[lane] ? kernel.pair_left_plus_coefficients.front()
                                                        : kernel.pair_left_minus_coefficients.front();
            base_coeff[lane] = coefficient.real();
        }
        std::size_t pair_idx = 0;
        for (std::size_t block = 0; block < dim; block += step) {
            for (std::size_t offset = 0; offset < selector; offset += 4) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = block + selector + (offset ^ chunk_mask);
                const __m256d phase = _mm256_loadu_pd(kernel.real_pair_flip_basis_phase_signs.data() + pair_idx);
                for (int lane = 0; lane < tile.count; ++lane) {
                    const __m256d q = _mm256_mul_pd(phase, _mm256_set1_pd(base_coeff[lane]));
                    const __m256d r0 = _mm256_loadu_pd(tile.re[lane] + i0);
                    const __m256d im0 = _mm256_loadu_pd(tile.im[lane] + i0);
                    const __m256d r1 = permute_lanes_xor2_tile<LaneMask>(_mm256_loadu_pd(tile.re[lane] + i1));
                    const __m256d im1 = permute_lanes_xor2_tile<LaneMask>(_mm256_loadu_pd(tile.im[lane] + i1));
                    _mm256_storeu_pd(tile.re[lane] + i0, _mm256_fnmadd_pd(q, r1, _mm256_mul_pd(vc, r0)));
                    _mm256_storeu_pd(tile.im[lane] + i0, _mm256_fnmadd_pd(q, im1, _mm256_mul_pd(vc, im0)));
                    const __m256d out_r1 = _mm256_fmadd_pd(q, r0, _mm256_mul_pd(vc, r1));
                    const __m256d out_im1 = _mm256_fmadd_pd(q, im0, _mm256_mul_pd(vc, im1));
                    _mm256_storeu_pd(tile.re[lane] + i1, permute_lanes_xor2_tile<LaneMask>(out_r1));
                    _mm256_storeu_pd(tile.im[lane] + i1, permute_lanes_xor2_tile<LaneMask>(out_im1));
                }
                pair_idx += 4;
            }
        }
    }
}

template <std::size_t LaneMask>
void rotate_general_tiled_batch_avx2(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits,
    BatchSignMode mode,
    std::size_t dim,
    int tiled_shots,
    std::size_t lower_mask) {
    const double c = kernel.cos_theta;
    const std::size_t selector = std::size_t{1} << kernel.pair_bit;
    const std::size_t step = selector << 1;
    const std::size_t chunk_mask = lower_mask & ~std::size_t{3};
    const __m256d vc = _mm256_set1_pd(c);
    if (mode != BatchSignMode::Mixed) {
        const auto* left = reinterpret_cast<const double*>(
            mode == BatchSignMode::AllPlus ? kernel.pair_left_plus_coefficients.data() : kernel.pair_left_minus_coefficients.data());
        const auto* right = reinterpret_cast<const double*>(
            mode == BatchSignMode::AllPlus ? kernel.pair_right_plus_coefficients.data() : kernel.pair_right_minus_coefficients.data());
        for (int first_shot = 0; first_shot < tiled_shots; first_shot += kShotMajorRotationTile) {
            const ShotMajorRotationTile tile = make_rotation_tile(runtime, sign_bits, mode, first_shot, tiled_shots);
            std::size_t pair_idx = 0;
            for (std::size_t block = 0; block < dim; block += step) {
                for (std::size_t offset = 0; offset < selector; offset += 4) {
                    const std::size_t i0 = block + offset;
                    const std::size_t i1 = block + selector + (offset ^ chunk_mask);
                    const __m256d lr = detail::load_complex_re4_inline(left + 2 * pair_idx);
                    const __m256d li = detail::load_complex_im4_inline(left + 2 * pair_idx);
                    const __m256d rr = detail::load_complex_re4_inline(right + 2 * pair_idx);
                    const __m256d ri = detail::load_complex_im4_inline(right + 2 * pair_idx);
                    for (int lane = 0; lane < tile.count; ++lane) {
                        const __m256d r0 = _mm256_loadu_pd(tile.re[lane] + i0);
                        const __m256d im0 = _mm256_loadu_pd(tile.im[lane] + i0);
                        const __m256d r1 = permute_lanes_xor2_tile<LaneMask>(_mm256_loadu_pd(tile.re[lane] + i1));
                        const __m256d im1 = permute_lanes_xor2_tile<LaneMask>(_mm256_loadu_pd(tile.im[lane] + i1));
                        __m256d out_r0 = _mm256_fmadd_pd(vc, r0, _mm256_mul_pd(rr, r1));
                        out_r0 = _mm256_fnmadd_pd(ri, im1, out_r0);
                        __m256d out_i0 = _mm256_fmadd_pd(vc, im0, _mm256_mul_pd(rr, im1));
                        out_i0 = _mm256_fmadd_pd(ri, r1, out_i0);
                        __m256d out_r1 = _mm256_fmadd_pd(vc, r1, _mm256_mul_pd(lr, r0));
                        out_r1 = _mm256_fnmadd_pd(li, im0, out_r1);
                        __m256d out_i1 = _mm256_fmadd_pd(vc, im1, _mm256_mul_pd(lr, im0));
                        out_i1 = _mm256_fmadd_pd(li, r0, out_i1);
                        _mm256_storeu_pd(tile.re[lane] + i0, out_r0);
                        _mm256_storeu_pd(tile.im[lane] + i0, out_i0);
                        _mm256_storeu_pd(tile.re[lane] + i1, permute_lanes_xor2_tile<LaneMask>(out_r1));
                        _mm256_storeu_pd(tile.im[lane] + i1, permute_lanes_xor2_tile<LaneMask>(out_i1));
                    }
                    pair_idx += 4;
                }
            }
        }
        return;
    }

    for (int first_shot = 0; first_shot < tiled_shots; first_shot += kShotMajorRotationTile) {
        const ShotMajorRotationTile tile = make_rotation_tile(runtime, sign_bits, mode, first_shot, tiled_shots);
        const Complex* left_coeff[kShotMajorRotationTile]{};
        const Complex* right_coeff[kShotMajorRotationTile]{};
        for (int lane = 0; lane < tile.count; ++lane) {
            left_coeff[lane] = tile.sign[lane] ? kernel.pair_left_plus_coefficients.data()
                                               : kernel.pair_left_minus_coefficients.data();
            right_coeff[lane] = tile.sign[lane] ? kernel.pair_right_plus_coefficients.data()
                                                : kernel.pair_right_minus_coefficients.data();
        }
        std::size_t pair_idx = 0;
        for (std::size_t block = 0; block < dim; block += step) {
            for (std::size_t offset = 0; offset < selector; offset += 4) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = block + selector + (offset ^ chunk_mask);
                for (int lane = 0; lane < tile.count; ++lane) {
                    const auto* left = reinterpret_cast<const double*>(left_coeff[lane]);
                    const auto* right = reinterpret_cast<const double*>(right_coeff[lane]);
                    const __m256d r0 = _mm256_loadu_pd(tile.re[lane] + i0);
                    const __m256d im0 = _mm256_loadu_pd(tile.im[lane] + i0);
                    const __m256d r1 = permute_lanes_xor2_tile<LaneMask>(_mm256_loadu_pd(tile.re[lane] + i1));
                    const __m256d im1 = permute_lanes_xor2_tile<LaneMask>(_mm256_loadu_pd(tile.im[lane] + i1));
                    const __m256d lr = detail::load_complex_re4_inline(left + 2 * pair_idx);
                    const __m256d li = detail::load_complex_im4_inline(left + 2 * pair_idx);
                    const __m256d rr = detail::load_complex_re4_inline(right + 2 * pair_idx);
                    const __m256d ri = detail::load_complex_im4_inline(right + 2 * pair_idx);
                    __m256d out_r0 = _mm256_fmadd_pd(vc, r0, _mm256_mul_pd(rr, r1));
                    out_r0 = _mm256_fnmadd_pd(ri, im1, out_r0);
                    __m256d out_i0 = _mm256_fmadd_pd(vc, im0, _mm256_mul_pd(rr, im1));
                    out_i0 = _mm256_fmadd_pd(ri, r1, out_i0);
                    __m256d out_r1 = _mm256_fmadd_pd(vc, r1, _mm256_mul_pd(lr, r0));
                    out_r1 = _mm256_fnmadd_pd(li, im0, out_r1);
                    __m256d out_i1 = _mm256_fmadd_pd(vc, im1, _mm256_mul_pd(lr, im0));
                    out_i1 = _mm256_fmadd_pd(li, r0, out_i1);
                    _mm256_storeu_pd(tile.re[lane] + i0, out_r0);
                    _mm256_storeu_pd(tile.im[lane] + i0, out_i0);
                    _mm256_storeu_pd(tile.re[lane] + i1, permute_lanes_xor2_tile<LaneMask>(out_r1));
                    _mm256_storeu_pd(tile.im[lane] + i1, permute_lanes_xor2_tile<LaneMask>(out_i1));
                }
                pair_idx += 4;
            }
        }
    }
}

template <typename Fn>
void dispatch_lane_mask(std::size_t lane_mask, Fn&& fn) {
    switch (lane_mask) {
    case 0:
        fn.template operator()<0>();
        break;
    case 1:
        fn.template operator()<1>();
        break;
    case 2:
        fn.template operator()<2>();
        break;
    default:
        fn.template operator()<3>();
        break;
    }
}
#endif

void rotate_diagonal_tiled_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits,
    BatchSignMode mode,
    std::size_t dim,
    int tiled_shots) {
    const double c = kernel.cos_theta;
    for (int first_shot = 0; first_shot < tiled_shots; first_shot += kShotMajorRotationTile) {
        const ShotMajorRotationTile tile = make_rotation_tile(runtime, sign_bits, mode, first_shot, tiled_shots);
#if SYMFT_INLINE_AVX2
        std::size_t basis = 0;
        const __m256d vc = _mm256_set1_pd(c);
        if (mode != BatchSignMode::Mixed) {
            const auto& coefficients = mode == BatchSignMode::AllPlus
                                           ? kernel.diagonal_plus_coefficients
                                           : kernel.diagonal_minus_coefficients;
            const auto* coeff = reinterpret_cast<const double*>(coefficients.data());
            for (; basis + 4 <= dim; basis += 4) {
                const __m256d fr = _mm256_add_pd(vc, detail::load_complex_re4_inline(coeff + 2 * basis));
                const __m256d fi = detail::load_complex_im4_inline(coeff + 2 * basis);
                for (int lane = 0; lane < tile.count; ++lane) {
                    const __m256d r = _mm256_loadu_pd(tile.re[lane] + basis);
                    const __m256d i = _mm256_loadu_pd(tile.im[lane] + basis);
                    _mm256_storeu_pd(tile.re[lane] + basis, _mm256_fnmadd_pd(fi, i, _mm256_mul_pd(fr, r)));
                    _mm256_storeu_pd(tile.im[lane] + basis, _mm256_fmadd_pd(fi, r, _mm256_mul_pd(fr, i)));
                }
            }
            for (; basis < dim; ++basis) {
                const double fr = c + coeff[2 * basis];
                const double fi = coeff[2 * basis + 1];
                for (int lane = 0; lane < tile.count; ++lane) {
                    const double r = tile.re[lane][basis];
                    const double i = tile.im[lane][basis];
                    tile.re[lane][basis] = fr * r - fi * i;
                    tile.im[lane][basis] = fr * i + fi * r;
                }
            }
            continue;
        }
        for (; basis + 4 <= dim; basis += 4) {
            for (int lane = 0; lane < tile.count; ++lane) {
                const auto& coefficients = tile.sign[lane] ? kernel.diagonal_plus_coefficients : kernel.diagonal_minus_coefficients;
                const auto* coeff = reinterpret_cast<const double*>(coefficients.data());
                const __m256d fr = _mm256_add_pd(vc, detail::load_complex_re4_inline(coeff + 2 * basis));
                const __m256d fi = detail::load_complex_im4_inline(coeff + 2 * basis);
                const __m256d r = _mm256_loadu_pd(tile.re[lane] + basis);
                const __m256d i = _mm256_loadu_pd(tile.im[lane] + basis);
                _mm256_storeu_pd(tile.re[lane] + basis, _mm256_fnmadd_pd(fi, i, _mm256_mul_pd(fr, r)));
                _mm256_storeu_pd(tile.im[lane] + basis, _mm256_fmadd_pd(fi, r, _mm256_mul_pd(fr, i)));
            }
        }
        for (; basis < dim; ++basis) {
#else
        for (std::size_t basis = 0; basis < dim; ++basis) {
#endif
            for (int lane = 0; lane < tile.count; ++lane) {
                const auto& coefficients = tile.sign[lane] ? kernel.diagonal_plus_coefficients : kernel.diagonal_minus_coefficients;
                const auto* coeff = reinterpret_cast<const double*>(coefficients.data());
                const double fr = c + coeff[2 * basis];
                const double fi = coeff[2 * basis + 1];
                const double r = tile.re[lane][basis];
                const double i = tile.im[lane][basis];
                tile.re[lane][basis] = fr * r - fi * i;
                tile.im[lane][basis] = fr * i + fi * r;
            }
        }
    }
}

void rotate_uniform_imag_tiled_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits,
    BatchSignMode mode,
    std::size_t dim,
    int tiled_shots) {
    const double c = kernel.cos_theta;
    const std::size_t selector = std::size_t{1} << kernel.pair_bit;
    const std::size_t step = selector << 1;
#if SYMFT_INLINE_AVX2
    if (kernel.pair_bit >= 2 && dim >= 4) {
        const std::size_t lower_mask = static_cast<std::size_t>(kernel.action.xmask) & (selector - 1);
        const std::size_t lane_mask = lower_mask & std::size_t{3};
        dispatch_lane_mask(lane_mask, [&]<std::size_t LaneMask>() {
            rotate_uniform_imag_tiled_batch_avx2<LaneMask>(
                runtime, kernel, sign_bits, mode, dim, tiled_shots, lower_mask);
        });
        return;
    }
#endif
    for (int first_shot = 0; first_shot < tiled_shots; first_shot += kShotMajorRotationTile) {
        const ShotMajorRotationTile tile = make_rotation_tile(runtime, sign_bits, mode, first_shot, tiled_shots);
        double q[kShotMajorRotationTile]{};
        for (int lane = 0; lane < tile.count; ++lane) {
            const Complex coefficient = tile.sign[lane] ? kernel.pair_left_plus_coefficients.front()
                                                        : kernel.pair_left_minus_coefficients.front();
            q[lane] = coefficient.imag();
        }
        for (std::size_t block = 0; block < dim; block += step) {
            for (std::size_t offset = 0; offset < selector; ++offset) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = i0 ^ static_cast<std::size_t>(kernel.action.xmask);
                for (int lane = 0; lane < tile.count; ++lane) {
                    const double r0 = tile.re[lane][i0];
                    const double im0 = tile.im[lane][i0];
                    const double r1 = tile.re[lane][i1];
                    const double im1 = tile.im[lane][i1];
                    tile.re[lane][i0] = c * r0 - q[lane] * im1;
                    tile.im[lane][i0] = c * im0 + q[lane] * r1;
                    tile.re[lane][i1] = c * r1 - q[lane] * im0;
                    tile.im[lane][i1] = c * im1 + q[lane] * r0;
                }
            }
        }
    }
}

void rotate_real_pair_flip_tiled_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits,
    BatchSignMode mode,
    std::size_t dim,
    int tiled_shots) {
    const double c = kernel.cos_theta;
    const std::size_t selector = std::size_t{1} << kernel.pair_bit;
    const std::size_t step = selector << 1;
#if SYMFT_INLINE_AVX2
    if (kernel.pair_bit >= 2 && dim >= 4) {
        const std::size_t lower_mask = static_cast<std::size_t>(kernel.action.xmask) & (selector - 1);
        const std::size_t lane_mask = lower_mask & std::size_t{3};
        dispatch_lane_mask(lane_mask, [&]<std::size_t LaneMask>() {
            rotate_real_pair_flip_tiled_batch_avx2<LaneMask>(
                runtime, kernel, sign_bits, mode, dim, tiled_shots, lower_mask);
        });
        return;
    }
#endif
    for (int first_shot = 0; first_shot < tiled_shots; first_shot += kShotMajorRotationTile) {
        const ShotMajorRotationTile tile = make_rotation_tile(runtime, sign_bits, mode, first_shot, tiled_shots);
        double base_coeff[kShotMajorRotationTile]{};
        for (int lane = 0; lane < tile.count; ++lane) {
            const Complex coefficient = tile.sign[lane] ? kernel.pair_left_plus_coefficients.front()
                                                        : kernel.pair_left_minus_coefficients.front();
            base_coeff[lane] = coefficient.real();
        }
        std::size_t pair_idx = 0;
        for (std::size_t block = 0; block < dim; block += step) {
            for (std::size_t offset = 0; offset < selector; ++offset) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = i0 ^ static_cast<std::size_t>(kernel.action.xmask);
                const double phase_sign = kernel.real_pair_flip_basis_phase_signs[pair_idx++];
                for (int lane = 0; lane < tile.count; ++lane) {
                    const double q = phase_sign * base_coeff[lane];
                    const double r0 = tile.re[lane][i0];
                    const double im0 = tile.im[lane][i0];
                    const double r1 = tile.re[lane][i1];
                    const double im1 = tile.im[lane][i1];
                    tile.re[lane][i0] = c * r0 - q * r1;
                    tile.im[lane][i0] = c * im0 - q * im1;
                    tile.re[lane][i1] = c * r1 + q * r0;
                    tile.im[lane][i1] = c * im1 + q * im0;
                }
            }
        }
    }
}

void rotate_general_tiled_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits,
    BatchSignMode mode,
    std::size_t dim,
    int tiled_shots) {
    const double c = kernel.cos_theta;
    const std::size_t selector = std::size_t{1} << kernel.pair_bit;
    const std::size_t step = selector << 1;
#if SYMFT_INLINE_AVX2
    if (kernel.pair_bit >= 2 && dim >= 4) {
        const std::size_t lower_mask = static_cast<std::size_t>(kernel.action.xmask) & (selector - 1);
        const std::size_t lane_mask = lower_mask & std::size_t{3};
        dispatch_lane_mask(lane_mask, [&]<std::size_t LaneMask>() {
            rotate_general_tiled_batch_avx2<LaneMask>(
                runtime, kernel, sign_bits, mode, dim, tiled_shots, lower_mask);
        });
        return;
    }
#endif
    for (int first_shot = 0; first_shot < tiled_shots; first_shot += kShotMajorRotationTile) {
        const ShotMajorRotationTile tile = make_rotation_tile(runtime, sign_bits, mode, first_shot, tiled_shots);
        const Complex* left_coeff[kShotMajorRotationTile]{};
        const Complex* right_coeff[kShotMajorRotationTile]{};
        for (int lane = 0; lane < tile.count; ++lane) {
            left_coeff[lane] = tile.sign[lane] ? kernel.pair_left_plus_coefficients.data()
                                               : kernel.pair_left_minus_coefficients.data();
            right_coeff[lane] = tile.sign[lane] ? kernel.pair_right_plus_coefficients.data()
                                                : kernel.pair_right_minus_coefficients.data();
        }
        std::size_t pair_idx = 0;
        for (std::size_t block = 0; block < dim; block += step) {
            for (std::size_t offset = 0; offset < selector; ++offset) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = i0 ^ static_cast<std::size_t>(kernel.action.xmask);
                for (int lane = 0; lane < tile.count; ++lane) {
                    const Complex lc = left_coeff[lane][pair_idx];
                    const Complex rc = right_coeff[lane][pair_idx];
                    const double r0 = tile.re[lane][i0];
                    const double im0 = tile.im[lane][i0];
                    const double r1 = tile.re[lane][i1];
                    const double im1 = tile.im[lane][i1];
                    tile.re[lane][i0] = c * r0 + rc.real() * r1 - rc.imag() * im1;
                    tile.im[lane][i0] = c * im0 + rc.real() * im1 + rc.imag() * r1;
                    tile.re[lane][i1] = c * r1 + lc.real() * r0 - lc.imag() * im0;
                    tile.im[lane][i1] = c * im1 + lc.real() * im0 + lc.imag() * r0;
                }
                ++pair_idx;
            }
        }
    }
}

int rotate_pauli_tiled_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits,
    BatchSignMode mode,
    std::size_t dim) {
#if !SYMFT_INLINE_AVX2
    (void)runtime;
    (void)kernel;
    (void)sign_bits;
    (void)mode;
    (void)dim;
    return 0;
#else
    const int tiled_shots = runtime.active_shots - (runtime.active_shots % kShotMajorRotationTile);
    if (tiled_shots <= 0) {
        return 0;
    }
    if (dim > kShotMajorTiledRotationMaxDim) {
        return 0;
    }
    if (!kernel.is_diagonal && kernel.pair_bit < 2) {
        return 0;
    }
    if (kernel.is_diagonal) {
        rotate_diagonal_tiled_batch(runtime, kernel, sign_bits, mode, dim, tiled_shots);
        return tiled_shots;
    }
    if (kernel.uniform_imag_pairs) {
        rotate_uniform_imag_tiled_batch(runtime, kernel, sign_bits, mode, dim, tiled_shots);
        return tiled_shots;
    }
    if (kernel.real_pair_flip) {
        rotate_real_pair_flip_tiled_batch(runtime, kernel, sign_bits, mode, dim, tiled_shots);
        return tiled_shots;
    }
    rotate_general_tiled_batch(runtime, kernel, sign_bits, mode, dim, tiled_shots);
    return tiled_shots;
#endif
}

void finish_active_measurement_branch(
    BatchFactoredExecutorState& runtime,
    int branch_condition,
    const std::vector<std::uint64_t>& branch_bits) {
    --runtime.k;
    ++runtime.ndormant;
    assign_batch_symbol(runtime, branch_condition, branch_bits);
}

double nondiagonal_probability_for_shot(
    const double* re,
    const double* im,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch) {
    double fast_probability = 0.0;
    if (detail::active_measurement_branch_probability_partner_permute_soa_inline(
            fast_probability,
            re,
            im,
            kernel,
            branch)) {
        return fast_probability;
    }
    const auto& sources0 = branch ? kernel.source0_true : kernel.source0_false;
    const auto& sources1 = branch ? kernel.source1_true : kernel.source1_false;
    const auto& coeffs0 = branch ? kernel.coeff0_true : kernel.coeff0_false;
    const auto& coeffs1 = branch ? kernel.coeff1_true : kernel.coeff1_false;
    const auto* coeff0 = reinterpret_cast<const double*>(coeffs0.data());
    const auto* coeff1 = reinterpret_cast<const double*>(coeffs1.data());
    double probability = 0.0;
    SYMFT_SINGLE_SIMD_LOOP
    for (std::size_t idx = 0; idx < sources0.size(); ++idx) {
        const std::size_t source0 = sources0[idx];
        const double c0r = coeff0[2 * idx];
        const double c0i = coeff0[2 * idx + 1];
        double ar = c0r * re[source0] - c0i * im[source0];
        double ai = c0r * im[source0] + c0i * re[source0];
        const std::size_t source1 = sources1[idx];
        if (source1 != detail::kNoSource) {
            const double c1r = coeff1[2 * idx];
            const double c1i = coeff1[2 * idx + 1];
            ar += c1r * re[source1] - c1i * im[source1];
            ai += c1r * im[source1] + c1i * re[source1];
        }
        probability += ar * ar + ai * ai;
    }
    return probability;
}

void project_nondiagonal_shot(
    double* re,
    double* im,
    double* scratch_re,
    double* scratch_im,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch,
    double invnorm) {
    if (detail::project_active_measurement_partner_permute_soa_inline(
            re,
            im,
            re,
            im,
            kernel,
            branch,
            invnorm)) {
        return;
    }
    const auto& sources0 = branch ? kernel.source0_true : kernel.source0_false;
    const auto& sources1 = branch ? kernel.source1_true : kernel.source1_false;
    const auto& coeffs0 = branch ? kernel.coeff0_true : kernel.coeff0_false;
    const auto& coeffs1 = branch ? kernel.coeff1_true : kernel.coeff1_false;
    const auto* coeff0 = reinterpret_cast<const double*>(coeffs0.data());
    const auto* coeff1 = reinterpret_cast<const double*>(coeffs1.data());
    const std::size_t out_dim = sources0.size();
    SYMFT_SINGLE_SIMD_LOOP
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const std::size_t source0 = sources0[idx];
        const double c0r = coeff0[2 * idx];
        const double c0i = coeff0[2 * idx + 1];
        double ar = c0r * re[source0] - c0i * im[source0];
        double ai = c0r * im[source0] + c0i * re[source0];
        const std::size_t source1 = sources1[idx];
        if (source1 != detail::kNoSource) {
            const double c1r = coeff1[2 * idx];
            const double c1i = coeff1[2 * idx + 1];
            ar += c1r * re[source1] - c1i * im[source1];
            ai += c1r * im[source1] + c1i * re[source1];
        }
        scratch_re[idx] = ar * invnorm;
        scratch_im[idx] = ai * invnorm;
    }
    std::copy_n(scratch_re, out_dim, re);
    std::copy_n(scratch_im, out_dim, im);
}

void rotate_pauli_shot(
    double* re,
    double* im,
    std::size_t dim,
    const PrecomputedActivePauliRotationKernel& kernel,
    bool sign,
    const simd::KernelTable& simd_table) {
    const double c = kernel.cos_theta;
    if (kernel.is_diagonal) {
        const auto& coefficients = sign ? kernel.diagonal_plus_coefficients : kernel.diagonal_minus_coefficients;
        if (dim < detail::kSimdPairRotationThreshold) {
            detail::mul_assign_soa_inline(re, im, coefficients.data(), c, dim);
        } else {
            simd_table.mul_assign_soa(re, im, coefficients.data(), c, dim);
        }
        return;
    }
    const std::size_t npairs = kernel.pair_count;
    if (kernel.uniform_imag_pairs) {
        const Complex coefficient = sign ? kernel.pair_left_plus_coefficients.front() : kernel.pair_left_minus_coefficients.front();
        if (npairs < detail::kSimdPairRotationThreshold) {
            detail::rotate_uniform_imag_pairs_soa_inline(re, im, dim, kernel.action.xmask, kernel.pair_bit, c, coefficient.imag());
        } else {
            simd_table.rotate_uniform_imag_pairs_soa(re, im, dim, kernel.action.xmask, kernel.pair_bit, c, coefficient.imag());
        }
        return;
    }
    if (kernel.real_pair_flip) {
        const Complex coefficient = sign ? kernel.pair_left_plus_coefficients.front() : kernel.pair_left_minus_coefficients.front();
        if (npairs < detail::kSimdPairRotationThreshold) {
            detail::rotate_real_pair_flip_soa_inline(
                re,
                im,
                dim,
                kernel.action.xmask,
                kernel.pair_bit,
                kernel.real_pair_flip_basis_phase_signs.data(),
                c,
                coefficient.real());
        } else {
            simd_table.rotate_real_pair_flip_soa(
                re,
                im,
                dim,
                kernel.action.xmask,
                kernel.pair_bit,
                kernel.real_pair_flip_basis_phase_signs.data(),
                c,
                coefficient.real());
        }
        return;
    }
    const auto& left_coeff = sign ? kernel.pair_left_plus_coefficients : kernel.pair_left_minus_coefficients;
    const auto& right_coeff = sign ? kernel.pair_right_plus_coefficients : kernel.pair_right_minus_coefficients;
    if (npairs < detail::kSimdPairRotationThreshold) {
        detail::rotate_general_pairs_soa_inline(re, im, dim, kernel.action.xmask, kernel.pair_bit, left_coeff.data(), right_coeff.data(), c);
    } else {
        simd_table.rotate_general_pairs_soa(
            re,
            im,
            dim,
            kernel.action.xmask,
            kernel.pair_bit,
            left_coeff.data(),
            right_coeff.data(),
            c);
    }
}

void rotate_pauli_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits) {
    if (kernel.action.nqubits != runtime.k) {
        fail("rotation kernel dimension does not match batch active state");
    }
    const std::size_t dim = active_length(runtime.k);
    const BatchSignMode mode = batch_sign_mode(runtime, sign_bits);
    const int first_tail_shot = rotate_pauli_tiled_batch(runtime, kernel, sign_bits, mode, dim);
    const auto& simd_table = simd::dispatch_table();
    for (int shot = first_tail_shot; shot < runtime.active_shots; ++shot) {
        const bool sign = sign_for_shot(mode, sign_bits, shot);
        rotate_pauli_shot(active_re_for_shot(runtime, shot), active_im_for_shot(runtime, shot), dim, kernel, sign, simd_table);
    }
}

void promote_first_dormant_rotation_batch(
    BatchFactoredExecutorState& runtime,
    double theta,
    const std::vector<std::uint64_t>& sign_bits) {
    if (runtime.ndormant <= 0) {
        fail("cannot promote a dormant qubit when none remain");
    }
    const std::size_t dim = active_length(runtime.k);
    const std::size_t promoted_dim = 2 * dim;
    if (promoted_dim > static_cast<std::size_t>(runtime.active_pitch)) {
        fail("batch active storage has too few columns for dormant promotion");
    }
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const BatchSignMode mode = batch_sign_mode(runtime, sign_bits);
    for (int shot = 0; shot < runtime.active_shots; ++shot) {
        const bool sign = mode == BatchSignMode::Mixed ? batch_bit_at(sign_bits, shot) : mode == BatchSignMode::AllPlus;
        const double q = sign ? s : -s;
        double* re = active_re_for_shot(runtime, shot);
        double* im = active_im_for_shot(runtime, shot);
        SYMFT_SINGLE_SIMD_LOOP
        for (std::size_t basis = 0; basis < dim; ++basis) {
            const double r = re[basis];
            const double i = im[basis];
            re[basis] = c * r;
            im[basis] = c * i;
            re[dim + basis] = -q * i;
            im[dim + basis] = q * r;
        }
    }
    ++runtime.k;
    --runtime.ndormant;
}

void sample_batch_measurement_branches_from_true(
    BatchFactoredExecutorState& runtime,
    std::vector<std::uint64_t>& branch_bits,
    std::vector<double>& prob_true,
    std::vector<double>& invnorms) {
    if (branch_bits.size() < runtime.batch_words) {
        branch_bits.resize(runtime.batch_words, 0);
    }
    const std::size_t nwords = runtime_batch_word_count(runtime);
    for (std::size_t word = 0; word < nwords; ++word) {
        const int base_shot = static_cast<int>(word << 6);
        const int live = std::min(64, runtime.active_shots - base_shot);
        std::uint64_t packed = 0;
        for (int bit = 0; bit < live; ++bit) {
            const int shot = base_shot + bit;
            const double pt = std::clamp(prob_true[static_cast<std::size_t>(shot)], 0.0, 1.0);
            const bool branch = sample_bernoulli(runtime.rng_state, pt);
            if (branch) {
                packed |= std::uint64_t{1} << bit;
            }
            const double probability = branch ? pt : 1.0 - pt;
            if (probability <= 0.0) {
                fail("sampled an impossible active measurement branch");
            }
            invnorms[static_cast<std::size_t>(shot)] = 1.0 / std::sqrt(probability);
        }
        branch_bits[word] = packed;
    }
    for (std::size_t word = nwords; word < runtime.batch_words; ++word) {
        branch_bits[word] = 0;
    }
}

void measure_diagonal_active_pauli_branch_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition) {
    const auto& source_false = kernel.source0_false;
    const auto& source_true = kernel.source0_true;
    const std::size_t out_dim = source_false.size();
    for (int shot = 0; shot < runtime.active_shots; ++shot) {
        runtime.branch_prob_true[static_cast<std::size_t>(shot)] = detail::norm_sum_soa_inline(
            active_re_for_shot(runtime, shot),
            active_im_for_shot(runtime, shot),
            source_true.data(),
            out_dim);
    }
    sample_batch_measurement_branches_from_true(
        runtime,
        runtime.eval_scratch,
        runtime.branch_prob_true,
        runtime.branch_invnorms);
    const auto& branch_bits = runtime.eval_scratch;
    for (int shot = 0; shot < runtime.active_shots; ++shot) {
        const bool branch = batch_bit_at(branch_bits, shot);
        const auto& sources = branch ? source_true : source_false;
        const double invnorm = runtime.branch_invnorms[static_cast<std::size_t>(shot)];
        double* re = active_re_for_shot(runtime, shot);
        double* im = active_im_for_shot(runtime, shot);
        SYMFT_SINGLE_SIMD_LOOP
        for (std::size_t idx = 0; idx < out_dim; ++idx) {
            const std::size_t source = sources[idx];
            re[idx] = re[source] * invnorm;
            im[idx] = im[source] * invnorm;
        }
    }
    finish_active_measurement_branch(runtime, branch_condition, branch_bits);
}

void measure_nondiagonal_active_pauli_branch_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition) {
    for (int shot = 0; shot < runtime.active_shots; ++shot) {
        runtime.branch_prob_true[static_cast<std::size_t>(shot)] = nondiagonal_probability_for_shot(
            active_re_for_shot(runtime, shot),
            active_im_for_shot(runtime, shot),
            kernel,
            true);
    }
    sample_batch_measurement_branches_from_true(
        runtime,
        runtime.eval_scratch,
        runtime.branch_prob_true,
        runtime.branch_invnorms);
    const auto& branch_bits = runtime.eval_scratch;
    const std::size_t pitch = static_cast<std::size_t>(runtime.active_pitch);
    for (int shot = 0; shot < runtime.active_shots; ++shot) {
        const bool branch = batch_bit_at(branch_bits, shot);
        const double invnorm = runtime.branch_invnorms[static_cast<std::size_t>(shot)];
        const std::size_t base = static_cast<std::size_t>(shot) * pitch;
        project_nondiagonal_shot(
            runtime.active_re.data() + base,
            runtime.active_im.data() + base,
            runtime.scratch_re.data() + base,
            runtime.scratch_im.data() + base,
            kernel,
            branch,
            invnorm);
    }
    finish_active_measurement_branch(runtime, branch_condition, branch_bits);
}

void measure_precomputed_active_pauli_branch_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition) {
    if (runtime.k <= 0) {
        fail("cannot measure an active Pauli when k == 0");
    }
    if (kernel.action.nqubits != runtime.k) {
        fail("measurement kernel dimension does not match batch active state");
    }
    if (kernel.is_diagonal) {
        measure_diagonal_active_pauli_branch_batch(runtime, kernel, branch_condition);
    } else {
        measure_nondiagonal_active_pauli_branch_batch(runtime, kernel, branch_condition);
    }
}

void measure_precomputed_active_pauli_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition,
    const SymbolicBoolEvaluationPlan& outcome_plan,
    std::optional<int> record,
    std::optional<int> record_condition) {
    measure_precomputed_active_pauli_branch_batch(runtime, kernel, branch_condition);
    if (write_direct_branch_measurement_record(
            runtime,
            branch_condition,
            outcome_plan,
            record,
            record_condition)) {
        return;
    }
    eval_symbolic_bool_batch(runtime.eval_scratch, outcome_plan, runtime);
    write_batch_measurement_record(runtime, record, runtime.eval_scratch, record_condition);
}

} // namespace symft
