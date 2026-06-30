#include "batch_internal.hpp"
#include "sampler/active_kernels.hpp"
#include "simd/simd.hpp"

namespace symft {

const std::vector<double>& fill_rotation_coefficients(
    BatchFactoredExecutorState& runtime,
    const std::vector<std::uint64_t>& sign_bits,
    double minus_coeff,
    double plus_coeff) {
    const int lanes = runtime.active_pitch;
    if (runtime.rotation_coefficients.size() < static_cast<std::size_t>(lanes)) {
        runtime.rotation_coefficients.resize(static_cast<std::size_t>(lanes), 0.0);
    }
    const std::size_t nwords = runtime_batch_word_count(runtime);
    for (std::size_t word = 0; word < nwords; ++word) {
        const std::uint64_t bits = word < sign_bits.size() ? sign_bits[word] : 0;
        const int base_shot = static_cast<int>(word << 6);
        const int live = std::min(64, runtime.active_shots - base_shot);
        SYMFT_BATCH_SIMD_LOOP
        for (int bit = 0; bit < live; ++bit) {
            runtime.rotation_coefficients[static_cast<std::size_t>(base_shot + bit)] =
                ((bits >> bit) & 1ULL) != 0 ? plus_coeff : minus_coeff;
        }
    }
    return runtime.rotation_coefficients;
}

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

void rotate_contiguous_active(
    double* re,
    double* im,
    std::size_t dim,
    const PrecomputedActivePauliRotationKernel& kernel,
    bool sign) {
    const double c = kernel.cos_theta;
    const auto& simd_table = simd::dispatch_table();
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
        detail::rotate_general_pairs_soa_inline(
            re,
            im,
            dim,
            kernel.action.xmask,
            kernel.pair_bit,
            left_coeff.data(),
            right_coeff.data(),
            c);
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

double diagonal_probability_contiguous(
    const double* re,
    const double* im,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch) {
    const auto& sources = branch ? kernel.source0_true : kernel.source0_false;
    const double probability = sources.size() < detail::kSimdPairRotationThreshold
                                   ? detail::norm_sum_soa_inline(re, im, sources.data(), sources.size())
                                   : simd::dispatch_table().norm_sum_soa(re, im, sources.data(), sources.size());
    return std::clamp(probability, 0.0, 1.0);
}

double nondiagonal_probability_contiguous(
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
        return std::clamp(fast_probability, 0.0, 1.0);
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
    return std::clamp(probability, 0.0, 1.0);
}

void project_diagonal_contiguous(
    double* re,
    double* im,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch,
    double invnorm) {
    const auto& sources = branch ? kernel.source0_true : kernel.source0_false;
    SYMFT_SINGLE_SIMD_LOOP
    for (std::size_t idx = 0; idx < sources.size(); ++idx) {
        const std::size_t source = sources[idx];
        re[idx] = re[source] * invnorm;
        im[idx] = im[source] * invnorm;
    }
}

void project_nondiagonal_contiguous(
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

bool can_use_nondiagonal_xmask_measurement(const PrecomputedActivePauliMeasurementKernel& kernel) {
    return detail::highest_set_bit64(kernel.action.xmask) == kernel.pivot;
}

void finish_active_measurement_branch(
    BatchFactoredExecutorState& runtime,
    int branch_condition,
    const std::vector<std::uint64_t>& branch_bits) {
    --runtime.k;
    ++runtime.ndormant;
    assign_batch_symbol(runtime, branch_condition, branch_bits);
}

void copy_projected_active_prefix_from_scratch(BatchFactoredExecutorState& runtime, std::size_t out_dim) {
    const std::size_t pitch = static_cast<std::size_t>(runtime.active_pitch);
    const std::size_t live = static_cast<std::size_t>(runtime.active_shots);
    for (std::size_t basis = 0; basis < out_dim; ++basis) {
        const std::size_t base = basis * pitch;
        std::copy_n(runtime.scratch_re.data() + base, live, runtime.active_re.data() + base);
        std::copy_n(runtime.scratch_im.data() + base, live, runtime.active_im.data() + base);
    }
}

void measure_nondiagonal_true_prob_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    std::size_t out_dim,
    bool use_xmask_kernel) {
    if (use_xmask_kernel) {
        batch_simd::scalar_table().nondiagonal_xmask_measure_true_prob(
            runtime.active_re.data(),
            runtime.active_im.data(),
            static_cast<std::size_t>(runtime.active_pitch),
            runtime.active_shots,
            kernel.action.xmask,
            static_cast<unsigned>(kernel.pivot),
            out_dim,
            kernel.coeff1_false_real.data(),
            kernel.coeff1_false_imag.data(),
            runtime.branch_prob_true.data());
        return;
    }
    batch_simd::scalar_table().nondiagonal_measure_true_prob(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.active_pitch),
        runtime.active_shots,
        kernel.source0_false.data(),
        kernel.source1_false.data(),
        kernel.coeff1_false_real.data(),
        kernel.coeff1_false_imag.data(),
        out_dim,
        runtime.branch_prob_true.data());
}

bool project_nondiagonal_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    std::size_t out_dim,
    bool use_xmask_kernel,
    const std::vector<std::uint64_t>& branch_bits) {
    if (use_xmask_kernel) {
        batch_simd::scalar_table().nondiagonal_xmask_project(
            runtime.active_re.data(),
            runtime.active_im.data(),
            runtime.active_re.data(),
            runtime.active_im.data(),
            static_cast<std::size_t>(runtime.active_pitch),
            runtime.active_shots,
            kernel.action.xmask,
            static_cast<unsigned>(kernel.pivot),
            out_dim,
            kernel.coeff1_false_real.data(),
            kernel.coeff1_false_imag.data(),
            branch_bits.data(),
            runtime.branch_invnorms.data());
        return true;
    }
    batch_simd::scalar_table().nondiagonal_project(
        runtime.active_re.data(),
        runtime.active_im.data(),
        runtime.scratch_re.data(),
        runtime.scratch_im.data(),
        static_cast<std::size_t>(runtime.active_pitch),
        runtime.active_shots,
        kernel.source0_false.data(),
        kernel.source1_false.data(),
        kernel.coeff1_false_real.data(),
        kernel.coeff1_false_imag.data(),
        out_dim,
        branch_bits.data(),
        runtime.branch_invnorms.data());
    return false;
}

void rotate_uniform_imag_pairs_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits) {
    const BatchSignMode mode = batch_sign_mode(runtime, sign_bits);
    if (mode != BatchSignMode::Mixed) {
        const double q = mode == BatchSignMode::AllPlus
                             ? kernel.pair_left_plus_coefficients.front().imag()
                             : kernel.pair_left_minus_coefficients.front().imag();
        batch_simd::scalar_table().rotate_uniform_imag_xmask_const(
            runtime.active_re.data(),
            runtime.active_im.data(),
            static_cast<std::size_t>(runtime.active_pitch),
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            kernel.pair_count,
            kernel.cos_theta,
            q);
        return;
    }
    const auto& coeffs = fill_rotation_coefficients(
        runtime,
        sign_bits,
        kernel.pair_left_minus_coefficients.front().imag(),
        kernel.pair_left_plus_coefficients.front().imag());
    if (kernel.pair_count < kXmaskRotationPairThreshold && runtime.active_pitch != 2) {
        batch_simd::scalar_table().rotate_uniform_imag_pairs(
            runtime.active_re.data(),
            runtime.active_im.data(),
            static_cast<std::size_t>(runtime.active_pitch),
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            kernel.pair_count,
            kernel.cos_theta,
            coeffs.data());
        return;
    }
    batch_simd::scalar_table().rotate_uniform_imag_xmask(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.active_pitch),
        runtime.active_shots,
        kernel.action.xmask,
        kernel.pair_bit,
        kernel.pair_count,
        kernel.cos_theta,
        coeffs.data());
}

void rotate_real_pair_flip_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits) {
    const BatchSignMode mode = batch_sign_mode(runtime, sign_bits);
    if (mode != BatchSignMode::Mixed) {
        const double q = mode == BatchSignMode::AllPlus
                             ? kernel.pair_left_plus_coefficients.front().real()
                             : kernel.pair_left_minus_coefficients.front().real();
        batch_simd::scalar_table().rotate_real_pair_flip_xmask_const(
            runtime.active_re.data(),
            runtime.active_im.data(),
            static_cast<std::size_t>(runtime.active_pitch),
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            kernel.real_pair_flip_basis_phase_signs.data(),
            kernel.pair_count,
            kernel.cos_theta,
            q);
        return;
    }
    const auto& coeffs = fill_rotation_coefficients(
        runtime,
        sign_bits,
        kernel.pair_left_minus_coefficients.front().real(),
        kernel.pair_left_plus_coefficients.front().real());
    if (kernel.pair_count < kXmaskRotationPairThreshold) {
        batch_simd::scalar_table().rotate_real_pair_flip(
            runtime.active_re.data(),
            runtime.active_im.data(),
            static_cast<std::size_t>(runtime.active_pitch),
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            kernel.real_pair_flip_basis_phase_signs.data(),
            kernel.pair_count,
            kernel.cos_theta,
            coeffs.data());
        return;
    }
    batch_simd::scalar_table().rotate_real_pair_flip_xmask(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.active_pitch),
        runtime.active_shots,
        kernel.action.xmask,
        kernel.pair_bit,
        kernel.real_pair_flip_basis_phase_signs.data(),
        kernel.pair_count,
        kernel.cos_theta,
        coeffs.data());
}

void rotate_pauli_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits) {
    if (kernel.action.nqubits != runtime.k) {
        fail("rotation kernel dimension does not match batch active state");
    }
    const std::size_t dim = active_length(runtime.k);
    if (runtime.dense_shot_major_active) {
        for (int shot = 0; shot < runtime.active_shots; ++shot) {
            rotate_contiguous_active(
                batch_active_re_for_shot(runtime, shot),
                batch_active_im_for_shot(runtime, shot),
                dim,
                kernel,
                batch_bit_at(sign_bits, shot));
        }
        return;
    }
    if (runtime.active_pitch == 1) {
        rotate_contiguous_active(
            runtime.active_re.data(),
            runtime.active_im.data(),
            dim,
            kernel,
            batch_bit_at(sign_bits, 0));
        return;
    }
    const double c = kernel.cos_theta;
    if (kernel.is_diagonal) {
        const BatchSignMode mode = batch_sign_mode(runtime, sign_bits);
        if (mode == BatchSignMode::AllMinus) {
            batch_simd::scalar_table().rotate_diagonal_const(
                runtime.active_re.data(),
                runtime.active_im.data(),
                static_cast<std::size_t>(runtime.active_pitch),
                runtime.active_shots,
                dim,
                kernel.diagonal_minus_coefficients.data(),
                c);
        } else if (mode == BatchSignMode::AllPlus) {
            batch_simd::scalar_table().rotate_diagonal_const(
                runtime.active_re.data(),
                runtime.active_im.data(),
                static_cast<std::size_t>(runtime.active_pitch),
                runtime.active_shots,
                dim,
                kernel.diagonal_plus_coefficients.data(),
                c);
        } else {
            batch_simd::scalar_table().rotate_diagonal_mixed(
                runtime.active_re.data(),
                runtime.active_im.data(),
                static_cast<std::size_t>(runtime.active_pitch),
                runtime.active_shots,
                dim,
                kernel.diagonal_minus_coefficients.data(),
                kernel.diagonal_plus_coefficients.data(),
                sign_bits.data(),
                c);
        }
        return;
    }
    if (kernel.uniform_imag_pairs) {
        rotate_uniform_imag_pairs_batch(runtime, kernel, sign_bits);
        return;
    }
    if (kernel.real_pair_flip) {
        rotate_real_pair_flip_batch(runtime, kernel, sign_bits);
        return;
    }
    const BatchSignMode mode = batch_sign_mode(runtime, sign_bits);
    if (mode == BatchSignMode::AllMinus) {
        batch_simd::scalar_table().rotate_general_xmask_const(
            runtime.active_re.data(),
            runtime.active_im.data(),
            static_cast<std::size_t>(runtime.active_pitch),
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            kernel.pair_count,
            kernel.pair_left_minus_coefficients.data(),
            kernel.pair_right_minus_coefficients.data(),
            c);
    } else if (mode == BatchSignMode::AllPlus) {
        batch_simd::scalar_table().rotate_general_xmask_const(
            runtime.active_re.data(),
            runtime.active_im.data(),
            static_cast<std::size_t>(runtime.active_pitch),
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            kernel.pair_count,
            kernel.pair_left_plus_coefficients.data(),
            kernel.pair_right_plus_coefficients.data(),
            c);
    } else {
        batch_simd::scalar_table().rotate_general_xmask_mixed(
            runtime.active_re.data(),
            runtime.active_im.data(),
            static_cast<std::size_t>(runtime.active_pitch),
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            kernel.pair_count,
            kernel.pair_left_minus_coefficients.data(),
            kernel.pair_right_minus_coefficients.data(),
            kernel.pair_left_plus_coefficients.data(),
            kernel.pair_right_plus_coefficients.data(),
            sign_bits.data(),
            c);
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
    if (runtime.dense_shot_major_active && runtime.active_stride < promoted_dim) {
        fail("batch active shot-major stride is too short for dormant promotion");
    }
    if (!runtime.dense_shot_major_active &&
        runtime.active_re.size() < promoted_dim * static_cast<std::size_t>(runtime.active_pitch)) {
        fail("batch active storage has too few columns for dormant promotion");
    }
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    if (runtime.dense_shot_major_active) {
        for (int shot = 0; shot < runtime.active_shots; ++shot) {
            const double q = batch_bit_at(sign_bits, shot) ? s : -s;
            double* re = batch_active_re_for_shot(runtime, shot);
            double* im = batch_active_im_for_shot(runtime, shot);
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
        return;
    }
    if (runtime.active_pitch == 1) {
        const double q = batch_bit_at(sign_bits, 0) ? s : -s;
        SYMFT_SINGLE_SIMD_LOOP
        for (std::size_t basis = 0; basis < dim; ++basis) {
            const double r = runtime.active_re[basis];
            const double i = runtime.active_im[basis];
            runtime.active_re[basis] = c * r;
            runtime.active_im[basis] = c * i;
            runtime.active_re[dim + basis] = -q * i;
            runtime.active_im[dim + basis] = q * r;
        }
        ++runtime.k;
        --runtime.ndormant;
        return;
    }
    const auto& coeffs = fill_rotation_coefficients(runtime, sign_bits, -s, s);
    batch_simd::scalar_table().promote_first_dormant_rotation(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.active_pitch),
        runtime.active_shots,
        dim,
        c,
        coeffs.data());
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
    std::fill(
        invnorms.begin() + static_cast<std::ptrdiff_t>(runtime.active_shots),
        invnorms.begin() + static_cast<std::ptrdiff_t>(runtime.active_pitch),
        1.0);
}

template <typename ProbabilityFn, typename ProjectFn>
void measure_shot_major_active_branch_batch(
    BatchFactoredExecutorState& runtime,
    int branch_condition,
    ProbabilityFn probability,
    ProjectFn project) {
    const std::size_t nwords = runtime_batch_word_count(runtime);
    for (std::size_t word = 0; word < nwords; ++word) {
        const int base_shot = static_cast<int>(word << 6);
        const int live = std::min(64, runtime.active_shots - base_shot);
        std::uint64_t packed = 0;
        for (int bit = 0; bit < live; ++bit) {
            const int shot = base_shot + bit;
            double* re = batch_active_re_for_shot(runtime, shot);
            double* im = batch_active_im_for_shot(runtime, shot);
            const double pt = probability(re, im);
            const bool branch = sample_bernoulli(runtime.rng_state, pt);
            if (branch) {
                packed |= std::uint64_t{1} << bit;
            }
            const double probability_sampled = branch ? pt : 1.0 - pt;
            if (probability_sampled <= 0.0) {
                fail("sampled an impossible active measurement branch");
            }
            project(shot, re, im, branch, 1.0 / std::sqrt(probability_sampled));
        }
        runtime.eval_scratch[word] = packed;
    }
    for (std::size_t word = nwords; word < runtime.batch_words; ++word) {
        runtime.eval_scratch[word] = 0;
    }
    finish_active_measurement_branch(runtime, branch_condition, runtime.eval_scratch);
}

void measure_diagonal_active_pauli_branch_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition) {
    const auto& source_false = kernel.source0_false;
    const auto& source_true = kernel.source0_true;
    const std::size_t out_dim = source_false.size();
    if (runtime.dense_shot_major_active) {
        measure_shot_major_active_branch_batch(
            runtime,
            branch_condition,
            [&](double* re, double* im) {
                return diagonal_probability_contiguous(re, im, kernel, true);
            },
            [&](int, double* re, double* im, bool branch, double invnorm) {
                project_diagonal_contiguous(re, im, kernel, branch, invnorm);
            });
        return;
    }
    if (runtime.active_pitch == 1) {
        runtime.branch_prob_true[0] = diagonal_probability_contiguous(
            runtime.active_re.data(),
            runtime.active_im.data(),
            kernel,
            true);
        sample_batch_measurement_branches_from_true(
            runtime,
            runtime.eval_scratch,
            runtime.branch_prob_true,
            runtime.branch_invnorms);
        const auto& branch_bits = runtime.eval_scratch;
        project_diagonal_contiguous(
            runtime.active_re.data(),
            runtime.active_im.data(),
            kernel,
            batch_bit_at(branch_bits, 0),
            runtime.branch_invnorms[0]);
        finish_active_measurement_branch(runtime, branch_condition, branch_bits);
        return;
    }
    batch_simd::scalar_table().diagonal_measure_true_prob(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.active_pitch),
        runtime.active_shots,
        source_true.data(),
        out_dim,
        runtime.branch_prob_true.data());
    sample_batch_measurement_branches_from_true(
        runtime,
        runtime.eval_scratch,
        runtime.branch_prob_true,
        runtime.branch_invnorms);
    const auto& branch_bits = runtime.eval_scratch;
    batch_simd::scalar_table().diagonal_project(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.active_pitch),
        runtime.active_shots,
        source_false.data(),
        source_true.data(),
        out_dim,
        branch_bits.data(),
        runtime.branch_invnorms.data());
    finish_active_measurement_branch(runtime, branch_condition, branch_bits);
}

void measure_nondiagonal_active_pauli_branch_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition) {
    const std::size_t out_dim = kernel.source0_false.size();
    if (runtime.dense_shot_major_active) {
        measure_shot_major_active_branch_batch(
            runtime,
            branch_condition,
            [&](double* re, double* im) {
                return nondiagonal_probability_contiguous(re, im, kernel, true);
            },
            [&](int shot, double* re, double* im, bool branch, double invnorm) {
                project_nondiagonal_contiguous(
                    re,
                    im,
                    batch_scratch_re_for_shot(runtime, shot),
                    batch_scratch_im_for_shot(runtime, shot),
                    kernel,
                    branch,
                    invnorm);
            });
        return;
    }
    if (runtime.active_pitch == 1) {
        runtime.branch_prob_true[0] = nondiagonal_probability_contiguous(
            runtime.active_re.data(),
            runtime.active_im.data(),
            kernel,
            true);
        sample_batch_measurement_branches_from_true(
            runtime,
            runtime.eval_scratch,
            runtime.branch_prob_true,
            runtime.branch_invnorms);
        const auto& branch_bits = runtime.eval_scratch;
        project_nondiagonal_contiguous(
            runtime.active_re.data(),
            runtime.active_im.data(),
            runtime.scratch_re.data(),
            runtime.scratch_im.data(),
            kernel,
            batch_bit_at(branch_bits, 0),
            runtime.branch_invnorms[0]);
        finish_active_measurement_branch(runtime, branch_condition, branch_bits);
        return;
    }
    const bool use_xmask_kernel = can_use_nondiagonal_xmask_measurement(kernel);
    measure_nondiagonal_true_prob_batch(runtime, kernel, out_dim, use_xmask_kernel);
    sample_batch_measurement_branches_from_true(
        runtime,
        runtime.eval_scratch,
        runtime.branch_prob_true,
        runtime.branch_invnorms);
    const auto& branch_bits = runtime.eval_scratch;
    if (!project_nondiagonal_batch(runtime, kernel, out_dim, use_xmask_kernel, branch_bits)) {
        copy_projected_active_prefix_from_scratch(runtime, out_dim);
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
