#include "batch_internal.hpp"

namespace symft {

const std::vector<double>& fill_rotation_coefficients(
    BatchFactoredExecutorState& runtime,
    const std::vector<std::uint64_t>& sign_bits,
    double minus_coeff,
    double plus_coeff) {
    if (runtime.rotation_coefficients.size() < static_cast<std::size_t>(runtime.active_shots)) {
        runtime.rotation_coefficients.resize(static_cast<std::size_t>(runtime.active_shots), 0.0);
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
    if (runtime.active_shots == runtime.active_pitch) {
        const std::size_t active_prefix_size = out_dim * static_cast<std::size_t>(runtime.active_pitch);
        std::copy_n(runtime.scratch_re.data(), active_prefix_size, runtime.active_re.data());
        std::copy_n(runtime.scratch_im.data(), active_prefix_size, runtime.active_im.data());
        return;
    }
    const std::size_t pitch = static_cast<std::size_t>(runtime.active_pitch);
    const auto live = static_cast<std::size_t>(runtime.active_shots);
    for (std::size_t basis = 0; basis < out_dim; ++basis) {
        const std::size_t base = basis * pitch;
        std::copy_n(runtime.scratch_re.data() + base, live, runtime.active_re.data() + base);
        std::copy_n(runtime.scratch_im.data() + base, live, runtime.active_im.data() + base);
    }
}

void swap_active_rows(BatchFactoredExecutorState& runtime, std::size_t row0, std::size_t row1) {
    const std::size_t pitch = static_cast<std::size_t>(runtime.active_pitch);
    double* r0 = runtime.active_re.data() + row0 * pitch;
    double* i0 = runtime.active_im.data() + row0 * pitch;
    double* r1 = runtime.active_re.data() + row1 * pitch;
    double* i1 = runtime.active_im.data() + row1 * pitch;
    SYMFT_BATCH_SIMD_LOOP
    for (int shot = 0; shot < runtime.active_shots; ++shot) {
        const auto idx = static_cast<std::size_t>(shot);
        const double tr = r0[idx];
        const double ti = i0[idx];
        r0[idx] = r1[idx];
        i0[idx] = i1[idx];
        r1[idx] = tr;
        i1[idx] = ti;
    }
}

bool can_permute_diagonal_measurement_in_place(
    const BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel) {
    return kernel.pivot == runtime.k - 1;
}

void permute_diagonal_measurement_eigenspaces_in_place(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel) {
    const std::size_t out_dim = kernel.source0_false.size();
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const std::size_t false_src = kernel.source0_false[idx];
        if (false_src != idx) {
            const std::size_t true_src = kernel.source0_true[idx];
            swap_active_rows(runtime, false_src, true_src);
        }
    }
}

void measure_diagonal_true_prob_from_false_prefix(
    BatchFactoredExecutorState& runtime,
    std::size_t out_dim) {
    std::fill(
        runtime.branch_prob_true.begin(),
        runtime.branch_prob_true.begin() + static_cast<std::ptrdiff_t>(runtime.active_shots),
        0.0);
    const std::size_t pitch = static_cast<std::size_t>(runtime.active_pitch);
    for (std::size_t basis = 0; basis < out_dim; ++basis) {
        const std::size_t base = basis * pitch;
        const double* rp = runtime.active_re.data() + base;
        const double* ip = runtime.active_im.data() + base;
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < runtime.active_shots; ++shot) {
            const double r = rp[shot];
            const double i = ip[shot];
            runtime.branch_prob_true[static_cast<std::size_t>(shot)] += r * r + i * i;
        }
    }
    SYMFT_BATCH_SIMD_LOOP
    for (int shot = 0; shot < runtime.active_shots; ++shot) {
        const auto idx = static_cast<std::size_t>(shot);
        runtime.branch_prob_true[idx] = 1.0 - std::clamp(runtime.branch_prob_true[idx], 0.0, 1.0);
    }
}

void project_diagonal_measurement_to_scratch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    const std::vector<std::uint64_t>& branch_bits) {
    const std::size_t out_dim = kernel.source0_false.size();
    const std::size_t pitch = static_cast<std::size_t>(runtime.active_pitch);
    for (std::size_t basis = 0; basis < out_dim; ++basis) {
        double* dst_r = runtime.scratch_re.data() + basis * pitch;
        double* dst_i = runtime.scratch_im.data() + basis * pitch;
        const double* false_r = runtime.active_re.data() + kernel.source0_false[basis] * pitch;
        const double* false_i = runtime.active_im.data() + kernel.source0_false[basis] * pitch;
        const double* true_r = runtime.active_re.data() + kernel.source0_true[basis] * pitch;
        const double* true_i = runtime.active_im.data() + kernel.source0_true[basis] * pitch;
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < runtime.active_shots; ++shot) {
            const auto shot_idx = static_cast<std::size_t>(shot);
            const bool branch = (branch_bits[batch_shot_word(shot)] & batch_shot_mask(shot)) != 0;
            const double n = runtime.branch_invnorms[shot_idx];
            dst_r[shot_idx] = (branch ? true_r[shot_idx] : false_r[shot_idx]) * n;
            dst_i[shot_idx] = (branch ? true_i[shot_idx] : false_i[shot_idx]) * n;
        }
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

void project_nondiagonal_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    std::size_t out_dim,
    bool use_xmask_kernel,
    const std::vector<std::uint64_t>& branch_bits) {
    if (use_xmask_kernel) {
        batch_simd::scalar_table().nondiagonal_xmask_project(
            runtime.active_re.data(),
            runtime.active_im.data(),
            runtime.scratch_re.data(),
            runtime.scratch_im.data(),
            static_cast<std::size_t>(runtime.active_pitch),
            runtime.active_shots,
            kernel.action.xmask,
            static_cast<unsigned>(kernel.pivot),
            out_dim,
            kernel.coeff1_false_real.data(),
            kernel.coeff1_false_imag.data(),
            branch_bits.data(),
            runtime.branch_invnorms.data());
        return;
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
    if (kernel.pair_count < kXmaskRotationPairThreshold) {
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
    const double c = kernel.cos_theta;
    if (kernel.is_diagonal) {
        const std::size_t dim = active_length(runtime.k);
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
    if (runtime.active_re.size() < promoted_dim * static_cast<std::size_t>(runtime.active_pitch)) {
        fail("batch active storage has too few columns for dormant promotion");
    }
    const double c = std::cos(theta);
    const double s = std::sin(theta);
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
    fill_batch_bits(branch_bits, runtime, false);
    for (int shot = 0; shot < runtime.active_shots; ++shot) {
        const double pt = std::clamp(prob_true[static_cast<std::size_t>(shot)], 0.0, 1.0);
        const bool branch = sample_bernoulli(runtime.rng_state, pt);
        if (branch) {
            set_batch_bit(branch_bits, shot);
        }
        const double probability = branch ? pt : 1.0 - pt;
        if (probability <= 0.0) {
            fail("sampled an impossible active measurement branch");
        }
        invnorms[static_cast<std::size_t>(shot)] = 1.0 / std::sqrt(probability);
    }
}

void measure_active_last_z_branch_batch(
    BatchFactoredExecutorState& runtime,
    int branch_condition) {
    if (runtime.k <= 0) {
        fail("cannot measure the last active qubit when k == 0");
    }
    const int new_k = runtime.k - 1;
    const std::size_t dim = active_length(new_k);
    batch_simd::scalar_table().last_z_measure_true_prob(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.active_pitch),
        runtime.active_shots,
        dim,
        runtime.branch_prob_true.data());
    sample_batch_measurement_branches_from_true(
        runtime,
        runtime.eval_scratch,
        runtime.branch_prob_true,
        runtime.branch_invnorms);
    const auto& branch_bits = runtime.eval_scratch;
    batch_simd::scalar_table().last_z_project(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.active_pitch),
        runtime.active_shots,
        dim,
        branch_bits.data(),
        runtime.branch_invnorms.data());
    finish_active_measurement_branch(runtime, branch_condition, branch_bits);
}

void measure_active_last_z_batch(
    BatchFactoredExecutorState& runtime,
    int branch_condition,
    const SymbolicBoolEvaluationPlan& outcome_plan,
    std::optional<int> record,
    std::optional<int> record_condition) {
    measure_active_last_z_branch_batch(runtime, branch_condition);
    eval_symbolic_bool_batch(runtime.eval_scratch, outcome_plan, runtime);
    write_batch_measurement_record(runtime, record, runtime.eval_scratch, record_condition);
}

void measure_diagonal_active_pauli_branch_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition) {
    const std::size_t out_dim = kernel.source0_false.size();
    const bool in_place_permuted = can_permute_diagonal_measurement_in_place(runtime, kernel);
    if (in_place_permuted) {
        permute_diagonal_measurement_eigenspaces_in_place(runtime, kernel);
        measure_diagonal_true_prob_from_false_prefix(runtime, out_dim);
    } else {
        batch_simd::scalar_table().diagonal_measure_true_prob(
            runtime.active_re.data(),
            runtime.active_im.data(),
            static_cast<std::size_t>(runtime.active_pitch),
            runtime.active_shots,
            kernel.source0_true.data(),
            out_dim,
            runtime.branch_prob_true.data());
    }
    sample_batch_measurement_branches_from_true(
        runtime,
        runtime.eval_scratch,
        runtime.branch_prob_true,
        runtime.branch_invnorms);
    const auto& branch_bits = runtime.eval_scratch;
    if (in_place_permuted) {
        batch_simd::scalar_table().last_z_project(
            runtime.active_re.data(),
            runtime.active_im.data(),
            static_cast<std::size_t>(runtime.active_pitch),
            runtime.active_shots,
            out_dim,
            branch_bits.data(),
            runtime.branch_invnorms.data());
    } else {
        project_diagonal_measurement_to_scratch(runtime, kernel, branch_bits);
        copy_projected_active_prefix_from_scratch(runtime, out_dim);
    }
    finish_active_measurement_branch(runtime, branch_condition, branch_bits);
}

void measure_nondiagonal_active_pauli_branch_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition) {
    const std::size_t out_dim = kernel.source0_false.size();
    const bool use_xmask_kernel = can_use_nondiagonal_xmask_measurement(kernel);
    measure_nondiagonal_true_prob_batch(runtime, kernel, out_dim, use_xmask_kernel);
    sample_batch_measurement_branches_from_true(
        runtime,
        runtime.eval_scratch,
        runtime.branch_prob_true,
        runtime.branch_invnorms);
    const auto& branch_bits = runtime.eval_scratch;
    project_nondiagonal_batch(runtime, kernel, out_dim, use_xmask_kernel, branch_bits);
    copy_projected_active_prefix_from_scratch(runtime, out_dim);
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
    eval_symbolic_bool_batch(runtime.eval_scratch, outcome_plan, runtime);
    write_batch_measurement_record(runtime, record, runtime.eval_scratch, record_condition);
}

} // namespace symft
