#include "batch_internal.hpp"

namespace symft {

constexpr double kBatchInvSqrt2 = 0.70710678118654752440;

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

bool batch_vector_bit(const std::vector<std::uint64_t>& bits, int shot) {
    return (bits[batch_shot_word(shot)] & batch_shot_mask(shot)) != 0;
}

template <typename Body>
void for_each_masked_live_shot(
    const BatchFactoredExecutorState& runtime,
    const std::vector<std::uint64_t>& live_bits,
    Body&& body) {
    const std::size_t nwords = runtime_batch_word_count(runtime);
    for (std::size_t word = 0; word < nwords; ++word) {
        std::uint64_t mask = live_bits[word] & batch_live_word_mask(runtime, word);
        while (mask != 0) {
            const int bit = trailing_zeros64(mask);
            body(static_cast<int>(word << 6) + bit);
            mask &= mask - 1;
        }
    }
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
            static_cast<std::size_t>(runtime.batches),
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            kernel.pair_left_indices.size(),
            kernel.cos_theta,
            q);
        return;
    }
    const auto& coeffs = fill_rotation_coefficients(
        runtime,
        sign_bits,
        kernel.pair_left_minus_coefficients.front().imag(),
        kernel.pair_left_plus_coefficients.front().imag());
    if (kernel.pair_left_indices.size() >= kXmaskRotationPairThreshold) {
        batch_simd::scalar_table().rotate_uniform_imag_xmask(
            runtime.active_re.data(),
            runtime.active_im.data(),
            static_cast<std::size_t>(runtime.batches),
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            kernel.pair_left_indices.size(),
            kernel.cos_theta,
            coeffs.data());
        return;
    }
    batch_simd::scalar_table().rotate_uniform_imag_pairs(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.batches),
        runtime.active_shots,
        kernel.pair_left_indices.data(),
        kernel.pair_right_indices.data(),
        kernel.pair_left_indices.size(),
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
            static_cast<std::size_t>(runtime.batches),
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            kernel.real_pair_flip_basis_phase_signs.data(),
            kernel.pair_left_indices.size(),
            kernel.cos_theta,
            q);
        return;
    }
    const auto& coeffs = fill_rotation_coefficients(
        runtime,
        sign_bits,
        kernel.pair_left_minus_coefficients.front().real(),
        kernel.pair_left_plus_coefficients.front().real());
    if (kernel.pair_left_indices.size() >= kXmaskRotationPairThreshold) {
        batch_simd::scalar_table().rotate_real_pair_flip_xmask(
            runtime.active_re.data(),
            runtime.active_im.data(),
            static_cast<std::size_t>(runtime.batches),
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            kernel.real_pair_flip_basis_phase_signs.data(),
            kernel.pair_left_indices.size(),
            kernel.cos_theta,
            coeffs.data());
        return;
    }
    batch_simd::scalar_table().rotate_real_pair_flip(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.batches),
        runtime.active_shots,
        kernel.pair_left_indices.data(),
        kernel.pair_right_indices.data(),
        kernel.real_pair_flip_basis_phase_signs.data(),
        kernel.pair_left_indices.size(),
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
                static_cast<std::size_t>(runtime.batches),
                runtime.active_shots,
                dim,
                kernel.diagonal_minus_coefficients.data(),
                c);
        } else if (mode == BatchSignMode::AllPlus) {
            batch_simd::scalar_table().rotate_diagonal_const(
                runtime.active_re.data(),
                runtime.active_im.data(),
                static_cast<std::size_t>(runtime.batches),
                runtime.active_shots,
                dim,
                kernel.diagonal_plus_coefficients.data(),
                c);
        } else {
            batch_simd::scalar_table().rotate_diagonal_mixed(
                runtime.active_re.data(),
                runtime.active_im.data(),
                static_cast<std::size_t>(runtime.batches),
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
            static_cast<std::size_t>(runtime.batches),
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            kernel.pair_left_indices.size(),
            kernel.pair_left_minus_coefficients.data(),
            kernel.pair_right_minus_coefficients.data(),
            c);
    } else if (mode == BatchSignMode::AllPlus) {
        batch_simd::scalar_table().rotate_general_xmask_const(
            runtime.active_re.data(),
            runtime.active_im.data(),
            static_cast<std::size_t>(runtime.batches),
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            kernel.pair_left_indices.size(),
            kernel.pair_left_plus_coefficients.data(),
            kernel.pair_right_plus_coefficients.data(),
            c);
    } else {
        batch_simd::scalar_table().rotate_general_xmask_mixed(
            runtime.active_re.data(),
            runtime.active_im.data(),
            static_cast<std::size_t>(runtime.batches),
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            kernel.pair_left_indices.size(),
            kernel.pair_left_minus_coefficients.data(),
            kernel.pair_right_minus_coefficients.data(),
            kernel.pair_left_plus_coefficients.data(),
            kernel.pair_right_plus_coefficients.data(),
            sign_bits.data(),
            c);
    }
}

void rotate_pauli_batch_masked(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits,
    const std::vector<std::uint64_t>& live_bits) {
    if (kernel.action.nqubits != runtime.k) {
        fail("rotation kernel dimension does not match batch active state");
    }
    const double c = kernel.cos_theta;
    const std::size_t pitch = static_cast<std::size_t>(runtime.batches);
    if (kernel.is_diagonal) {
        const std::size_t dim = active_length(runtime.k);
        for (std::size_t basis = 0; basis < dim; ++basis) {
            double* rp = runtime.active_re.data() + basis * pitch;
            double* ip = runtime.active_im.data() + basis * pitch;
            for_each_masked_live_shot(runtime, live_bits, [&](int shot) {
                const bool sign = batch_vector_bit(sign_bits, shot);
                const auto& coeff = sign ? kernel.diagonal_plus_coefficients[basis]
                                         : kernel.diagonal_minus_coefficients[basis];
                const double fr = c + coeff.real();
                const double fi = coeff.imag();
                const double r = rp[shot];
                const double v = ip[shot];
                rp[shot] = fr * r - fi * v;
                ip[shot] = fr * v + fi * r;
            });
        }
        return;
    }

    if (kernel.uniform_imag_pairs) {
        const double minus_q = kernel.pair_left_minus_coefficients.front().imag();
        const double plus_q = kernel.pair_left_plus_coefficients.front().imag();
        for (std::size_t idx = 0; idx < kernel.pair_left_indices.size(); ++idx) {
            double* r0p = runtime.active_re.data() + kernel.pair_left_indices[idx] * pitch;
            double* i0p = runtime.active_im.data() + kernel.pair_left_indices[idx] * pitch;
            double* r1p = runtime.active_re.data() + kernel.pair_right_indices[idx] * pitch;
            double* i1p = runtime.active_im.data() + kernel.pair_right_indices[idx] * pitch;
            for_each_masked_live_shot(runtime, live_bits, [&](int shot) {
                const double q = batch_vector_bit(sign_bits, shot) ? plus_q : minus_q;
                const double r0 = r0p[shot];
                const double i0 = i0p[shot];
                const double r1 = r1p[shot];
                const double i1 = i1p[shot];
                r0p[shot] = c * r0 - q * i1;
                i0p[shot] = c * i0 + q * r1;
                r1p[shot] = c * r1 - q * i0;
                i1p[shot] = c * i1 + q * r0;
            });
        }
        return;
    }

    if (kernel.real_pair_flip) {
        const double minus_q = kernel.pair_left_minus_coefficients.front().real();
        const double plus_q = kernel.pair_left_plus_coefficients.front().real();
        for (std::size_t idx = 0; idx < kernel.pair_left_indices.size(); ++idx) {
            double* r0p = runtime.active_re.data() + kernel.pair_left_indices[idx] * pitch;
            double* i0p = runtime.active_im.data() + kernel.pair_left_indices[idx] * pitch;
            double* r1p = runtime.active_re.data() + kernel.pair_right_indices[idx] * pitch;
            double* i1p = runtime.active_im.data() + kernel.pair_right_indices[idx] * pitch;
            const double basis_phase_sign = kernel.real_pair_flip_basis_phase_signs[idx];
            for_each_masked_live_shot(runtime, live_bits, [&](int shot) {
                const double sign_q = batch_vector_bit(sign_bits, shot) ? plus_q : minus_q;
                const double q = basis_phase_sign * sign_q;
                const double r0 = r0p[shot];
                const double i0 = i0p[shot];
                const double r1 = r1p[shot];
                const double i1 = i1p[shot];
                r0p[shot] = c * r0 - q * r1;
                i0p[shot] = c * i0 - q * i1;
                r1p[shot] = c * r1 + q * r0;
                i1p[shot] = c * i1 + q * i0;
            });
        }
        return;
    }

    for (std::size_t idx = 0; idx < kernel.pair_left_indices.size(); ++idx) {
        double* r0p = runtime.active_re.data() + kernel.pair_left_indices[idx] * pitch;
        double* i0p = runtime.active_im.data() + kernel.pair_left_indices[idx] * pitch;
        double* r1p = runtime.active_re.data() + kernel.pair_right_indices[idx] * pitch;
        double* i1p = runtime.active_im.data() + kernel.pair_right_indices[idx] * pitch;
        for_each_masked_live_shot(runtime, live_bits, [&](int shot) {
            const bool sign = batch_vector_bit(sign_bits, shot);
            const auto& left_coeff = sign ? kernel.pair_left_plus_coefficients[idx]
                                          : kernel.pair_left_minus_coefficients[idx];
            const auto& right_coeff = sign ? kernel.pair_right_plus_coefficients[idx]
                                           : kernel.pair_right_minus_coefficients[idx];
            const double lr = left_coeff.real();
            const double li = left_coeff.imag();
            const double rr = right_coeff.real();
            const double ri = right_coeff.imag();
            const double r0 = r0p[shot];
            const double im0 = i0p[shot];
            const double r1 = r1p[shot];
            const double im1 = i1p[shot];
            r0p[shot] = c * r0 + rr * r1 - ri * im1;
            i0p[shot] = c * im0 + rr * im1 + ri * r1;
            r1p[shot] = c * r1 + lr * r0 - li * im0;
            i1p[shot] = c * im1 + lr * im0 + li * r0;
        });
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
    if (runtime.active_re.size() < promoted_dim * static_cast<std::size_t>(runtime.batches)) {
        fail("batch active storage has too few columns for dormant promotion");
    }
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    const auto& coeffs = fill_rotation_coefficients(runtime, sign_bits, -s, s);
    batch_simd::scalar_table().promote_first_dormant_rotation(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.batches),
        runtime.active_shots,
        dim,
        c,
        coeffs.data());
    ++runtime.k;
    --runtime.ndormant;
}

void promote_first_dormant_rotation_batch_masked(
    BatchFactoredExecutorState& runtime,
    double theta,
    const std::vector<std::uint64_t>& sign_bits,
    const std::vector<std::uint64_t>& live_bits) {
    if (runtime.ndormant <= 0) {
        fail("cannot promote a dormant qubit when none remain");
    }
    const std::size_t dim = active_length(runtime.k);
    const std::size_t promoted_dim = 2 * dim;
    const std::size_t pitch = static_cast<std::size_t>(runtime.batches);
    if (runtime.active_re.size() < promoted_dim * pitch) {
        fail("batch active storage has too few columns for dormant promotion");
    }
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    for (std::size_t basis = 0; basis < dim; ++basis) {
        double* r0p = runtime.active_re.data() + basis * pitch;
        double* i0p = runtime.active_im.data() + basis * pitch;
        double* r1p = runtime.active_re.data() + (dim + basis) * pitch;
        double* i1p = runtime.active_im.data() + (dim + basis) * pitch;
        for_each_masked_live_shot(runtime, live_bits, [&](int shot) {
            const double q = batch_vector_bit(sign_bits, shot) ? s : -s;
            const double r = r0p[shot];
            const double i = i0p[shot];
            r0p[shot] = c * r;
            i0p[shot] = c * i;
            r1p[shot] = -q * i;
            i1p[shot] = q * r;
        });
    }
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

void sample_batch_measurement_branches_from_true_masked(
    BatchFactoredExecutorState& runtime,
    std::vector<std::uint64_t>& branch_bits,
    std::vector<double>& prob_true,
    std::vector<double>& invnorms,
    const std::vector<std::uint64_t>& live_bits) {
    fill_batch_bits(branch_bits, runtime, false);
    for_each_masked_live_shot(runtime, live_bits, [&](int shot) {
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
    });
}

void measure_active_last_z_batch(
    BatchFactoredExecutorState& runtime,
    int branch_condition,
    const SymbolicBoolEvaluationPlan& outcome_plan,
    std::optional<int> record,
    std::optional<int> record_condition) {
    if (runtime.k <= 0) {
        fail("cannot measure the last active qubit when k == 0");
    }
    const int new_k = runtime.k - 1;
    const std::size_t dim = active_length(new_k);
    batch_simd::scalar_table().last_z_measure_true_prob(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.batches),
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
        static_cast<std::size_t>(runtime.batches),
        runtime.active_shots,
        dim,
        branch_bits.data(),
        runtime.branch_invnorms.data());
    runtime.k = new_k;
    ++runtime.ndormant;
    assign_batch_symbol(runtime, branch_condition, branch_bits);
    eval_symbolic_bool_batch(runtime.eval_scratch, outcome_plan, runtime);
    write_batch_measurement_record(runtime, record, runtime.eval_scratch, record_condition);
}

void measure_active_last_z_batch_masked(
    BatchFactoredExecutorState& runtime,
    int branch_condition,
    const SymbolicBoolEvaluationPlan& outcome_plan,
    std::optional<int> record,
    std::optional<int> record_condition,
    const std::vector<std::uint64_t>& live_bits) {
    if (runtime.k <= 0) {
        fail("cannot measure the last active qubit when k == 0");
    }
    const int new_k = runtime.k - 1;
    const std::size_t dim = active_length(new_k);
    const std::size_t pitch = static_cast<std::size_t>(runtime.batches);
    std::fill(
        runtime.branch_prob_true.begin(),
        runtime.branch_prob_true.begin() + static_cast<std::ptrdiff_t>(runtime.active_shots),
        0.0);
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const double* rp = runtime.active_re.data() + (dim + basis) * pitch;
        const double* ip = runtime.active_im.data() + (dim + basis) * pitch;
        for_each_masked_live_shot(runtime, live_bits, [&](int shot) {
            const double r = rp[shot];
            const double i = ip[shot];
            runtime.branch_prob_true[static_cast<std::size_t>(shot)] += r * r + i * i;
        });
    }
    sample_batch_measurement_branches_from_true_masked(
        runtime,
        runtime.eval_scratch,
        runtime.branch_prob_true,
        runtime.branch_invnorms,
        live_bits);
    const auto& branch_bits = runtime.eval_scratch;
    for (std::size_t basis = 0; basis < dim; ++basis) {
        double* dst_r = runtime.active_re.data() + basis * pitch;
        double* dst_i = runtime.active_im.data() + basis * pitch;
        const double* false_r = runtime.active_re.data() + basis * pitch;
        const double* false_i = runtime.active_im.data() + basis * pitch;
        const double* true_r = runtime.active_re.data() + (dim + basis) * pitch;
        const double* true_i = runtime.active_im.data() + (dim + basis) * pitch;
        for_each_masked_live_shot(runtime, live_bits, [&](int shot) {
            const bool branch = batch_vector_bit(branch_bits, shot);
            const double n = runtime.branch_invnorms[static_cast<std::size_t>(shot)];
            dst_r[shot] = (branch ? true_r[shot] : false_r[shot]) * n;
            dst_i[shot] = (branch ? true_i[shot] : false_i[shot]) * n;
        });
    }
    runtime.k = new_k;
    ++runtime.ndormant;
    assign_batch_symbol_masked(runtime, branch_condition, branch_bits, live_bits);
    eval_symbolic_bool_batch(runtime.eval_scratch, outcome_plan, runtime);
    write_batch_measurement_record_masked(runtime, record, runtime.eval_scratch, record_condition, live_bits);
}

void measure_diagonal_active_pauli_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition,
    const SymbolicBoolEvaluationPlan& outcome_plan,
    std::optional<int> record,
    std::optional<int> record_condition) {
    const auto& source_false = kernel.source0_false;
    const auto& source_true = kernel.source0_true;
    const std::size_t out_dim = source_false.size();
    batch_simd::scalar_table().diagonal_measure_true_prob(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.batches),
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
        static_cast<std::size_t>(runtime.batches),
        runtime.active_shots,
        source_false.data(),
        source_true.data(),
        out_dim,
        branch_bits.data(),
        runtime.branch_invnorms.data());
    --runtime.k;
    ++runtime.ndormant;
    assign_batch_symbol(runtime, branch_condition, branch_bits);
    eval_symbolic_bool_batch(runtime.eval_scratch, outcome_plan, runtime);
    write_batch_measurement_record(runtime, record, runtime.eval_scratch, record_condition);
}

void measure_diagonal_active_pauli_batch_masked(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition,
    const SymbolicBoolEvaluationPlan& outcome_plan,
    std::optional<int> record,
    std::optional<int> record_condition,
    const std::vector<std::uint64_t>& live_bits) {
    const auto& source_false = kernel.source0_false;
    const auto& source_true = kernel.source0_true;
    const std::size_t out_dim = source_false.size();
    const std::size_t pitch = static_cast<std::size_t>(runtime.batches);
    std::fill(
        runtime.branch_prob_true.begin(),
        runtime.branch_prob_true.begin() + static_cast<std::ptrdiff_t>(runtime.active_shots),
        0.0);
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const double* rp = runtime.active_re.data() + source_true[idx] * pitch;
        const double* ip = runtime.active_im.data() + source_true[idx] * pitch;
        for_each_masked_live_shot(runtime, live_bits, [&](int shot) {
            const double r = rp[shot];
            const double i = ip[shot];
            runtime.branch_prob_true[static_cast<std::size_t>(shot)] += r * r + i * i;
        });
    }
    sample_batch_measurement_branches_from_true_masked(
        runtime,
        runtime.eval_scratch,
        runtime.branch_prob_true,
        runtime.branch_invnorms,
        live_bits);
    const auto& branch_bits = runtime.eval_scratch;
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        double* dst_r = runtime.active_re.data() + idx * pitch;
        double* dst_i = runtime.active_im.data() + idx * pitch;
        const double* false_r = runtime.active_re.data() + source_false[idx] * pitch;
        const double* false_i = runtime.active_im.data() + source_false[idx] * pitch;
        const double* true_r = runtime.active_re.data() + source_true[idx] * pitch;
        const double* true_i = runtime.active_im.data() + source_true[idx] * pitch;
        for_each_masked_live_shot(runtime, live_bits, [&](int shot) {
            const bool branch = batch_vector_bit(branch_bits, shot);
            const double n = runtime.branch_invnorms[static_cast<std::size_t>(shot)];
            dst_r[shot] = (branch ? true_r[shot] : false_r[shot]) * n;
            dst_i[shot] = (branch ? true_i[shot] : false_i[shot]) * n;
        });
    }
    --runtime.k;
    ++runtime.ndormant;
    assign_batch_symbol_masked(runtime, branch_condition, branch_bits, live_bits);
    eval_symbolic_bool_batch(runtime.eval_scratch, outcome_plan, runtime);
    write_batch_measurement_record_masked(runtime, record, runtime.eval_scratch, record_condition, live_bits);
}

void measure_nondiagonal_active_pauli_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition,
    const SymbolicBoolEvaluationPlan& outcome_plan,
    std::optional<int> record,
    std::optional<int> record_condition) {
    const std::size_t out_dim = kernel.source0_false.size();
    batch_simd::scalar_table().nondiagonal_measure_true_prob(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.batches),
        runtime.active_shots,
        kernel.source0_false.data(),
        kernel.source1_false.data(),
        kernel.coeff1_false_real.data(),
        kernel.coeff1_false_imag.data(),
        out_dim,
        runtime.branch_prob_true.data());
    sample_batch_measurement_branches_from_true(
        runtime,
        runtime.eval_scratch,
        runtime.branch_prob_true,
        runtime.branch_invnorms);
    const auto& branch_bits = runtime.eval_scratch;
    batch_simd::scalar_table().nondiagonal_project(
        runtime.active_re.data(),
        runtime.active_im.data(),
        runtime.scratch_re.data(),
        runtime.scratch_im.data(),
        static_cast<std::size_t>(runtime.batches),
        runtime.active_shots,
        kernel.source0_false.data(),
        kernel.source1_false.data(),
        kernel.coeff1_false_real.data(),
        kernel.coeff1_false_imag.data(),
        out_dim,
        branch_bits.data(),
        runtime.branch_invnorms.data());
    const std::size_t active_prefix_size = out_dim * static_cast<std::size_t>(runtime.batches);
    std::copy_n(runtime.scratch_re.data(), active_prefix_size, runtime.active_re.data());
    std::copy_n(runtime.scratch_im.data(), active_prefix_size, runtime.active_im.data());
    --runtime.k;
    ++runtime.ndormant;
    assign_batch_symbol(runtime, branch_condition, branch_bits);
    eval_symbolic_bool_batch(runtime.eval_scratch, outcome_plan, runtime);
    write_batch_measurement_record(runtime, record, runtime.eval_scratch, record_condition);
}

void measure_nondiagonal_active_pauli_batch_masked(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition,
    const SymbolicBoolEvaluationPlan& outcome_plan,
    std::optional<int> record,
    std::optional<int> record_condition,
    const std::vector<std::uint64_t>& live_bits) {
    const std::size_t out_dim = kernel.source0_false.size();
    const std::size_t pitch = static_cast<std::size_t>(runtime.batches);
    std::fill(
        runtime.branch_prob_true.begin(),
        runtime.branch_prob_true.begin() + static_cast<std::ptrdiff_t>(runtime.active_shots),
        0.0);
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const double* r0p = runtime.active_re.data() + kernel.source0_false[idx] * pitch;
        const double* i0p = runtime.active_im.data() + kernel.source0_false[idx] * pitch;
        const double* r1p = runtime.active_re.data() + kernel.source1_false[idx] * pitch;
        const double* i1p = runtime.active_im.data() + kernel.source1_false[idx] * pitch;
        const double c1r = kernel.coeff1_false_real[idx];
        const double c1i = kernel.coeff1_false_imag[idx];
        for_each_masked_live_shot(runtime, live_bits, [&](int shot) {
            const double prod_r = c1r * r1p[shot] - c1i * i1p[shot];
            const double prod_i = c1r * i1p[shot] + c1i * r1p[shot];
            const double amp_r = kBatchInvSqrt2 * r0p[shot] - prod_r;
            const double amp_i = kBatchInvSqrt2 * i0p[shot] - prod_i;
            runtime.branch_prob_true[static_cast<std::size_t>(shot)] += amp_r * amp_r + amp_i * amp_i;
        });
    }
    sample_batch_measurement_branches_from_true_masked(
        runtime,
        runtime.eval_scratch,
        runtime.branch_prob_true,
        runtime.branch_invnorms,
        live_bits);
    const auto& branch_bits = runtime.eval_scratch;
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const double* r0p = runtime.active_re.data() + kernel.source0_false[idx] * pitch;
        const double* i0p = runtime.active_im.data() + kernel.source0_false[idx] * pitch;
        const double* r1p = runtime.active_re.data() + kernel.source1_false[idx] * pitch;
        const double* i1p = runtime.active_im.data() + kernel.source1_false[idx] * pitch;
        double* dst_r = runtime.scratch_re.data() + idx * pitch;
        double* dst_i = runtime.scratch_im.data() + idx * pitch;
        const double c1r = kernel.coeff1_false_real[idx];
        const double c1i = kernel.coeff1_false_imag[idx];
        for_each_masked_live_shot(runtime, live_bits, [&](int shot) {
            const double branch_sign = batch_vector_bit(branch_bits, shot) ? -1.0 : 1.0;
            const double prod_r = c1r * r1p[shot] - c1i * i1p[shot];
            const double prod_i = c1r * i1p[shot] + c1i * r1p[shot];
            const double n = runtime.branch_invnorms[static_cast<std::size_t>(shot)];
            dst_r[shot] = (kBatchInvSqrt2 * r0p[shot] + branch_sign * prod_r) * n;
            dst_i[shot] = (kBatchInvSqrt2 * i0p[shot] + branch_sign * prod_i) * n;
        });
    }
    const std::size_t active_prefix_size = out_dim * pitch;
    std::copy_n(runtime.scratch_re.data(), active_prefix_size, runtime.active_re.data());
    std::copy_n(runtime.scratch_im.data(), active_prefix_size, runtime.active_im.data());
    --runtime.k;
    ++runtime.ndormant;
    assign_batch_symbol_masked(runtime, branch_condition, branch_bits, live_bits);
    eval_symbolic_bool_batch(runtime.eval_scratch, outcome_plan, runtime);
    write_batch_measurement_record_masked(runtime, record, runtime.eval_scratch, record_condition, live_bits);
}

void measure_precomputed_active_pauli_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition,
    const SymbolicBoolEvaluationPlan& outcome_plan,
    std::optional<int> record,
    std::optional<int> record_condition) {
    if (runtime.k <= 0) {
        fail("cannot measure an active Pauli when k == 0");
    }
    if (kernel.action.nqubits != runtime.k) {
        fail("measurement kernel dimension does not match batch active state");
    }
    if (kernel.is_diagonal) {
        measure_diagonal_active_pauli_batch(runtime, kernel, branch_condition, outcome_plan, record, record_condition);
    } else {
        measure_nondiagonal_active_pauli_batch(runtime, kernel, branch_condition, outcome_plan, record, record_condition);
    }
}

void measure_precomputed_active_pauli_batch_masked(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition,
    const SymbolicBoolEvaluationPlan& outcome_plan,
    std::optional<int> record,
    std::optional<int> record_condition,
    const std::vector<std::uint64_t>& live_bits) {
    if (runtime.k <= 0) {
        fail("cannot measure an active Pauli when k == 0");
    }
    if (kernel.action.nqubits != runtime.k) {
        fail("measurement kernel dimension does not match batch active state");
    }
    if (kernel.is_diagonal) {
        measure_diagonal_active_pauli_batch_masked(
            runtime,
            kernel,
            branch_condition,
            outcome_plan,
            record,
            record_condition,
            live_bits);
    } else {
        measure_nondiagonal_active_pauli_batch_masked(
            runtime,
            kernel,
            branch_condition,
            outcome_plan,
            record,
            record_condition,
            live_bits);
    }
}


} // namespace symft
