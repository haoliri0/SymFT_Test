#include "batch_internal.hpp"

#include <utility>

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

template <typename RangeFunc>
bool run_parallel_outer(BatchFactoredExecutorState& runtime, std::size_t outer_items, RangeFunc&& func) {
    if (batch_parallel_workers(runtime.threads, outer_items, runtime.active_shots) <= 1) {
        return false;
    }
    batch_parallel_for_outer(runtime.threads, outer_items, runtime.active_shots, std::forward<RangeFunc>(func));
    return true;
}

template <typename AccumulateFunc>
bool accumulate_prob_true_parallel(
    BatchFactoredExecutorState& runtime,
    std::size_t outer_items,
    AccumulateFunc&& accumulate) {
    const int workers = batch_parallel_workers(runtime.threads, outer_items, runtime.active_shots);
    if (workers <= 1) {
        return false;
    }
    const std::size_t stride = static_cast<std::size_t>(runtime.batches);
    const std::size_t partial_size = static_cast<std::size_t>(workers) * stride;
    if (runtime.branch_prob_partials.size() < partial_size) {
        runtime.branch_prob_partials.resize(partial_size, 0.0);
    }
    for (int worker = 0; worker < workers; ++worker) {
        double* local = runtime.branch_prob_partials.data() + static_cast<std::size_t>(worker) * stride;
        std::fill(local, local + runtime.active_shots, 0.0);
    }
    batch_parallel_for_outer(
        runtime.threads,
        outer_items,
        runtime.active_shots,
        [&](std::size_t begin, std::size_t end, int worker) {
            double* local = runtime.branch_prob_partials.data() + static_cast<std::size_t>(worker) * stride;
            accumulate(begin, end, local);
        });
    std::fill(
        runtime.branch_prob_true.begin(),
        runtime.branch_prob_true.begin() + runtime.active_shots,
        0.0);
    for (int worker = 0; worker < workers; ++worker) {
        const double* local = runtime.branch_prob_partials.data() + static_cast<std::size_t>(worker) * stride;
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < runtime.active_shots; ++shot) {
            runtime.branch_prob_true[static_cast<std::size_t>(shot)] += local[shot];
        }
    }
    return true;
}

void rotate_uniform_imag_pairs_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits) {
    const auto& table = batch_simd::scalar_table();
    const std::size_t npairs = kernel.pair_left_indices.size();
    const std::size_t leading_shots = static_cast<std::size_t>(runtime.batches);
    const BatchSignMode mode = batch_sign_mode(runtime, sign_bits);
    if (mode != BatchSignMode::Mixed) {
        const double q = mode == BatchSignMode::AllPlus
                             ? kernel.pair_left_plus_coefficients.front().imag()
                             : kernel.pair_left_minus_coefficients.front().imag();
        if (run_parallel_outer(runtime, npairs, [&](std::size_t begin, std::size_t end, int) {
                table.rotate_uniform_imag_pairs_const(
                    runtime.active_re.data(),
                    runtime.active_im.data(),
                    leading_shots,
                    runtime.active_shots,
                    kernel.pair_left_indices.data() + begin,
                    kernel.pair_right_indices.data() + begin,
                    end - begin,
                    kernel.cos_theta,
                    q);
            })) {
            return;
        }
        table.rotate_uniform_imag_xmask_const(
            runtime.active_re.data(),
            runtime.active_im.data(),
            leading_shots,
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            npairs,
            kernel.cos_theta,
            q);
        return;
    }
    const auto& coeffs = fill_rotation_coefficients(
        runtime,
        sign_bits,
        kernel.pair_left_minus_coefficients.front().imag(),
        kernel.pair_left_plus_coefficients.front().imag());
    if (run_parallel_outer(runtime, npairs, [&](std::size_t begin, std::size_t end, int) {
            table.rotate_uniform_imag_pairs(
                runtime.active_re.data(),
                runtime.active_im.data(),
                leading_shots,
                runtime.active_shots,
                kernel.pair_left_indices.data() + begin,
                kernel.pair_right_indices.data() + begin,
                end - begin,
                kernel.cos_theta,
                coeffs.data());
        })) {
        return;
    }
    if (npairs >= kXmaskRotationPairThreshold) {
        table.rotate_uniform_imag_xmask(
            runtime.active_re.data(),
            runtime.active_im.data(),
            leading_shots,
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            npairs,
            kernel.cos_theta,
            coeffs.data());
        return;
    }
    table.rotate_uniform_imag_pairs(
        runtime.active_re.data(),
        runtime.active_im.data(),
        leading_shots,
        runtime.active_shots,
        kernel.pair_left_indices.data(),
        kernel.pair_right_indices.data(),
        npairs,
        kernel.cos_theta,
        coeffs.data());
}

void rotate_real_pair_flip_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits) {
    const auto& table = batch_simd::scalar_table();
    const std::size_t npairs = kernel.pair_left_indices.size();
    const std::size_t leading_shots = static_cast<std::size_t>(runtime.batches);
    const BatchSignMode mode = batch_sign_mode(runtime, sign_bits);
    if (mode != BatchSignMode::Mixed) {
        const double q = mode == BatchSignMode::AllPlus
                             ? kernel.pair_left_plus_coefficients.front().real()
                             : kernel.pair_left_minus_coefficients.front().real();
        if (run_parallel_outer(runtime, npairs, [&](std::size_t begin, std::size_t end, int) {
                table.rotate_real_pair_flip_pairs_const(
                    runtime.active_re.data(),
                    runtime.active_im.data(),
                    leading_shots,
                    runtime.active_shots,
                    kernel.pair_left_indices.data() + begin,
                    kernel.pair_right_indices.data() + begin,
                    kernel.real_pair_flip_basis_phase_signs.data() + begin,
                    end - begin,
                    kernel.cos_theta,
                    q);
            })) {
            return;
        }
        table.rotate_real_pair_flip_xmask_const(
            runtime.active_re.data(),
            runtime.active_im.data(),
            leading_shots,
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            kernel.real_pair_flip_basis_phase_signs.data(),
            npairs,
            kernel.cos_theta,
            q);
        return;
    }
    const auto& coeffs = fill_rotation_coefficients(
        runtime,
        sign_bits,
        kernel.pair_left_minus_coefficients.front().real(),
        kernel.pair_left_plus_coefficients.front().real());
    if (run_parallel_outer(runtime, npairs, [&](std::size_t begin, std::size_t end, int) {
            table.rotate_real_pair_flip(
                runtime.active_re.data(),
                runtime.active_im.data(),
                leading_shots,
                runtime.active_shots,
                kernel.pair_left_indices.data() + begin,
                kernel.pair_right_indices.data() + begin,
                kernel.real_pair_flip_basis_phase_signs.data() + begin,
                end - begin,
                kernel.cos_theta,
                coeffs.data());
        })) {
        return;
    }
    if (npairs >= kXmaskRotationPairThreshold) {
        table.rotate_real_pair_flip_xmask(
            runtime.active_re.data(),
            runtime.active_im.data(),
            leading_shots,
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            kernel.real_pair_flip_basis_phase_signs.data(),
            npairs,
            kernel.cos_theta,
            coeffs.data());
        return;
    }
    table.rotate_real_pair_flip(
        runtime.active_re.data(),
        runtime.active_im.data(),
        leading_shots,
        runtime.active_shots,
        kernel.pair_left_indices.data(),
        kernel.pair_right_indices.data(),
        kernel.real_pair_flip_basis_phase_signs.data(),
        npairs,
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
    const auto& table = batch_simd::scalar_table();
    const std::size_t leading_shots = static_cast<std::size_t>(runtime.batches);
    const double c = kernel.cos_theta;
    if (kernel.is_diagonal) {
        const std::size_t dim = active_length(runtime.k);
        const BatchSignMode mode = batch_sign_mode(runtime, sign_bits);
        if (mode == BatchSignMode::AllMinus) {
            if (run_parallel_outer(runtime, dim, [&](std::size_t begin, std::size_t end, int) {
                    table.rotate_diagonal_const(
                        runtime.active_re.data() + begin * leading_shots,
                        runtime.active_im.data() + begin * leading_shots,
                        leading_shots,
                        runtime.active_shots,
                        end - begin,
                        kernel.diagonal_minus_coefficients.data() + begin,
                        c);
                })) {
                return;
            }
            table.rotate_diagonal_const(
                runtime.active_re.data(),
                runtime.active_im.data(),
                leading_shots,
                runtime.active_shots,
                dim,
                kernel.diagonal_minus_coefficients.data(),
                c);
        } else if (mode == BatchSignMode::AllPlus) {
            if (run_parallel_outer(runtime, dim, [&](std::size_t begin, std::size_t end, int) {
                    table.rotate_diagonal_const(
                        runtime.active_re.data() + begin * leading_shots,
                        runtime.active_im.data() + begin * leading_shots,
                        leading_shots,
                        runtime.active_shots,
                        end - begin,
                        kernel.diagonal_plus_coefficients.data() + begin,
                        c);
                })) {
                return;
            }
            table.rotate_diagonal_const(
                runtime.active_re.data(),
                runtime.active_im.data(),
                leading_shots,
                runtime.active_shots,
                dim,
                kernel.diagonal_plus_coefficients.data(),
                c);
        } else {
            if (run_parallel_outer(runtime, dim, [&](std::size_t begin, std::size_t end, int) {
                    table.rotate_diagonal_mixed(
                        runtime.active_re.data() + begin * leading_shots,
                        runtime.active_im.data() + begin * leading_shots,
                        leading_shots,
                        runtime.active_shots,
                        end - begin,
                        kernel.diagonal_minus_coefficients.data() + begin,
                        kernel.diagonal_plus_coefficients.data() + begin,
                        sign_bits.data(),
                        c);
                })) {
                return;
            }
            table.rotate_diagonal_mixed(
                runtime.active_re.data(),
                runtime.active_im.data(),
                leading_shots,
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
    const std::size_t npairs = kernel.pair_left_indices.size();
    const BatchSignMode mode = batch_sign_mode(runtime, sign_bits);
    if (mode == BatchSignMode::AllMinus) {
        if (run_parallel_outer(runtime, npairs, [&](std::size_t begin, std::size_t end, int) {
                table.rotate_general_pairs_const(
                    runtime.active_re.data(),
                    runtime.active_im.data(),
                    leading_shots,
                    runtime.active_shots,
                    kernel.pair_left_indices.data() + begin,
                    kernel.pair_right_indices.data() + begin,
                    end - begin,
                    kernel.pair_left_minus_coefficients.data() + begin,
                    kernel.pair_right_minus_coefficients.data() + begin,
                    c);
            })) {
            return;
        }
        table.rotate_general_xmask_const(
            runtime.active_re.data(),
            runtime.active_im.data(),
            leading_shots,
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            npairs,
            kernel.pair_left_minus_coefficients.data(),
            kernel.pair_right_minus_coefficients.data(),
            c);
    } else if (mode == BatchSignMode::AllPlus) {
        if (run_parallel_outer(runtime, npairs, [&](std::size_t begin, std::size_t end, int) {
                table.rotate_general_pairs_const(
                    runtime.active_re.data(),
                    runtime.active_im.data(),
                    leading_shots,
                    runtime.active_shots,
                    kernel.pair_left_indices.data() + begin,
                    kernel.pair_right_indices.data() + begin,
                    end - begin,
                    kernel.pair_left_plus_coefficients.data() + begin,
                    kernel.pair_right_plus_coefficients.data() + begin,
                    c);
            })) {
            return;
        }
        table.rotate_general_xmask_const(
            runtime.active_re.data(),
            runtime.active_im.data(),
            leading_shots,
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            npairs,
            kernel.pair_left_plus_coefficients.data(),
            kernel.pair_right_plus_coefficients.data(),
            c);
    } else {
        if (run_parallel_outer(runtime, npairs, [&](std::size_t begin, std::size_t end, int) {
                table.rotate_general_pairs_mixed(
                    runtime.active_re.data(),
                    runtime.active_im.data(),
                    leading_shots,
                    runtime.active_shots,
                    kernel.pair_left_indices.data() + begin,
                    kernel.pair_right_indices.data() + begin,
                    end - begin,
                    kernel.pair_left_minus_coefficients.data() + begin,
                    kernel.pair_right_minus_coefficients.data() + begin,
                    kernel.pair_left_plus_coefficients.data() + begin,
                    kernel.pair_right_plus_coefficients.data() + begin,
                    sign_bits.data(),
                    c);
            })) {
            return;
        }
        table.rotate_general_xmask_mixed(
            runtime.active_re.data(),
            runtime.active_im.data(),
            leading_shots,
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            npairs,
            kernel.pair_left_minus_coefficients.data(),
            kernel.pair_right_minus_coefficients.data(),
            kernel.pair_left_plus_coefficients.data(),
            kernel.pair_right_plus_coefficients.data(),
            sign_bits.data(),
            c);
    }
}

void apply_active_basis_change_batch(BatchFactoredExecutorState& runtime, char kind, int q) {
    if (q < 0 || q >= runtime.k) {
        fail("active basis-change qubit is out of range");
    }
    const std::size_t dim = active_length(runtime.k);
    const std::size_t mask = std::size_t{1} << q;
    if (kind == 'H') {
        const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
        batch_parallel_for_outer(runtime.threads, dim, runtime.active_shots, [&](std::size_t begin, std::size_t end, int) {
            for (std::size_t base = begin; base < end; ++base) {
                if ((base & mask) != 0) {
                    continue;
                }
                const std::size_t paired = base | mask;
                SYMFT_BATCH_SIMD_LOOP
                for (int shot = 0; shot < runtime.active_shots; ++shot) {
                    const std::size_t off0 = batch_active_offset(runtime, base, shot);
                    const std::size_t off1 = batch_active_offset(runtime, paired, shot);
                    const double r0 = runtime.active_re[off0];
                    const double i0 = runtime.active_im[off0];
                    const double r1 = runtime.active_re[off1];
                    const double i1 = runtime.active_im[off1];
                    runtime.active_re[off0] = (r0 + r1) * inv_sqrt2;
                    runtime.active_im[off0] = (i0 + i1) * inv_sqrt2;
                    runtime.active_re[off1] = (r0 - r1) * inv_sqrt2;
                    runtime.active_im[off1] = (i0 - i1) * inv_sqrt2;
                }
            }
        });
    } else if (kind == 'S') {
        batch_parallel_for_outer(runtime.threads, dim, runtime.active_shots, [&](std::size_t begin, std::size_t end, int) {
            for (std::size_t basis = begin; basis < end; ++basis) {
                if ((basis & mask) == 0) {
                    continue;
                }
                SYMFT_BATCH_SIMD_LOOP
                for (int shot = 0; shot < runtime.active_shots; ++shot) {
                    const std::size_t off = batch_active_offset(runtime, basis, shot);
                    const double r = runtime.active_re[off];
                    const double i = runtime.active_im[off];
                    runtime.active_re[off] = -i;
                    runtime.active_im[off] = r;
                }
            }
        });
    } else {
        fail("unsupported active basis change");
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
    const auto& table = batch_simd::scalar_table();
    const std::size_t leading_shots = static_cast<std::size_t>(runtime.batches);
    if (run_parallel_outer(runtime, dim, [&](std::size_t begin, std::size_t end, int) {
            table.promote_first_dormant_rotation_range(
                runtime.active_re.data(),
                runtime.active_im.data(),
                leading_shots,
                runtime.active_shots,
                dim,
                begin,
                end,
                c,
                coeffs.data());
        })) {
        ++runtime.k;
        --runtime.ndormant;
        return;
    }
    table.promote_first_dormant_rotation(
        runtime.active_re.data(),
        runtime.active_im.data(),
        leading_shots,
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
    const auto& table = batch_simd::scalar_table();
    const std::size_t leading_shots = static_cast<std::size_t>(runtime.batches);
    if (!accumulate_prob_true_parallel(
            runtime,
            dim,
            [&](std::size_t begin, std::size_t end, double* prob_true) {
                table.last_z_measure_true_prob_range(
                    runtime.active_re.data(),
                    runtime.active_im.data(),
                    leading_shots,
                    runtime.active_shots,
                    dim,
                    begin,
                    end,
                    prob_true);
            })) {
        table.last_z_measure_true_prob(
            runtime.active_re.data(),
            runtime.active_im.data(),
            leading_shots,
            runtime.active_shots,
            dim,
            runtime.branch_prob_true.data());
    }
    sample_batch_measurement_branches_from_true(
        runtime,
        runtime.eval_scratch,
        runtime.branch_prob_true,
        runtime.branch_invnorms);
    const auto& branch_bits = runtime.eval_scratch;
    if (run_parallel_outer(runtime, dim, [&](std::size_t begin, std::size_t end, int) {
            table.last_z_project_range(
                runtime.active_re.data(),
                runtime.active_im.data(),
                leading_shots,
                runtime.active_shots,
                dim,
                begin,
                end,
                branch_bits.data(),
                runtime.branch_invnorms.data());
        })) {
        runtime.k = new_k;
        ++runtime.ndormant;
        assign_batch_symbol(runtime, branch_condition, branch_bits);
        eval_symbolic_bool_batch(runtime.eval_scratch, outcome_plan, runtime);
        write_batch_measurement_record(runtime, record, runtime.eval_scratch, record_condition);
        return;
    }
    table.last_z_project(
        runtime.active_re.data(),
        runtime.active_im.data(),
        leading_shots,
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
    const auto& table = batch_simd::scalar_table();
    const std::size_t leading_shots = static_cast<std::size_t>(runtime.batches);
    if (!accumulate_prob_true_parallel(
            runtime,
            out_dim,
            [&](std::size_t begin, std::size_t end, double* prob_true) {
                table.diagonal_measure_true_prob_range(
                    runtime.active_re.data(),
                    runtime.active_im.data(),
                    leading_shots,
                    runtime.active_shots,
                    source_true.data(),
                    begin,
                    end,
                    prob_true);
            })) {
        table.diagonal_measure_true_prob(
            runtime.active_re.data(),
            runtime.active_im.data(),
            leading_shots,
            runtime.active_shots,
            source_true.data(),
            out_dim,
            runtime.branch_prob_true.data());
    }
    sample_batch_measurement_branches_from_true(
        runtime,
        runtime.eval_scratch,
        runtime.branch_prob_true,
        runtime.branch_invnorms);
    const auto& branch_bits = runtime.eval_scratch;
    table.diagonal_project(
        runtime.active_re.data(),
        runtime.active_im.data(),
        leading_shots,
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

void measure_nondiagonal_active_pauli_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition,
    const SymbolicBoolEvaluationPlan& outcome_plan,
    std::optional<int> record,
    std::optional<int> record_condition) {
    const std::size_t out_dim = kernel.source0_false.size();
    const auto& table = batch_simd::scalar_table();
    const std::size_t leading_shots = static_cast<std::size_t>(runtime.batches);
    if (!accumulate_prob_true_parallel(
            runtime,
            out_dim,
            [&](std::size_t begin, std::size_t end, double* prob_true) {
                table.nondiagonal_measure_true_prob_range(
                    runtime.active_re.data(),
                    runtime.active_im.data(),
                    leading_shots,
                    runtime.active_shots,
                    kernel.source0_false.data(),
                    kernel.source1_false.data(),
                    kernel.coeff1_false_real.data(),
                    kernel.coeff1_false_imag.data(),
                    begin,
                    end,
                    prob_true);
            })) {
        table.nondiagonal_measure_true_prob(
            runtime.active_re.data(),
            runtime.active_im.data(),
            leading_shots,
            runtime.active_shots,
            kernel.source0_false.data(),
            kernel.source1_false.data(),
            kernel.coeff1_false_real.data(),
            kernel.coeff1_false_imag.data(),
            out_dim,
            runtime.branch_prob_true.data());
    }
    sample_batch_measurement_branches_from_true(
        runtime,
        runtime.eval_scratch,
        runtime.branch_prob_true,
        runtime.branch_invnorms);
    const auto& branch_bits = runtime.eval_scratch;
    const bool projected_parallel = run_parallel_outer(runtime, out_dim, [&](std::size_t begin, std::size_t end, int) {
        table.nondiagonal_project_range(
            runtime.active_re.data(),
            runtime.active_im.data(),
            runtime.scratch_re.data(),
            runtime.scratch_im.data(),
            leading_shots,
            runtime.active_shots,
            kernel.source0_false.data(),
            kernel.source1_false.data(),
            kernel.coeff1_false_real.data(),
            kernel.coeff1_false_imag.data(),
            begin,
            end,
            branch_bits.data(),
            runtime.branch_invnorms.data());
    });
    if (!projected_parallel) {
        table.nondiagonal_project(
            runtime.active_re.data(),
            runtime.active_im.data(),
            runtime.scratch_re.data(),
            runtime.scratch_im.data(),
            leading_shots,
            runtime.active_shots,
            kernel.source0_false.data(),
            kernel.source1_false.data(),
            kernel.coeff1_false_real.data(),
            kernel.coeff1_false_imag.data(),
            out_dim,
            branch_bits.data(),
            runtime.branch_invnorms.data());
    }
    const std::size_t active_prefix_size = out_dim * static_cast<std::size_t>(runtime.batches);
    std::copy_n(runtime.scratch_re.data(), active_prefix_size, runtime.active_re.data());
    std::copy_n(runtime.scratch_im.data(), active_prefix_size, runtime.active_im.data());
    --runtime.k;
    ++runtime.ndormant;
    assign_batch_symbol(runtime, branch_condition, branch_bits);
    eval_symbolic_bool_batch(runtime.eval_scratch, outcome_plan, runtime);
    write_batch_measurement_record(runtime, record, runtime.eval_scratch, record_condition);
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


} // namespace symft
