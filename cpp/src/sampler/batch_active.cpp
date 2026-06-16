#include "batch_internal.hpp"

namespace symft {

namespace {

inline constexpr std::size_t kBatchThreadPairGrain = std::size_t{1} << 18;
inline constexpr std::size_t kBatchThreadBasisGrain = std::size_t{1} << 18;

bool packed_batch_bit(const std::uint64_t* bits, int shot) {
    return (bits[static_cast<std::size_t>(shot >> 6)] & (std::uint64_t{1} << (shot & 63))) != 0;
}

bool has_parallel_work(
    const BatchFactoredExecutorState& runtime,
    std::size_t items,
    std::size_t min_items_per_worker) {
    return batch_parallel_worker_count(runtime, items, min_items_per_worker) > 1;
}

void rotate_diagonal_const_threaded(
    BatchFactoredExecutorState& runtime,
    const Complex* coeff,
    std::size_t dim,
    double c) {
    const std::size_t leading = static_cast<std::size_t>(runtime.batches);
    batch_parallel_for(runtime, dim, kBatchThreadBasisGrain, [&](int, std::size_t first, std::size_t last) {
        for (std::size_t basis = first; basis < last; ++basis) {
            const double fr = c + coeff[basis].real();
            const double fi = coeff[basis].imag();
            double* rp = runtime.active_re.data() + basis * leading;
            double* ip = runtime.active_im.data() + basis * leading;
            SYMFT_BATCH_SIMD_LOOP
            for (int shot = 0; shot < runtime.active_shots; ++shot) {
                const double r = rp[shot];
                const double v = ip[shot];
                rp[shot] = fr * r - fi * v;
                ip[shot] = fr * v + fi * r;
            }
        }
    });
}

void rotate_diagonal_mixed_threaded(
    BatchFactoredExecutorState& runtime,
    const Complex* minus_coeff,
    const Complex* plus_coeff,
    const std::uint64_t* sign_bits,
    std::size_t dim,
    double c) {
    const std::size_t leading = static_cast<std::size_t>(runtime.batches);
    batch_parallel_for(runtime, dim, kBatchThreadBasisGrain, [&](int, std::size_t first, std::size_t last) {
        for (std::size_t basis = first; basis < last; ++basis) {
            const double minus_fr = c + minus_coeff[basis].real();
            const double minus_fi = minus_coeff[basis].imag();
            const double plus_fr = c + plus_coeff[basis].real();
            const double plus_fi = plus_coeff[basis].imag();
            double* rp = runtime.active_re.data() + basis * leading;
            double* ip = runtime.active_im.data() + basis * leading;
            SYMFT_BATCH_SIMD_LOOP
            for (int shot = 0; shot < runtime.active_shots; ++shot) {
                const bool sign = packed_batch_bit(sign_bits, shot);
                const double fr = sign ? plus_fr : minus_fr;
                const double fi = sign ? plus_fi : minus_fi;
                const double r = rp[shot];
                const double v = ip[shot];
                rp[shot] = fr * r - fi * v;
                ip[shot] = fr * v + fi * r;
            }
        }
    });
}

void rotate_uniform_imag_pairs_const_threaded(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    double q) {
    const std::size_t leading = static_cast<std::size_t>(runtime.batches);
    const auto* left = kernel.pair_left_indices.data();
    const auto* right = kernel.pair_right_indices.data();
    batch_parallel_for(
        runtime,
        kernel.pair_left_indices.size(),
        kBatchThreadPairGrain,
        [&](int, std::size_t first, std::size_t last) {
            for (std::size_t idx = first; idx < last; ++idx) {
                double* r0p = runtime.active_re.data() + left[idx] * leading;
                double* i0p = runtime.active_im.data() + left[idx] * leading;
                double* r1p = runtime.active_re.data() + right[idx] * leading;
                double* i1p = runtime.active_im.data() + right[idx] * leading;
                SYMFT_BATCH_SIMD_LOOP
                for (int shot = 0; shot < runtime.active_shots; ++shot) {
                    const double r0 = r0p[shot];
                    const double i0 = i0p[shot];
                    const double r1 = r1p[shot];
                    const double i1 = i1p[shot];
                    r0p[shot] = kernel.cos_theta * r0 - q * i1;
                    i0p[shot] = kernel.cos_theta * i0 + q * r1;
                    r1p[shot] = kernel.cos_theta * r1 - q * i0;
                    i1p[shot] = kernel.cos_theta * i1 + q * r0;
                }
            }
        });
}

void rotate_uniform_imag_pairs_mixed_threaded(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const double* q_by_shot) {
    const std::size_t leading = static_cast<std::size_t>(runtime.batches);
    const auto* left = kernel.pair_left_indices.data();
    const auto* right = kernel.pair_right_indices.data();
    batch_parallel_for(
        runtime,
        kernel.pair_left_indices.size(),
        kBatchThreadPairGrain,
        [&](int, std::size_t first, std::size_t last) {
            for (std::size_t idx = first; idx < last; ++idx) {
                double* r0p = runtime.active_re.data() + left[idx] * leading;
                double* i0p = runtime.active_im.data() + left[idx] * leading;
                double* r1p = runtime.active_re.data() + right[idx] * leading;
                double* i1p = runtime.active_im.data() + right[idx] * leading;
                SYMFT_BATCH_SIMD_LOOP
                for (int shot = 0; shot < runtime.active_shots; ++shot) {
                    const double q = q_by_shot[static_cast<std::size_t>(shot)];
                    const double r0 = r0p[shot];
                    const double i0 = i0p[shot];
                    const double r1 = r1p[shot];
                    const double i1 = i1p[shot];
                    r0p[shot] = kernel.cos_theta * r0 - q * i1;
                    i0p[shot] = kernel.cos_theta * i0 + q * r1;
                    r1p[shot] = kernel.cos_theta * r1 - q * i0;
                    i1p[shot] = kernel.cos_theta * i1 + q * r0;
                }
            }
        });
}

void rotate_real_pair_flip_const_threaded(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    double q) {
    const std::size_t leading = static_cast<std::size_t>(runtime.batches);
    const auto* left = kernel.pair_left_indices.data();
    const auto* right = kernel.pair_right_indices.data();
    const auto* phase = kernel.real_pair_flip_basis_phase_signs.data();
    batch_parallel_for(
        runtime,
        kernel.pair_left_indices.size(),
        kBatchThreadPairGrain,
        [&](int, std::size_t first, std::size_t last) {
            for (std::size_t idx = first; idx < last; ++idx) {
                const double signed_q = phase[idx] * q;
                double* r0p = runtime.active_re.data() + left[idx] * leading;
                double* i0p = runtime.active_im.data() + left[idx] * leading;
                double* r1p = runtime.active_re.data() + right[idx] * leading;
                double* i1p = runtime.active_im.data() + right[idx] * leading;
                SYMFT_BATCH_SIMD_LOOP
                for (int shot = 0; shot < runtime.active_shots; ++shot) {
                    const double r0 = r0p[shot];
                    const double i0 = i0p[shot];
                    const double r1 = r1p[shot];
                    const double i1 = i1p[shot];
                    r0p[shot] = kernel.cos_theta * r0 - signed_q * r1;
                    i0p[shot] = kernel.cos_theta * i0 - signed_q * i1;
                    r1p[shot] = kernel.cos_theta * r1 + signed_q * r0;
                    i1p[shot] = kernel.cos_theta * i1 + signed_q * i0;
                }
            }
        });
}

void rotate_real_pair_flip_mixed_threaded(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const double* q_by_shot) {
    const std::size_t leading = static_cast<std::size_t>(runtime.batches);
    const auto* left = kernel.pair_left_indices.data();
    const auto* right = kernel.pair_right_indices.data();
    const auto* phase = kernel.real_pair_flip_basis_phase_signs.data();
    batch_parallel_for(
        runtime,
        kernel.pair_left_indices.size(),
        kBatchThreadPairGrain,
        [&](int, std::size_t first, std::size_t last) {
            for (std::size_t idx = first; idx < last; ++idx) {
                const double basis_phase_sign = phase[idx];
                double* r0p = runtime.active_re.data() + left[idx] * leading;
                double* i0p = runtime.active_im.data() + left[idx] * leading;
                double* r1p = runtime.active_re.data() + right[idx] * leading;
                double* i1p = runtime.active_im.data() + right[idx] * leading;
                SYMFT_BATCH_SIMD_LOOP
                for (int shot = 0; shot < runtime.active_shots; ++shot) {
                    const double q = basis_phase_sign * q_by_shot[static_cast<std::size_t>(shot)];
                    const double r0 = r0p[shot];
                    const double i0 = i0p[shot];
                    const double r1 = r1p[shot];
                    const double i1 = i1p[shot];
                    r0p[shot] = kernel.cos_theta * r0 - q * r1;
                    i0p[shot] = kernel.cos_theta * i0 - q * i1;
                    r1p[shot] = kernel.cos_theta * r1 + q * r0;
                    i1p[shot] = kernel.cos_theta * i1 + q * i0;
                }
            }
        });
}

void rotate_general_pairs_const_threaded(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const Complex* left_coeff,
    const Complex* right_coeff) {
    const std::size_t leading = static_cast<std::size_t>(runtime.batches);
    const auto* left = kernel.pair_left_indices.data();
    const auto* right = kernel.pair_right_indices.data();
    batch_parallel_for(
        runtime,
        kernel.pair_left_indices.size(),
        kBatchThreadPairGrain,
        [&](int, std::size_t first, std::size_t last) {
            for (std::size_t idx = first; idx < last; ++idx) {
                const double lr = left_coeff[idx].real();
                const double li = left_coeff[idx].imag();
                const double rr = right_coeff[idx].real();
                const double ri = right_coeff[idx].imag();
                double* r0p = runtime.active_re.data() + left[idx] * leading;
                double* i0p = runtime.active_im.data() + left[idx] * leading;
                double* r1p = runtime.active_re.data() + right[idx] * leading;
                double* i1p = runtime.active_im.data() + right[idx] * leading;
                SYMFT_BATCH_SIMD_LOOP
                for (int shot = 0; shot < runtime.active_shots; ++shot) {
                    const double r0 = r0p[shot];
                    const double im0 = i0p[shot];
                    const double r1 = r1p[shot];
                    const double im1 = i1p[shot];
                    r0p[shot] = kernel.cos_theta * r0 + rr * r1 - ri * im1;
                    i0p[shot] = kernel.cos_theta * im0 + rr * im1 + ri * r1;
                    r1p[shot] = kernel.cos_theta * r1 + lr * r0 - li * im0;
                    i1p[shot] = kernel.cos_theta * im1 + lr * im0 + li * r0;
                }
            }
        });
}

void rotate_general_pairs_mixed_threaded(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::uint64_t* sign_bits) {
    const std::size_t leading = static_cast<std::size_t>(runtime.batches);
    const auto* left = kernel.pair_left_indices.data();
    const auto* right = kernel.pair_right_indices.data();
    batch_parallel_for(
        runtime,
        kernel.pair_left_indices.size(),
        kBatchThreadPairGrain,
        [&](int, std::size_t first, std::size_t last) {
            for (std::size_t idx = first; idx < last; ++idx) {
                const double lmr = kernel.pair_left_minus_coefficients[idx].real();
                const double lmi = kernel.pair_left_minus_coefficients[idx].imag();
                const double rmr = kernel.pair_right_minus_coefficients[idx].real();
                const double rmi = kernel.pair_right_minus_coefficients[idx].imag();
                const double lpr = kernel.pair_left_plus_coefficients[idx].real();
                const double lpi = kernel.pair_left_plus_coefficients[idx].imag();
                const double rpr = kernel.pair_right_plus_coefficients[idx].real();
                const double rpi = kernel.pair_right_plus_coefficients[idx].imag();
                double* r0p = runtime.active_re.data() + left[idx] * leading;
                double* i0p = runtime.active_im.data() + left[idx] * leading;
                double* r1p = runtime.active_re.data() + right[idx] * leading;
                double* i1p = runtime.active_im.data() + right[idx] * leading;
                SYMFT_BATCH_SIMD_LOOP
                for (int shot = 0; shot < runtime.active_shots; ++shot) {
                    const bool sign = packed_batch_bit(sign_bits, shot);
                    const double lr = sign ? lpr : lmr;
                    const double li = sign ? lpi : lmi;
                    const double rr = sign ? rpr : rmr;
                    const double ri = sign ? rpi : rmi;
                    const double r0 = r0p[shot];
                    const double im0 = i0p[shot];
                    const double r1 = r1p[shot];
                    const double im1 = i1p[shot];
                    r0p[shot] = kernel.cos_theta * r0 + rr * r1 - ri * im1;
                    i0p[shot] = kernel.cos_theta * im0 + rr * im1 + ri * r1;
                    r1p[shot] = kernel.cos_theta * r1 + lr * r0 - li * im0;
                    i1p[shot] = kernel.cos_theta * im1 + lr * im0 + li * r0;
                }
            }
        });
}

void promote_first_dormant_rotation_threaded(
    BatchFactoredExecutorState& runtime,
    std::size_t dim,
    double c,
    const double* q_by_shot) {
    const std::size_t leading = static_cast<std::size_t>(runtime.batches);
    batch_parallel_for(runtime, dim, kBatchThreadBasisGrain, [&](int, std::size_t first, std::size_t last) {
        for (std::size_t basis = first; basis < last; ++basis) {
            double* r0p = runtime.active_re.data() + basis * leading;
            double* i0p = runtime.active_im.data() + basis * leading;
            double* r1p = runtime.active_re.data() + (dim + basis) * leading;
            double* i1p = runtime.active_im.data() + (dim + basis) * leading;
            SYMFT_BATCH_SIMD_LOOP
            for (int shot = 0; shot < runtime.active_shots; ++shot) {
                const double q = q_by_shot[static_cast<std::size_t>(shot)];
                const double r = r0p[shot];
                const double i = i0p[shot];
                r0p[shot] = c * r;
                i0p[shot] = c * i;
                r1p[shot] = -q * i;
                i1p[shot] = q * r;
            }
        }
    });
}

void accumulate_true_prob_threads(
    BatchFactoredExecutorState& runtime,
    std::size_t items,
    auto&& accumulate_range) {
    const int workers = batch_parallel_worker_count(runtime, items, kBatchThreadBasisGrain);
    std::fill(runtime.branch_prob_true.begin(), runtime.branch_prob_true.begin() + runtime.active_shots, 0.0);
    if (workers <= 1) {
        accumulate_range(runtime.branch_prob_true.data(), std::size_t{0}, items);
        return;
    }
    std::vector<double> partials(static_cast<std::size_t>(workers) * static_cast<std::size_t>(runtime.active_shots), 0.0);
    batch_parallel_for_workers(runtime, workers, items, [&](int worker, std::size_t first, std::size_t last) {
        accumulate_range(partials.data() + static_cast<std::size_t>(worker) * static_cast<std::size_t>(runtime.active_shots), first, last);
    });
    for (int worker = 0; worker < workers; ++worker) {
        const double* local = partials.data() + static_cast<std::size_t>(worker) * static_cast<std::size_t>(runtime.active_shots);
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < runtime.active_shots; ++shot) {
            runtime.branch_prob_true[static_cast<std::size_t>(shot)] += local[shot];
        }
    }
}

void last_z_measure_true_prob_threaded(BatchFactoredExecutorState& runtime, std::size_t dim) {
    const std::size_t leading = static_cast<std::size_t>(runtime.batches);
    accumulate_true_prob_threads(runtime, dim, [&](double* local, std::size_t first, std::size_t last) {
        for (std::size_t basis = first; basis < last; ++basis) {
            const double* rp = runtime.active_re.data() + (dim + basis) * leading;
            const double* ip = runtime.active_im.data() + (dim + basis) * leading;
            SYMFT_BATCH_SIMD_LOOP
            for (int shot = 0; shot < runtime.active_shots; ++shot) {
                const double r = rp[shot];
                const double i = ip[shot];
                local[shot] += r * r + i * i;
            }
        }
    });
}

void last_z_project_threaded(
    BatchFactoredExecutorState& runtime,
    std::size_t dim,
    const std::uint64_t* branch_bits,
    const double* invnorms) {
    const std::size_t leading = static_cast<std::size_t>(runtime.batches);
    batch_parallel_for(runtime, dim, kBatchThreadBasisGrain, [&](int, std::size_t first, std::size_t last) {
        for (std::size_t basis = first; basis < last; ++basis) {
            double* dst_r = runtime.active_re.data() + basis * leading;
            double* dst_i = runtime.active_im.data() + basis * leading;
            const double* false_r = runtime.active_re.data() + basis * leading;
            const double* false_i = runtime.active_im.data() + basis * leading;
            const double* true_r = runtime.active_re.data() + (dim + basis) * leading;
            const double* true_i = runtime.active_im.data() + (dim + basis) * leading;
            SYMFT_BATCH_SIMD_LOOP
            for (int shot = 0; shot < runtime.active_shots; ++shot) {
                const bool branch = packed_batch_bit(branch_bits, shot);
                const double n = invnorms[shot];
                dst_r[shot] = (branch ? true_r[shot] : false_r[shot]) * n;
                dst_i[shot] = (branch ? true_i[shot] : false_i[shot]) * n;
            }
        }
    });
}

void diagonal_measure_true_prob_threaded(
    BatchFactoredExecutorState& runtime,
    const std::size_t* source_true,
    std::size_t out_dim) {
    const std::size_t leading = static_cast<std::size_t>(runtime.batches);
    accumulate_true_prob_threads(runtime, out_dim, [&](double* local, std::size_t first, std::size_t last) {
        for (std::size_t idx = first; idx < last; ++idx) {
            const double* rp = runtime.active_re.data() + source_true[idx] * leading;
            const double* ip = runtime.active_im.data() + source_true[idx] * leading;
            SYMFT_BATCH_SIMD_LOOP
            for (int shot = 0; shot < runtime.active_shots; ++shot) {
                const double r = rp[shot];
                const double i = ip[shot];
                local[shot] += r * r + i * i;
            }
        }
    });
}

void diagonal_project_threaded(
    BatchFactoredExecutorState& runtime,
    const std::size_t* source_false,
    const std::size_t* source_true,
    std::size_t out_dim,
    const std::uint64_t* branch_bits,
    const double* invnorms) {
    const std::size_t leading = static_cast<std::size_t>(runtime.batches);
    batch_parallel_for(runtime, out_dim, kBatchThreadBasisGrain, [&](int, std::size_t first, std::size_t last) {
        for (std::size_t idx = first; idx < last; ++idx) {
            double* dst_r = runtime.scratch_re.data() + idx * leading;
            double* dst_i = runtime.scratch_im.data() + idx * leading;
            const double* false_r = runtime.active_re.data() + source_false[idx] * leading;
            const double* false_i = runtime.active_im.data() + source_false[idx] * leading;
            const double* true_r = runtime.active_re.data() + source_true[idx] * leading;
            const double* true_i = runtime.active_im.data() + source_true[idx] * leading;
            SYMFT_BATCH_SIMD_LOOP
            for (int shot = 0; shot < runtime.active_shots; ++shot) {
                const bool branch = packed_batch_bit(branch_bits, shot);
                const double n = invnorms[shot];
                dst_r[shot] = (branch ? true_r[shot] : false_r[shot]) * n;
                dst_i[shot] = (branch ? true_i[shot] : false_i[shot]) * n;
            }
        }
    });
    const std::size_t active_prefix_size = out_dim * leading;
    std::copy_n(runtime.scratch_re.data(), active_prefix_size, runtime.active_re.data());
    std::copy_n(runtime.scratch_im.data(), active_prefix_size, runtime.active_im.data());
}

void nondiagonal_measure_true_prob_threaded(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    std::size_t out_dim) {
    const std::size_t leading = static_cast<std::size_t>(runtime.batches);
    constexpr double inv_sqrt2 = 0.70710678118654752440;
    accumulate_true_prob_threads(runtime, out_dim, [&](double* local, std::size_t first, std::size_t last) {
        for (std::size_t idx = first; idx < last; ++idx) {
            const double* r0p = runtime.active_re.data() + kernel.source0_false[idx] * leading;
            const double* i0p = runtime.active_im.data() + kernel.source0_false[idx] * leading;
            const double* r1p = runtime.active_re.data() + kernel.source1_false[idx] * leading;
            const double* i1p = runtime.active_im.data() + kernel.source1_false[idx] * leading;
            const double c1r = kernel.coeff1_false_real[idx];
            const double c1i = kernel.coeff1_false_imag[idx];
            SYMFT_BATCH_SIMD_LOOP
            for (int shot = 0; shot < runtime.active_shots; ++shot) {
                const double prod_r = c1r * r1p[shot] - c1i * i1p[shot];
                const double prod_i = c1r * i1p[shot] + c1i * r1p[shot];
                const double amp_r = inv_sqrt2 * r0p[shot] - prod_r;
                const double amp_i = inv_sqrt2 * i0p[shot] - prod_i;
                local[shot] += amp_r * amp_r + amp_i * amp_i;
            }
        }
    });
}

void nondiagonal_project_threaded(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    std::size_t out_dim,
    const std::uint64_t* branch_bits,
    const double* invnorms) {
    const std::size_t leading = static_cast<std::size_t>(runtime.batches);
    constexpr double inv_sqrt2 = 0.70710678118654752440;
    batch_parallel_for(runtime, out_dim, kBatchThreadBasisGrain, [&](int, std::size_t first, std::size_t last) {
        for (std::size_t idx = first; idx < last; ++idx) {
            const double* r0p = runtime.active_re.data() + kernel.source0_false[idx] * leading;
            const double* i0p = runtime.active_im.data() + kernel.source0_false[idx] * leading;
            const double* r1p = runtime.active_re.data() + kernel.source1_false[idx] * leading;
            const double* i1p = runtime.active_im.data() + kernel.source1_false[idx] * leading;
            double* dst_r = runtime.scratch_re.data() + idx * leading;
            double* dst_i = runtime.scratch_im.data() + idx * leading;
            const double c1r = kernel.coeff1_false_real[idx];
            const double c1i = kernel.coeff1_false_imag[idx];
            SYMFT_BATCH_SIMD_LOOP
            for (int shot = 0; shot < runtime.active_shots; ++shot) {
                const double branch_sign = packed_batch_bit(branch_bits, shot) ? -1.0 : 1.0;
                const double prod_r = c1r * r1p[shot] - c1i * i1p[shot];
                const double prod_i = c1r * i1p[shot] + c1i * r1p[shot];
                const double n = invnorms[shot];
                dst_r[shot] = (inv_sqrt2 * r0p[shot] + branch_sign * prod_r) * n;
                dst_i[shot] = (inv_sqrt2 * i0p[shot] + branch_sign * prod_i) * n;
            }
        }
    });
}

} // namespace

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

void rotate_uniform_imag_pairs_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits) {
    const BatchSignMode mode = batch_sign_mode(runtime, sign_bits);
    if (mode != BatchSignMode::Mixed) {
        const double q = mode == BatchSignMode::AllPlus
                             ? kernel.pair_left_plus_coefficients.front().imag()
                             : kernel.pair_left_minus_coefficients.front().imag();
        if (has_parallel_work(runtime, kernel.pair_left_indices.size(), kBatchThreadPairGrain)) {
            rotate_uniform_imag_pairs_const_threaded(runtime, kernel, q);
            return;
        }
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
    if (has_parallel_work(runtime, kernel.pair_left_indices.size(), kBatchThreadPairGrain)) {
        rotate_uniform_imag_pairs_mixed_threaded(runtime, kernel, coeffs.data());
        return;
    }
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
        if (has_parallel_work(runtime, kernel.pair_left_indices.size(), kBatchThreadPairGrain)) {
            rotate_real_pair_flip_const_threaded(runtime, kernel, q);
            return;
        }
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
    if (has_parallel_work(runtime, kernel.pair_left_indices.size(), kBatchThreadPairGrain)) {
        rotate_real_pair_flip_mixed_threaded(runtime, kernel, coeffs.data());
        return;
    }
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
            if (has_parallel_work(runtime, dim, kBatchThreadBasisGrain)) {
                rotate_diagonal_const_threaded(runtime, kernel.diagonal_minus_coefficients.data(), dim, c);
                return;
            }
            batch_simd::scalar_table().rotate_diagonal_const(
                runtime.active_re.data(),
                runtime.active_im.data(),
                static_cast<std::size_t>(runtime.batches),
                runtime.active_shots,
                dim,
                kernel.diagonal_minus_coefficients.data(),
                c);
        } else if (mode == BatchSignMode::AllPlus) {
            if (has_parallel_work(runtime, dim, kBatchThreadBasisGrain)) {
                rotate_diagonal_const_threaded(runtime, kernel.diagonal_plus_coefficients.data(), dim, c);
                return;
            }
            batch_simd::scalar_table().rotate_diagonal_const(
                runtime.active_re.data(),
                runtime.active_im.data(),
                static_cast<std::size_t>(runtime.batches),
                runtime.active_shots,
                dim,
                kernel.diagonal_plus_coefficients.data(),
                c);
        } else {
            if (has_parallel_work(runtime, dim, kBatchThreadBasisGrain)) {
                rotate_diagonal_mixed_threaded(
                    runtime,
                    kernel.diagonal_minus_coefficients.data(),
                    kernel.diagonal_plus_coefficients.data(),
                    sign_bits.data(),
                    dim,
                    c);
                return;
            }
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
        if (has_parallel_work(runtime, kernel.pair_left_indices.size(), kBatchThreadPairGrain)) {
            rotate_general_pairs_const_threaded(
                runtime,
                kernel,
                kernel.pair_left_minus_coefficients.data(),
                kernel.pair_right_minus_coefficients.data());
            return;
        }
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
        if (has_parallel_work(runtime, kernel.pair_left_indices.size(), kBatchThreadPairGrain)) {
            rotate_general_pairs_const_threaded(
                runtime,
                kernel,
                kernel.pair_left_plus_coefficients.data(),
                kernel.pair_right_plus_coefficients.data());
            return;
        }
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
        if (has_parallel_work(runtime, kernel.pair_left_indices.size(), kBatchThreadPairGrain)) {
            rotate_general_pairs_mixed_threaded(runtime, kernel, sign_bits.data());
            return;
        }
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

void apply_active_basis_change_batch(BatchFactoredExecutorState& runtime, char kind, int q) {
    if (q < 0 || q >= runtime.k) {
        fail("active basis-change qubit is out of range");
    }
    const std::size_t dim = active_length(runtime.k);
    const std::size_t mask = std::size_t{1} << q;
    if (kind == 'H') {
        const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
        batch_parallel_for(runtime, dim, kBatchThreadBasisGrain, [&](int, std::size_t first, std::size_t last) {
            for (std::size_t base = first; base < last; ++base) {
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
        batch_parallel_for(runtime, dim, kBatchThreadBasisGrain, [&](int, std::size_t first, std::size_t last) {
            for (std::size_t basis = first; basis < last; ++basis) {
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
    if (has_parallel_work(runtime, dim, kBatchThreadBasisGrain)) {
        promote_first_dormant_rotation_threaded(runtime, dim, c, coeffs.data());
        ++runtime.k;
        --runtime.ndormant;
        return;
    }
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
    if (has_parallel_work(runtime, dim, kBatchThreadBasisGrain)) {
        last_z_measure_true_prob_threaded(runtime, dim);
    } else {
        batch_simd::scalar_table().last_z_measure_true_prob(
            runtime.active_re.data(),
            runtime.active_im.data(),
            static_cast<std::size_t>(runtime.batches),
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
    if (has_parallel_work(runtime, dim, kBatchThreadBasisGrain)) {
        last_z_project_threaded(runtime, dim, branch_bits.data(), runtime.branch_invnorms.data());
    } else {
        batch_simd::scalar_table().last_z_project(
            runtime.active_re.data(),
            runtime.active_im.data(),
            static_cast<std::size_t>(runtime.batches),
            runtime.active_shots,
            dim,
            branch_bits.data(),
            runtime.branch_invnorms.data());
    }
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
    if (has_parallel_work(runtime, out_dim, kBatchThreadBasisGrain)) {
        diagonal_measure_true_prob_threaded(runtime, source_true.data(), out_dim);
    } else {
        batch_simd::scalar_table().diagonal_measure_true_prob(
            runtime.active_re.data(),
            runtime.active_im.data(),
            static_cast<std::size_t>(runtime.batches),
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
    if (has_parallel_work(runtime, out_dim, kBatchThreadBasisGrain)) {
        diagonal_project_threaded(
            runtime,
            source_false.data(),
            source_true.data(),
            out_dim,
            branch_bits.data(),
            runtime.branch_invnorms.data());
    } else {
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
    }
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
    if (has_parallel_work(runtime, out_dim, kBatchThreadBasisGrain)) {
        nondiagonal_measure_true_prob_threaded(runtime, kernel, out_dim);
    } else {
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
    }
    sample_batch_measurement_branches_from_true(
        runtime,
        runtime.eval_scratch,
        runtime.branch_prob_true,
        runtime.branch_invnorms);
    const auto& branch_bits = runtime.eval_scratch;
    if (has_parallel_work(runtime, out_dim, kBatchThreadBasisGrain)) {
        nondiagonal_project_threaded(runtime, kernel, out_dim, branch_bits.data(), runtime.branch_invnorms.data());
    } else {
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
