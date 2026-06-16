#include "active_kernels.hpp"

#include "symft/simd.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace symft {
using namespace detail;

ActiveState::ActiveState(int k_) : k(checked_nqubits(k_)), alpha(active_length(k), Complex(0.0, 0.0)) {
    alpha[0] = Complex(1.0, 0.0);
}

ActiveState::ActiveState(int k_, std::vector<Complex> alpha_) : k(checked_nqubits(k_)), alpha(std::move(alpha_)) {
    if (alpha.size() != active_length(k)) {
        fail("active vector length does not match active qubit count");
    }
}

std::size_t ActiveState::dim() const {
    return active_length(k);
}

void ActiveState::reserve_for_k(int max_k) {
    const std::size_t max_dim = active_length(max_k);
    if (alpha.size() < max_dim) {
        alpha.resize(max_dim, Complex(0.0, 0.0));
    }
}

ActivePauliAction::ActivePauliAction(const PauliString& pauli) : nqubits(pauli.nqubits) {
    if (nqubits >= 63) {
        fail("Pauli string has too many qubits for active-state basis indexing");
    }
    if (!pauli_squares_to_identity(pauli)) {
        fail("Pauli rotation requires P^2 == I");
    }
    xmask = active_mask_x(pauli);
    zmask = active_mask_z(pauli);
    even_phase = phase_factor(pauli.phase_exponent());
    odd_phase = -even_phase;
    xz_overlap_odd = is_odd_popcount(xmask & zmask);
}

PrecomputedActivePauliRotationKernel::PrecomputedActivePauliRotationKernel(const ActivePauliAction& action_, double theta_)
    : action(action_), is_diagonal(action.xmask == 0), theta(theta_), cos_theta(std::cos(theta_)) {
    const Complex minus_i_s(0.0, -std::sin(theta));
    const Complex plus_i_s(0.0, std::sin(theta));
    const std::size_t dim = active_length(action.nqubits);
    if (is_diagonal) {
        diagonal_minus_coefficients.resize(dim);
        diagonal_plus_coefficients.resize(dim);
        for (std::size_t basis = 0; basis < dim; ++basis) {
            const Complex phase = active_action_phase(action, basis);
            diagonal_minus_coefficients[basis] = minus_i_s * phase;
            diagonal_plus_coefficients[basis] = plus_i_s * phase;
        }
        return;
    }
    uniform_imag_pairs = action.zmask == 0;
    real_pair_flip = can_rotate_real_pair_flip(action);
    pair_bit = static_cast<unsigned>(trailing_zeros64(action.xmask));
    const std::size_t pair_selector = action.xmask & (~action.xmask + 1);
    for (std::size_t left = 0; left < dim; ++left) {
        if ((left & pair_selector) != 0) {
            continue;
        }
        const std::size_t right = left ^ action.xmask;
        pair_left_indices.push_back(left);
        pair_right_indices.push_back(right);
        if (real_pair_flip) {
            real_pair_flip_basis_phase_signs.push_back(
                is_odd_popcount(static_cast<std::uint64_t>(left) & action.zmask) ? -1.0 : 1.0);
        }
        const Complex left_phase = active_action_phase(action, left);
        const Complex right_phase = active_action_phase(action, right);
        pair_left_minus_coefficients.push_back(minus_i_s * left_phase);
        pair_right_minus_coefficients.push_back(minus_i_s * right_phase);
        pair_left_plus_coefficients.push_back(plus_i_s * left_phase);
        pair_right_plus_coefficients.push_back(plus_i_s * right_phase);
    }
}

namespace {

PrecomputedActivePauliMeasurementKernel precomputed_nondiagonal_measurement_kernel(const ActivePauliAction& action) {
    PrecomputedActivePauliMeasurementKernel out;
    out.action = action;
    out.is_diagonal = false;
    out.pivot = trailing_zeros64(action.xmask);
    const std::size_t dim = active_length(action.nqubits);
    const std::size_t out_dim = dim >> 1;
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    out.source0_false.reserve(out_dim);
    out.source1_false.reserve(out_dim);
    out.coeff0_false.reserve(out_dim);
    out.coeff1_false.reserve(out_dim);
    out.coeff1_false_real.reserve(out_dim);
    out.coeff1_false_imag.reserve(out_dim);
    out.source0_true.reserve(out_dim);
    out.source1_true.reserve(out_dim);
    out.coeff0_true.reserve(out_dim);
    out.coeff1_true.reserve(out_dim);
    for (std::size_t packed = 0; packed < out_dim; ++packed) {
        const std::size_t x0 = insert_zero_bit(packed, out.pivot);
        const std::size_t x1 = x0 ^ action.xmask;
        const Complex eta = active_action_phase(action, x0);
        const Complex coeff = (Complex(1.0, 0.0) / eta) * inv_sqrt2;
        out.source0_false.push_back(x0);
        out.source1_false.push_back(x1);
        out.coeff0_false.push_back(Complex(inv_sqrt2, 0.0));
        out.coeff1_false.push_back(coeff);
        out.coeff1_false_real.push_back(coeff.real());
        out.coeff1_false_imag.push_back(coeff.imag());
        out.source0_true.push_back(x0);
        out.source1_true.push_back(x1);
        out.coeff0_true.push_back(Complex(inv_sqrt2, 0.0));
        out.coeff1_true.push_back(-coeff);
    }
    return out;
}

PrecomputedActivePauliMeasurementKernel precomputed_diagonal_measurement_kernel(const ActivePauliAction& action) {
    PrecomputedActivePauliMeasurementKernel out;
    out.action = action;
    out.is_diagonal = true;
    out.pivot = trailing_zeros64(action.zmask);
    const std::size_t dim = active_length(action.nqubits);
    const std::size_t out_dim = dim >> 1;
    const bool negative_phase = std::abs(action.even_phase.real() + 1.0) < 1e-12 &&
                                std::abs(action.even_phase.imag()) < 1e-12;
    const bool positive_phase = std::abs(action.even_phase.real() - 1.0) < 1e-12 &&
                                std::abs(action.even_phase.imag()) < 1e-12;
    if (!negative_phase && !positive_phase) {
        fail("diagonal active measurement Pauli must have real eigenvalues");
    }
    const int phase_bit = negative_phase ? 1 : 0;
    const std::uint64_t z_without_pivot = action.zmask & ~(std::uint64_t{1} << out.pivot);
    out.source0_false.reserve(out_dim);
    out.source0_true.reserve(out_dim);
    out.coeff0_false.assign(out_dim, Complex(1.0, 0.0));
    out.coeff1_false.assign(out_dim, Complex(0.0, 0.0));
    out.coeff0_true.assign(out_dim, Complex(1.0, 0.0));
    out.coeff1_true.assign(out_dim, Complex(0.0, 0.0));
    out.source1_false.assign(out_dim, kNoSource);
    out.source1_true.assign(out_dim, kNoSource);
    for (std::size_t packed = 0; packed < out_dim; ++packed) {
        const std::size_t x_without_pivot = insert_zero_bit(packed, out.pivot);
        const int parity = popcount64(static_cast<std::uint64_t>(x_without_pivot) & z_without_pivot) & 1;
        const int false_pivot = phase_bit ^ parity;
        const int true_pivot = false_pivot ^ 1;
        out.source0_false.push_back(x_without_pivot | (std::size_t(false_pivot) << out.pivot));
        out.source0_true.push_back(x_without_pivot | (std::size_t(true_pivot) << out.pivot));
    }
    return out;
}

} // namespace

PrecomputedActivePauliMeasurementKernel::PrecomputedActivePauliMeasurementKernel(const PauliString& pauli)
    : PrecomputedActivePauliMeasurementKernel(ActivePauliAction(pauli)) {}

PrecomputedActivePauliMeasurementKernel::PrecomputedActivePauliMeasurementKernel(const ActivePauliAction& action_) {
    if (action_.nqubits <= 0) {
        fail("cannot build an active measurement kernel for k == 0");
    }
    if (action_.xmask != 0) {
        *this = precomputed_nondiagonal_measurement_kernel(action_);
    } else if (action_.zmask != 0) {
        *this = precomputed_diagonal_measurement_kernel(action_);
    } else {
        fail("cannot build an active measurement kernel for identity Pauli");
    }
}

void apply_pauli(std::vector<Complex>& out, const PauliString& pauli, const std::vector<Complex>& alpha) {
    if (pauli.nqubits >= 63) {
        fail("Pauli string has too many qubits for active-state basis indexing");
    }
    const std::size_t dim = active_length(pauli.nqubits);
    if (alpha.size() < dim || out.size() < dim || out.data() == alpha.data()) {
        fail("invalid active Pauli application buffers");
    }
    const std::uint64_t xmask = active_mask_x(pauli);
    const std::uint64_t zmask = active_mask_z(pauli);
    const Complex even = phase_factor(pauli.phase_exponent());
    const Complex odd = -even;
    for (std::size_t src = 0; src < dim; ++src) {
        const std::size_t dst = src ^ xmask;
        out[dst] = (is_odd_popcount(static_cast<std::uint64_t>(src) & zmask) ? odd : even) * alpha[src];
    }
}

void apply_pauli(ActiveState& state, const PauliString& pauli) {
    if (pauli.nqubits != state.k) {
        fail("Pauli string dimension does not match active state");
    }
    const ActivePauliAction action(pauli);
    const std::size_t dim = state.dim();
    if (action.xmask == 0) {
        for (std::size_t src = 0; src < dim; ++src) {
            state.alpha[src] *= active_action_phase(action, src);
        }
        return;
    }
    const std::size_t selector = action.xmask & (~action.xmask + 1);
    for (std::size_t base = 0; base < dim; ++base) {
        if ((base & selector) != 0) {
            continue;
        }
        const std::size_t paired = base ^ action.xmask;
        const Complex a0 = state.alpha[base];
        const Complex a1 = state.alpha[paired];
        const bool parity0 = active_action_phase_odd(action, base);
        const Complex phase0 = parity0 ? action.odd_phase : action.even_phase;
        const Complex phase1 = (parity0 != action.xz_overlap_odd) ? action.odd_phase : action.even_phase;
        state.alpha[base] = phase1 * a1;
        state.alpha[paired] = phase0 * a0;
    }
}

void rotate_pauli(ActiveState& state, const PauliString& pauli, double theta) {
    rotate_pauli(state, ActivePauliAction(pauli), theta);
}

void rotate_pauli(ActiveState& state, const ActivePauliAction& action, double theta) {
    if (action.nqubits != state.k) {
        fail("Pauli action dimension does not match active state");
    }
    const std::size_t dim = state.dim();
    const double c = std::cos(theta);
    const Complex minus_i_s(0.0, -std::sin(theta));
    if (action.xmask == 0) {
        for (std::size_t basis = 0; basis < dim; ++basis) {
            state.alpha[basis] *= Complex(c, 0.0) + minus_i_s * active_action_phase(action, basis);
        }
        return;
    }
    const std::size_t selector = action.xmask & (~action.xmask + 1);
    for (std::size_t base = 0; base < dim; ++base) {
        if ((base & selector) != 0) {
            continue;
        }
        const std::size_t paired = base ^ action.xmask;
        const Complex a0 = state.alpha[base];
        const Complex a1 = state.alpha[paired];
        const bool parity0 = active_action_phase_odd(action, base);
        const Complex phase0 = parity0 ? action.odd_phase : action.even_phase;
        const Complex phase1 = (parity0 != action.xz_overlap_odd) ? action.odd_phase : action.even_phase;
        state.alpha[base] = c * a0 + minus_i_s * phase1 * a1;
        state.alpha[paired] = c * a1 + minus_i_s * phase0 * a0;
    }
}

void rotate_pauli(ActiveState& state, const PrecomputedActivePauliRotationKernel& kernel, bool sign) {
    if (kernel.action.nqubits != state.k) {
        fail("rotation kernel dimension does not match active state");
    }
    const double c = kernel.cos_theta;
    if (kernel.is_diagonal) {
        const auto& coefficients = sign ? kernel.diagonal_plus_coefficients : kernel.diagonal_minus_coefficients;
        simd::dispatch_table().mul_assign(state.alpha.data(), coefficients.data(), c, coefficients.size());
        return;
    }
    if (kernel.uniform_imag_pairs) {
        const Complex coefficient = sign ? kernel.pair_left_plus_coefficients.front() : kernel.pair_left_minus_coefficients.front();
        const std::size_t npairs = kernel.pair_left_indices.size();
        if (npairs < kSimdPairRotationThreshold) {
            rotate_uniform_imag_pairs_scalar(
                state.alpha.data(),
                kernel.pair_left_indices.data(),
                kernel.pair_right_indices.data(),
                npairs,
                c,
                coefficient.imag());
        } else {
            simd::dispatch_table().rotate_uniform_imag_pairs(
                state.alpha.data(),
                kernel.pair_left_indices.data(),
                kernel.pair_right_indices.data(),
                npairs,
                c,
                coefficient.imag());
        }
        return;
    }
    if (kernel.real_pair_flip) {
        const Complex coefficient = sign ? kernel.pair_left_plus_coefficients.front() : kernel.pair_left_minus_coefficients.front();
        const std::size_t npairs = kernel.pair_left_indices.size();
        if (npairs < kSimdPairRotationThreshold) {
            rotate_real_pair_flip_scalar(
                state.alpha.data(),
                kernel.pair_left_indices.data(),
                kernel.pair_right_indices.data(),
                npairs,
                c,
                coefficient.real(),
                kernel.action.zmask);
        } else {
            simd::dispatch_table().rotate_real_pair_flip(
                state.alpha.data(),
                kernel.pair_left_indices.data(),
                kernel.pair_right_indices.data(),
                npairs,
                c,
                coefficient.real(),
                kernel.action.zmask);
        }
        return;
    }
    const auto& left_coeff = sign ? kernel.pair_left_plus_coefficients : kernel.pair_left_minus_coefficients;
    const auto& right_coeff = sign ? kernel.pair_right_plus_coefficients : kernel.pair_right_minus_coefficients;
    for (std::size_t idx = 0; idx < kernel.pair_left_indices.size(); ++idx) {
        const std::size_t i0 = kernel.pair_left_indices[idx];
        const std::size_t i1 = kernel.pair_right_indices[idx];
        const Complex a0 = state.alpha[i0];
        const Complex a1 = state.alpha[i1];
        state.alpha[i0] = c * a0 + right_coeff[idx] * a1;
        state.alpha[i1] = c * a1 + left_coeff[idx] * a0;
    }
}

std::string active_simd_backend() {
    return simd::dispatch_name();
}

} // namespace symft
