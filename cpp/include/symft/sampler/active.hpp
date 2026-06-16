#pragma once

#include "symft/core/pauli.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace symft {

struct ActiveState {
    int k = 0;
    std::vector<Complex> alpha;

    ActiveState();
    explicit ActiveState(int k);
    ActiveState(int k, std::vector<Complex> alpha);

    std::size_t dim() const;
    void reserve_for_k(int max_k);
};

struct ActivePauliAction {
    int nqubits = 0;
    std::uint64_t xmask = 0;
    std::uint64_t zmask = 0;
    Complex even_phase = Complex(1.0, 0.0);
    Complex odd_phase = Complex(-1.0, 0.0);
    bool xz_overlap_odd = false;

    ActivePauliAction() = default;
    explicit ActivePauliAction(const PauliString& pauli);
};

struct PrecomputedActivePauliRotationKernel {
    ActivePauliAction action;
    bool is_diagonal = true;
    bool uniform_imag_pairs = false;
    bool real_pair_flip = false;
    unsigned pair_bit = 0;
    double theta = 0.0;
    double cos_theta = 1.0;
    std::vector<Complex> diagonal_minus_coefficients;
    std::vector<Complex> diagonal_plus_coefficients;
    std::vector<std::size_t> pair_left_indices;
    std::vector<std::size_t> pair_right_indices;
    std::vector<double> real_pair_flip_basis_phase_signs;
    std::vector<Complex> pair_left_minus_coefficients;
    std::vector<Complex> pair_right_minus_coefficients;
    std::vector<Complex> pair_left_plus_coefficients;
    std::vector<Complex> pair_right_plus_coefficients;

    PrecomputedActivePauliRotationKernel() = default;
    PrecomputedActivePauliRotationKernel(const ActivePauliAction& action, double theta);
};

struct PrecomputedActivePauliMeasurementKernel {
    ActivePauliAction action;
    int pivot = 0;
    bool is_diagonal = true;
    std::vector<std::size_t> source0_false;
    std::vector<std::size_t> source1_false;
    std::vector<Complex> coeff0_false;
    std::vector<Complex> coeff1_false;
    std::vector<double> coeff1_false_real;
    std::vector<double> coeff1_false_imag;
    std::vector<std::size_t> source0_true;
    std::vector<std::size_t> source1_true;
    std::vector<Complex> coeff0_true;
    std::vector<Complex> coeff1_true;

    PrecomputedActivePauliMeasurementKernel() = default;
    explicit PrecomputedActivePauliMeasurementKernel(const PauliString& pauli);
    explicit PrecomputedActivePauliMeasurementKernel(const ActivePauliAction& action);
};

void apply_pauli(std::vector<Complex>& out, const PauliString& pauli, const std::vector<Complex>& alpha);
void apply_pauli(ActiveState& state, const PauliString& pauli);
void rotate_pauli(ActiveState& state, const PauliString& pauli, double theta);
void rotate_pauli(ActiveState& state, const ActivePauliAction& action, double theta);
void rotate_pauli(ActiveState& state, const PrecomputedActivePauliRotationKernel& kernel, bool sign);
std::string active_simd_backend();

} // namespace symft
