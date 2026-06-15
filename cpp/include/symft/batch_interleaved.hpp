#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>

namespace symft::batch_interleaved_avx2 {

using Complex = std::complex<double>;

void rotate_uniform_imag_xmask(
    double* active,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    double c,
    const double* q_by_shot);

void rotate_real_pair_flip_xmask(
    double* active,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    double c,
    const double* q_by_shot,
    std::uint64_t zmask);

void diagonal_measure_true_prob(
    const double* active,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source_true,
    std::size_t out_dim,
    double* prob_true);

void diagonal_project(
    const double* active,
    double* scratch,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source_false,
    const std::size_t* source_true,
    std::size_t out_dim,
    const std::uint64_t* branch_bits,
    const double* invnorms);

void nondiagonal_measure_true_prob(
    const double* active,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source0,
    const std::size_t* source1,
    const Complex* coeff1_false,
    std::size_t out_dim,
    double* prob_true);

void nondiagonal_project(
    const double* active,
    double* scratch,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source0,
    const std::size_t* source1,
    const Complex* coeff1_false,
    std::size_t out_dim,
    const std::uint64_t* branch_bits,
    const double* invnorms);

void promote_first_dormant_rotation(
    double* active,
    std::size_t leading_shots,
    int active_shots,
    std::size_t dim,
    double c,
    const double* q_by_shot);

} // namespace symft::batch_interleaved_avx2
