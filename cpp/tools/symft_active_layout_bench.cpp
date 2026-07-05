#include "simd/simd.hpp"
#include "sampler/active.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Complex = symft::Complex;

constexpr double kPi = 3.141592653589793238462643383279502884;

#if defined(__clang__)
#define SYMFT_BENCH_SIMD_LOOP _Pragma("clang loop vectorize(enable) interleave(enable)")
#elif defined(__GNUC__)
#define SYMFT_BENCH_SIMD_LOOP _Pragma("GCC ivdep")
#else
#define SYMFT_BENCH_SIMD_LOOP
#endif

#if defined(__AVX2__) && defined(__FMA__)
#define SYMFT_BENCH_INLINE_AVX2 1
#else
#define SYMFT_BENCH_INLINE_AVX2 0
#endif

struct SoAState {
    std::vector<double> re;
    std::vector<double> im;
};

struct BenchCase {
    std::string name;
    symft::PrecomputedActivePauliRotationKernel kernel;
};

struct BenchResult {
    double seconds = 0.0;
    double ns_per_apply = 0.0;
    double ns_per_amplitude = 0.0;
    double checksum = 0.0;
};

double seconds_between(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double>(stop - start).count();
}

std::size_t active_dim(int k) {
    return std::size_t{1} << k;
}

symft::PauliString uniform_pauli(int k, int pair_bit, bool noncontiguous) {
    symft::PauliString pauli(k);
    pauli.set_xbit(pair_bit);
    if (noncontiguous) {
        pauli.set_xbit(pair_bit - 1);
    }
    return pauli;
}

symft::PauliString real_pauli(int k, int pair_bit, bool noncontiguous) {
    symft::PauliString pauli(k);
    pauli.set_xbit(pair_bit);
    pauli.set_zbit(pair_bit);
    pauli.set_phase(1);
    if (noncontiguous) {
        pauli.set_xbit(pair_bit - 1);
    }
    return pauli;
}

symft::PauliString general_pauli(int k, int pair_bit, bool noncontiguous) {
    symft::PauliString pauli(k);
    pauli.set_xbit(pair_bit);
    if (noncontiguous) {
        pauli.set_xbit(pair_bit - 1);
    }
    pauli.set_zbit((pair_bit + 1) % k);
    return pauli;
}

symft::PauliString uniform_pauli_with_lower_mask(int k, int pair_bit, std::uint64_t lower_mask) {
    symft::PauliString pauli(k);
    pauli.set_xbit(pair_bit);
    for (int bit = 0; bit < pair_bit; ++bit) {
        if ((lower_mask & (std::uint64_t{1} << bit)) != 0) {
            pauli.set_xbit(bit);
        }
    }
    return pauli;
}

symft::PauliString real_pauli_with_lower_mask(int k, int pair_bit, std::uint64_t lower_mask) {
    symft::PauliString pauli = uniform_pauli_with_lower_mask(k, pair_bit, lower_mask);
    pauli.set_zbit(pair_bit);
    pauli.set_phase(1);
    return pauli;
}

symft::PauliString general_pauli_with_lower_mask(int k, int pair_bit, std::uint64_t lower_mask) {
    symft::PauliString pauli = uniform_pauli_with_lower_mask(k, pair_bit, lower_mask);
    int zbit = 0;
    while (zbit == pair_bit || (lower_mask & (std::uint64_t{1} << zbit)) != 0) {
        ++zbit;
    }
    pauli.set_zbit(zbit);
    return pauli;
}

std::vector<Complex> initial_alpha(int k) {
    const std::size_t dim = active_dim(k);
    std::vector<Complex> alpha(dim);
    double norm2 = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        const double r = std::sin(0.17 * static_cast<double>(i + 1)) + 0.25 * std::cos(0.03 * static_cast<double>(i + 3));
        const double im = std::cos(0.11 * static_cast<double>(i + 5)) - 0.15 * std::sin(0.07 * static_cast<double>(i + 2));
        alpha[i] = Complex(r, im);
        norm2 += r * r + im * im;
    }
    const double invnorm = 1.0 / std::sqrt(norm2);
    for (auto& value : alpha) {
        value *= invnorm;
    }
    return alpha;
}

SoAState to_soa(const std::vector<Complex>& alpha) {
    SoAState out;
    out.re.resize(alpha.size());
    out.im.resize(alpha.size());
    for (std::size_t i = 0; i < alpha.size(); ++i) {
        out.re[i] = alpha[i].real();
        out.im[i] = alpha[i].imag();
    }
    return out;
}

double checksum_aos(const std::vector<Complex>& alpha) {
    double out = 0.0;
    for (std::size_t i = 0; i < alpha.size(); ++i) {
        out += (1.0 + static_cast<double>(i & 15)) * (alpha[i].real() + 0.5 * alpha[i].imag());
    }
    return out;
}

double checksum_soa(const SoAState& state) {
    double out = 0.0;
    for (std::size_t i = 0; i < state.re.size(); ++i) {
        out += (1.0 + static_cast<double>(i & 15)) * (state.re[i] + 0.5 * state.im[i]);
    }
    return out;
}

void apply_soa_block(SoAState& state, const symft::PrecomputedActivePauliRotationKernel& kernel, bool sign) {
    const double c = kernel.cos_kernel_angle;
    double* re = state.re.data();
    double* im = state.im.data();
    const std::size_t dim = state.re.size();
    if (kernel.is_diagonal) {
        const auto& coefficients = sign ? kernel.diagonal_plus_coefficients : kernel.diagonal_minus_coefficients;
        const auto* coeff = reinterpret_cast<const double*>(coefficients.data());
        SYMFT_BENCH_SIMD_LOOP
        for (std::size_t basis = 0; basis < coefficients.size(); ++basis) {
            const double fr = c + coeff[2 * basis];
            const double fi = coeff[2 * basis + 1];
            const double r = re[basis];
            const double v = im[basis];
            re[basis] = fr * r - fi * v;
            im[basis] = fr * v + fi * r;
        }
        return;
    }
    const std::size_t selector = std::size_t{1} << kernel.pair_bit;
    const std::size_t step = selector << 1;
    if (kernel.uniform_imag_pairs) {
        const Complex coefficient = sign ? kernel.pair_left_plus_coefficients.front() : kernel.pair_left_minus_coefficients.front();
        const double q = coefficient.imag();
#if SYMFT_BENCH_INLINE_AVX2
        if (selector == 1 && kernel.action.xmask == 1 && dim >= 4) {
            const __m256d vc = _mm256_set1_pd(c);
            const __m256d vq = _mm256_set1_pd(q);
            std::size_t basis = 0;
            for (; basis + 4 <= dim; basis += 4) {
                const __m256d r = _mm256_loadu_pd(re + basis);
                const __m256d v = _mm256_loadu_pd(im + basis);
                const __m256d swapped_r = _mm256_permute_pd(r, 0b0101);
                const __m256d swapped_v = _mm256_permute_pd(v, 0b0101);
                _mm256_storeu_pd(re + basis, _mm256_fnmadd_pd(vq, swapped_v, _mm256_mul_pd(vc, r)));
                _mm256_storeu_pd(im + basis, _mm256_fmadd_pd(vq, swapped_r, _mm256_mul_pd(vc, v)));
            }
            for (; basis < dim; basis += 2) {
                const double r0 = re[basis];
                const double im0 = im[basis];
                const double r1 = re[basis + 1];
                const double im1 = im[basis + 1];
                re[basis] = c * r0 - q * im1;
                im[basis] = c * im0 + q * r1;
                re[basis + 1] = c * r1 - q * im0;
                im[basis + 1] = c * im1 + q * r0;
            }
            return;
        }
#endif
        for (std::size_t block = 0; block < dim; block += step) {
            SYMFT_BENCH_SIMD_LOOP
            for (std::size_t offset = 0; offset < selector; ++offset) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = i0 ^ static_cast<std::size_t>(kernel.action.xmask);
                const double r0 = re[i0];
                const double im0 = im[i0];
                const double r1 = re[i1];
                const double im1 = im[i1];
                re[i0] = c * r0 - q * im1;
                im[i0] = c * im0 + q * r1;
                re[i1] = c * r1 - q * im0;
                im[i1] = c * im1 + q * r0;
            }
        }
        return;
    }
    if (kernel.real_pair_flip) {
        const Complex coefficient = sign ? kernel.pair_left_plus_coefficients.front() : kernel.pair_left_minus_coefficients.front();
        const double base_coeff = coefficient.real();
#if SYMFT_BENCH_INLINE_AVX2
        if (selector == 1 && kernel.action.xmask == 1 && dim >= 4) {
            const __m256d vc = _mm256_set1_pd(c);
            std::size_t basis = 0;
            for (; basis + 4 <= dim; basis += 4) {
                const double q0 = kernel.real_pair_flip_basis_phase_signs[basis >> 1] * base_coeff;
                const double q1 = kernel.real_pair_flip_basis_phase_signs[(basis >> 1) + 1] * base_coeff;
                const __m256d signed_q = _mm256_set_pd(q1, -q1, q0, -q0);
                const __m256d r = _mm256_loadu_pd(re + basis);
                const __m256d v = _mm256_loadu_pd(im + basis);
                const __m256d swapped_r = _mm256_permute_pd(r, 0b0101);
                const __m256d swapped_v = _mm256_permute_pd(v, 0b0101);
                _mm256_storeu_pd(re + basis, _mm256_fmadd_pd(signed_q, swapped_r, _mm256_mul_pd(vc, r)));
                _mm256_storeu_pd(im + basis, _mm256_fmadd_pd(signed_q, swapped_v, _mm256_mul_pd(vc, v)));
            }
            for (; basis < dim; basis += 2) {
                const double q = kernel.real_pair_flip_basis_phase_signs[basis >> 1] * base_coeff;
                const double r0 = re[basis];
                const double im0 = im[basis];
                const double r1 = re[basis + 1];
                const double im1 = im[basis + 1];
                re[basis] = c * r0 - q * r1;
                im[basis] = c * im0 - q * im1;
                re[basis + 1] = c * r1 + q * r0;
                im[basis + 1] = c * im1 + q * im0;
            }
            return;
        }
#endif
        std::size_t pair_idx = 0;
        for (std::size_t block = 0; block < dim; block += step) {
            SYMFT_BENCH_SIMD_LOOP
            for (std::size_t offset = 0; offset < selector; ++offset) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = i0 ^ static_cast<std::size_t>(kernel.action.xmask);
                const double q = kernel.real_pair_flip_basis_phase_signs[pair_idx++] * base_coeff;
                const double r0 = re[i0];
                const double im0 = im[i0];
                const double r1 = re[i1];
                const double im1 = im[i1];
                re[i0] = c * r0 - q * r1;
                im[i0] = c * im0 - q * im1;
                re[i1] = c * r1 + q * r0;
                im[i1] = c * im1 + q * im0;
            }
        }
        return;
    }
    const auto& left_coeff = sign ? kernel.pair_left_plus_coefficients : kernel.pair_left_minus_coefficients;
    const auto& right_coeff = sign ? kernel.pair_right_plus_coefficients : kernel.pair_right_minus_coefficients;
    const auto* left = reinterpret_cast<const double*>(left_coeff.data());
    const auto* right = reinterpret_cast<const double*>(right_coeff.data());
#if SYMFT_BENCH_INLINE_AVX2
    if (selector == 1 && kernel.action.xmask == 1 && dim >= 4) {
        const __m256d vc = _mm256_set1_pd(c);
        std::size_t basis = 0;
        for (; basis + 4 <= dim; basis += 4) {
            const std::size_t pair_idx = basis >> 1;
            const __m256d coeff_r =
                _mm256_set_pd(left[2 * (pair_idx + 1)], right[2 * (pair_idx + 1)], left[2 * pair_idx], right[2 * pair_idx]);
            const __m256d coeff_i = _mm256_set_pd(
                left[2 * (pair_idx + 1) + 1],
                right[2 * (pair_idx + 1) + 1],
                left[2 * pair_idx + 1],
                right[2 * pair_idx + 1]);
            const __m256d r = _mm256_loadu_pd(re + basis);
            const __m256d v = _mm256_loadu_pd(im + basis);
            const __m256d swapped_r = _mm256_permute_pd(r, 0b0101);
            const __m256d swapped_v = _mm256_permute_pd(v, 0b0101);
            __m256d out_r = _mm256_fmadd_pd(vc, r, _mm256_mul_pd(coeff_r, swapped_r));
            out_r = _mm256_fnmadd_pd(coeff_i, swapped_v, out_r);
            __m256d out_i = _mm256_fmadd_pd(vc, v, _mm256_mul_pd(coeff_r, swapped_v));
            out_i = _mm256_fmadd_pd(coeff_i, swapped_r, out_i);
            _mm256_storeu_pd(re + basis, out_r);
            _mm256_storeu_pd(im + basis, out_i);
        }
        for (; basis < dim; basis += 2) {
            const std::size_t pair_idx = basis >> 1;
            const double r0 = re[basis];
            const double im0 = im[basis];
            const double r1 = re[basis + 1];
            const double im1 = im[basis + 1];
            const double lr = left[2 * pair_idx];
            const double li = left[2 * pair_idx + 1];
            const double rr = right[2 * pair_idx];
            const double ri = right[2 * pair_idx + 1];
            re[basis] = c * r0 + rr * r1 - ri * im1;
            im[basis] = c * im0 + rr * im1 + ri * r1;
            re[basis + 1] = c * r1 + lr * r0 - li * im0;
            im[basis + 1] = c * im1 + lr * im0 + li * r0;
        }
        return;
    }
#endif
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        SYMFT_BENCH_SIMD_LOOP
        for (std::size_t offset = 0; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = i0 ^ static_cast<std::size_t>(kernel.action.xmask);
            const double r0 = re[i0];
            const double im0 = im[i0];
            const double r1 = re[i1];
            const double im1 = im[i1];
            const double lr = left[2 * pair_idx];
            const double li = left[2 * pair_idx + 1];
            const double rr = right[2 * pair_idx];
            const double ri = right[2 * pair_idx + 1];
            re[i0] = c * r0 + rr * r1 - ri * im1;
            im[i0] = c * im0 + rr * im1 + ri * r1;
            re[i1] = c * r1 + lr * r0 - li * im0;
            im[i1] = c * im1 + lr * im0 + li * r0;
            ++pair_idx;
        }
    }
}

void apply_soa_dispatch(SoAState& state, const symft::PrecomputedActivePauliRotationKernel& kernel, bool sign) {
    const auto& table = symft::simd::dispatch_table();
    const double c = kernel.cos_kernel_angle;
    if (kernel.is_diagonal) {
        const auto& coefficients = sign ? kernel.diagonal_plus_coefficients : kernel.diagonal_minus_coefficients;
        table.mul_assign_soa(state.re.data(), state.im.data(), coefficients.data(), c, coefficients.size());
        return;
    }
    if (kernel.uniform_imag_pairs) {
        const Complex coefficient = sign ? kernel.pair_left_plus_coefficients.front() : kernel.pair_left_minus_coefficients.front();
        table.rotate_uniform_imag_pairs_soa(
            state.re.data(),
            state.im.data(),
            state.re.size(),
            kernel.action.xmask,
            kernel.pair_bit,
            c,
            coefficient.imag());
        return;
    }
    if (kernel.real_pair_flip) {
        const Complex coefficient = sign ? kernel.pair_left_plus_coefficients.front() : kernel.pair_left_minus_coefficients.front();
        table.rotate_real_pair_flip_soa(
            state.re.data(),
            state.im.data(),
            state.re.size(),
            kernel.action.xmask,
            kernel.pair_bit,
            kernel.real_pair_flip_basis_phase_signs.data(),
            c,
            coefficient.real());
        return;
    }
    const auto& left_coeff = sign ? kernel.pair_left_plus_coefficients : kernel.pair_left_minus_coefficients;
    const auto& right_coeff = sign ? kernel.pair_right_plus_coefficients : kernel.pair_right_minus_coefficients;
    table.rotate_general_pairs_soa(
        state.re.data(),
        state.im.data(),
        state.re.size(),
        kernel.action.xmask,
        kernel.pair_bit,
        left_coeff.data(),
        right_coeff.data(),
        c);
}

BenchResult bench_aos(
    const std::vector<Complex>& initial,
    const symft::PrecomputedActivePauliRotationKernel& kernel,
    int k,
    int iterations) {
    symft::ActiveState state(k, initial);
    const auto start = Clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        symft::rotate_pauli(state, kernel, (iter & 1) != 0);
    }
    const auto stop = Clock::now();
    const double seconds = seconds_between(start, stop);
    return {
        seconds,
        1.0e9 * seconds / static_cast<double>(iterations),
        1.0e9 * seconds / static_cast<double>(iterations * static_cast<int>(initial.size())),
        checksum_aos(state.alpha),
    };
}

template <typename Apply>
BenchResult bench_soa(const std::vector<Complex>& initial, const symft::PrecomputedActivePauliRotationKernel& kernel, int iterations, Apply apply) {
    SoAState state = to_soa(initial);
    const auto start = Clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        apply(state, kernel, (iter & 1) != 0);
    }
    const auto stop = Clock::now();
    const double seconds = seconds_between(start, stop);
    return {
        seconds,
        1.0e9 * seconds / static_cast<double>(iterations),
        1.0e9 * seconds / static_cast<double>(iterations * static_cast<int>(initial.size())),
        checksum_soa(state),
    };
}

std::vector<BenchCase> make_cases(int k) {
    std::vector<BenchCase> cases;
    cases.push_back({"diag_z0", symft::PrecomputedActivePauliRotationKernel(symft::ActivePauliAction(symft::pauli_z(k, 0)), kPi / 8.0)});
    cases.push_back({"uniform_x0", symft::PrecomputedActivePauliRotationKernel(symft::ActivePauliAction(symft::pauli_x(k, 0)), kPi / 8.0)});

    std::vector<int> pair_bits = {1, std::min(3, k - 1), k - 1};
    std::sort(pair_bits.begin(), pair_bits.end());
    pair_bits.erase(std::unique(pair_bits.begin(), pair_bits.end()), pair_bits.end());
    for (const int pair_bit : pair_bits) {
        cases.push_back(
            {"uniform_p" + std::to_string(pair_bit) + "_contig",
             symft::PrecomputedActivePauliRotationKernel(symft::ActivePauliAction(uniform_pauli(k, pair_bit, false)), kPi / 8.0)});
        cases.push_back(
            {"uniform_p" + std::to_string(pair_bit) + "_noncontig",
             symft::PrecomputedActivePauliRotationKernel(symft::ActivePauliAction(uniform_pauli(k, pair_bit, true)), kPi / 8.0)});
        cases.push_back(
            {"real_p" + std::to_string(pair_bit) + "_contig",
             symft::PrecomputedActivePauliRotationKernel(symft::ActivePauliAction(real_pauli(k, pair_bit, false)), kPi / 8.0)});
        cases.push_back(
            {"real_p" + std::to_string(pair_bit) + "_noncontig",
             symft::PrecomputedActivePauliRotationKernel(symft::ActivePauliAction(real_pauli(k, pair_bit, true)), kPi / 8.0)});
        cases.push_back(
            {"general_p" + std::to_string(pair_bit) + "_contig",
             symft::PrecomputedActivePauliRotationKernel(symft::ActivePauliAction(general_pauli(k, pair_bit, false)), kPi / 8.0)});
        cases.push_back(
            {"general_p" + std::to_string(pair_bit) + "_noncontig",
             symft::PrecomputedActivePauliRotationKernel(symft::ActivePauliAction(general_pauli(k, pair_bit, true)), kPi / 8.0)});
    }
    if (k >= 8) {
        std::vector<int> stress_pair_bits = {std::min(5, k - 1), k - 1};
        std::sort(stress_pair_bits.begin(), stress_pair_bits.end());
        stress_pair_bits.erase(std::unique(stress_pair_bits.begin(), stress_pair_bits.end()), stress_pair_bits.end());
        for (const int pair_bit : stress_pair_bits) {
            for (std::uint64_t lower_mask = 1; lower_mask <= 3; ++lower_mask) {
                cases.push_back(
                    {"uniform_p" + std::to_string(pair_bit) + "_low" + std::to_string(lower_mask),
                     symft::PrecomputedActivePauliRotationKernel(
                         symft::ActivePauliAction(uniform_pauli_with_lower_mask(k, pair_bit, lower_mask)),
                         kPi / 8.0)});
                cases.push_back(
                    {"real_p" + std::to_string(pair_bit) + "_low" + std::to_string(lower_mask),
                     symft::PrecomputedActivePauliRotationKernel(
                         symft::ActivePauliAction(real_pauli_with_lower_mask(k, pair_bit, lower_mask)),
                         kPi / 8.0)});
                cases.push_back(
                    {"general_p" + std::to_string(pair_bit) + "_low" + std::to_string(lower_mask),
                     symft::PrecomputedActivePauliRotationKernel(
                         symft::ActivePauliAction(general_pauli_with_lower_mask(k, pair_bit, lower_mask)),
                         kPi / 8.0)});
            }
        }
    }
    return cases;
}

const char* pairing_name(const symft::PrecomputedActivePauliRotationKernel& kernel) {
    if (kernel.is_diagonal) {
        return "diagonal";
    }
    const std::size_t selector = std::size_t{1} << kernel.pair_bit;
    return (static_cast<std::size_t>(kernel.action.xmask) & (selector - 1)) == 0 ? "contiguous" : "noncontiguous";
}

void print_result(
    int k,
    const BenchCase& bench_case,
    const std::string& layout,
    int iterations,
    const BenchResult& result) {
    std::cout << "k " << k << " dim " << active_dim(k) << " kernel " << bench_case.name << " layout " << layout
              << " pair_bit " << bench_case.kernel.pair_bit
              << " xmask 0x" << std::hex << bench_case.kernel.action.xmask << std::dec
              << " pairing " << pairing_name(bench_case.kernel)
              << " iterations " << iterations << " seconds " << result.seconds << " ns_per_apply " << result.ns_per_apply
              << " ns_per_amplitude " << result.ns_per_amplitude << " checksum " << std::setprecision(17) << result.checksum
              << std::setprecision(6) << "\n";
}

int parse_int(const char* raw, const char* name) {
    try {
        std::size_t consumed = 0;
        const int value = std::stoi(raw, &consumed);
        if (consumed != std::string(raw).size()) {
            throw std::runtime_error("trailing characters");
        }
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + raw);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 4) {
            std::cerr << "usage: symft_active_layout_bench [target_amplitude_updates] [repeats] [max_k]\n";
            return 2;
        }
        const int target_updates = argc >= 2 ? parse_int(argv[1], "target_amplitude_updates") : (1 << 22);
        const int repeats = argc >= 3 ? parse_int(argv[2], "repeats") : 3;
        const int max_k = argc >= 4 ? parse_int(argv[3], "max_k") : 10;
        if (target_updates <= 0 || repeats <= 0 || max_k < 4) {
            throw std::runtime_error("target updates and repeats must be positive; max_k must be at least 4");
        }

        std::cout << "SymFT C++ active-state layout microbenchmark\n";
        std::cout << "simd_backend " << symft::active_simd_backend() << "\n";
        std::cout << "target_amplitude_updates " << target_updates << "\n";
        std::cout << "repeats " << repeats << "\n";

        for (int k = 4; k <= max_k; k += 2) {
            const auto initial = initial_alpha(k);
            const int iterations = std::max(16, target_updates / static_cast<int>(active_dim(k)));
            const auto cases = make_cases(k);
            for (const auto& bench_case : cases) {
                for (int repeat = 0; repeat < repeats; ++repeat) {
                    print_result(k, bench_case, "aos_complex", iterations, bench_aos(initial, bench_case.kernel, k, iterations));
                    print_result(
                        k,
                        bench_case,
                        "soa_block",
                        iterations,
                        bench_soa(initial, bench_case.kernel, iterations, apply_soa_block));
                    print_result(
                        k,
                        bench_case,
                        "soa_dispatch",
                        iterations,
                        bench_soa(initial, bench_case.kernel, iterations, apply_soa_dispatch));
                    std::cout << "repeat_done " << repeat + 1 << "\n";
                }
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "symft_active_layout_bench: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
