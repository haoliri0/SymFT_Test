#include "symft/symft.hpp"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << "\n";
        std::exit(1);
    }
}

bool approx(symft::Complex a, symft::Complex b, double eps = 1e-10) {
    return std::abs(a - b) <= eps;
}

void test_pauli_algebra() {
    using namespace symft;
    require(pauli_x(1, 0) * pauli_x(1, 0) == pauli_identity(1), "X squares to I");
    require(pauli_y(1, 0) * pauli_y(1, 0) == pauli_identity(1), "Y squares to I");
    auto xz = pauli_y(1, 0);
    xz.phase_shift(3);
    require(pauli_x(1, 0) * pauli_z(1, 0) == xz, "packed XZ phase");
    require(pauli_anticommutes(pauli_x(1, 0), pauli_z(1, 0)), "X anticommutes with Z");
    const auto ps = pauli_string("IXYZ");
    require(!ps.xbit(0) && !ps.zbit(0), "I bits");
    require(ps.xbit(1) && !ps.zbit(1), "X bits");
    require(ps.xbit(2) && ps.zbit(2), "Y bits");
    require(!ps.xbit(3) && ps.zbit(3), "Z bits");
}

void test_clifford_frame() {
    using namespace symft;
    CliffordFrame cf(2);
    left_CX(cf, 0, 1);
    require(preimage(cf, pauli_x(2, 0)) == pauli_string("XX"), "CX maps Xc");
    require(preimage(cf, pauli_z(2, 1)) == pauli_string("ZZ"), "CX maps Zt");

    CliffordFrame h(1);
    left_H(h, 0);
    left_S(h, 0);
    require(preimage(h, pauli_x(1, 0)) == pauli_y(1, 0), "left composition order");
}

void test_active_rotation() {
    using namespace symft;
    ActiveState st(1);
    rotate_pauli(st, pauli_x(1, 0), M_PI / 4.0);
    require(approx(st.alpha[0], {std::cos(M_PI / 4.0), 0.0}), "X rotation amplitude 0");
    require(approx(st.alpha[1], {0.0, -std::sin(M_PI / 4.0)}), "X rotation amplitude 1");

    ActiveState y(1);
    rotate_pauli(y, pauli_y(1, 0), M_PI / 6.0);
    require(approx(y.alpha[0], {std::cos(M_PI / 6.0), 0.0}), "Y rotation amplitude 0");
    require(approx(y.alpha[1], {std::sin(M_PI / 6.0), 0.0}), "Y rotation amplitude 1");

    ActiveState z(1, {{3.0, 4.0}, {5.0, -2.0}});
    auto before = z.alpha;
    ActivePauliAction action(pauli_z(1, 0));
    PrecomputedActivePauliRotationKernel kernel(action, 0.31);
    rotate_pauli(z, kernel, false);
    require(approx(z.alpha[0], std::exp(symft::Complex(0.0, -0.31)) * before[0]), "precomputed Z rotation 0");
    require(approx(z.alpha[1], std::exp(symft::Complex(0.0, 0.31)) * before[1]), "precomputed Z rotation 1");

    ActiveState x_expected(1, {{0.2, -0.3}, {0.7, 0.4}});
    ActiveState x_fast(1, x_expected.alpha);
    const auto x_pauli = pauli_x(1, 0);
    rotate_pauli(x_expected, x_pauli, 0.23);
    rotate_pauli(x_fast, PrecomputedActivePauliRotationKernel(ActivePauliAction(x_pauli), 0.23), false);
    require(approx(x_fast.alpha[0], x_expected.alpha[0]) && approx(x_fast.alpha[1], x_expected.alpha[1]), "precomputed X fast rotation");

    ActiveState y_expected(1, {{0.1, 0.8}, {-0.6, 0.2}});
    ActiveState y_fast(1, y_expected.alpha);
    const auto y_pauli = pauli_y(1, 0);
    rotate_pauli(y_expected, y_pauli, -0.19);
    rotate_pauli(y_fast, PrecomputedActivePauliRotationKernel(ActivePauliAction(y_pauli), 0.19), true);
    require(approx(y_fast.alpha[0], y_expected.alpha[0]) && approx(y_fast.alpha[1], y_expected.alpha[1]), "precomputed Y fast signed rotation");

    std::vector<Complex> alpha6(64);
    for (std::size_t i = 0; i < alpha6.size(); ++i) {
        alpha6[i] = {0.01 * static_cast<double>(i + 1), -0.02 * static_cast<double>((i % 7) + 1)};
    }
    ActiveState x6_expected(6, alpha6);
    ActiveState x6_fast(6, alpha6);
    const auto x6_pauli = pauli_x(6, 5);
    rotate_pauli(x6_expected, x6_pauli, 0.17);
    rotate_pauli(x6_fast, PrecomputedActivePauliRotationKernel(ActivePauliAction(x6_pauli), 0.17), false);
    for (std::size_t i = 0; i < alpha6.size(); ++i) {
        require(approx(x6_fast.alpha[i], x6_expected.alpha[i]), "precomputed multi-qubit X fast rotation");
    }

    ActiveState y6_expected(6, alpha6);
    ActiveState y6_fast(6, alpha6);
    const auto y6_pauli = pauli_y(6, 4);
    rotate_pauli(y6_expected, y6_pauli, 0.11);
    rotate_pauli(y6_fast, PrecomputedActivePauliRotationKernel(ActivePauliAction(y6_pauli), 0.11), false);
    for (std::size_t i = 0; i < alpha6.size(); ++i) {
        require(approx(y6_fast.alpha[i], y6_expected.alpha[i]), "precomputed multi-qubit Y fast rotation");
    }
}

void test_parser_feedback() {
    using namespace symft;
    const auto parsed = parse_stim_text("M !0\nCX rec[-1] 1\nM 1\n");
    PendingFactoredState pending(parsed.state);
    const auto program = plan_factored_updates(pending);
    const auto records = sample_measurements(program);
    require(program.nrecords == 2, "feedback record count");
    require(packed_bit(records, 0) && packed_bit(records, 1), "feedback deterministic records");
}

void test_t_gate_exact_rotation() {
    using namespace symft;
    const auto parsed = parse_stim_text("H 0\nT 0\nM 0\n");
    PendingFactoredState pending(parsed.state);
    const auto program = plan_factored_updates(pending);
    require(program.max_k == 1, "T promotes one active qubit");
    int ones = 0;
    const auto records = sample_measurements(program, 200, 7);
    for (const auto& shot : records) {
        ones += packed_bit(shot, 0) ? 1 : 0;
    }
    require(ones > 50 && ones < 150, "T then MX produces non-deterministic X measurement");
}

void test_presampled_exogenous() {
    using namespace symft;
    const auto parsed = parse_stim_text("X_ERROR(1) 0\nM 0\n");
    PendingFactoredState pending(parsed.state);
    const auto program = plan_factored_updates(pending);
    const auto samples = presample_exogenous(program, 4, 17);
    require(samples.nshots == 4, "presampled shot count");
    require(samples.nsymbols == program.nsymbols, "presampled symbol count");

    FactoredExecutorState runtime(program, samples.next_rng_state);
    for (int shot = 0; shot < samples.nshots; ++shot) {
        reset_executor(runtime, program);
        execute_in_place(runtime, program, samples, shot);
        require(packed_bit(runtime.measurement_words, 0), "presampled deterministic X error");
    }
}

void test_batch_sampler() {
    using namespace symft;
    const auto parsed = parse_stim_text("X_ERROR(1) 0\nM 0\n");
    PendingFactoredState pending(parsed.state);
    const auto program = plan_factored_updates(pending);
    const auto records = sample_measurements_batch(program, 9, 4, 17);
    require(records.size() == 9, "batch sampler shot count");
    for (const auto& shot : records) {
        require(packed_bit(shot, 0), "batch presampled deterministic X error");
    }

    const auto feedback = parse_stim_text("M !0\nCX rec[-1] 1\nM 1\n");
    PendingFactoredState pending_feedback(feedback.state);
    const auto feedback_program = plan_factored_updates(pending_feedback);
    const auto feedback_records = sample_measurements_batch(feedback_program, 7, 3, 19);
    for (const auto& shot : feedback_records) {
        require(packed_bit(shot, 0) && packed_bit(shot, 1), "batch feedback deterministic records");
    }

    const auto t_circuit = parse_stim_text("H 0\nT 0\nM 0\n");
    PendingFactoredState pending_t(t_circuit.state);
    const auto t_program = plan_factored_updates(pending_t);
    const auto t_records = sample_measurements_batch(t_program, 200, 32, 23);
    int ones = 0;
    for (const auto& shot : t_records) {
        ones += packed_bit(shot, 0) ? 1 : 0;
    }
    require(ones > 50 && ones < 150, "batch T then MX produces non-deterministic X measurement");

    const std::vector<BatchKernelBackend> backends = {
        BatchKernelBackend::SoaAutovec,
        BatchKernelBackend::SoaDispatch,
        BatchKernelBackend::InterleavedAvx2,
    };
    for (const auto backend : backends) {
        try {
            const auto backend_records = sample_measurements_batch(program, 9, 4, 17, backend);
            for (const auto& shot : backend_records) {
                require(packed_bit(shot, 0), "batch backend deterministic X error");
            }
            const auto backend_t_records = sample_measurements_batch(t_program, 200, 32, 23, backend);
            int backend_ones = 0;
            for (const auto& shot : backend_t_records) {
                backend_ones += packed_bit(shot, 0) ? 1 : 0;
            }
            require(
                backend_ones > 50 && backend_ones < 150,
                std::string("batch backend T stochastic range: ") + batch_kernel_backend_name(backend));
        } catch (const Error& ex) {
            const bool optional_unavailable =
                backend == BatchKernelBackend::InterleavedAvx2 &&
                std::string(ex.what()).find("unavailable") != std::string::npos;
            if (!optional_unavailable) {
                throw;
            }
        }
    }
}

void test_detectors() {
    using namespace symft;
    const auto accepted = parse_stim_text("M !0\nOBSERVABLE_INCLUDE(0) rec[-1]\n");
    const auto summary = estimate_stim_logical_error_rate(accepted, 5);
    require(summary.shots == 5, "summary shot count");
    require(summary.discarded == 0, "summary no discard");
    require(summary.logical_errors == 5, "summary logical count");

    const auto rejected = parse_stim_text("M !0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n");
    const auto summary2 = estimate_stim_logical_error_rate(rejected, 5);
    require(summary2.discarded == 5, "summary discard count");
    require(summary2.accepted == 0, "summary accepted count");
}

} // namespace

int main() {
    test_pauli_algebra();
    test_clifford_frame();
    test_active_rotation();
    test_parser_feedback();
    test_t_gate_exact_rotation();
    test_presampled_exogenous();
    test_batch_sampler();
    test_detectors();
    std::cout << "symft_cpp_tests passed (SIMD backend: " << symft::active_simd_backend() << ")\n";
    return 0;
}
