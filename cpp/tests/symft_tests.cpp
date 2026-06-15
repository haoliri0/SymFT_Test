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
}

void test_parser_feedback() {
    using namespace symft;
    const auto parsed = parse_stim_text("M !0\nCX rec[-1] 1\nM 1\n");
    PendingFactoredState pending(parsed.state);
    const auto program = plan_factored_updates(pending);
    const auto records = sample_measurements(program);
    require(records.size() == 2, "feedback record count");
    require(records[0] && records[1], "feedback deterministic records");
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
        ones += shot[0] ? 1 : 0;
    }
    require(ones > 50 && ones < 150, "T then MX produces non-deterministic X measurement");
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
    test_detectors();
    std::cout << "symft_cpp_tests passed (SIMD backend: " << symft::active_simd_backend() << ")\n";
    return 0;
}
