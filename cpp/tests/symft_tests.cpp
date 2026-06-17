#include "frontend/stim.hpp"
#include "sampler/batch_sampler.hpp"
#include "sampler/exogenous.hpp"
#include "sampler/single_shot.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
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

std::optional<int> instruction_record(const symft::FactoredInstruction& instruction) {
    return std::visit(
        [](const auto& inst) -> std::optional<int> {
            using T = std::decay_t<decltype(inst)>;
            if constexpr (
                std::is_same_v<T, symft::RecordMeasurement> ||
                std::is_same_v<T, symft::MeasureActiveLastZ> ||
                std::is_same_v<T, symft::MeasurePrecomputedActivePauli> ||
                std::is_same_v<T, symft::IntroduceDormantMeasurementBranch>) {
                return inst.record;
            } else {
                return std::nullopt;
            }
        },
        instruction);
}

symft::BatchDetectorPostselectionPlan single_detector_plan(
    const symft::FactoredInstructionProgram& program,
    int record) {
    symft::BatchDetectorPostselectionPlan plan;
    plan.instruction_records_by_index.reserve(program.instructions.size());
    int producer_instruction = -1;
    for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
        const int produced_record = instruction_record(program.instructions[idx]).value_or(0);
        plan.instruction_records_by_index.push_back(produced_record);
        if (produced_record == record) {
            producer_instruction = static_cast<int>(idx);
        }
    }
    require(producer_instruction >= 0, "postselection test record producer");
    plan.detectors_by_record.resize(static_cast<std::size_t>(program.nrecords + 1));
    plan.detectors_by_record[static_cast<std::size_t>(record)].push_back({record});
    plan.condition_last_use_by_index.assign(static_cast<std::size_t>(program.nsymbols + 1), -1);
    plan.record_last_use_by_index.assign(static_cast<std::size_t>(program.nrecords + 1), -1);
    plan.record_last_use_by_index[static_cast<std::size_t>(record)] = producer_instruction;
    return plan;
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

void test_active_h_rewrite_stays_virtual() {
    using namespace symft;
    FrameFactoredState state(2, 1);
    apply_pauli_rotation(state, pauli_z(2, 0) * pauli_x(2, 1), M_PI / 2.0);
    apply_pauli_measurement(state, pauli_z(2, 1));
    PendingFactoredState pending(state);
    const auto program = plan_factored_updates(pending);
    require(program.max_k == 2, "virtual active H rewrite promotes dormant qubit");

    const auto records = sample_measurements(program, 8, 17);
    for (const auto& shot : records) {
        require(packed_bit(shot, 0), "virtual active H rewrite preserves deterministic dormant rotation");
    }
}

void test_dormant_measurement_tableau_reuse() {
    using namespace symft;
    FrameFactoredState state(2, 1);
    const PauliString measured = pauli_z(2, 0) * pauli_x(2, 1);
    apply_pauli_measurement(state, measured);
    apply_pauli_measurement(state, measured);
    PendingFactoredState pending(state);
    const auto program = plan_factored_updates(pending);
    require(program.max_k == 1, "dormant measurement tableau does not touch alpha");

    const auto records = sample_measurements(program, 64, 123);
    for (const auto& shot : records) {
        require(
            packed_bit(shot, 0) == packed_bit(shot, 1),
            "dormant measurement tableau reuses fixed stabilizer outcome");
    }
}

void test_dormant_measurement_sign_feeds_promotion() {
    using namespace symft;
    FrameFactoredState state(1, 0);
    apply_pauli_measurement(state, pauli_x(1, 0));
    apply_pauli_rotation(state, pauli_z(1, 0), M_PI / 2.0);
    apply_pauli_measurement(state, pauli_x(1, 0));
    PendingFactoredState pending(state);
    const auto program = plan_factored_updates(pending);
    require(program.max_k == 1, "dormant sign promotion test promotes one active qubit");

    const auto records = sample_measurements(program, 64, 456);
    for (const auto& shot : records) {
        require(
            packed_bit(shot, 0) != packed_bit(shot, 1),
            "dormant measurement sign feeds later promotion");
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

void test_stim_frontend_circuit_lowering() {
    using namespace symft;
    const auto circuit = parse_stim_circuit_text("REPEAT 2 {\nM !0\n}\nDETECTOR rec[-1] rec[-2]\n");
    require(circuit.nqubits == 1, "circuit qubit count");
    require(circuit.nrecords == 2, "circuit record count");
    require(circuit.instructions.size() == 2, "circuit flattened repeat instructions");
    require(circuit.instructions[0].kind == CircuitInstructionKind::MZ, "circuit measurement kind");
    require(circuit.instructions[0].measurement_targets[0].inverted, "circuit inverted measurement target");
    require(circuit.detectors.size() == 1, "circuit detector count");
    require(circuit.detectors[0].records[0] == 2 && circuit.detectors[0].records[1] == 1, "circuit detector records resolved");

    const auto lowered = lower_circuit_to_factored(circuit);
    require(lowered.measurement_records.size() == 2, "lowered record count");
    PendingFactoredState pending(lowered.state);
    const auto program = plan_factored_updates(pending);
    const auto records = sample_measurements(program);
    require(packed_bit(records, 0) && packed_bit(records, 1), "lowered circuit deterministic records");

    const auto cy_feedback = parse_stim_text("M !0\nCY rec[-1] 1\nM 1\n");
    PendingFactoredState pending_cy(cy_feedback.state);
    const auto cy_program = plan_factored_updates(pending_cy);
    const auto cy_records = sample_measurements(cy_program);
    require(packed_bit(cy_records, 0) && packed_bit(cy_records, 1), "CY feedback deterministic records");

    const auto ordinary_cy = parse_stim_text("X 0\nCY 0 1\nM 1\n");
    PendingFactoredState pending_ordinary_cy(ordinary_cy.state);
    const auto ordinary_cy_program = plan_factored_updates(pending_ordinary_cy);
    const auto ordinary_cy_records = sample_measurements(ordinary_cy_program);
    require(packed_bit(ordinary_cy_records, 0), "ordinary CY deterministic target record");
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

    const auto packed_samples = presample_exogenous_packed(program, 4, 17);
    require(packed_samples.nshots == 4, "packed presampled shot count");
    require(packed_samples.nsymbols == program.nsymbols, "packed presampled symbol count");
    SingleShotPresampledExpressionPlan packed_expression_plan;
    prepare_single_shot_presampled_expression_plan(packed_expression_plan, program, packed_samples);
    SingleShotPresampledExpressionBlock packed_expression_block;
    evaluate_single_shot_presampled_expression_block(
        packed_expression_block,
        packed_expression_plan,
        packed_samples);
    for (int shot = 0; shot < packed_samples.nshots; ++shot) {
        reset_executor(runtime, program);
        execute_in_place(runtime, program, packed_expression_plan, packed_expression_block, shot);
        require(packed_bit(runtime.measurement_words, 0), "packed presampled expression deterministic X error");
    }

    require(default_single_shot_sample_chunk_shots() == 1024, "single-shot default sample chunk");
    const auto chunked_records = sample_measurements(program, 9, 17, 3);
    require(chunked_records.size() == 9, "chunked single-shot sample count");
    for (const auto& shot : chunked_records) {
        require(packed_bit(shot, 0), "chunked single-shot deterministic X error");
    }

    const auto random_error = parse_stim_text("X_ERROR(0.5) 0\nM 0\n");
    PendingFactoredState pending_random(random_error.state);
    const auto random_program = plan_factored_updates(pending_random);
    const auto scalar_samples = presample_exogenous(random_program, 11, 31);
    const auto packed_random_samples = presample_exogenous_packed(random_program, 11, 31);
    PresampledExogenous scalar_block;
    scalar_block.nshots = 5;
    scalar_block.nsymbols = scalar_samples.nsymbols;
    scalar_block.nwords = scalar_samples.nwords;
    scalar_block.next_rng_state = scalar_samples.next_rng_state;
    scalar_block.exogenous_assigned_words = scalar_samples.exogenous_assigned_words;
    scalar_block.value_words.resize(5 * scalar_block.nwords);
    std::copy_n(
        scalar_samples.value_words.begin() + static_cast<std::ptrdiff_t>(3 * scalar_block.nwords),
        5 * scalar_block.nwords,
        scalar_block.value_words.begin());
    BatchFactoredExecutorState scalar_runtime(random_program, 5, 41);
    execute_batch_in_place(scalar_runtime, random_program, scalar_block);
    BatchFactoredExecutorState packed_runtime(random_program, 5, 41);
    execute_batch_in_place(packed_runtime, random_program, packed_random_samples, 3);
    require(
        scalar_runtime.measurement_words == packed_runtime.measurement_words,
        "packed batch exogenous slice matches shot-major presample");
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
}

void test_batch_postselection() {
    using namespace symft;
    {
        const auto parsed = parse_stim_text("M !0\nDETECTOR rec[-1]\n");
        PendingFactoredState pending(parsed.state);
        const auto program = plan_factored_updates(pending);
        const auto samples = presample_exogenous(program, 8, 17);
        BatchFactoredExecutorState runtime(program, 8, 19);
        BatchDetectorPostselectionScratch scratch;
        const auto result = execute_batch_postselected_in_place(
            runtime,
            program,
            samples,
            single_detector_plan(program, 1),
            scratch);
        require(result.discarded == 8, "batch postselection rejects fired detector");
        require(result.accepted == 0, "batch postselection no accepted shots after rejection");
        require(runtime.active_shots == 0, "batch postselection compacts all rejected shots");
    }
    {
        const auto parsed = parse_stim_text("M 0\nDETECTOR rec[-1]\n");
        PendingFactoredState pending(parsed.state);
        const auto program = plan_factored_updates(pending);
        const auto samples = presample_exogenous(program, 8, 23);
        BatchFactoredExecutorState runtime(program, 8, 29);
        BatchDetectorPostselectionScratch scratch;
        const auto result = execute_batch_postselected_in_place(
            runtime,
            program,
            samples,
            single_detector_plan(program, 1),
            scratch);
        require(result.discarded == 0, "batch postselection keeps quiet detector");
        require(result.accepted == 8, "batch postselection accepted count");
        require(runtime.active_shots == 8, "batch postselection keeps active shots");
        require(runtime.measurement_words[0] == 0, "batch postselection keeps quiet measurement records");
    }
    {
        const auto parsed = parse_stim_text("X_ERROR(0.5) 0\nM 0\nDETECTOR rec[-1]\nM !1\n");
        PendingFactoredState pending(parsed.state);
        const auto program = plan_factored_updates(pending);
        const auto samples = presample_exogenous(program, 8, 31);
        BatchDetectorPostselectionScratch default_scratch;
        BatchFactoredExecutorState default_runtime(program, 8, 37);
        const auto default_result = execute_batch_postselected_in_place(
            default_runtime,
            program,
            samples,
            single_detector_plan(program, 1),
            default_scratch);
        BatchDetectorPostselectionScratch hybrid_scratch;
        BatchFactoredExecutorState hybrid_runtime(program, 8, 37);
        const auto hybrid_result = execute_batch_postselected_in_place(
            hybrid_runtime,
            program,
            samples,
            single_detector_plan(program, 1),
            hybrid_scratch,
            BatchDetectorPostselectionOptions{false, false, 8});
        require(default_result.discarded == hybrid_result.discarded, "hybrid postselection discarded count");
        require(default_result.accepted == hybrid_result.accepted, "hybrid postselection accepted count");
        require(default_runtime.measurement_words == hybrid_runtime.measurement_words, "hybrid postselection records");
    }
    {
        const auto parsed = parse_stim_text(
            "X_ERROR(0.125) 0\n"
            "M 0\n"
            "DETECTOR rec[-1]\n"
            "H 1\n"
            "T 1\n"
            "T_DAG 1\n"
            "H 1\n"
            "M 1\n");
        PendingFactoredState pending(parsed.state);
        const auto program = plan_factored_updates(pending);
        const auto samples = presample_exogenous(program, 64, 41);
        BatchDetectorPostselectionScratch default_scratch;
        BatchFactoredExecutorState default_runtime(program, 64, 43);
        const auto default_result = execute_batch_postselected_in_place(
            default_runtime,
            program,
            samples,
            single_detector_plan(program, 1),
            default_scratch);
        BatchDetectorPostselectionScratch masked_scratch;
        BatchFactoredExecutorState masked_runtime(program, 64, 43);
        const auto masked_result = execute_batch_postselected_in_place(
            masked_runtime,
            program,
            samples,
            single_detector_plan(program, 1),
            masked_scratch,
            BatchDetectorPostselectionOptions{false, true, 0});
        require(masked_result.discarded > 0, "masked postselection test kills at least one lane");
        require(masked_result.discarded * 4 < 64, "masked postselection test stays dirty before final compaction");
        require(default_result.discarded == masked_result.discarded, "masked postselection discarded count");
        require(default_result.accepted == masked_result.accepted, "masked postselection accepted count");
        require(default_runtime.measurement_words == masked_runtime.measurement_words, "masked postselection records");
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
    test_active_h_rewrite_stays_virtual();
    test_dormant_measurement_tableau_reuse();
    test_dormant_measurement_sign_feeds_promotion();
    test_parser_feedback();
    test_stim_frontend_circuit_lowering();
    test_t_gate_exact_rotation();
    test_presampled_exogenous();
    test_batch_sampler();
    test_batch_postselection();
    test_detectors();
    std::cout << "symft_cpp_tests passed (SIMD backend: " << symft::active_simd_backend() << ")\n";
    return 0;
}
