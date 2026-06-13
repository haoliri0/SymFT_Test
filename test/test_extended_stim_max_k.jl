@testset "Stim parser symbolic noise variables" begin
    result = parse_stim_text("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        X_ERROR(0.25) 0
        DEPOLARIZE1(0.3) 0
        DEPOLARIZE2(0.15) 0 1
        M(0.125) 0
        MX(0.25) 1
        MPP(0.5) !X0*Z1
        T 0
        T_DAG 1
    """)

    state = stim_state(result)
    context = symbolic_context(state)
    bernoulli = bernoulli_probabilities(context)
    categorical = categorical_distributions(context)

    @test nqubits(state) == 2
    @test length(pending_measurements(state)) == 3
    @test length(pending_rotations(state)) == 2
    @test length(stim_measurement_records(result)) == 3
    measurements = pending_measurements(state)
    records = stim_measurement_records(result)
    @test getproperty.(measurements, :record) == [1, 2, 3]
    @test getproperty.(measurements, :record_condition) == only.(condition_ids.(records))
    @test count(==(0.25), values(bernoulli)) >= 2
    @test 0.125 in values(bernoulli)
    @test 0.5 in values(bernoulli)

    @test length(categorical) == 2
    @test length(categorical[1].conditions) == 2
    @test categorical[1].assignments == Vector{Bool}[
        [false, false],
        [false, true],
        [true, false],
        [true, true],
    ]
    @test categorical[1].probabilities ≈ [0.7, 0.1, 0.1, 0.1]
    @test length(categorical[2].conditions) == 4
    @test categorical[2].assignments[1] == Bool[false, false, false, false]
    @test categorical[2].probabilities ≈ vcat([0.85], fill(0.01, 15))
end

@testset "Stim parser links record feedback to pending measurement outcomes" begin
    result = parse_stim_text("""
        M 0
        CX rec[-1] 1
        M 1
    """)
    state = stim_state(result)
    records = stim_measurement_records(result)
    record_conditions = only.(condition_ids.(records))
    measurements = pending_measurements(state)

    @test length(records) == 2
    @test getproperty.(measurements, :record) == [1, 2]
    @test getproperty.(measurements, :record_condition) == record_conditions
    @test symbolic_sign(measurements[2].pauli) == records[1]

    pending = PendingFactoredState(state)
    program = plan_factored_updates!(pending)
    @test program.instructions == FactoredInstruction[
        RecordMeasurement(SymbolicBool(false), 1, record_conditions[1]),
        RecordMeasurement(records[1], 2, record_conditions[2]),
    ]
end

@testset "Stim parser annotations and repeat blocks" begin
    result = parse_stim_text("""
        M 0
        DETECTOR(1, 2) rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
        REPEAT 2 {
            MX 1
            DETECTOR rec[-1]
        }
    """)

    @test nqubits(stim_state(result)) == 2
    @test length(stim_measurement_records(result)) == 3
    @test stim_detectors(result) == StimDetector[
        StimDetector([1], [1.0, 2.0], 2),
        StimDetector([2], Float64[], 6),
        StimDetector([3], Float64[], 6),
    ]
    @test stim_observable_includes(result) == StimObservableInclude[
        StimObservableInclude(0, [1], 3),
    ]
end

@testset "Stim parser repeated measurement resets and shifted detectors" begin
    result = parse_stim_text("""
        X 0
        MR 0
        REPEAT 2 {
            X 0
            MR 0
            SHIFT_COORDS(0, 0, 1)
            DETECTOR(5, 6, 0) rec[-1] rec[-2]
        }
    """)

    @test nqubits(stim_state(result)) == 1
    @test length(stim_measurement_records(result)) == 3
    detectors = stim_detectors(result)
    @test getproperty.(detectors, :records) == [[2, 1], [3, 2]]
    @test getproperty.(detectors, :coords) == [[5.0, 6.0, 1.0], [5.0, 6.0, 2.0]]

    program = plan_factored_updates!(PendingFactoredState(stim_state(result)))
    records = sample_measurements(program, 4; batches = 2)
    @test records == trues(4, 3)
end

@testset "Stim parser keeps readout error out of post-measurement state" begin
    result = parse_stim_text("""
        M(1) 0
        M 0
    """)
    program = plan_factored_updates!(PendingFactoredState(stim_state(result)))
    records = sample_measurements(program, 4; batches = 2)

    @test records[:, 1] == trues(4)
    @test records[:, 2] == falses(4)
end

@testset "Stim parser exact reset uses hidden records" begin
    result = parse_stim_text("""
        X 0
        R 0
        M 0
    """)
    measurements = pending_measurements(stim_state(result))
    @test length(stim_measurement_records(result)) == 1
    @test length(measurements) == 2
    @test measurements[1].record === nothing
    @test measurements[1].record_condition !== nothing
    @test measurements[2].record == 1

    program = plan_factored_updates!(PendingFactoredState(stim_state(result)))
    @test program.nrecords == 1
    records = sample_measurements(program, 8; batches = 4)
    @test records == falses(8, 1)
end

@testset "Stim batch estimator counts detector discards and logicals" begin
    accepted = parse_stim_text("""
        M !0
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    summary = estimate_stim_logical_error_rate(accepted, 5; batches = 2)
    @test summary.shots == 5
    @test summary.discarded == 0
    @test summary.accepted == 5
    @test summary.logical_errors == 5
    @test discard_rate(summary) == 0.0
    @test logical_error_rate(summary) == 1.0

    rejected = parse_stim_text("""
        M !0
        DETECTOR rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    summary = estimate_stim_logical_error_rate(rejected, 5; batches = 2)
    @test summary.discarded == 5
    @test summary.accepted == 0
    @test summary.logical_errors == 0
    @test discard_rate(summary) == 1.0
    @test isnan(logical_error_rate(summary))
end

@testset "d3 extended Stim pending max active width" begin
    result = parse_stim_file(joinpath(@__DIR__, "..", "d3.stim"))
    pending = PendingFactoredState(stim_state(result))
    program = plan_factored_updates!(pending)

    @test nqubits(stim_state(result)) == 15
    @test !has_pending_operations(pending)
    @test max_active_qubits(pending) == 4
    @test max_active_qubits(program) == 4
end

@testset "d5 extended Stim pending max active width" begin
    result = parse_stim_file(joinpath(@__DIR__, "..", "d5.stim"))
    pending = PendingFactoredState(stim_state(result))
    program = plan_factored_updates!(pending)

    @test nqubits(stim_state(result)) == 42
    @test !has_pending_operations(pending)
    @test max_active_qubits(pending) == 10
    @test max_active_qubits(program) == 10
end
