@testset "FactoredInstructionProgram single-shot deterministic records" begin
    result = parse_stim_text("""
        M !0
        CX rec[-1] 1
        M 1
    """)
    pending = PendingFactoredState(stim_state(result))
    program = plan_factored_updates!(pending)

    @test sample_measurements(program) == Bool[true, true]
end

@testset "FactoredInstructionProgram single-shot packed symbolic records" begin
    conditions = collect(1:40)
    expr = SymbolicBool(false, conditions)
    program = FactoredInstructionProgram(
        0,
        0,
        ActiveState(0),
        FactoredInstruction[RecordMeasurement(expr, 1)],
        0,
    )

    @test_throws ArgumentError sample_measurements(program)

    values = [condition => isodd(condition) for condition in conditions]

    @test sample_measurements(program; values) == Bool[false]
end

@testset "FactoredInstructionProgram single-shot exogenous noise" begin
    result = parse_stim_text("M(1) 0")
    pending = PendingFactoredState(stim_state(result))
    program = plan_factored_updates!(pending)

    @test sample_measurements(program) == Bool[true]
end

mutable struct SequenceRNG
    values::Vector{Float64}
    cursor::Int
end

SequenceRNG(values::AbstractVector{<:Real}) = SequenceRNG(Float64[value for value in values], 1)

function Base.rand(rng::SequenceRNG)
    value = rng.values[rng.cursor]
    rng.cursor += 1
    return value
end

@testset "FactoredInstructionProgram pre-samples rare correlated symbols" begin
    context = SymbolicContext()
    condition_sets = Vector{Int}[]
    assignments = Vector{Bool}[
        [false, false],
        [true, false],
        [false, true],
    ]
    probabilities = [0.99, 0.005, 0.005]
    for _ in 1:5
        push!(condition_sets, fresh_categorical_conditions!(context, assignments, probabilities))
    end
    program = FactoredInstructionProgram(0, 0, ActiveState(0), FactoredInstruction[], 0, context)

    @test isempty(program.sampled_categorical_distributions)
    @test length(program.sampled_rare_categorical_groups) == 1
    @test length(program.sampled_rare_categorical_groups[1].conditions) == 5

    p = 0.01
    hit_third_draw = exp(2.5 * log1p(-p))
    exit_afterwards = exp(10.0 * log1p(-p))
    runtime = FactoredExecutorState(program; rng = SequenceRNG([hit_third_draw, 0.25, exit_afterwards]))
    execute!(runtime, program)

    @test all(runtime.assigned)
    @test !any(runtime.values[condition_sets[1]])
    @test !any(runtime.values[condition_sets[2]])
    @test runtime.values[condition_sets[3]] == Bool[true, false]
    @test !any(runtime.values[condition_sets[4]])
    @test !any(runtime.values[condition_sets[5]])
end

@testset "FactoredInstructionProgram single-shot active measurement" begin
    program = FactoredInstructionProgram(
        1,
        1,
        ActiveState(1, ComplexF64[0.0 + 0.0im, 1.0 + 0.0im]),
        FactoredInstruction[MeasureActiveLastZ(1, symbolic_bool(1), 1)],
        1,
    )
    runtime = FactoredExecutorState(program)

    @test execute!(runtime, program) == Bool[true]
    @test runtime.active.k == 0
    @test runtime.k == 0
    @test runtime.ndormant == 1
    @test runtime.values[1]
end

@testset "FactoredInstructionProgram single-shot active rotation" begin
    program = FactoredInstructionProgram(
        1,
        1,
        ActiveState(1),
        FactoredInstruction[
            ApplyPrecomputedActivePauliRotation(pauli_x(1, 0), pi / 2, SymbolicBool(false)),
            MeasureActiveLastZ(1, symbolic_bool(1), 1),
        ],
        1,
    )

    @test sample_measurements(program) == Bool[true]
end

@testset "FactoredInstructionProgram single-shot precomputed active rotation" begin
    program = FactoredInstructionProgram(
        8,
        8,
        ActiveState(8),
        FactoredInstruction[
            ApplyPrecomputedActivePauliRotation(pauli_x(8, 7), pi / 2, SymbolicBool(false)),
            MeasureActiveLastZ(1, symbolic_bool(1), 1),
        ],
        8,
    )

    @test sample_measurements(program) == Bool[true]
end

@testset "FactoredInstructionProgram presampled rare Bernoulli symbols" begin
    context = SymbolicContext()
    condition = fresh_bernoulli_condition!(context, 0.01)
    program = FactoredInstructionProgram(
        0,
        0,
        ActiveState(0),
        FactoredInstruction[RecordMeasurement(symbolic_bool(condition), 1)],
        0,
        context,
    )

    p = 0.01
    hit_second_shot = exp(1.5 * log1p(-p))
    exit_afterwards = exp(2.0 * log1p(-p))
    samples = presample_exogenous(
        program,
        3;
        rng = SequenceRNG([hit_second_shot, exit_afterwards]),
    )
    runtime = FactoredExecutorState(program)

    @test samples.value_words == UInt64[0, 1, 0]
    @test sample_measurements!(runtime, program, samples, 1) == Bool[false]
    @test sample_measurements!(runtime, program, samples, 2) == Bool[true]
    @test sample_measurements!(runtime, program, samples, 3) == Bool[false]
end

@testset "FactoredInstructionProgram presampled correlated rare symbols" begin
    context = SymbolicContext()
    assignments = Vector{Bool}[
        [false, false],
        [true, false],
        [false, true],
    ]
    probabilities = [0.99, 0.005, 0.005]
    conditions1 = fresh_categorical_conditions!(context, assignments, probabilities)
    conditions2 = fresh_categorical_conditions!(context, assignments, probabilities)
    instructions = FactoredInstruction[
        RecordMeasurement(symbolic_bool(conditions1[1]), 1),
        RecordMeasurement(symbolic_bool(conditions1[2]), 2),
        RecordMeasurement(symbolic_bool(conditions2[1]), 3),
        RecordMeasurement(symbolic_bool(conditions2[2]), 4),
    ]
    program = FactoredInstructionProgram(0, 0, ActiveState(0), instructions, 0, context)

    p = 0.01
    hit_second_draw = exp(1.5 * log1p(-p))
    choose_first_event_row = 0.25
    exit_afterwards = exp(10.0 * log1p(-p))
    samples = presample_exogenous(
        program,
        2;
        rng = SequenceRNG([hit_second_draw, choose_first_event_row, exit_afterwards]),
    )
    runtime = FactoredExecutorState(program)

    @test samples.value_words == UInt64[4, 0]
    @test sample_measurements!(runtime, program, samples, 1) == Bool[false, false, true, false]
    @test sample_measurements!(runtime, program, samples, 2) == Bool[false, false, false, false]
end

@testset "FactoredInstructionProgram presampled measurement error" begin
    result = parse_stim_text("M(1) 0")
    pending = PendingFactoredState(stim_state(result))
    program = plan_factored_updates!(pending)
    samples = presample_exogenous(program, 2)
    runtime = FactoredExecutorState(program)

    @test sample_measurements!(runtime, program, samples, 1) == Bool[true]
    @test sample_measurements!(runtime, program, samples, 2) == Bool[true]
end

@testset "FactoredInstructionProgram single-shot dormant promotion" begin
    program = FactoredInstructionProgram(
        1,
        0,
        ActiveState(0),
        FactoredInstruction[
            PromoteDormantRotation(pi / 2, SymbolicBool(false)),
            MeasureActiveLastZ(1, symbolic_bool(1), 1),
        ],
        1,
    )

    @test sample_measurements(program) == Bool[true]
end

@testset "FactoredExecutorState reserves active buffers to max_k" begin
    program = FactoredInstructionProgram(
        4,
        0,
        ActiveState(0),
        FactoredInstruction[PromoteDormantRotation(0.125, SymbolicBool(false)) for _ in 1:4],
        4,
    )
    runtime = FactoredExecutorState(program)
    @test runtime.active.k == 0
    @test length(runtime.active.alpha) == 16
    @test length(runtime.active_scratch) == 16

    sample_measurements!(runtime, program)
    bytes = @allocated sample_measurements!(runtime, program)

    @test runtime.active.k == 4
    @test length(runtime.active.alpha) == 16
    @test length(runtime.active_scratch) == 16
    @test bytes == 0
end
