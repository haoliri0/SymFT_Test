function _reset_batch_avx2_state!()
    SymFT._BATCH_AVX2_TRIED[] = false
    SymFT._BATCH_AVX2_SYMBOL[] = Base.C_NULL
    SymFT._BATCH_AVX2_UNIFORM_XMASK_SYMBOL[] = Base.C_NULL
    SymFT._BATCH_AVX2_REAL_PAIR_FLIP_SYMBOL[] = Base.C_NULL
    SymFT._BATCH_AVX2_REAL_PAIR_FLIP_XMASK_SYMBOL[] = Base.C_NULL
    SymFT._BATCH_AVX2_DIAGONAL_MEASURE_PROB_SYMBOL[] = Base.C_NULL
    SymFT._BATCH_AVX2_DIAGONAL_MEASURE_PROJECT_SYMBOL[] = Base.C_NULL
    SymFT._BATCH_AVX2_NONDIAGONAL_MEASURE_PROB_SYMBOL[] = Base.C_NULL
    SymFT._BATCH_AVX2_NONDIAGONAL_MEASURE_PROJECT_SYMBOL[] = Base.C_NULL
    SymFT._BATCH_AVX2_PROMOTE_SYMBOL[] = Base.C_NULL
    SymFT._BATCH_AVX2_HANDLE[] = nothing
    return nothing
end

@testset "BatchFactoredExecutorState deterministic records" begin
    result = parse_stim_text("""
        M !0
        CX rec[-1] 1
        M 1
    """)
    pending = PendingFactoredState(stim_state(result))
    program = plan_factored_updates!(pending)

    records = sample_measurements(program, 5; batches = 2)

    @test size(records) == (5, 2)
    @test records == trues(5, 2)
end

@testset "BatchFactoredExecutorState kernel backend selects active layout" begin
    program = FactoredInstructionProgram(
        1,
        1,
        ActiveState(1),
        FactoredInstruction[],
        1,
    )

    turbo_runtime = BatchFactoredExecutorState(program; batches = 4, kernel_backend = :turbo)
    c_runtime = BatchFactoredExecutorState(program; batches = 4, kernel_backend = :c)
    scalar_runtime = BatchFactoredExecutorState(program; batches = 4, kernel_backend = :scalar)

    @test turbo_runtime.kernel_backend == :structarray
    @test !(turbo_runtime.active isa Matrix{ComplexF64})
    @test c_runtime.kernel_backend == :c
    @test c_runtime.active isa Matrix{ComplexF64}
    @test scalar_runtime.kernel_backend == :scalar
    @test scalar_runtime.active isa Matrix{ComplexF64}
    @test_throws ArgumentError BatchFactoredExecutorState(
        program;
        batches = 4,
        kernel_backend = :c,
        active_layout = :struct,
    )
end

@testset "BatchFactoredExecutorState packed symbolic records" begin
    conditions = collect(1:40)
    expr = SymbolicBool(false, conditions)
    program = FactoredInstructionProgram(
        0,
        0,
        ActiveState(0),
        FactoredInstruction[RecordMeasurement(expr, 1)],
        0,
    )

    @test_throws ArgumentError sample_measurements(program, 3; batches = 2)

    values = [condition => isodd(condition) for condition in conditions]

    @test sample_measurements(program, 3; batches = 2, values) == falses(3, 1)
end

@testset "BatchFactoredExecutorState active basis change" begin
    program = FactoredInstructionProgram(
        1,
        1,
        ActiveState(1),
        FactoredInstruction[ApplyActiveBasisChange(:H, 0)],
        1,
    )
    runtime = BatchFactoredExecutorState(program; batches = 4)
    execute!(runtime, program)

    expected = ComplexF64[inv(sqrt(2.0)), inv(sqrt(2.0))]
    @test runtime.active[1:4, 1] ≈ fill(expected[1], 4)
    @test runtime.active[1:4, 2] ≈ fill(expected[2], 4)
end

@testset "BatchFactoredExecutorState active measurement branches by shot" begin
    program = FactoredInstructionProgram(
        1,
        1,
        ActiveState(1, ComplexF64[sqrt(0.25) + 0.0im, sqrt(0.75) + 0.0im]),
        FactoredInstruction[
            MeasureActiveLastZ(1, symbolic_bool(1), 1),
        ],
        1,
    )
    runtime = BatchFactoredExecutorState(program; batches = 4, rng = SequenceRNG([0.1, 0.8, 0.7]))

    records = sample_measurements!(runtime, program, 3)

    @test records[:, 1] == Bool[true, false, true]
    @test runtime.k == 0
    @test runtime.ndormant == 1
end

@testset "SharedActiveBatchExecutorState keeps identical active states shared" begin
    program = FactoredInstructionProgram(
        1,
        1,
        ActiveState(1),
        FactoredInstruction[
            ApplyPrecomputedActivePauliRotation(pauli_x(1, 0), pi / 8, SymbolicBool(false)),
        ],
        1,
    )
    runtime = SharedActiveBatchExecutorState(program; batches = 8)

    execute!(runtime, program)

    @test active_packet_count(runtime) == 1
end

@testset "SharedActiveBatchExecutorState splits active states by rotation sign" begin
    context = SymbolicContext()
    sign = fresh_bernoulli_condition!(context, 0.5)
    program = FactoredInstructionProgram(
        1,
        1,
        ActiveState(1),
        FactoredInstruction[
            ApplyPrecomputedActivePauliRotation(pauli_x(1, 0), pi / 8, symbolic_bool(sign)),
        ],
        1,
        context,
    )
    runtime = SharedActiveBatchExecutorState(
        program;
        batches = 4,
        rng = SequenceRNG([0.6, 0.4, 0.8, 0.3]),
    )

    execute!(runtime, program)

    @test active_packet_count(runtime) == 2
end

@testset "SharedActiveBatchExecutorState starts materialized for saturation-prone programs" begin
    context = SymbolicContext()
    sign1 = fresh_bernoulli_condition!(context, 0.5)
    sign2 = fresh_bernoulli_condition!(context, 0.5)
    program = FactoredInstructionProgram(
        8,
        1,
        ActiveState(1),
        FactoredInstruction[
            ApplyPrecomputedActivePauliRotation(pauli_x(1, 0), pi / 8, symbolic_bool(sign1)),
            ApplyPrecomputedActivePauliRotation(pauli_z(1, 0), pi / 8, symbolic_bool(sign2)),
        ],
        8,
        context,
    )

    runtime = SharedActiveBatchExecutorState(program; batches = 4)

    @test active_packet_count(runtime) == 4
end

@testset "SharedActiveBatchExecutorState active measurement branches by shot" begin
    program = FactoredInstructionProgram(
        1,
        1,
        ActiveState(1, ComplexF64[sqrt(0.25) + 0.0im, sqrt(0.75) + 0.0im]),
        FactoredInstruction[
            MeasureActiveLastZ(1, symbolic_bool(1), 1),
        ],
        1,
    )
    runtime = SharedActiveBatchExecutorState(program; batches = 4, rng = SequenceRNG([0.1, 0.8, 0.7]))

    records = sample_measurements!(runtime, program, 3)

    @test records[:, 1] == Bool[true, false, true]
    @test runtime.base.k == 0
    @test runtime.base.ndormant == 1
    @test active_packet_count(runtime) == 2
end

@testset "Precomputed diagonal active measurement matches single shots" begin
    alpha = ComplexF64[
        ComplexF64(sin(0.13 * basis), cos(0.17 * basis)) for basis in 1:8
    ]
    alpha ./= sqrt(sum(abs2, alpha))
    program = FactoredInstructionProgram(
        3,
        3,
        ActiveState(3, alpha),
        FactoredInstruction[
            MeasurePrecomputedActivePauli(pauli_z(3, 0), 1, symbolic_bool(1), 1),
        ],
        3,
    )
    draws = [0.05, 0.35, 0.65, 0.95]
    expected = Bool[
        sample_measurements!(FactoredExecutorState(program; rng = SequenceRNG([draw])), program)[1] for
        draw in draws
    ]

    batch_runtime = BatchFactoredExecutorState(program; batches = 4, rng = SequenceRNG(draws))
    @test sample_measurements!(batch_runtime, program, 4)[:, 1] == expected

    shared_runtime = SharedActiveBatchExecutorState(program; batches = 4, rng = SequenceRNG(draws))
    @test sample_measurements!(shared_runtime, program, 4)[:, 1] == expected
end

@testset "BatchFactoredExecutorState AVX2 diagonal active measurement matches Julia fallback" begin
    alpha = ComplexF64[
        ComplexF64(sin(0.013 * basis), cos(0.017 * basis)) for basis in 1:1024
    ]
    alpha ./= sqrt(sum(abs2, alpha))
    program = FactoredInstructionProgram(
        10,
        10,
        ActiveState(10, alpha),
        FactoredInstruction[
            MeasurePrecomputedActivePauli(
                product(pauli_z(10, 3), pauli_z(10, 9)),
                1,
                symbolic_bool(1),
                1,
            ),
        ],
        10,
    )
    draws = [mod(0.137 * shot, 1.0) for shot in 1:128]
    runtime_avx2 =
        BatchFactoredExecutorState(program; batches = 128, kernel_backend = :c, rng = SequenceRNG(draws))
    runtime_julia =
        BatchFactoredExecutorState(program; batches = 128, kernel_backend = :c, rng = SequenceRNG(draws))

    old_disable = get(ENV, "SYMFT_DISABLE_AVX2_BATCH_ROT", nothing)
    ENV["SYMFT_DISABLE_AVX2_BATCH_ROT"] = "1"
    _reset_batch_avx2_state!()
    expected_records = falses(0, 0)
    expected_active = Matrix{ComplexF64}(undef, 0, 0)
    try
        records_julia = sample_measurements!(runtime_julia, program, 128)
        expected_records = copy(records_julia)
        expected_active = copy(runtime_julia.active[:, 1:512])
    finally
        if old_disable === nothing
            delete!(ENV, "SYMFT_DISABLE_AVX2_BATCH_ROT")
        else
            ENV["SYMFT_DISABLE_AVX2_BATCH_ROT"] = old_disable
        end
        _reset_batch_avx2_state!()
    end

    records_avx2 = sample_measurements!(runtime_avx2, program, 128)

    @test records_avx2 == expected_records
    @test runtime_avx2.active[:, 1:512] ≈ expected_active
end

@testset "BatchFactoredExecutorState AVX2 non-diagonal active measurement matches Julia fallback" begin
    alpha = ComplexF64[
        ComplexF64(sin(0.019 * basis), cos(0.011 * basis)) for basis in 1:1024
    ]
    alpha ./= sqrt(sum(abs2, alpha))
    program = FactoredInstructionProgram(
        10,
        10,
        ActiveState(10, alpha),
        FactoredInstruction[
            MeasurePrecomputedActivePauli(
                pauli_string("XIIIIIIIIY"),
                1,
                symbolic_bool(1),
                1,
            ),
        ],
        10,
    )
    draws = [mod(0.193 * shot, 1.0) for shot in 1:128]
    runtime_avx2 =
        BatchFactoredExecutorState(program; batches = 128, kernel_backend = :c, rng = SequenceRNG(draws))
    runtime_julia =
        BatchFactoredExecutorState(program; batches = 128, kernel_backend = :c, rng = SequenceRNG(draws))

    old_disable = get(ENV, "SYMFT_DISABLE_AVX2_BATCH_ROT", nothing)
    ENV["SYMFT_DISABLE_AVX2_BATCH_ROT"] = "1"
    _reset_batch_avx2_state!()
    expected_records = falses(0, 0)
    expected_active = Matrix{ComplexF64}(undef, 0, 0)
    try
        records_julia = sample_measurements!(runtime_julia, program, 128)
        expected_records = copy(records_julia)
        expected_active = copy(runtime_julia.active[:, 1:512])
    finally
        if old_disable === nothing
            delete!(ENV, "SYMFT_DISABLE_AVX2_BATCH_ROT")
        else
            ENV["SYMFT_DISABLE_AVX2_BATCH_ROT"] = old_disable
        end
        _reset_batch_avx2_state!()
    end

    records_avx2 = sample_measurements!(runtime_avx2, program, 128)

    @test records_avx2 == expected_records
    @test runtime_avx2.active[:, 1:512] ≈ expected_active
end

@testset "BatchFactoredExecutorState active rotation signs vary by shot" begin
    context = SymbolicContext()
    sign = fresh_bernoulli_condition!(context, 0.5)
    program = FactoredInstructionProgram(
        1,
        1,
        ActiveState(1),
        FactoredInstruction[
            ApplyPrecomputedActivePauliRotation(pauli_x(1, 0), pi / 4, symbolic_bool(sign)),
            MeasurePrecomputedActivePauli(neg(pauli_y(1, 0)), 2, symbolic_bool(2), 1),
        ],
        1,
        context,
    )
    runtime = BatchFactoredExecutorState(
        program;
        batches = 4,
        rng = SequenceRNG([0.6, 0.4, 0.8, 0.3, 0.5, 0.5, 0.5, 0.5]),
    )

    records = sample_measurements!(runtime, program, 4)

    @test records[:, 1] == Bool[false, true, false, true]
end

@testset "BatchFactoredExecutorState dormant promotion signs vary by shot" begin
    context = SymbolicContext()
    sign = fresh_bernoulli_condition!(context, 0.5)
    program = FactoredInstructionProgram(
        1,
        0,
        ActiveState(0),
        FactoredInstruction[
            PromoteDormantRotation(pi / 4, symbolic_bool(sign)),
            MeasurePrecomputedActivePauli(neg(pauli_y(1, 0)), 2, symbolic_bool(2), 1),
        ],
        1,
        context,
    )
    runtime = BatchFactoredExecutorState(
        program;
        batches = 4,
        rng = SequenceRNG([0.6, 0.4, 0.8, 0.3, 0.5, 0.5, 0.5, 0.5]),
    )

    records = sample_measurements!(runtime, program, 4)

    @test records[:, 1] == Bool[false, true, false, true]
end

@testset "BatchFactoredExecutorState AVX2 dormant promotion matches Julia fallback" begin
    program = FactoredInstructionProgram(
        10,
        9,
        ActiveState(9),
        FactoredInstruction[],
        10,
    )
    runtime_avx2 = BatchFactoredExecutorState(program; batches = 128, kernel_backend = :c)
    runtime_julia = BatchFactoredExecutorState(program; batches = 128, kernel_backend = :c)
    runtime_avx2.active_shots = 128
    runtime_julia.active_shots = 128
    sign_bits = UInt64[0xa55aa55aa55aa55a, 0x5aa55aa55aa55aa5]

    @inbounds for basis in 1:512
        for shot in 1:runtime_avx2.active_shots
            value = ComplexF64(sin(0.009 * shot + 0.017 * basis), cos(0.011 * shot - 0.013 * basis))
            runtime_avx2.active[shot, basis] = value
            runtime_julia.active[shot, basis] = value
        end
    end

    old_disable = get(ENV, "SYMFT_DISABLE_AVX2_BATCH_ROT", nothing)
    ENV["SYMFT_DISABLE_AVX2_BATCH_ROT"] = "1"
    _reset_batch_avx2_state!()
    try
        SymFT._batch_promote_first_dormant_rotation!(runtime_julia, pi / 8, sign_bits)
    finally
        if old_disable === nothing
            delete!(ENV, "SYMFT_DISABLE_AVX2_BATCH_ROT")
        else
            ENV["SYMFT_DISABLE_AVX2_BATCH_ROT"] = old_disable
        end
        _reset_batch_avx2_state!()
    end

    SymFT._batch_promote_first_dormant_rotation!(runtime_avx2, pi / 8, sign_bits)

    @test runtime_avx2.k == runtime_julia.k == 10
    @test runtime_avx2.ndormant == runtime_julia.ndormant == 0
    @test runtime_avx2.active[:, 1:1024] ≈ runtime_julia.active[:, 1:1024]
end

@testset "BatchFactoredExecutorState precomputed active rotation partial batch" begin
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

    records = sample_measurements(program, 5; batches = 3)

    @test records == trues(5, 1)
end

@testset "BatchFactoredExecutorState AVX2 active rotation matches Julia fallback" begin
    program = FactoredInstructionProgram(
        10,
        10,
        ActiveState(10),
        FactoredInstruction[],
        10,
    )
    runtime_avx2 = BatchFactoredExecutorState(program; batches = 128, kernel_backend = :c)
    runtime_julia = BatchFactoredExecutorState(program; batches = 128, kernel_backend = :c)
    runtime_avx2.active_shots = 128
    runtime_julia.active_shots = 128
    pauli = PauliString(10)
    SymFT._set_xbit!(pauli, 0)
    SymFT._set_xbit!(pauli, 1)
    SymFT._set_xbit!(pauli, 7)
    kernel = ApplyPrecomputedActivePauliRotation(pauli, pi / 8, SymbolicBool(false)).rotation_kernel
    sign_bits = UInt64[0xa55aa55aa55aa55a, 0x5aa55aa55aa55aa5]

    @inbounds for basis in 1:size(runtime_avx2.active, 2)
        for shot in 1:runtime_avx2.active_shots
            value = ComplexF64(sin(0.013 * shot + 0.017 * basis), cos(0.019 * shot - 0.011 * basis))
            runtime_avx2.active[shot, basis] = value
            runtime_julia.active[shot, basis] = value
        end
    end

    old_disable = get(ENV, "SYMFT_DISABLE_AVX2_BATCH_ROT", nothing)
    ENV["SYMFT_DISABLE_AVX2_BATCH_ROT"] = "1"
    _reset_batch_avx2_state!()
    try
        SymFT._batch_rotate_pauli!(runtime_julia, kernel, sign_bits)
    finally
        if old_disable === nothing
            delete!(ENV, "SYMFT_DISABLE_AVX2_BATCH_ROT")
        else
            ENV["SYMFT_DISABLE_AVX2_BATCH_ROT"] = old_disable
        end
        _reset_batch_avx2_state!()
    end

    SymFT._batch_rotate_pauli!(runtime_avx2, kernel, sign_bits)

    @test runtime_avx2.active ≈ runtime_julia.active
end

@testset "BatchFactoredExecutorState AVX2 real pair-flip rotation matches Julia fallback" begin
    program = FactoredInstructionProgram(
        10,
        10,
        ActiveState(10),
        FactoredInstruction[],
        10,
    )
    runtime_avx2 = BatchFactoredExecutorState(program; batches = 128, kernel_backend = :c)
    runtime_julia = BatchFactoredExecutorState(program; batches = 128, kernel_backend = :c)
    runtime_avx2.active_shots = 128
    runtime_julia.active_shots = 128
    pauli = pauli_string("XXYIIIIIII")
    kernel = ApplyPrecomputedActivePauliRotation(pauli, pi / 8, SymbolicBool(false)).rotation_kernel
    @test SymFT._can_rotate_real_pair_flip(kernel)
    sign_bits = UInt64[0xa55aa55aa55aa55a, 0x5aa55aa55aa55aa5]

    @inbounds for basis in 1:size(runtime_avx2.active, 2)
        for shot in 1:runtime_avx2.active_shots
            value = ComplexF64(sin(0.021 * shot + 0.015 * basis), cos(0.023 * shot - 0.009 * basis))
            runtime_avx2.active[shot, basis] = value
            runtime_julia.active[shot, basis] = value
        end
    end

    old_disable = get(ENV, "SYMFT_DISABLE_AVX2_BATCH_ROT", nothing)
    ENV["SYMFT_DISABLE_AVX2_BATCH_ROT"] = "1"
    _reset_batch_avx2_state!()
    try
        SymFT._batch_rotate_pauli!(runtime_julia, kernel, sign_bits)
    finally
        if old_disable === nothing
            delete!(ENV, "SYMFT_DISABLE_AVX2_BATCH_ROT")
        else
            ENV["SYMFT_DISABLE_AVX2_BATCH_ROT"] = old_disable
        end
        _reset_batch_avx2_state!()
    end

    SymFT._batch_rotate_pauli!(runtime_avx2, kernel, sign_bits)

    @test runtime_avx2.active ≈ runtime_julia.active
end

@testset "BatchFactoredExecutorState real pair-flip active rotation matches single shots" begin
    pauli = pauli_string("XXY")
    kernel = ApplyPrecomputedActivePauliRotation(pauli, pi / 8, SymbolicBool(false)).rotation_kernel
    @test SymFT._can_rotate_real_pair_flip(kernel)
    program = FactoredInstructionProgram(
        3,
        3,
        ActiveState(3),
        FactoredInstruction[],
        3,
    )
    runtime = BatchFactoredExecutorState(program; batches = 4)
    runtime.active_shots = 4
    sign_bits = UInt64[0b1010]

    @inbounds for basis in 1:size(runtime.active, 2)
        for shot in 1:runtime.active_shots
            runtime.active[shot, basis] =
                ComplexF64(sin(0.13 * basis + 0.07 * shot), cos(0.11 * basis - 0.03 * shot))
        end
    end
    expected = copy(runtime.active)

    SymFT._batch_rotate_pauli!(runtime, kernel, sign_bits)
    @inbounds for shot in 1:4
        state = ActiveState(3, vec(expected[shot, :]))
        rotate_pauli!(state, pauli, SymFT._batch_bit(sign_bits, shot) ? -pi / 8 : pi / 8)
        @test runtime.active[shot, :] ≈ state.alpha
    end
end

@testset "SharedActiveBatchExecutorState real pair-flip active rotation keeps packet shared" begin
    pauli = pauli_string("XXY")
    kernel = ApplyPrecomputedActivePauliRotation(pauli, pi / 8, SymbolicBool(false)).rotation_kernel
    program = FactoredInstructionProgram(
        3,
        3,
        ActiveState(3),
        FactoredInstruction[],
        3,
    )
    runtime = SharedActiveBatchExecutorState(program; batches = 4)
    runtime.base.active_shots = 4
    runtime.npackets = 1
    runtime.materialized = false
    @inbounds for basis in 1:size(runtime.active, 1)
        runtime.active[basis, 1] = ComplexF64(sin(0.17 * basis), cos(0.19 * basis))
    end
    before = copy(runtime.active[:, 1])
    sign_bits = UInt64[0]

    SymFT._shared_rotate_pauli!(runtime, kernel, sign_bits)

    expected = ActiveState(3, before)
    rotate_pauli!(expected, pauli, pi / 8)
    @test active_packet_count(runtime) == 1
    @test runtime.active[:, 1] ≈ expected.alpha
end

@testset "BatchFactoredExecutorState zero shots" begin
    program = FactoredInstructionProgram(
        1,
        1,
        ActiveState(1),
        FactoredInstruction[MeasureActiveLastZ(1, symbolic_bool(1), 1)],
        1,
    )

    records = sample_measurements(program, 0; batches = 4)

    @test size(records) == (0, 1)
end
