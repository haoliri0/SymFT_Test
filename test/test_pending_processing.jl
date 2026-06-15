@testset "PendingFactoredState processing boundary" begin
    state = PendingFactoredState(2, 1)
    @test !has_pending_operations(state)
    @test pending_operation_count(state) == 0
    @test max_active_qubits(state) == 1
    @test process_next_pending_operation!(state) === nothing
    @test process_pending_operations!(state) == FactoredInstruction[]
    @test factored_instructions(state) == FactoredInstruction[]
    @test hasmethod(process_pending_operation!, Tuple{PendingFactoredState,PendingPauliRotation})
    @test hasmethod(process_pending_operation!, Tuple{PendingFactoredState,PendingPauliMeasurement})
    @test hasmethod(process_pending_rotation!, Tuple{PendingFactoredState,PendingPauliRotation})
    @test hasmethod(process_pending_measurement!, Tuple{PendingFactoredState,PendingPauliMeasurement})

    full_state = FrameFactoredState(2, 1)
    left_H!(full_state, 0)
    add_pauli!(full_state.active_frame, pauli_x(2, 0), 3)
    apply_pauli_measurement!(full_state, pauli_z(2, 1))
    state = PendingFactoredState(full_state)
    @test nqubits(state) == 2
    @test nactive(state) == 1
    @test ndormant(state) == 1
    @test max_active_qubits(state) == 1
    @test :clifford ∉ propertynames(state)
    @test :active_frame ∉ propertynames(state)
    @test pending_operations(state) == pending_operations(full_state)
    @test symbolic_context(state) === symbolic_context(full_state)

    program = factored_instruction_program(state)
    @test program.n == 2
    @test program.initial_k == 1
    @test program.max_k == 1
    @test max_active_qubits(program) == 1
    @test program.initial_active.alpha == state.active.alpha
    @test isempty(program.instructions)
end

@testset "PendingFactoredState pending measurement planning" begin
    state = PendingFactoredState(3, 1)
    measurement = PendingPauliMeasurement(SymbolicPauliString(pauli_string("IZZ"), SymbolicBool(true, [2])))
    instruction = process_pending_measurement!(state, measurement)
    @test instruction == RecordMeasurement(SymbolicBool(true, [2]), 1)
    @test factored_instructions(state) == FactoredInstruction[instruction]
    @test pending_operation_count(state) == 0
    @test nactive(state) == 1

    state = PendingFactoredState(1, 0)
    negative_z = phase_shift!(pauli_z(1, 0), 2)
    instruction = process_pending_measurement!(state, PendingPauliMeasurement(SymbolicPauliString(negative_z)))
    @test instruction == RecordMeasurement(SymbolicBool(true), 1)

    state = PendingFactoredState(1, 0)
    instruction = process_pending_measurement!(
        state,
        PendingPauliMeasurement(SymbolicPauliString(pauli_z(1, 0)), 3, 9),
    )
    @test instruction == RecordMeasurement(SymbolicBool(false), 3, 9)
    @test next_condition(state) == 10

    state = PendingFactoredState(1, 1)
    state.active = ActiveState(1, ComplexF64[sqrt(0.25), sqrt(0.75)])
    before_alpha = copy(state.active.alpha)
    instruction = process_pending_measurement!(state, PendingPauliMeasurement(SymbolicPauliString(pauli_z(1, 0))))
    @test instruction == MeasurePrecomputedActivePauli(pauli_z(1, 0), 1, symbolic_bool(1), 1)
    @test nactive(state) == 0
    @test ndormant(state) == 1
    @test max_active_qubits(state) == 1
    @test state.active.alpha == before_alpha
    @test dormant_bits(state.dormant) == [SymbolicBool(false)]

    state = PendingFactoredState(1, 1)
    push!(state.pending_operations, PendingPauliMeasurement(SymbolicPauliString(pauli_z(1, 0))))
    push!(state.pending_operations, PendingPauliRotation(0.125, SymbolicPauliString(pauli_z(1, 0))))
    instruction = process_next_pending_operation!(state)
    @test instruction == MeasurePrecomputedActivePauli(pauli_z(1, 0), 1, symbolic_bool(1), 1)
    @test pending_operations(state) == PendingFactoredOperation[
        PendingPauliRotation(0.125, SymbolicPauliString(pauli_z(1, 0), [1])),
    ]

    state = PendingFactoredState(2, 1)
    instruction = process_pending_measurement!(
        state,
        PendingPauliMeasurement(SymbolicPauliString(pauli_string("ZI"), SymbolicBool(false, [5]))),
    )
    @test instruction == MeasurePrecomputedActivePauli(pauli_z(1, 0), 6, SymbolicBool(false, [5, 6]), 1)
    @test nactive(state) == 0
    @test dormant_bits(state.dormant) == [SymbolicBool(false), SymbolicBool(false)]

    state = PendingFactoredState(2, 2)
    push!(state.pending_operations, PendingPauliMeasurement(SymbolicPauliString(pauli_x(2, 1))))
    push!(state.pending_operations, PendingPauliRotation(0.125, SymbolicPauliString(pauli_z(2, 1))))
    instructions = process_next_pending_operation!(state)
    @test instructions == MeasurePrecomputedActivePauli(pauli_x(2, 1), 1, symbolic_bool(1), 1)
    @test nactive(state) == 1
    @test ndormant(state) == 1
    @test state.active.alpha == ComplexF64[1.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im]
    @test pending_operations(state) == PendingFactoredOperation[
        PendingPauliRotation(0.125, SymbolicPauliString(pauli_x(2, 1))),
    ]

    state = PendingFactoredState(2, 2)
    push!(state.pending_operations, PendingPauliMeasurement(SymbolicPauliString(pauli_x(2, 0))))
    push!(state.pending_operations, PendingPauliRotation(0.125, SymbolicPauliString(pauli_z(2, 1))))
    instructions = process_next_pending_operation!(state)
    @test instructions == MeasurePrecomputedActivePauli(pauli_x(2, 0), 1, symbolic_bool(1), 1)
    @test pending_operations(state) == PendingFactoredOperation[
        PendingPauliRotation(0.125, SymbolicPauliString(pauli_z(2, 0))),
    ]

    state = PendingFactoredState(1, 1)
    instruction = process_pending_measurement!(state, PendingPauliMeasurement(SymbolicPauliString(pauli_y(1, 0))))
    @test factored_instructions(state) == FactoredInstruction[
        MeasurePrecomputedActivePauli(pauli_y(1, 0), 1, symbolic_bool(1), 1),
    ]
    @test instruction == MeasurePrecomputedActivePauli(pauli_y(1, 0), 1, symbolic_bool(1), 1)

    state = PendingFactoredState(2, 0)
    push!(state.pending_operations, PendingPauliRotation(0.25, SymbolicPauliString(pauli_z(2, 1))))
    instruction = process_pending_measurement!(
        state,
        PendingPauliMeasurement(SymbolicPauliString(pauli_x(2, 1))),
    )
    @test instruction == IntroduceDormantMeasurementBranch(1, symbolic_bool(1), 1)
    @test dormant_bits(state.dormant) == [SymbolicBool(false), SymbolicBool(false)]
    @test pending_operations(state) == PendingFactoredOperation[
        PendingPauliRotation(0.25, SymbolicPauliString(pauli_string("IX"))),
    ]

    state = PendingFactoredState(1, 0)
    instruction = process_pending_measurement!(state, PendingPauliMeasurement(SymbolicPauliString(pauli_y(1, 0))))
    @test instruction == IntroduceDormantMeasurementBranch(1, SymbolicBool(true, [1]), 1)
    @test dormant_bits(state.dormant) == [SymbolicBool(false)]

    state = PendingFactoredState(2, 1)
    push!(state.pending_operations, PendingPauliRotation(0.125, SymbolicPauliString(pauli_z(2, 0))))
    instruction = process_pending_measurement!(
        state,
        PendingPauliMeasurement(SymbolicPauliString(pauli_string("XX"))),
    )
    @test instruction == IntroduceDormantMeasurementBranch(1, symbolic_bool(1), 1)
    @test pending_operations(state) == PendingFactoredOperation[
        PendingPauliRotation(0.125, SymbolicPauliString(pauli_string("ZX"))),
    ]

    state = PendingFactoredState(2, 1)
    instruction = process_pending_measurement!(
        state,
        PendingPauliMeasurement(SymbolicPauliString(pauli_string("ZX"))),
    )
    @test factored_instructions(state) == FactoredInstruction[
        ApplyActiveBasisChange(:H, 0),
        IntroduceDormantMeasurementBranch(1, symbolic_bool(1), 1),
    ]
    @test instruction == IntroduceDormantMeasurementBranch(1, symbolic_bool(1), 1)
    @test state.active.alpha == ComplexF64[1.0 + 0.0im, 0.0 + 0.0im]

    full_state = FrameFactoredState(1, 0)
    apply_pauli_measurement!(full_state, pauli_z(1, 0))
    state = PendingFactoredState(full_state)
    instruction = process_next_pending_operation!(state)
    @test instruction == RecordMeasurement(SymbolicBool(false), 1)
    @test !has_pending_operations(state)

    bad_pauli = phase_shift!(pauli_x(1, 0), 1)
    @test_throws ArgumentError process_pending_measurement!(
        PendingFactoredState(1, 1),
        PendingPauliMeasurement(SymbolicPauliString(bad_pauli)),
    )
end

@testset "PendingFactoredState pending rotation planning" begin
    theta = 0.125

    state = PendingFactoredState(1, 1)
    before_alpha = copy(state.active.alpha)
    instruction = process_pending_rotation!(state, PendingPauliRotation(theta, SymbolicPauliString(pauli_x(1, 0))))
    @test instruction == ApplyPrecomputedActivePauliRotation(pauli_x(1, 0), theta, SymbolicBool(false))
    @test state.active.alpha == before_alpha

    state = PendingFactoredState(8, 8)
    instruction = process_pending_rotation!(state, PendingPauliRotation(theta, SymbolicPauliString(pauli_x(8, 7))))
    @test instruction == ApplyPrecomputedActivePauliRotation(pauli_x(8, 7), theta, SymbolicBool(false))

    state = PendingFactoredState(1, 1)
    negative_x = phase_shift!(pauli_x(1, 0), 2)
    instruction = process_pending_rotation!(state, PendingPauliRotation(theta, SymbolicPauliString(negative_x)))
    @test instruction == ApplyPrecomputedActivePauliRotation(pauli_x(1, 0), theta, SymbolicBool(true))

    state = PendingFactoredState(2, 1)
    instruction = process_pending_rotation!(
        state,
        PendingPauliRotation(theta, SymbolicPauliString(pauli_string("XZ"))),
    )
    @test instruction == ApplyPrecomputedActivePauliRotation(pauli_x(1, 0), theta, SymbolicBool(false))
    @test nactive(state) == 1
    @test dormant_bits(state.dormant) == [SymbolicBool(false)]

    state = PendingFactoredState(2, 1)
    rotation = PendingPauliRotation(theta, SymbolicPauliString(pauli_string("XZ")))
    measurement = PendingPauliMeasurement(SymbolicPauliString(pauli_z(2, 0)))
    push!(state.pending_operations, rotation)
    push!(state.pending_operations, measurement)
    instructions = process_pending_operations!(state)
    @test instructions == FactoredInstruction[
        ApplyPrecomputedActivePauliRotation(pauli_x(1, 0), theta, SymbolicBool(false)),
        MeasurePrecomputedActivePauli(pauli_z(1, 0), 1, symbolic_bool(1), 1),
    ]
    @test pending_operations(state) == PendingFactoredOperation[]

    state = PendingFactoredState(1, 0)
    instruction = process_pending_rotation!(state, PendingPauliRotation(theta, SymbolicPauliString(pauli_x(1, 0))))
    @test instruction == PromoteDormantRotation(theta, SymbolicBool(false))
    @test nactive(state) == 1
    @test ndormant(state) == 0
    @test max_active_qubits(state) == 1
    @test state.active.alpha == ComplexF64[1.0 + 0.0im]

    state = PendingFactoredState(1, 0)
    instruction = process_pending_rotation!(state, PendingPauliRotation(theta, SymbolicPauliString(pauli_y(1, 0))))
    @test instruction == PromoteDormantRotation(theta, SymbolicBool(true))
    @test nactive(state) == 1

    state = PendingFactoredState(1, 0)
    instruction = process_pending_rotation!(
        state,
        PendingPauliRotation(theta, SymbolicPauliString(pauli_x(1, 0), [7])),
    )
    @test instruction == PromoteDormantRotation(theta, symbolic_bool(7))

    state = PendingFactoredState(2, 0)
    instruction = process_pending_rotation!(
        state,
        PendingPauliRotation(theta, SymbolicPauliString(pauli_string("XX"))),
    )
    @test instruction == PromoteDormantRotation(theta, SymbolicBool(false))
    @test nactive(state) == 1
    @test ndormant(state) == 1
    @test dormant_bits(state.dormant) == [SymbolicBool(false)]

    state = PendingFactoredState(2, 1)
    rotation = PendingPauliRotation(theta, SymbolicPauliString(pauli_string("XX")))
    measurement = PendingPauliMeasurement(SymbolicPauliString(pauli_z(2, 0)))
    push!(state.pending_operations, rotation)
    push!(state.pending_operations, measurement)
    instruction = process_next_pending_operation!(state)
    @test instruction == PromoteDormantRotation(theta, SymbolicBool(false))
    @test nactive(state) == 2
    @test ndormant(state) == 0
    @test max_active_qubits(state) == 2
    @test state.active.alpha == ComplexF64[1.0 + 0.0im, 0.0 + 0.0im]
    @test pending_operations(state) == PendingFactoredOperation[
        PendingPauliMeasurement(SymbolicPauliString(pauli_string("ZZ"))),
    ]

    program = plan_factored_updates!(state)
    @test program isa FactoredInstructionProgram
    @test program.n == 2
    @test program.initial_k == 1
    @test program.max_k == 2
    @test max_active_qubits(program) == 2
    @test program.initial_active.alpha == ComplexF64[1.0 + 0.0im, 0.0 + 0.0im]
    @test program.instructions == FactoredInstruction[
        PromoteDormantRotation(theta, SymbolicBool(false)),
        MeasurePrecomputedActivePauli(pauli_string("ZZ"), 1, symbolic_bool(1), 1),
    ]
    @test !has_pending_operations(state)
end
