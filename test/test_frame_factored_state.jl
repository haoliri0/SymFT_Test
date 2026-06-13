@testset "FrameFactoredState constructors" begin
    state = FrameFactoredState(5, 2)
    @test nqubits(state) == 5
    @test nactive(state) == 2
    @test ndormant(state) == 3
    @test nqubits(state.clifford) == 5
    @test state.active.k == 2
    @test nqubits(state.active_frame) == 5
    @test ndormant(state.dormant) == 3
    @test symbolic_context(state) === symbolic_context(state.active_frame)
    @test symbolic_context(state) === symbolic_context(state.dormant)
    @test isempty(pending_operations(state))
    @test isempty(pending_rotations(state))
    @test isempty(pending_measurements(state))

    state = FrameFactoredState(4)
    @test nactive(state) == 0
    @test ndormant(state) == 4

    context = SymbolicContext()
    state = FrameFactoredState(3, 1, context)
    @test symbolic_context(state) === context

    @test_throws ArgumentError FrameFactoredState(2, 3)
    @test_throws DimensionMismatch FrameFactoredState(
        2,
        1,
        CliffordFrame(3),
        ActiveState(1),
        ActivePauliFrame(2, SymbolicContext()),
        DormantState(1, SymbolicContext()),
        SymbolicContext(),
    )

    context = SymbolicContext()
    @test_throws ArgumentError FrameFactoredState(
        2,
        1,
        CliffordFrame(2),
        ActiveState(1),
        ActivePauliFrame(2, SymbolicContext()),
        DormantState(1, context),
        context,
    )

    pending = PendingFactoredOperation[
        PendingPauliRotation(0.25, SymbolicPauliString(pauli_x(2, 0), [6])),
        PendingPauliMeasurement(SymbolicPauliString(pauli_z(2, 1), [8])),
    ]
    state = FrameFactoredState(
        2,
        1,
        CliffordFrame(2),
        ActiveState(1),
        ActivePauliFrame(2, context),
        DormantState(1, context),
        context,
        pending,
    )
    @test pending_operations(state) == pending
    @test pending_rotations(state) == [pending[1]]
    @test pending_measurements(state) == [pending[2]]
    @test next_condition(state) == 9
end

@testset "FrameFactoredState Clifford gates" begin
    state = FrameFactoredState(2, 1)
    left_H!(state, 0)
    @test preimage(state.clifford, pauli_x(2, 0)) == pauli_z(2, 0)

    left_CX!(state, 0, 1)
    reference = CliffordFrame(2)
    left_H!(reference, 0)
    left_CX!(reference, 0, 1)
    @test state.clifford.data == reference.data

    @test_throws BoundsError left_Z!(state, 2)
    @test_throws ArgumentError left_CZ!(state, 1, 1)
end

@testset "FrameFactoredState symbolic Pauli application" begin
    state = FrameFactoredState(4, 2)
    apply_pauli!(state, ConditionalPauliString(pauli_string("XYZI"), 7))

    @test length(state.active_frame.terms) == 1
    @test state.active_frame.terms[1] == ConditionalPauliString(pauli_string("XYZI"), 7)
    @test dormant_bits(state.dormant) == [SymbolicBool(false), SymbolicBool(false)]
    @test next_condition(state) == 8

    apply_pauli!(state, pauli_string("IIZX"), 8)
    @test length(state.active_frame.terms) == 2
    @test state.active_frame.terms[2] == ConditionalPauliString(pauli_string("IIZX"), 8)
    @test dormant_bit(state.dormant, 0) == SymbolicBool(false)
    @test dormant_bit(state.dormant, 1) == SymbolicBool(false)
    @test next_condition(state) == 9

    apply_pauli!(state, pauli_string("IIYI"), 9)
    @test length(state.active_frame.terms) == 3
    @test state.active_frame.terms[3] == ConditionalPauliString(pauli_string("IIYI"), 9)
    @test dormant_bit(state.dormant, 0) == SymbolicBool(false)
    @test dormant_bit(state.dormant, 1) == SymbolicBool(false)
    @test next_condition(state) == 10

    apply_pauli!(state, pauli_string("IIZI"), 12)
    @test length(state.active_frame.terms) == 4
    @test state.active_frame.terms[4] == ConditionalPauliString(pauli_string("IIZI"), 12)
    @test dormant_bit(state.dormant, 0) == SymbolicBool(false)
    @test next_condition(state) == 13

    @test_throws DimensionMismatch apply_pauli!(state, pauli_x(5, 0), 13)

    state = FrameFactoredState(3, 0)
    apply_pauli!(state, pauli_x(3, 2), 1)
    @test state.active_frame.terms == [ConditionalPauliString(pauli_x(3, 2), 1)]
    @test dormant_bit(state.dormant, 2) == SymbolicBool(false)

    state = FrameFactoredState(2, 2)
    apply_pauli!(state, pauli_string("ZY"), 2)
    @test state.active_frame.terms == [ConditionalPauliString(pauli_string("ZY"), 2)]
    @test ndormant(state.dormant) == 0

    state = FrameFactoredState(2, 0)
    apply_pauli!(state, pauli_x(2, 0), SymbolicBool(true, [3, 5]))
    @test length(state.active_frame.terms) == 3
    @test bernoulli_probability(symbolic_context(state), 1) == 1.0
    @test state.active_frame.terms[2] == ConditionalPauliString(pauli_x(2, 0), 3)
    @test state.active_frame.terms[3] == ConditionalPauliString(pauli_x(2, 0), 5)
end

@testset "FrameFactoredState Pauli through Clifford frame" begin
    state = FrameFactoredState(2, 1)
    left_CX!(state, 0, 1)

    apply_pauli!(state, pauli_x(2, 0), 3)
    @test length(state.active_frame.terms) == 1
    @test state.active_frame.terms[1] == ConditionalPauliString(pauli_string("XX"), 3)
    @test dormant_bit(state.dormant, 0) == SymbolicBool(false)

    state = FrameFactoredState(2, 1)
    left_H!(state, 0)
    apply_pauli!(state, pauli_x(2, 0), 4)
    @test state.active_frame.terms[1] == ConditionalPauliString(pauli_string("ZI"), 4)
    @test dormant_bits(state.dormant) == [SymbolicBool(false)]
end

@testset "FrameFactoredState pending factored operations" begin
    state = FrameFactoredState(3, 1)
    before_active = copy(state.active.alpha)
    before_dormant = dormant_bits(state.dormant)

    rotation = apply_pauli_rotation!(state, pauli_string("XZI"), 0.125)
    @test rotation.theta == 0.125
    @test rotation.pauli == SymbolicPauliString(pauli_string("XZI"))
    @test pending_operations(state) == PendingFactoredOperation[rotation]
    @test pending_rotations(state) == [rotation]
    @test isempty(pending_measurements(state))
    @test state.active.alpha == before_active
    @test dormant_bits(state.dormant) == before_dormant
    @test isempty(state.active_frame.terms)

    context = SymbolicContext()
    state = FrameFactoredState(2, 1, context)
    add_pauli!(state.active_frame, pauli_x(2, 0), 3)
    rotation = apply_pauli_rotation!(state, pauli_z(2, 0), 0.5)
    @test rotation.pauli == SymbolicPauliString(pauli_z(2, 0), [3])
    @test next_condition(state) == 4

    measurement = apply_pauli_measurement!(state, pauli_z(2, 0))
    @test measurement.pauli == SymbolicPauliString(pauli_z(2, 0), [3])
    @test pending_operations(state) == PendingFactoredOperation[rotation, measurement]
    @test pending_rotations(state) == [rotation]
    @test pending_measurements(state) == [measurement]

    state = FrameFactoredState(3, 1)
    apply_pauli!(state, pauli_string("IXI"), 5)
    rotation = apply_pauli_rotation!(state, pauli_string("IZI"), 0.25)
    @test rotation.pauli == SymbolicPauliString(pauli_string("IZI"), [5])
    measurement = apply_pauli_measurement!(state, pauli_string("IIX"))
    @test measurement.pauli == SymbolicPauliString(pauli_string("IIX"))
    apply_pauli!(state, pauli_string("IIZ"), 6)
    measurement = apply_pauli_measurement!(state, pauli_string("IIX"))
    @test measurement.pauli == SymbolicPauliString(pauli_string("IIX"), [6])
    @test dormant_bits(state.dormant) == [SymbolicBool(false), SymbolicBool(false)]

    state = FrameFactoredState(2, 1)
    left_CX!(state, 0, 1)
    rotation = apply_pauli_rotation!(state, pauli_x(2, 0), pi / 8)
    @test rotation.theta == Float64(pi / 8)
    @test rotation.pauli == SymbolicPauliString(pauli_string("XX"))

    measurement = apply_pauli_measurement!(state, pauli_z(2, 1))
    @test measurement.pauli == SymbolicPauliString(pauli_string("ZZ"))
    @test pending_operations(state) == PendingFactoredOperation[rotation, measurement]

    measurement = apply_pauli_measurement!(state, pauli_x(2, 1), SymbolicBool(true, [7]))
    @test measurement.pauli == SymbolicPauliString(pauli_string("IX"), SymbolicBool(true, [7]))

    state = FrameFactoredState(1, 1)
    rotation1 = apply_pauli_rotation!(state, pauli_x(1, 0), 0.1)
    measurement = apply_pauli_measurement!(state, pauli_z(1, 0))
    rotation2 = apply_pauli_rotation!(state, pauli_y(1, 0), 0.2)
    @test pending_operations(state) == PendingFactoredOperation[rotation1, measurement, rotation2]
    @test pending_rotations(state) == [rotation1, rotation2]
    @test pending_measurements(state) == [measurement]

    state = FrameFactoredState(3, 1)
    apply_pauli_rotation!(state, pauli_string("IZI"), 0.25)
    apply_pauli_measurement!(state, pauli_string("IIX"))

    apply_pauli!(state, pauli_string("IXI"), 5)
    @test pending_operations(state) == PendingFactoredOperation[
        PendingPauliRotation(0.25, SymbolicPauliString(pauli_string("IZI"))),
        PendingPauliMeasurement(SymbolicPauliString(pauli_string("IIX"))),
    ]
    @test state.active_frame.terms == [ConditionalPauliString(pauli_string("IXI"), 5)]
    @test dormant_bits(state.dormant) == [SymbolicBool(false), SymbolicBool(false)]

    apply_pauli!(state, pauli_string("IIZ"), 6)
    @test pending_operations(state) == PendingFactoredOperation[
        PendingPauliRotation(0.25, SymbolicPauliString(pauli_string("IZI"))),
        PendingPauliMeasurement(SymbolicPauliString(pauli_string("IIX"))),
    ]
    @test state.active_frame.terms == [
        ConditionalPauliString(pauli_string("IXI"), 5),
        ConditionalPauliString(pauli_string("IIZ"), 6),
    ]
    @test dormant_bits(state.dormant) == [SymbolicBool(false), SymbolicBool(false)]

    apply_pauli!(state, pauli_string("IXI"), 5)
    @test pending_operations(state) == PendingFactoredOperation[
        PendingPauliRotation(0.25, SymbolicPauliString(pauli_string("IZI"))),
        PendingPauliMeasurement(SymbolicPauliString(pauli_string("IIX"))),
    ]
    @test state.active_frame.terms == [
        ConditionalPauliString(pauli_string("IXI"), 5),
        ConditionalPauliString(pauli_string("IIZ"), 6),
        ConditionalPauliString(pauli_string("IXI"), 5),
    ]
    @test dormant_bits(state.dormant) == [SymbolicBool(false), SymbolicBool(false)]

    state = FrameFactoredState(2, 0)
    apply_pauli_rotation!(state, pauli_x(2, 0), 0.125)
    apply_pauli_rotation!(state, pauli_y(2, 0), 0.25)
    apply_pauli!(state, pauli_y(2, 0), 7)
    @test pending_operations(state) == PendingFactoredOperation[
        PendingPauliRotation(0.125, SymbolicPauliString(pauli_x(2, 0))),
        PendingPauliRotation(0.25, SymbolicPauliString(pauli_y(2, 0))),
    ]
    @test state.active_frame.terms == [ConditionalPauliString(pauli_y(2, 0), 7)]
    @test dormant_bit(state.dormant, 0) == SymbolicBool(false)

    @test_throws DimensionMismatch apply_pauli_rotation!(state, pauli_x(3, 0), 0.1)
    @test_throws DimensionMismatch apply_pauli_measurement!(state, pauli_x(3, 0))
end
