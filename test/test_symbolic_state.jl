@testset "ActivePauliFrame symbolic conjugation" begin
    cp = ConditionalPauliString(pauli_x(1, 0), 7)

    out = conjugate_by(cp, pauli_z(1, 0))
    @test out.pauli == pauli_z(1, 0)
    @test symbolic_sign(out) == symbolic_bool(7)
    @test sign_conditions(out) == [7]

    out = conjugate_by(cp, pauli_x(1, 0))
    @test out.pauli == pauli_x(1, 0)
    @test symbolic_sign(out) == SymbolicBool(false)
    @test isempty(sign_conditions(out))

    out = conjugate_by(cp, SymbolicPauliString(pauli_z(1, 0), SymbolicBool(true, [3, 7])))
    @test out.pauli == pauli_z(1, 0)
    @test symbolic_sign(out) == SymbolicBool(true, [3])
    @test sign_conditions(out) == [3]

    sp = SymbolicPauliString(pauli_x(1, 0), SymbolicBool(true, [4, 4]))
    @test sp.sign == SymbolicBool(true)
    @test isempty(sign_conditions(sp))
    @test constant_term(symbolic_sign(sp))

    frame = ActivePauliFrame(2)
    sx = add_pauli!(frame, pauli_x(2, 0))
    sz = add_pauli!(frame, pauli_z(2, 1))
    @test condition_id(sx) == 1
    @test condition_id(sz) == 2
    @test next_condition(frame) == 3

    out = conjugate_by(frame, pauli_string("ZY"))
    @test out.pauli == pauli_string("ZY")
    @test sign_conditions(out) == [1, 2]

    out = conjugate_by(frame, pauli_string("ZI"))
    @test sign_conditions(out) == [1]

    add_pauli!(frame, pauli_x(2, 0), 1)
    out = conjugate_by(frame, pauli_string("ZI"))
    @test isempty(sign_conditions(out))

    @test_throws DimensionMismatch add_pauli!(frame, pauli_x(3, 0))
    @test_throws DimensionMismatch conjugate_by(frame, pauli_x(1, 0))
    @test_throws ArgumentError ConditionalPauliString(pauli_x(1, 0), 0)
end

@testset "ActivePauliFrame packed word boundaries" begin
    frame = ActivePauliFrame(130)
    cp = add_pauli!(frame, pauli_x(130, 64))

    out = conjugate_by(frame, pauli_z(130, 64))
    @test sign_conditions(out) == [condition_id(cp)]
    @test out.pauli == pauli_z(130, 64)

    out = conjugate_by(frame, pauli_z(130, 65))
    @test isempty(sign_conditions(out))
end

@testset "SymbolicBool expressions" begin
    @test SymbolicBool(false) == SymbolicBool(false, Int[])
    @test symbolic_bool(3) == SymbolicBool(false, [3])
    @test SymbolicBool(true, [2, 1, 2, 4]) == SymbolicBool(true, [1, 4])

    expr = xor(SymbolicBool(true, [1, 3]), SymbolicBool(false, [3, 5]))
    @test constant_term(expr)
    @test condition_ids(expr) == [1, 5]
    @test !expr == SymbolicBool(false, [1, 5])
    @test xor(expr, true) == SymbolicBool(false, [1, 5])
    context = SymbolicContext()
    @test fresh_condition!(context) == 1
    @test fresh_symbolic_bool!(context) == symbolic_bool(2)
    @test next_condition(context) == 3

    b = fresh_bernoulli_bool!(context, 0.25)
    @test b == symbolic_bool(3)
    @test bernoulli_probability(context, 3) == 0.25
    @test isnothing(bernoulli_probability(context, 2))

    conditions = fresh_categorical_conditions!(
        context,
        Vector{Bool}[[false, false], [false, true], [true, false], [true, true]],
        [0.7, 0.1, 0.1, 0.1],
    )
    @test conditions == [4, 5]
    distribution = categorical_distribution(context, 4)
    @test distribution == categorical_distribution(context, 5)
    @test distribution.conditions == [4, 5]
    @test distribution.assignments == Vector{Bool}[[false, false], [false, true], [true, false], [true, true]]
    @test distribution.probabilities ≈ [0.7, 0.1, 0.1, 0.1]
    @test length(categorical_distributions(context)) == 1
    @test_throws ArgumentError fresh_bernoulli_condition!(context, -0.1)
    @test_throws ArgumentError fresh_categorical_conditions!(context, Vector{Bool}[[false], [true]], [0.25, 0.25])

    @test_throws ArgumentError symbolic_bool(0)
    @test_throws ArgumentError SymbolicContext(0)
end

@testset "DormantState symbolic basis" begin
    state = DormantState(3)
    @test ndormant(state) == 3
    @test nqubits(state) == 3
    @test dormant_bits(state) == [SymbolicBool(false), SymbolicBool(false), SymbolicBool(false)]
    @test sprint(show, state) == "|0,0,0>"

    s1 = assign_dormant_symbol!(state, 0)
    s2 = assign_dormant_symbol!(state, 2)
    @test condition_ids(s1) == [1]
    @test condition_ids(s2) == [2]
    @test next_condition(state) == 3
    @test dormant_bit(state, 0) == symbolic_bool(1)
    @test dormant_bit(state, 2) == symbolic_bool(2)

    set_dormant_bit!(state, 1, SymbolicBool(true, [2, 4]))
    @test next_condition(state) == 5
    @test dormant_bit(state, 1) == SymbolicBool(true, [2, 4])

    xor_dormant_bit!(state, 1, symbolic_bool(2))
    @test dormant_bit(state, 1) == SymbolicBool(true, [4])

    flip_dormant_bit!(state, 1)
    @test dormant_bit(state, 1) == SymbolicBool(false, [4])

    set_dormant_bit!(state, 0, false)
    @test dormant_bit(state, 0) == SymbolicBool(false)

    state2 = DormantState(5, 2)
    @test ndormant(state2) == 3

    state3 = DormantState(SymbolicBool[SymbolicBool(false, [7]), SymbolicBool(true)])
    @test next_condition(state3) == 8
    @test dormant_bits(state3) == [SymbolicBool(false, [7]), SymbolicBool(true)]

    copied = copy(state3)
    set_dormant_bit!(copied, 0, false)
    @test dormant_bit(state3, 0) == SymbolicBool(false, [7])
    @test dormant_bit(copied, 0) == SymbolicBool(false)
    @test symbolic_context(copied) === symbolic_context(state3)

    @test_throws BoundsError dormant_bit(state, 3)
    @test_throws ArgumentError DormantState(2, 3)
end

@testset "shared symbolic context" begin
    context = SymbolicContext()
    frame = ActivePauliFrame(2, context)
    dormant = DormantState(3, context)

    cp = add_pauli!(frame, pauli_x(2, 0))
    bit = assign_dormant_symbol!(dormant, 1)

    @test condition_id(cp) == 1
    @test bit == symbolic_bool(2)
    @test next_condition(context) == 3
    @test next_condition(frame) == 3
    @test next_condition(dormant) == 3
    @test symbolic_context(frame) === context
    @test symbolic_context(dormant) === context

    set_dormant_bit!(dormant, 0, SymbolicBool(false, [8]))
    @test next_condition(context) == 9

    cp = add_pauli!(frame, pauli_z(2, 1))
    @test condition_id(cp) == 9
    @test next_condition(context) == 10

    copied = copy(dormant)
    copied_bit = assign_dormant_symbol!(copied, 2)
    @test copied_bit == symbolic_bool(10)
    @test next_condition(frame) == 11
    @test dormant_bit(dormant, 2) == SymbolicBool(false)
end
