@testset "ActiveState constructors" begin
    st = ActiveState()
    @test st.k == 0
    @test st.alpha == ComplexF64[1.0 + 0.0im]

    st = ActiveState(2)
    @test st.k == 2
    @test st.alpha == ComplexF64[1.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im]

    @test_throws ArgumentError ActiveState(2, ComplexF64[1.0 + 0.0im])
end

@testset "ActiveState Pauli action" begin
    out = zeros(ComplexF64, 4)
    alpha = ComplexF64[1.0 + 0.0im, 2.0 + 0.0im, 3.0 + 0.0im, 4.0 + 0.0im]

    apply_pauli!(out, pauli_x(2, 0), alpha)
    @test out == ComplexF64[2.0 + 0.0im, 1.0 + 0.0im, 4.0 + 0.0im, 3.0 + 0.0im]

    apply_pauli!(out, pauli_z(2, 1), alpha)
    @test out == ComplexF64[1.0 + 0.0im, 2.0 + 0.0im, -3.0 + 0.0im, -4.0 + 0.0im]

    out1 = zeros(ComplexF64, 2)
    apply_pauli!(out1, pauli_y(1, 0), ComplexF64[1.0 + 0.0im, 0.0 + 0.0im])
    @test out1 == ComplexF64[0.0 + 0.0im, 0.0 + 1.0im]

    apply_pauli!(out1, pauli_y(1, 0), ComplexF64[0.0 + 0.0im, 1.0 + 0.0im])
    @test out1 == ComplexF64[0.0 - 1.0im, 0.0 + 0.0im]

    st = ActiveState(2, alpha)
    apply_pauli!(st, pauli_string("XY"))
    expected = zeros(ComplexF64, 4)
    apply_pauli!(expected, pauli_string("XY"), alpha)
    @test st.alpha == expected

    @test_throws ArgumentError apply_pauli!(alpha, pauli_x(2, 0), alpha)
    @test_throws DimensionMismatch apply_pauli!(ActiveState(1), pauli_x(2, 0))
end

@testset "ActiveState Pauli rotations" begin
    st = ActiveState(1)
    rotate_pauli!(st, pauli_x(1, 0), pi / 4)
    @test st.alpha ≈ ComplexF64[cos(pi / 4), -1im * sin(pi / 4)]

    st = ActiveState(1, ComplexF64[3.0 + 4.0im, 5.0 - 2.0im])
    theta = 0.31
    before = copy(st.alpha)
    rotate_pauli!(st, pauli_z(1, 0), theta)
    @test st.alpha ≈ ComplexF64[exp(-1im * theta) * before[1], exp(1im * theta) * before[2]]

    st = ActiveState(1)
    rotate_pauli!(st, pauli_y(1, 0), pi / 6)
    @test st.alpha ≈ ComplexF64[cos(pi / 6), sin(pi / 6)]

    st = ActiveState(2, ComplexF64[1.0 + 0.0im, 2.0 - 1.0im, -0.5 + 0.25im, 0.0 + 3.0im])
    p = pauli_string("XY")
    theta = 0.125
    rotate_pauli!(st, p, theta)
    pauli_rotation!(st, p, -theta)
    @test st.alpha ≈ ComplexF64[1.0 + 0.0im, 2.0 - 1.0im, -0.5 + 0.25im, 0.0 + 3.0im]

    @test_throws ArgumentError rotate_pauli!(ActiveState(1), pauli_x(1, 0) * pauli_z(1, 0), 0.1)

    st = ActiveState(3, ComplexF64[complex(i, -i / 10) for i in 1:8])
    p = pauli_string("XYZ")
    rotate_pauli!(st, p, 0.1)
    @test (@allocated rotate_pauli!(st, p, 0.2)) == 0

    st = ActiveState(1, ComplexF64[0.6 + 0.1im, -0.2 + 0.7im])
    resize!(st.alpha, 8)
    st.alpha[3:8] .= ComplexF64[10 + im, 11 + im, 12 + im, 13 + im, 14 + im, 15 + im]
    before_tail = copy(st.alpha[3:8])
    expected = ActiveState(1, st.alpha[1:2])
    rotate_pauli!(expected, pauli_y(1, 0), 0.37)
    rotate_pauli!(st, pauli_y(1, 0), 0.37)
    @test st.alpha[1:2] ≈ expected.alpha
    @test st.alpha[3:8] == before_tail

    st = ActiveState(2, ComplexF64[0.2 + 0.1im, 0.3 - 0.4im, -0.5 + 0.2im, 0.7 + 0.9im])
    p = pauli_string("XZ")
    plan = SymFT.ActivePauliRotationPlan(p)
    expected = ActiveState(2, copy(st.alpha))
    rotate_pauli!(expected, p, -0.41)
    rotate_pauli!(st, plan, cos(-0.41), ComplexF64(0.0, -sin(-0.41)))
    @test st.alpha ≈ expected.alpha

    st = ActiveState(2, ComplexF64[0.2 + 0.1im, 0.3 - 0.4im, -0.5 + 0.2im, 0.7 + 0.9im])
    p = pauli_string("ZZ")
    plan = SymFT.ActivePauliRotationPlan(p)
    expected = ActiveState(2, copy(st.alpha))
    rotate_pauli!(expected, p, 0.23)
    rotate_pauli!(st, plan, cos(0.23), ComplexF64(0.0, -sin(0.23)))
    @test st.alpha ≈ expected.alpha

    st = ActiveState(3, ComplexF64[complex(sin(i), cos(i / 3)) for i in 1:8])
    p = pauli_string("XZY")
    kernel = SymFT.PrecomputedActivePauliRotationKernel(SymFT.ActivePauliAction(p), 0.19)
    expected = ActiveState(3, copy(st.alpha))
    rotate_pauli!(expected, p, 0.19)
    rotate_pauli!(st, kernel, false)
    @test st.alpha ≈ expected.alpha

    st = ActiveState(3, ComplexF64[complex(cos(i / 5), sin(i / 7)) for i in 1:8])
    expected = ActiveState(3, copy(st.alpha))
    rotate_pauli!(expected, p, -0.19)
    rotate_pauli!(st, kernel, true)
    @test st.alpha ≈ expected.alpha

    st = ActiveState(3, ComplexF64[complex(sin(i / 3), cos(i / 4)) for i in 1:8])
    p = pauli_string("XXY")
    kernel = SymFT.PrecomputedActivePauliRotationKernel(SymFT.ActivePauliAction(p), 0.23)
    @test SymFT._can_rotate_real_pair_flip(kernel)
    expected = ActiveState(3, copy(st.alpha))
    rotate_pauli!(expected, p, -0.23)
    rotate_pauli!(st, kernel, true)
    @test st.alpha ≈ expected.alpha
end
