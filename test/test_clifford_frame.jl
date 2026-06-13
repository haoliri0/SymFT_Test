@testset "single-qubit left Clifford gates" begin
    x = pauli_x(1, 0)
    y = pauli_y(1, 0)
    z = pauli_z(1, 0)

    cf = CliffordFrame(1)
    left_H!(cf, 0)
    @test preimage(cf, x) == z
    @test preimage(cf, z) == x

    cf = CliffordFrame(1)
    left_S!(cf, 0)
    @test preimage(cf, x) == neg(y)
    @test preimage(cf, z) == z

    cf = CliffordFrame(1)
    left_SDG!(cf, 0)
    @test preimage(cf, x) == y
    @test preimage(cf, z) == z

    cf = CliffordFrame(1)
    left_X!(cf, 0)
    @test preimage(cf, x) == x
    @test preimage(cf, z) == neg(z)

    cf = CliffordFrame(1)
    left_Z!(cf, 0)
    @test preimage(cf, x) == neg(x)
    @test preimage(cf, z) == z
end

@testset "two-qubit left Clifford gates" begin
    x0 = pauli_x(2, 0)
    x1 = pauli_x(2, 1)
    z0 = pauli_z(2, 0)
    z1 = pauli_z(2, 1)

    cf = CliffordFrame(2)
    left_CX!(cf, 0, 1)
    @test preimage(cf, x0) == product(x0, x1)
    @test preimage(cf, x1) == x1
    @test preimage(cf, z0) == z0
    @test preimage(cf, z1) == product(z0, z1)

    cf = CliffordFrame(2)
    left_CZ!(cf, 0, 1)
    @test preimage(cf, x0) == product(x0, z1)
    @test preimage(cf, x1) == product(z0, x1)
    @test preimage(cf, z0) == z0
    @test preimage(cf, z1) == z1

    cf = CliffordFrame(2)
    left_SWAP!(cf, 0, 1)
    @test preimage(cf, x0) == x1
    @test preimage(cf, x1) == x0
    @test preimage(cf, z0) == z1
    @test preimage(cf, z1) == z0
end

@testset "single-qubit right Clifford gates" begin
    x = pauli_x(1, 0)
    y = pauli_y(1, 0)
    z = pauli_z(1, 0)

    cf = CliffordFrame(1)
    right_H!(cf, 0)
    @test preimage(cf, x) == z
    @test preimage(cf, z) == x

    cf = CliffordFrame(1)
    right_S!(cf, 0)
    @test preimage(cf, x) == neg(y)
    @test preimage(cf, y) == x
    @test preimage(cf, z) == z

    cf = CliffordFrame(1)
    right_SDG!(cf, 0)
    @test preimage(cf, x) == y
    @test preimage(cf, y) == neg(x)
    @test preimage(cf, z) == z

    cf = CliffordFrame(1)
    right_X!(cf, 0)
    @test preimage(cf, x) == x
    @test preimage(cf, z) == neg(z)

    cf = CliffordFrame(1)
    right_Z!(cf, 0)
    @test preimage(cf, x) == neg(x)
    @test preimage(cf, z) == z
end

@testset "two-qubit right Clifford gates" begin
    x0 = pauli_x(2, 0)
    x1 = pauli_x(2, 1)
    z0 = pauli_z(2, 0)
    z1 = pauli_z(2, 1)

    cf = CliffordFrame(2)
    right_CX!(cf, 0, 1)
    @test preimage(cf, x0) == product(x0, x1)
    @test preimage(cf, x1) == x1
    @test preimage(cf, z0) == z0
    @test preimage(cf, z1) == product(z0, z1)

    cf = CliffordFrame(2)
    right_CZ!(cf, 0, 1)
    @test preimage(cf, x0) == product(x0, z1)
    @test preimage(cf, x1) == product(z0, x1)
    @test preimage(cf, z0) == z0
    @test preimage(cf, z1) == z1

    cf = CliffordFrame(2)
    right_SWAP!(cf, 0, 1)
    @test preimage(cf, x0) == x1
    @test preimage(cf, x1) == x0
    @test preimage(cf, z0) == z1
    @test preimage(cf, z1) == z0
end

@testset "left composition order" begin
    cf = CliffordFrame(1)
    left_H!(cf, 0)
    left_S!(cf, 0)
    @test preimage(cf, pauli_x(1, 0)) == pauli_y(1, 0)
end

@testset "right composition order" begin
    cf = CliffordFrame(1)
    right_H!(cf, 0)
    right_S!(cf, 0)
    @test preimage(cf, pauli_x(1, 0)) == pauli_z(1, 0)
    @test preimage(cf, pauli_z(1, 0)) == neg(pauli_y(1, 0))
end

@testset "right gates transform existing rows" begin
    expect_right_matches_reference(right_H!, left_H!, 3, 1)
    expect_right_matches_reference(right_S!, left_S!, 3, 1)
    expect_right_matches_reference(right_SDG!, left_SDG!, 3, 1)
    expect_right_matches_reference(right_X!, left_X!, 3, 1)
    expect_right_matches_reference(right_Z!, left_Z!, 3, 1)
    expect_right_matches_reference(right_CX!, left_CX!, 3, 0, 2)
    expect_right_matches_reference(right_CZ!, left_CZ!, 3, 0, 2)
    expect_right_matches_reference(right_SWAP!, left_SWAP!, 3, 0, 2)
end

@testset "0-based qubit validation" begin
    @test_throws BoundsError left_H!(CliffordFrame(1), 1)
    @test_throws BoundsError right_H!(CliffordFrame(1), 1)
    @test_throws ArgumentError left_CX!(CliffordFrame(2), 0, 0)
    @test_throws ArgumentError right_CX!(CliffordFrame(2), 0, 0)
end

@testset "packed Int64 word boundaries" begin
    @test sprint(show, PauliString(0)) == "I"

    ps = pauli_x(65, 63)
    @test xbit(ps, 63)
    @test !xbit(ps, 64)

    cf = CliffordFrame(65)
    left_H!(cf, 63)
    @test preimage(cf, pauli_x(65, 63)) == pauli_z(65, 63)

    expect_right_matches_reference(right_CX!, left_CX!, 65, 63, 64)
end

