@testset "PauliString algebra" begin
    @test pauli_x(1, 0) * pauli_x(1, 0) == pauli_identity(1)
    @test pauli_y(1, 0) * pauli_y(1, 0) == pauli_identity(1)
    @test pauli_x(1, 0) * pauli_z(1, 0) == phase_shift!(copy(pauli_y(1, 0)), 3)
    @test pauli_z(1, 0) * pauli_x(1, 0) == phase_shift!(copy(pauli_y(1, 0)), 1)
    @test pauli_anticommutes(pauli_x(1, 0), pauli_z(1, 0))
    @test !pauli_anticommutes(pauli_y(1, 0), pauli_y(1, 0))

    ps = pauli_string("IXYZ")
    @test !xbit(ps, 0) && !zbit(ps, 0)
    @test xbit(ps, 1) && !zbit(ps, 1)
    @test xbit(ps, 2) && zbit(ps, 2)
    @test !xbit(ps, 3) && zbit(ps, 3)
end
