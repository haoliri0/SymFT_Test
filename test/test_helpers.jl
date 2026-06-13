using Test
using SymFT

neg(p::PauliString) = phase_shift!(copy(p), 2)

function product(ps::PauliString...)
    isempty(ps) && error("product needs at least one Pauli string")
    acc = pauli_identity(nqubits(ps[1]))
    for p in ps
        acc = acc * p
    end
    return acc
end

function expect_right_matches_reference(right_gate!, left_gate!, n::Integer, args...)
    old = CliffordFrame(n)
    if n >= 1
        left_H!(old, 0)
        left_S!(old, n - 1)
    end
    if n >= 2
        left_CX!(old, 0, n - 1)
        left_CZ!(old, 0, 1)
    end

    new = copy(old)
    right_gate!(new, args...)

    g = CliffordFrame(n)
    left_gate!(g, args...)

    probes = PauliString[]
    for q in 0:(n - 1)
        push!(probes, pauli_x(n, q))
        push!(probes, pauli_z(n, q))
    end
    for p in probes
        @test preimage(new, p) == preimage(g, preimage(old, p))
    end
end
