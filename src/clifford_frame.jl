"""
    CliffordFrame(nqubits)

Packed Clifford frame storing preimages of physical Pauli generators. Row
`q + 1` stores `C' * X_q * C`; row `nqubits + q + 1` stores
`C' * Z_q * C`, where `q` is a 0-based physical qubit index. Each row has the
same packed layout as `PauliString`.
"""
struct CliffordFrame
    nqubits::Int
    nwords::Int
    data::Matrix{Int64}

    function CliffordFrame(nqubits::Integer, data::AbstractMatrix{<:Integer})
        n = _checked_nqubits(nqubits)
        nw = _nwords(n)
        size(data) == (2 * n, 2 * nw + 1) ||
            throw(ArgumentError("packed Clifford frame has size $(size(data)); expected $((2 * n, 2 * nw + 1))"))
        packed = Matrix{Int64}(data)
        phase_col = 2 * nw + 1
        if nw > 0 && (n & 63) != 0
            mask = _tail_mask(n)
            @inbounds for row in axes(packed, 1)
                packed[row, nw] &= mask
                packed[row, 2 * nw] &= mask
                packed[row, phase_col] &= Int64(3)
            end
        else
            @inbounds for row in axes(packed, 1)
                packed[row, phase_col] &= Int64(3)
            end
        end
        return new(n, nw, packed)
    end
end

function CliffordFrame(nqubits::Integer)
    n = _checked_nqubits(nqubits)
    nw = _nwords(n)
    data = zeros(Int64, 2 * n, 2 * nw + 1)
    for q in 0:(n - 1)
        word = _word_index(q)
        mask = _bit_mask(q)
        data[q + 1, word] = mask
        data[n + q + 1, nw + word] = mask
    end
    return CliffordFrame(n, data)
end

nqubits(cf::CliffordFrame)::Int = cf.nqubits
Base.copy(cf::CliffordFrame) = CliffordFrame(cf.nqubits, copy(cf.data))

_xrow(q::Int)::Int = q + 1
_zrow(cf::CliffordFrame, q::Int)::Int = cf.nqubits + q + 1

function _check_frame_qubit(cf::CliffordFrame, q::Integer)::Int
    return _check_qubit(cf.nqubits, q)
end

function _check_distinct_frame_qubits(cf::CliffordFrame, a::Integer, b::Integer)
    ai = _check_frame_qubit(cf, a)
    bi = _check_frame_qubit(cf, b)
    ai != bi || throw(ArgumentError("two-qubit Clifford gate requires distinct qubits"))
    return ai, bi
end

function _swap_rows!(cf::CliffordFrame, row_a::Int, row_b::Int)
    row_a == row_b && return cf
    @inbounds for col in axes(cf.data, 2)
        tmp = cf.data[row_a, col]
        cf.data[row_a, col] = cf.data[row_b, col]
        cf.data[row_b, col] = tmp
    end
    return cf
end

function _add_row_phase!(cf::CliffordFrame, row::Int, delta::Integer)
    cf.data[row, 2 * cf.nwords + 1] = Int64(Int(cf.data[row, 2 * cf.nwords + 1]) + Int(delta)) & Int64(3)
    return cf
end

function _mul_rows!(cf::CliffordFrame, dst::Int, lhs::Int, rhs::Int, extra_phase::Integer = 0)
    nw = cf.nwords
    carry = 0
    # Row multiplication mirrors packed Pauli multiplication but writes into a
    # matrix row. `extra_phase` accounts for Clifford conjugation identities
    # such as S' X S = -Y.
    @inbounds for w in 1:nw
        carry += count_ones(cf.data[lhs, nw + w] & cf.data[rhs, w])
    end
    phase = Int(cf.data[lhs, 2 * nw + 1]) + Int(cf.data[rhs, 2 * nw + 1]) +
        Int(extra_phase) + 2 * (carry & 1)
    @inbounds for w in 1:nw
        cf.data[dst, w] = xor(cf.data[lhs, w], cf.data[rhs, w])
        cf.data[dst, nw + w] = xor(cf.data[lhs, nw + w], cf.data[rhs, nw + w])
    end
    cf.data[dst, 2 * nw + 1] = Int64(phase) & Int64(3)
    return cf
end

function _rmul_row!(acc::PauliString, cf::CliffordFrame, row::Int)
    acc.nqubits == cf.nqubits || throw(DimensionMismatch("Pauli string and Clifford frame have different numbers of qubits"))
    nw = cf.nwords
    carry = 0
    @inbounds for w in 1:nw
        carry += count_ones(acc.data[nw + w] & cf.data[row, w])
    end
    phase = phase_exponent(acc) + Int(cf.data[row, 2 * nw + 1]) + 2 * (carry & 1)
    @inbounds for w in 1:nw
        acc.data[w] = xor(acc.data[w], cf.data[row, w])
        acc.data[nw + w] = xor(acc.data[nw + w], cf.data[row, nw + w])
    end
    _set_phase!(acc, phase)
    return acc
end

function preimage!(out::PauliString, cf::CliffordFrame, p::PauliString)
    out.nqubits == cf.nqubits == p.nqubits ||
        throw(DimensionMismatch("Pauli string and Clifford frame have different numbers of qubits"))
    fill!(out.data, 0)
    _set_phase!(out, phase_exponent(p))
    # Since the frame stores C'X_qC and C'Z_qC rows, the preimage of a packed
    # Pauli is the ordered product of the rows selected by its X/Z masks.
    for q in 0:(cf.nqubits - 1)
        xbit(p, q) && _rmul_row!(out, cf, _xrow(q))
        zbit(p, q) && _rmul_row!(out, cf, _zrow(cf, q))
    end
    return out
end

preimage(cf::CliffordFrame, p::PauliString) = preimage!(PauliString(cf.nqubits), cf, p)

function coordinates_in_frame(cf::CliffordFrame, p::PauliString)
    p.nqubits == cf.nqubits ||
        throw(DimensionMismatch("Pauli string and Clifford frame have different numbers of qubits"))
    out = PauliString(cf.nqubits)
    xrow = PauliString(cf.nqubits)
    zrow = PauliString(cf.nqubits)
    for q in 0:(cf.nqubits - 1)
        _copy_row_to_pauli!(xrow, cf, _xrow(q))
        _copy_row_to_pauli!(zrow, cf, _zrow(cf, q))
        pauli_anticommutes(p, zrow) && _set_xbit!(out, q)
        pauli_anticommutes(p, xrow) && _set_zbit!(out, q)
    end

    reconstructed = preimage(cf, out)
    _same_pauli_body(reconstructed, p) ||
        throw(ArgumentError("frame rows do not span the Pauli body"))
    _set_phase!(out, phase_exponent(p) - phase_exponent(reconstructed))
    return out
end

function _same_pauli_body(a::PauliString, b::PauliString)
    a.nqubits == b.nqubits || return false
    last_body_col = 2 * _nwords(a.nqubits)
    @inbounds for idx in 1:last_body_col
        a.data[idx] == b.data[idx] || return false
    end
    return true
end

function _copy_row_to_pauli!(out::PauliString, cf::CliffordFrame, row::Int)
    out.nqubits == cf.nqubits ||
        throw(DimensionMismatch("Pauli string and Clifford frame have different numbers of qubits"))
    @inbounds for col in axes(cf.data, 2)
        out.data[col] = cf.data[row, col]
    end
    return out
end

function _copy_pauli_to_row!(cf::CliffordFrame, row::Int, p::PauliString)
    p.nqubits == cf.nqubits ||
        throw(DimensionMismatch("Pauli string and Clifford frame have different numbers of qubits"))
    @inbounds for col in axes(cf.data, 2)
        cf.data[row, col] = p.data[col]
    end
    return cf
end

function _right_apply_clifford!(cf::CliffordFrame, g::CliffordFrame)
    cf.nqubits == g.nqubits ||
        throw(DimensionMismatch("Clifford frames act on different numbers of qubits"))
    row_pauli = PauliString(cf.nqubits)
    out = PauliString(cf.nqubits)
    # Right application C <- C*G composes every stored generator preimage with
    # G's preimage map.
    for row in axes(cf.data, 1)
        _copy_row_to_pauli!(row_pauli, cf, row)
        preimage!(out, g, row_pauli)
        _copy_pauli_to_row!(cf, row, out)
    end
    return cf
end

function _right_apply_left_gate!(cf::CliffordFrame, left_gate!, args...)
    g = CliffordFrame(cf.nqubits)
    left_gate!(g, args...)
    return _right_apply_clifford!(cf, g)
end

function left_H!(cf::CliffordFrame, q::Integer)
    qi = _check_frame_qubit(cf, q)
    _swap_rows!(cf, _xrow(qi), _zrow(cf, qi))
    return cf
end

function left_S!(cf::CliffordFrame, q::Integer)
    qi = _check_frame_qubit(cf, q)
    _mul_rows!(cf, _xrow(qi), _xrow(qi), _zrow(cf, qi), 3) # S' X S = -Y = i^3 XZ
    return cf
end

function left_SDG!(cf::CliffordFrame, q::Integer)
    qi = _check_frame_qubit(cf, q)
    _mul_rows!(cf, _xrow(qi), _xrow(qi), _zrow(cf, qi), 1) # S X S' = Y = i XZ
    return cf
end

function left_X!(cf::CliffordFrame, q::Integer)
    qi = _check_frame_qubit(cf, q)
    _add_row_phase!(cf, _zrow(cf, qi), 2) # X Z X = -Z
    return cf
end

function left_Z!(cf::CliffordFrame, q::Integer)
    qi = _check_frame_qubit(cf, q)
    _add_row_phase!(cf, _xrow(qi), 2) # Z X Z = -X
    return cf
end

function left_CX!(cf::CliffordFrame, c::Integer, t::Integer)
    ci, ti = _check_distinct_frame_qubits(cf, c, t)
    _mul_rows!(cf, _xrow(ci), _xrow(ci), _xrow(ti))               # X_c -> X_c X_t
    _mul_rows!(cf, _zrow(cf, ti), _zrow(cf, ci), _zrow(cf, ti))   # Z_t -> Z_c Z_t
    return cf
end

function left_CZ!(cf::CliffordFrame, a::Integer, b::Integer)
    ai, bi = _check_distinct_frame_qubits(cf, a, b)
    _mul_rows!(cf, _xrow(ai), _xrow(ai), _zrow(cf, bi)) # X_a -> X_a Z_b
    _mul_rows!(cf, _xrow(bi), _zrow(cf, ai), _xrow(bi)) # X_b -> Z_a X_b
    return cf
end

function left_SWAP!(cf::CliffordFrame, a::Integer, b::Integer)
    ai, bi = _check_distinct_frame_qubits(cf, a, b)
    _swap_rows!(cf, _xrow(ai), _xrow(bi))
    _swap_rows!(cf, _zrow(cf, ai), _zrow(cf, bi))
    return cf
end

function right_H!(cf::CliffordFrame, q::Integer)
    return _right_apply_left_gate!(cf, left_H!, q)
end

function right_S!(cf::CliffordFrame, q::Integer)
    return _right_apply_left_gate!(cf, left_S!, q)
end

function right_SDG!(cf::CliffordFrame, q::Integer)
    return _right_apply_left_gate!(cf, left_SDG!, q)
end

function right_X!(cf::CliffordFrame, q::Integer)
    return _right_apply_left_gate!(cf, left_X!, q)
end

function right_Z!(cf::CliffordFrame, q::Integer)
    return _right_apply_left_gate!(cf, left_Z!, q)
end

function right_CX!(cf::CliffordFrame, c::Integer, t::Integer)
    return _right_apply_left_gate!(cf, left_CX!, c, t)
end

function right_CZ!(cf::CliffordFrame, a::Integer, b::Integer)
    return _right_apply_left_gate!(cf, left_CZ!, a, b)
end

function right_SWAP!(cf::CliffordFrame, a::Integer, b::Integer)
    return _right_apply_left_gate!(cf, left_SWAP!, a, b)
end
