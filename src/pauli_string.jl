const WORD_BITS = 64

# All Pauli and Clifford bit positions are 0-based. Julia array positions are
# converted at the boundary by adding one in `_word_index` and callers that
# access per-qubit arrays.
function _checked_nqubits(nqubits::Integer)::Int
    n = Int(nqubits)
    n >= 0 || throw(ArgumentError("number of qubits must be nonnegative"))
    return n
end

_nwords(nqubits::Integer)::Int = cld(_checked_nqubits(nqubits), WORD_BITS)
_stride(nqubits::Integer)::Int = 2 * _nwords(nqubits) + 1

function _check_qubit(nqubits::Int, q::Integer)::Int
    qi = Int(q)
    0 <= qi < nqubits || throw(BoundsError(0:(nqubits - 1), qi))
    return qi
end

_word_index(q::Int)::Int = (q >>> 6) + 1
_bit_mask(q::Int)::Int64 = Int64(1) << (q & 63)

function _tail_mask(nqubits::Int)::Int64
    used = nqubits & 63
    used == 0 && return Int64(-1)
    return Int64((UInt64(1) << used) - UInt64(1))
end

function _clear_unused_pauli_bits!(data::Vector{Int64}, nqubits::Int)
    nw = _nwords(nqubits)
    if nw > 0 && (nqubits & 63) != 0
        mask = _tail_mask(nqubits)
        data[nw] &= mask
        data[2 * nw] &= mask
    end
    return data
end

"""
    PauliString(nqubits)

Packed Pauli string on 0-based qubit positions. The backing data is a
`Vector{Int64}` laid out as `[x_words..., z_words..., phase]`, representing

    i^phase * prod_q X_q^x[q] Z_q^z[q]

with the product ordered by increasing 0-based qubit index.
"""
struct PauliString
    nqubits::Int
    data::Vector{Int64}

    function PauliString(nqubits::Integer, data::AbstractVector{<:Integer})
        n = _checked_nqubits(nqubits)
        length(data) == _stride(n) ||
            throw(ArgumentError("packed Pauli data has length $(length(data)); expected $(_stride(n))"))
        packed = Int64.(data)
        packed[end] &= Int64(3)
        _clear_unused_pauli_bits!(packed, n)
        return new(n, packed)
    end
end

PauliString(nqubits::Integer) = PauliString(nqubits, zeros(Int64, _stride(nqubits)))

nqubits(ps::PauliString)::Int = ps.nqubits
phase_exponent(ps::PauliString)::Int = Int(ps.data[end] & Int64(3))

function _set_phase!(ps::PauliString, phase::Integer)
    ps.data[end] = Int64(phase) & Int64(3)
    return ps
end

function phase_shift!(ps::PauliString, delta::Integer)
    return _set_phase!(ps, phase_exponent(ps) + Int(delta))
end

function xbit(ps::PauliString, q::Integer)::Bool
    qi = _check_qubit(ps.nqubits, q)
    return (ps.data[_word_index(qi)] & _bit_mask(qi)) != 0
end

function zbit(ps::PauliString, q::Integer)::Bool
    qi = _check_qubit(ps.nqubits, q)
    nw = _nwords(ps.nqubits)
    return (ps.data[nw + _word_index(qi)] & _bit_mask(qi)) != 0
end

function _set_xbit!(ps::PauliString, q::Int)
    ps.data[_word_index(q)] |= _bit_mask(q)
    return ps
end

function _set_zbit!(ps::PauliString, q::Int)
    ps.data[_nwords(ps.nqubits) + _word_index(q)] |= _bit_mask(q)
    return ps
end

pauli_identity(nqubits::Integer) = PauliString(nqubits)

function pauli_x(nqubits::Integer, q::Integer)
    ps = PauliString(nqubits)
    _set_xbit!(ps, _check_qubit(ps.nqubits, q))
    return ps
end

function pauli_z(nqubits::Integer, q::Integer)
    ps = PauliString(nqubits)
    _set_zbit!(ps, _check_qubit(ps.nqubits, q))
    return ps
end

function pauli_y(nqubits::Integer, q::Integer)
    ps = PauliString(nqubits)
    qi = _check_qubit(ps.nqubits, q)
    _set_xbit!(ps, qi)
    _set_zbit!(ps, qi)
    _set_phase!(ps, 1)
    return ps
end

function pauli_string(ops::AbstractString)
    ps = PauliString(length(ops))
    for (j, ch0) in enumerate(ops)
        q = j - 1
        ch = uppercase(ch0)
        if ch == 'I' || ch == '_'
            continue
        elseif ch == 'X'
            _set_xbit!(ps, q)
        elseif ch == 'Z'
            _set_zbit!(ps, q)
        elseif ch == 'Y'
            _set_xbit!(ps, q)
            _set_zbit!(ps, q)
            phase_shift!(ps, 1)
        else
            throw(ArgumentError("unsupported Pauli character '$ch0' at 0-based qubit $q"))
        end
    end
    return ps
end

Base.copy(ps::PauliString) = PauliString(ps.nqubits, copy(ps.data))
Base.:(==)(a::PauliString, b::PauliString) = a.nqubits == b.nqubits && a.data == b.data
Base.hash(ps::PauliString, h::UInt) = hash(ps.data, hash(ps.nqubits, h))

function Base.:*(lhs::PauliString, rhs::PauliString)
    out = copy(lhs)
    return _rmul!(out, rhs)
end

function _rmul!(lhs::PauliString, rhs::PauliString)
    lhs.nqubits == rhs.nqubits || throw(DimensionMismatch("Pauli strings act on different numbers of qubits"))
    nw = _nwords(lhs.nqubits)
    carry = 0
    # Multiplying X/Z masks is xor on the body; every lhs-Z crossing a rhs-X
    # contributes one -1, encoded as a phase increment of two.
    @inbounds for w in 1:nw
        carry += count_ones(lhs.data[nw + w] & rhs.data[w])
    end
    phase = phase_exponent(lhs) + phase_exponent(rhs) + 2 * (carry & 1)
    @inbounds for w in 1:nw
        lhs.data[w] = xor(lhs.data[w], rhs.data[w])
        lhs.data[nw + w] = xor(lhs.data[nw + w], rhs.data[nw + w])
    end
    _set_phase!(lhs, phase)
    return lhs
end

function pauli_anticommutes(a::PauliString, b::PauliString)::Bool
    a.nqubits == b.nqubits || throw(DimensionMismatch("Pauli strings act on different numbers of qubits"))
    nw = _nwords(a.nqubits)
    parity = false
    @inbounds for w in 1:nw
        parity = xor(parity, isodd(count_ones(a.data[w] & b.data[nw + w])))
        parity = xor(parity, isodd(count_ones(a.data[nw + w] & b.data[w])))
    end
    return parity
end

function Base.show(io::IO, ps::PauliString)
    nw = _nwords(ps.nqubits)
    ny = 0
    @inbounds for w in 1:nw
        ny += count_ones(ps.data[w] & ps.data[nw + w])
    end
    # The stored phase multiplies ordered XZ bodies. Display converts it to the
    # conventional coefficient multiplying I/X/Y/Z letters.
    coeff_phase = mod(phase_exponent(ps) - ny, 4)
    coeff = coeff_phase == 0 ? "" : coeff_phase == 1 ? "i*" : coeff_phase == 2 ? "-" : "-i*"

    chars = Vector{Char}(undef, ps.nqubits)
    for q in 0:(ps.nqubits - 1)
        x = xbit(ps, q)
        z = zbit(ps, q)
        chars[q + 1] = x ? (z ? 'Y' : 'X') : (z ? 'Z' : 'I')
    end
    body = ps.nqubits == 0 ? "I" : String(chars)
    print(io, coeff, body)
end
