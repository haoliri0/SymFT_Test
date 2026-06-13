"""
    ConditionalPauliString(pauli, condition)

Single symbolic Pauli-frame term `pauli^s_condition`. A full
`SymbolicBool` condition is represented by multiple terms plus, when needed,
a deterministic condition elsewhere in the shared `SymbolicContext`.
"""
struct ConditionalPauliString
    pauli::PauliString
    condition::Int

    function ConditionalPauliString(pauli::PauliString, condition::Integer)
        c = Int(condition)
        c > 0 || throw(ArgumentError("condition id must be positive"))
        return new(copy(pauli), c)
    end
end

condition_id(cp::ConditionalPauliString)::Int = cp.condition
nqubits(cp::ConditionalPauliString)::Int = cp.pauli.nqubits
Base.copy(cp::ConditionalPauliString) = ConditionalPauliString(cp.pauli, cp.condition)
Base.:(==)(a::ConditionalPauliString, b::ConditionalPauliString) =
    a.condition == b.condition && a.pauli == b.pauli

"""
    ActivePauliFrame(k, context)

Symbolic product of conditional Pauli strings on a `k`-qubit virtual register.
For `FrameFactoredState` this `k` is currently the full virtual register size,
because dormant parts of conditional Paulis are intentionally not pushed into
the dormant basis during circuit ingestion.
"""
mutable struct ActivePauliFrame
    k::Int
    terms::Vector{ConditionalPauliString}
    context::SymbolicContext

    function ActivePauliFrame(k::Integer, context::SymbolicContext)
        ki = _checked_nqubits(k)
        return new(ki, ConditionalPauliString[], context)
    end
end

ActivePauliFrame(k::Integer) = ActivePauliFrame(k, SymbolicContext())

nqubits(frame::ActivePauliFrame)::Int = frame.k
symbolic_context(frame::ActivePauliFrame) = frame.context
next_condition(frame::ActivePauliFrame)::Int = next_condition(frame.context)
Base.copy(frame::ActivePauliFrame) =
    ActivePauliFrame(frame.k, copy.(frame.terms), frame.context)

function ActivePauliFrame(k::Integer, terms::AbstractVector{ConditionalPauliString}, context::SymbolicContext)
    frame = ActivePauliFrame(k, context)
    for term in terms
        add_pauli!(frame, term)
    end
    return frame
end

function _check_frame_pauli(frame::ActivePauliFrame, pauli::PauliString)
    pauli.nqubits == frame.k ||
        throw(DimensionMismatch("Pauli string acts on $(pauli.nqubits) qubits; symbolic Pauli frame has $(frame.k) qubits"))
    return nothing
end

function _check_conditioned_pauli_dimensions(q::PauliString, cp::ConditionalPauliString)
    q.nqubits == cp.pauli.nqubits ||
        throw(DimensionMismatch("Pauli strings act on different numbers of qubits"))
    return nothing
end

"""
    add_pauli!(frame, pauli)

Append `pauli^s` to the symbolic Pauli frame, assigning a fresh positive
integer id to `s`. Returns the stored `ConditionalPauliString`.
"""
function add_pauli!(frame::ActivePauliFrame, pauli::PauliString)
    cp = ConditionalPauliString(pauli, fresh_condition!(frame.context))
    add_pauli!(frame, cp)
    return cp
end

function add_pauli!(frame::ActivePauliFrame, pauli::PauliString, condition::Integer)
    cp = ConditionalPauliString(pauli, condition)
    add_pauli!(frame, cp)
    return cp
end

function add_pauli!(frame::ActivePauliFrame, cp::ConditionalPauliString)
    _check_frame_pauli(frame, cp.pauli)
    push!(frame.terms, copy(cp))
    # Terms may be imported with pre-existing ids, for example from Stim
    # measurement-record feedback. Keep future fresh ids above them.
    _bump_next_condition!(frame.context, cp.condition)
    return cp
end

"""
    SymbolicPauliString(pauli, sign)

Packed Pauli body with a symbolic sign `(-1)^sign`. Conjugating by symbolic
Pauli frames changes only this sign because Pauli conjugation preserves the
body up to a sign.
"""
struct SymbolicPauliString
    pauli::PauliString
    sign::SymbolicBool

    function SymbolicPauliString(pauli::PauliString, sign::SymbolicBool)
        return new(copy(pauli), copy(sign))
    end
end

SymbolicPauliString(pauli::PauliString, sign_conditions::AbstractVector{<:Integer}) =
    SymbolicPauliString(pauli, SymbolicBool(false, sign_conditions))
SymbolicPauliString(pauli::PauliString) = SymbolicPauliString(pauli, SymbolicBool(false))

nqubits(sp::SymbolicPauliString)::Int = sp.pauli.nqubits
symbolic_sign(sp::SymbolicPauliString) = copy(sp.sign)
sign_conditions(sp::SymbolicPauliString) = condition_ids(sp.sign)
Base.copy(sp::SymbolicPauliString) = SymbolicPauliString(sp.pauli, sp.sign)
Base.:(==)(a::SymbolicPauliString, b::SymbolicPauliString) =
    a.pauli == b.pauli && a.sign == b.sign

function _toggle_if_anticommutes(out::SymbolicPauliString, cp::ConditionalPauliString)
    _check_conditioned_pauli_dimensions(out.pauli, cp)
    pauli_anticommutes(cp.pauli, out.pauli) ||
        return out
    return SymbolicPauliString(out.pauli, xor(out.sign, symbolic_bool(cp.condition)))
end

"""
    conjugate_by(cp, q)

Compute `(P^s)' * q * P^s` symbolically, where `cp` stores `P^s`.
The packed Pauli body of `q` is unchanged; the returned `SymbolicPauliString`
stores a `SymbolicBool` sign expression determining whether a minus sign is
applied.
"""
function conjugate_by(cp::ConditionalPauliString, q::PauliString)
    out = SymbolicPauliString(q)
    return _toggle_if_anticommutes(out, cp)
end

function conjugate_by(cp::ConditionalPauliString, q::SymbolicPauliString)
    out = copy(q)
    return _toggle_if_anticommutes(out, cp)
end

conjugate_by(q::Union{PauliString,SymbolicPauliString}, cp::ConditionalPauliString) =
    conjugate_by(cp, q)

"""
    conjugate_by(frame, q)

Compute `E' * q * E` for the symbolic Pauli frame
`E = prod_i P_i^s_i`, returning a symbolic sign on top of `q`.
"""
function conjugate_by(frame::ActivePauliFrame, q::PauliString)
    q.nqubits == frame.k ||
        throw(DimensionMismatch("Pauli string acts on $(q.nqubits) qubits; symbolic Pauli frame has $(frame.k) qubits"))
    out = SymbolicPauliString(q)
    for cp in frame.terms
        out = _toggle_if_anticommutes(out, cp)
    end
    return out
end

function conjugate_by(frame::ActivePauliFrame, q::SymbolicPauliString)
    q.pauli.nqubits == frame.k ||
        throw(DimensionMismatch("Pauli string acts on $(q.pauli.nqubits) qubits; symbolic Pauli frame has $(frame.k) qubits"))
    out = copy(q)
    for cp in frame.terms
        out = _toggle_if_anticommutes(out, cp)
    end
    return out
end

conjugate_by(q::Union{PauliString,SymbolicPauliString}, frame::ActivePauliFrame) =
    conjugate_by(frame, q)

function Base.show(io::IO, cp::ConditionalPauliString)
    print(io, "(")
    show(io, cp.pauli)
    print(io, ")^s", cp.condition)
end

function Base.show(io::IO, sp::SymbolicPauliString)
    if !constant_term(sp.sign) && isempty(condition_ids(sp.sign))
        show(io, sp.pauli)
    else
        print(io, "(-1)^(")
        show(io, sp.sign)
        print(io, ") * ")
        show(io, sp.pauli)
    end
end
