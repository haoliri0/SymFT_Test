"""
    DormantState(bits, context)

Symbolic computational-basis bits for dormant qubits. These bits are not a
dense state; each entry is the symbolic value of one frozen basis qubit.
"""
mutable struct DormantState
    d::Int
    bits::Vector{SymbolicBool}
    context::SymbolicContext

    function DormantState(bits::AbstractVector{SymbolicBool}, context::SymbolicContext)
        d = length(bits)
        stored = copy.(bits)
        for bit in stored
            _bump_next_condition!(context, bit)
        end
        return new(d, stored, context)
    end
end

"""
    DormantState(d)

Symbolic computational-basis state for `d` dormant qubits. The initial state is
`|0, 0, ..., 0>`. Dormant qubit positions are 0-based.
"""
function DormantState(d::Integer)
    return DormantState(d, SymbolicContext())
end

function DormantState(d::Integer, context::SymbolicContext)
    di = _checked_nqubits(d)
    return DormantState([SymbolicBool(false) for _ in 1:di], context)
end

function DormantState(n::Integer, k::Integer)
    return DormantState(n, k, SymbolicContext())
end

function DormantState(n::Integer, k::Integer, context::SymbolicContext)
    ni = _checked_nqubits(n)
    ki = _checked_nqubits(k)
    ki <= ni || throw(ArgumentError("active qubit count k=$ki exceeds n=$ni"))
    return DormantState(ni - ki, context)
end

DormantState(bits::AbstractVector{SymbolicBool}) = DormantState(bits, SymbolicContext())
DormantState(bits::AbstractVector{SymbolicBool}, next_condition::Integer) =
    DormantState(bits, SymbolicContext(next_condition))

nqubits(state::DormantState)::Int = state.d
ndormant(state::DormantState)::Int = state.d
symbolic_context(state::DormantState) = state.context
next_condition(state::DormantState)::Int = next_condition(state.context)

function _check_dormant_qubit(state::DormantState, q::Integer)::Int
    return _check_qubit(state.d, q)
end

dormant_bits(state::DormantState) = copy.(state.bits)

function dormant_bit(state::DormantState, q::Integer)
    qi = _check_dormant_qubit(state, q)
    return copy(state.bits[qi + 1])
end

function set_dormant_bit!(state::DormantState, q::Integer, value)
    qi = _check_dormant_qubit(state, q)
    expr = _as_symbolic_bool(value)
    state.bits[qi + 1] = expr
    # Imported symbolic expressions may reference condition ids allocated by a
    # sibling frame object sharing the same context.
    _bump_next_condition!(state.context, expr)
    return state
end

function assign_dormant_symbol!(state::DormantState, q::Integer)
    qi = _check_dormant_qubit(state, q)
    expr = fresh_symbolic_bool!(state.context)
    state.bits[qi + 1] = expr
    return expr
end

function flip_dormant_bit!(state::DormantState, q::Integer)
    qi = _check_dormant_qubit(state, q)
    state.bits[qi + 1] = !state.bits[qi + 1]
    return state
end

function xor_dormant_bit!(state::DormantState, q::Integer, value)
    qi = _check_dormant_qubit(state, q)
    expr = xor(state.bits[qi + 1], _as_symbolic_bool(value))
    state.bits[qi + 1] = expr
    _bump_next_condition!(state.context, expr)
    return state
end

function Base.copy(state::DormantState)
    return DormantState(state.bits, state.context)
end

function Base.show(io::IO, state::DormantState)
    print(io, "|")
    for q in 0:(state.d - 1)
        q > 0 && print(io, ",")
        show(io, state.bits[q + 1])
    end
    print(io, ">")
end
