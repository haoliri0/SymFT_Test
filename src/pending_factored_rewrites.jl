function _pauli_body_y_count(p::PauliString)
    nw = _nwords(p.nqubits)
    count = 0
    @inbounds for w in 1:nw
        count += count_ones(p.data[w] & p.data[nw + w])
    end
    return count
end

function _measurement_phase_sign(p::PauliString)
    coeff_phase = mod(phase_exponent(p) - _pauli_body_y_count(p), 4)
    coeff_phase == 0 && return false
    coeff_phase == 2 && return true
    throw(ArgumentError("Pauli measurement requires a Hermitian Pauli with real coefficient"))
end

function _first_dormant_x_qubit(state::PendingFactoredState, p::PauliString)
    for q in state.k:(state.n - 1)
        xbit(p, q) && return q - state.k
    end
    return nothing
end

function _with_xored_symbolic_sign(sp::SymbolicPauliString, sign::SymbolicBool)
    return SymbolicPauliString(sp.pauli, xor(sp.sign, sign))
end

# The `_forward_conjugate_*` helpers return `G * p * G'` for the listed
# Clifford gate. The name is from the state rewrite point of view: when the
# planner inserts a Clifford `G` before a pending operation, remaining Paulis
# are pushed forward through that inserted Clifford.
function _forward_conjugate_H(p::PauliString, q::Integer)
    g = CliffordFrame(p.nqubits)
    left_H!(g, q)
    return preimage(g, p)
end

function _forward_conjugate_S(p::PauliString, q::Integer)
    g = CliffordFrame(p.nqubits)
    left_SDG!(g, q)
    return preimage(g, p)
end

function _forward_conjugate_CX(p::PauliString, c::Integer, t::Integer)
    g = CliffordFrame(p.nqubits)
    left_CX!(g, c, t)
    return preimage(g, p)
end

function _forward_conjugate_CZ(p::PauliString, a::Integer, b::Integer)
    g = CliffordFrame(p.nqubits)
    left_CZ!(g, a, b)
    return preimage(g, p)
end

function _forward_conjugate_SWAP(p::PauliString, a::Integer, b::Integer)
    g = CliffordFrame(p.nqubits)
    left_SWAP!(g, a, b)
    return preimage(g, p)
end

function _transform_pauli_by_basis_changes(p::PauliString, changes::Vector{Tuple})
    out = copy(p)
    for change in changes
        if change[1] === :H
            out = _forward_conjugate_H(out, change[2])
        elseif change[1] === :S
            out = _forward_conjugate_S(out, change[2])
        elseif change[1] === :CX
            out = _forward_conjugate_CX(out, change[2], change[3])
        elseif change[1] === :CZ
            out = _forward_conjugate_CZ(out, change[2], change[3])
        elseif change[1] === :SWAP
            out = _forward_conjugate_SWAP(out, change[2], change[3])
        else
            throw(ArgumentError("unsupported stabilizer-coordinate basis change $(change[1])"))
        end
    end
    return out
end

function _transform_pending_operation_by_basis_changes(operation::PendingPauliRotation, changes::Vector{Tuple})
    return PendingPauliRotation(
        operation.theta,
        SymbolicPauliString(_transform_pauli_by_basis_changes(operation.pauli.pauli, changes), operation.pauli.sign),
    )
end

function _transform_pending_operation_by_basis_changes(operation::PendingPauliMeasurement, changes::Vector{Tuple})
    return PendingPauliMeasurement(
        SymbolicPauliString(_transform_pauli_by_basis_changes(operation.pauli.pauli, changes), operation.pauli.sign),
        operation.record,
        operation.record_condition,
    )
end

function _transform_pending_operations_by_basis_changes!(state::PendingFactoredState, changes::Vector{Tuple})
    # Every pending operation is expressed in the current virtual basis. After
    # a coordinate-basis change in the stabilizer tableau, rewrite all pending
    # Paulis into the new basis so the FIFO remains valid. The dense active
    # vector is not updated here.
    for idx in eachindex(state.pending_operations)
        state.pending_operations[idx] = _transform_pending_operation_by_basis_changes(state.pending_operations[idx], changes)
    end
    return state
end

function _emit_active_basis_change_if_needed!(state::PendingFactoredState, change::Tuple)
    kind = change[1]
    (kind === :H || kind === :S) || return nothing
    q = Int(change[2])
    q < state.k || return nothing
    _push_instruction!(state, ApplyActiveBasisChange(kind, q))
    return nothing
end

function _rewrite_current_and_pending_by_basis_change!(
    state::PendingFactoredState,
    current::PendingFactoredOperation,
    change::Tuple,
)
    changes = Tuple[change]
    _emit_active_basis_change_if_needed!(state, change)
    current = _transform_pending_operation_by_basis_changes(current, changes)
    _transform_pending_operations_by_basis_changes!(state, changes)
    return current
end

function _xor_operation_sign_if_anticommutes(
    operation::PendingPauliRotation,
    pauli::PauliString,
    sign::SymbolicBool,
)
    pauli_anticommutes(pauli, operation.pauli.pauli) || return operation
    return PendingPauliRotation(operation.theta, _with_xored_symbolic_sign(operation.pauli, sign))
end

function _xor_operation_sign_if_anticommutes(
    operation::PendingPauliMeasurement,
    pauli::PauliString,
    sign::SymbolicBool,
)
    pauli_anticommutes(pauli, operation.pauli.pauli) || return operation
    return PendingPauliMeasurement(
        _with_xored_symbolic_sign(operation.pauli, sign),
        operation.record,
        operation.record_condition,
    )
end

function _push_symbolic_pauli_through_current_and_pending!(
    state::PendingFactoredState,
    current::PendingFactoredOperation,
    pauli::PauliString,
    sign::SymbolicBool,
)
    _bump_next_condition!(state.context, sign)
    current = _xor_operation_sign_if_anticommutes(current, pauli, sign)
    for idx in eachindex(state.pending_operations)
        state.pending_operations[idx] =
            _xor_operation_sign_if_anticommutes(state.pending_operations[idx], pauli, sign)
    end
    return current
end

function _push_symbolic_pauli_through_pending_from!(
    state::PendingFactoredState,
    first_index::Int,
    pauli::PauliString,
    sign::SymbolicBool,
)
    _bump_next_condition!(state.context, sign)
    # Applying X^s to a basis state is postponed by conjugating it through the
    # remaining queued Paulis. Only anticommuting operations acquire the sign s.
    for idx in first_index:length(state.pending_operations)
        state.pending_operations[idx] =
            _xor_operation_sign_if_anticommutes(state.pending_operations[idx], pauli, sign)
    end
    return state
end
