"""
    process_pending_rotation!(state, rotation)

Lower one pending `exp(-im * theta * P)` Pauli rotation into the post-circuit
pending factored state. Dormant qubits are kept in `|0,0,...,0>` at this
stage, so dormant `Z` support contributes no symbolic basis sign. Dormant X/Y
rotations reduce all non-diagonal support to one dormant `X`, move that qubit
to the first dormant position using Pauli rewrites only, and promote it into the
active subsystem.
"""
function process_pending_rotation!(state::PendingFactoredState, rotation::PendingPauliRotation)
    _check_pending_operation_state(state, rotation)
    current = copy(rotation)
    picked_dormant = _first_dormant_x_qubit(state, current.pauli.pauli)
    if picked_dormant === nothing
        return _process_diagonal_dormant_rotation!(state, current)
    end
    return _process_nondiagonal_dormant_rotation!(state, current, picked_dormant)
end

function _rotation_sign_from_pauli(pauli::SymbolicPauliString)
    return xor(pauli.sign, _measurement_phase_sign(pauli.pauli))
end

function _rotation_sign_with_dormant_z(_state::PendingFactoredState, pauli::SymbolicPauliString)
    return _rotation_sign_from_pauli(pauli)
end

function _active_rotation_instruction(
    active_body::PauliString,
    theta::Real,
    sign::SymbolicBool,
)
    return ApplyPrecomputedActivePauliRotation(active_body, theta, sign)
end

function _process_diagonal_dormant_rotation!(
    state::PendingFactoredState,
    current::PendingPauliRotation,
)
    active_body = _project_pauli_body(current.pauli.pauli, 0, state.k)
    sign = _rotation_sign_with_dormant_z(state, current.pauli)
    # Pure dormant-Z rotations are shot-global phases under the all-zero
    # dormant convention, so they produce no runtime instruction.
    _has_nonidentity_body(active_body) || return nothing
    _pauli_squares_to_identity(active_body) ||
        throw(ArgumentError("active rotation Pauli must square to identity"))
    return _push_instruction!(state, _active_rotation_instruction(active_body, current.theta, sign))
end

function _prepare_active_qubit_for_dormant_rotation!(
    state::PendingFactoredState,
    current::PendingPauliRotation,
    q::Int,
)
    xb = xbit(current.pauli.pauli, q)
    zb = zbit(current.pauli.pauli, q)
    if xb && zb
        return _rewrite_current_and_pending_by_basis_change!(state, current, (:S, q))
    elseif zb
        return _rewrite_current_and_pending_by_basis_change!(state, current, (:H, q))
    end
    return current
end

function _eliminate_active_x_for_dormant_rotation!(
    state::PendingFactoredState,
    current::PendingPauliRotation,
    picked_dormant::Int,
    target_active::Int,
)
    control = state.k + picked_dormant
    return _rewrite_current_and_pending_by_basis_change!(state, current, (:CX, control, target_active))
end

function _prepare_other_dormant_qubit_for_rotation!(
    state::PendingFactoredState,
    current::PendingPauliRotation,
    dormant_q::Int,
)
    q = state.k + dormant_q
    xb = xbit(current.pauli.pauli, q)
    zb = zbit(current.pauli.pauli, q)
    if xb && zb
        return _rewrite_current_and_pending_by_basis_change!(state, current, (:S, q))
    end
    return current
end

function _eliminate_other_dormant_support_for_rotation!(
    state::PendingFactoredState,
    current::PendingPauliRotation,
    picked_dormant::Int,
    target_dormant::Int,
)
    current = _prepare_other_dormant_qubit_for_rotation!(state, current, target_dormant)
    control = state.k + picked_dormant
    target = state.k + target_dormant

    # With a picked dormant control, CX removes target X support and CZ removes
    # target Z support, leaving only the picked dormant qubit non-diagonal.
    if xbit(current.pauli.pauli, target)
        current = _rewrite_current_and_pending_by_basis_change!(state, current, (:CX, control, target))
    end
    if zbit(current.pauli.pauli, target)
        current = _rewrite_current_and_pending_by_basis_change!(state, current, (:CZ, control, target))
    end
    return current
end

function _move_picked_dormant_to_front!(
    state::PendingFactoredState,
    current::PendingPauliRotation,
    picked_dormant::Int,
)
    picked_dormant == 0 && return current
    first = state.k
    picked = state.k + picked_dormant
    # Promotion always consumes the first dormant slot, so swap the selected
    # non-diagonal support there before emitting `PromoteDormantRotation`.
    current = _rewrite_current_and_pending_by_basis_change!(state, current, (:SWAP, first, picked))
    return current
end

function _normalize_first_dormant_rotation_to_x!(
    state::PendingFactoredState,
    current::PendingPauliRotation,
)
    q = state.k
    xb = xbit(current.pauli.pauli, q)
    zb = zbit(current.pauli.pauli, q)
    xb || throw(AssertionError("dormant rotation reduction lost first-dormant X support"))
    if zb
        current = _rewrite_current_and_pending_by_basis_change!(state, current, (:S, q))
    end
    xbit(current.pauli.pauli, q) && !zbit(current.pauli.pauli, q) ||
        throw(AssertionError("failed to normalize dormant rotation Pauli to first-dormant X"))
    return current
end

function _check_reduced_dormant_rotation(state::PendingFactoredState, current::PendingPauliRotation)
    p = current.pauli.pauli
    first = state.k
    for q in 0:(state.k - 1)
        (!xbit(p, q) && !zbit(p, q)) ||
            throw(AssertionError("dormant rotation reduction left active support"))
    end
    for d in 0:(ndormant(state) - 1)
        q = state.k + d
        if d == 0
            (xbit(p, q) && !zbit(p, q)) ||
                throw(AssertionError("dormant rotation reduction did not leave X on first dormant qubit"))
        else
            (!xbit(p, q) && !zbit(p, q)) ||
                throw(AssertionError("dormant rotation reduction left other dormant support"))
        end
    end
    xbit(p, first) || throw(AssertionError("dormant rotation reduction lost first dormant qubit"))
    return nothing
end

function _promote_first_dormant_rotation_state!(
    state::PendingFactoredState,
    theta::Float64,
    sign::SymbolicBool,
)
    ndormant(state) > 0 ||
        throw(ArgumentError("cannot promote a first dormant qubit when there are no dormant qubits"))
    instruction = _push_instruction!(state, PromoteDormantRotation(theta, sign))
    _set_planning_active_count!(state, state.k + 1)
    return instruction
end

function _process_nondiagonal_dormant_rotation!(
    state::PendingFactoredState,
    current::PendingPauliRotation,
    picked_dormant::Int,
)
    for q in 0:(state.k - 1)
        current = _prepare_active_qubit_for_dormant_rotation!(state, current, q)
        xbit(current.pauli.pauli, q) || continue
        # Active support is eliminated by pushing controlled operations through
        # the pending FIFO; active amplitudes are not updated during planning.
        current = _eliminate_active_x_for_dormant_rotation!(
            state,
            current,
            picked_dormant,
            q,
        )
    end

    for d in 0:(ndormant(state) - 1)
        d == picked_dormant && continue
        current = _eliminate_other_dormant_support_for_rotation!(
            state,
            current,
            picked_dormant,
            d,
        )
    end

    current = _move_picked_dormant_to_front!(state, current, picked_dormant)
    current = _normalize_first_dormant_rotation_to_x!(state, current)
    _check_reduced_dormant_rotation(state, current)

    sign = _rotation_sign_from_pauli(current.pauli)
    return _promote_first_dormant_rotation_state!(state, current.theta, sign)
end
