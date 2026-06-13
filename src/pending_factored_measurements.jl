"""
    process_pending_measurement!(state, measurement)

Lower one pending Pauli measurement into the post-circuit pending factored
state. Dormant computational-basis bits are kept in `|0,0,...,0>` at this
stage. Deterministic dormant-Z measurements therefore use only the symbolic
Pauli sign and Hermitian Pauli phase. Active-only measurements emit update
instructions that will compute branch probabilities and project the active
state at execution time. Dormant X/Y measurements reduce to one dormant
measurement, introduce a fresh symbolic branch bit, and push `X^s` through the
remaining pending operations without storing that bit in `state.dormant`.
"""
function process_pending_measurement!(state::PendingFactoredState, measurement::PendingPauliMeasurement)
    _check_pending_operation_state(state, measurement)
    queued_first = !isempty(state.pending_operations) && state.pending_operations[1] == measurement
    current = copy(measurement)
    pauli = current.pauli.pauli
    active_body = _project_pauli_body(pauli, 0, state.k)

    # Dormant X/Y support is the non-diagonal case: it creates a fresh symbolic
    # branch and is pushed through the remaining pending FIFO.
    picked_dormant = _first_dormant_x_qubit(state, pauli)
    if picked_dormant !== nothing
        return _measure_dormant_xy_pauli!(state, current, picked_dormant, queued_first)
    end

    base_outcome = _measurement_base_outcome(state, current.pauli)
    _has_nonidentity_body(active_body) ||
        return _record_deterministic_measurement!(state, current, base_outcome)

    return _measure_active_pauli_branches!(state, current, active_body, base_outcome, queued_first)
end

function _measurement_base_outcome(
    state::PendingFactoredState,
    pauli::SymbolicPauliString,
)
    return _measurement_base_outcome(state, pauli, nothing)
end

function _measurement_base_outcome(
    _state::PendingFactoredState,
    pauli::SymbolicPauliString,
    _skip_dormant::Union{Nothing,Int},
)
    # Pending dormant bits are fixed to zero in this planning stage, so dormant
    # Z support contributes no basis-dependent sign.
    return xor(pauli.sign, _measurement_phase_sign(pauli.pauli))
end

function _record_deterministic_measurement!(
    state::PendingFactoredState,
    measurement::PendingPauliMeasurement,
    outcome::SymbolicBool,
)
    record = _measurement_record!(state, measurement)
    return _push_instruction!(state, RecordMeasurement(outcome, record, measurement.record_condition))
end

function _prepare_active_qubit_for_dormant_measurement!(
    state::PendingFactoredState,
    current::PendingPauliMeasurement,
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

function _eliminate_dormant_x_with_picked_control!(
    state::PendingFactoredState,
    current::PendingPauliMeasurement,
    picked_dormant::Int,
    target_dormant::Int,
)
    control = state.k + picked_dormant
    target = state.k + target_dormant
    return _rewrite_current_and_pending_by_basis_change!(state, current, (:CX, control, target))
end

function _eliminate_active_x_with_picked_dormant_control!(
    state::PendingFactoredState,
    current::PendingPauliMeasurement,
    picked_dormant::Int,
    target_active::Int,
)
    control = state.k + picked_dormant
    return _rewrite_current_and_pending_by_basis_change!(state, current, (:CX, control, target_active))
end

function _normalize_picked_dormant_measurement_to_z!(
    state::PendingFactoredState,
    current::PendingPauliMeasurement,
    picked_dormant::Int,
)
    q = state.k + picked_dormant
    xb = xbit(current.pauli.pauli, q)
    zb = zbit(current.pauli.pauli, q)
    if xb && zb
        current = _rewrite_current_and_pending_by_basis_change!(state, current, (:S, q))
        return _rewrite_current_and_pending_by_basis_change!(state, current, (:H, q))
    elseif xb
        return _rewrite_current_and_pending_by_basis_change!(state, current, (:H, q))
    elseif zb
        return current
    end
    throw(AssertionError("dormant measurement reduction lost the picked Pauli support"))
end

function _check_reduced_dormant_measurement(
    state::PendingFactoredState,
    current::PendingPauliMeasurement,
    picked_dormant::Int,
)
    p = current.pauli.pauli
    picked = state.k + picked_dormant
    for q in 0:(state.k - 1)
        (!xbit(p, q) && !zbit(p, q)) ||
            throw(AssertionError("dormant measurement reduction left active support"))
    end
    for d in 0:(ndormant(state) - 1)
        q = state.k + d
        !xbit(p, q) ||
            throw(AssertionError("dormant measurement reduction left dormant X support"))
    end
    zbit(p, picked) ||
        throw(AssertionError("dormant measurement reduction did not leave Z on the picked dormant qubit"))
    return nothing
end

function _measure_dormant_xy_pauli!(
    state::PendingFactoredState,
    current::PendingPauliMeasurement,
    picked_dormant::Int,
    queued_first::Bool,
)
    for d in 0:(ndormant(state) - 1)
        d == picked_dormant && continue
        q = state.k + d
        xbit(current.pauli.pauli, q) || continue
        # A dormant control qubit is still in |0>; the state update is deferred
        # to symbolic pushback on the pending operations.
        current = _eliminate_dormant_x_with_picked_control!(
            state,
            current,
            picked_dormant,
            d,
        )
    end

    for q in 0:(state.k - 1)
        current = _prepare_active_qubit_for_dormant_measurement!(state, current, q)
        xbit(current.pauli.pauli, q) || continue
        current = _eliminate_active_x_with_picked_dormant_control!(
            state,
            current,
            picked_dormant,
            q,
        )
    end

    current = _normalize_picked_dormant_measurement_to_z!(state, current, picked_dormant)
    _check_reduced_dormant_measurement(state, current, picked_dormant)

    base_outcome = _measurement_base_outcome(state, current.pauli, picked_dormant)
    branch = fresh_condition!(state.context)
    branch_bit = symbolic_bool(branch)
    first_remaining = queued_first ? 2 : 1
    # The current operation is either at queue index 1 and will be popped after
    # processing, or it was supplied directly. Push the branch through only the
    # operations that remain after the current measurement.
    _push_symbolic_pauli_through_pending_from!(
        state,
        first_remaining,
        pauli_x(state.n, state.k + picked_dormant),
        branch_bit,
    )
    outcome = xor(base_outcome, branch_bit)
    record = _measurement_record!(state, current)
    return _push_instruction!(
        state,
        IntroduceDormantMeasurementBranch(branch, outcome, record, current.record_condition),
    )
end

function _measure_active_pauli_branches!(
    state::PendingFactoredState,
    current::PendingPauliMeasurement,
    active_body::PauliString,
    base_outcome::SymbolicBool,
    queued_first::Bool,
)
    _pauli_squares_to_identity(active_body) ||
        throw(ArgumentError("active measurement Pauli must square to identity"))
    kernel = PrecomputedActivePauliMeasurementKernel(active_body)
    frame = _active_measurement_coordinate_frame(state, active_body, kernel)
    _transform_pending_operations_by_frame!(state, frame)
    branch = fresh_condition!(state.context)
    branch_bit = symbolic_bool(branch)
    first_remaining = queued_first ? 2 : 1
    # After the tableau rewrite, the measured stabilizer is the first dormant
    # Z. Branch 1 is represented by the conjugate X on that coordinate.
    _push_symbolic_pauli_through_pending_from!(
        state,
        first_remaining,
        pauli_x(state.n, state.k - 1),
        branch_bit,
    )

    outcome = xor(base_outcome, branch_bit)
    record = _measurement_record!(state, current)
    instruction = _push_instruction!(
        state,
        MeasurePrecomputedActivePauli(active_body, branch, outcome, record, current.record_condition),
    )
    _set_planning_active_count!(state, state.k - 1)
    return instruction
end

function _active_measurement_coordinate_frame(
    state::PendingFactoredState,
    active_body::PauliString,
    kernel::PrecomputedActivePauliMeasurementKernel,
)
    k = state.k
    active_body.nqubits == k ||
        throw(DimensionMismatch("active measurement Pauli has $(active_body.nqubits) qubits; expected $k"))
    frame = CliffordFrame(state.n)
    pivot = kernel.pivot
    measured = _embed_active_pauli(state.n, active_body)
    fixed_x = kernel.is_diagonal ? pauli_x(state.n, pivot) : pauli_z(state.n, pivot)
    _copy_pauli_to_row!(frame, _xrow(k - 1), fixed_x)
    _copy_pauli_to_row!(frame, _zrow(frame, k - 1), measured)

    new_q = 0
    for old_q in 0:(k - 1)
        old_q == pivot && continue
        if kernel.is_diagonal
            zrow = pauli_z(state.n, old_q)
            xrow = pauli_x(state.n, old_q)
            zbit(active_body, old_q) && (xrow = xrow * pauli_x(state.n, pivot))
        else
            zrow = pauli_z(state.n, old_q)
            xbit(active_body, old_q) && (zrow = zrow * pauli_z(state.n, pivot))
            xrow = pauli_x(state.n, old_q)
            zbit(active_body, old_q) && (xrow = xrow * pauli_z(state.n, pivot))
        end
        _copy_pauli_to_row!(frame, _xrow(new_q), xrow)
        _copy_pauli_to_row!(frame, _zrow(frame, new_q), zrow)
        new_q += 1
    end
    new_q == k - 1 || throw(AssertionError("active measurement tableau dropped the wrong number of qubits"))
    return frame
end

function _embed_active_pauli(n::Int, active_body::PauliString)
    out = PauliString(n)
    for q in 0:(active_body.nqubits - 1)
        xbit(active_body, q) && _set_xbit!(out, q)
        zbit(active_body, q) && _set_zbit!(out, q)
    end
    _set_phase!(out, phase_exponent(active_body))
    return out
end

function _transform_pending_operation_by_frame(
    operation::PendingPauliRotation,
    frame::CliffordFrame,
)
    return PendingPauliRotation(
        operation.theta,
        SymbolicPauliString(coordinates_in_frame(frame, operation.pauli.pauli), operation.pauli.sign),
    )
end

function _transform_pending_operation_by_frame(
    operation::PendingPauliMeasurement,
    frame::CliffordFrame,
)
    return PendingPauliMeasurement(
        SymbolicPauliString(coordinates_in_frame(frame, operation.pauli.pauli), operation.pauli.sign),
        operation.record,
        operation.record_condition,
    )
end

function _transform_pending_operations_by_frame!(state::PendingFactoredState, frame::CliffordFrame)
    for idx in eachindex(state.pending_operations)
        state.pending_operations[idx] =
            _transform_pending_operation_by_frame(state.pending_operations[idx], frame)
    end
    return state
end
