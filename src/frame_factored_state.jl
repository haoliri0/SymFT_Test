abstract type PendingFactoredOperation end

"""
    PendingPauliRotation(theta, pauli)

Rotation queued at the frame-factored boundary after conjugating through the
current Clifford frame and symbolic Pauli frame.
"""
struct PendingPauliRotation <: PendingFactoredOperation
    theta::Float64
    pauli::SymbolicPauliString

    function PendingPauliRotation(theta::Real, pauli::SymbolicPauliString)
        return new(Float64(theta), copy(pauli))
    end
end

Base.copy(rotation::PendingPauliRotation) = PendingPauliRotation(rotation.theta, rotation.pauli)
Base.:(==)(lhs::PendingPauliRotation, rhs::PendingPauliRotation) =
    lhs.theta == rhs.theta && lhs.pauli == rhs.pauli

"""
    PendingPauliMeasurement(pauli, record, record_condition)

Measurement queued at the frame-factored boundary. `record` and
`record_condition` are optional because reset-like internal measurements need
not correspond to a Stim measurement record, while reporting measurements must
preserve their record-feedback alias.
"""
struct PendingPauliMeasurement <: PendingFactoredOperation
    pauli::SymbolicPauliString
    record::Union{Nothing,Int}
    record_condition::Union{Nothing,Int}

    function PendingPauliMeasurement(
        pauli::SymbolicPauliString,
        record::Union{Nothing,Integer} = nothing,
        record_condition::Union{Nothing,Integer} = nothing,
    )
        ri = _checked_measurement_record(record)
        condition = _checked_record_condition(record_condition)
        return new(copy(pauli), ri, condition)
    end
end

Base.copy(measurement::PendingPauliMeasurement) =
    PendingPauliMeasurement(measurement.pauli, measurement.record, measurement.record_condition)
Base.:(==)(lhs::PendingPauliMeasurement, rhs::PendingPauliMeasurement) =
    lhs.pauli == rhs.pauli &&
    lhs.record == rhs.record &&
    lhs.record_condition == rhs.record_condition

function _checked_measurement_record(record::Nothing)
    return nothing
end

function _checked_measurement_record(record::Integer)
    ri = Int(record)
    ri > 0 || throw(ArgumentError("measurement record id must be positive"))
    return ri
end

"""
    FrameFactoredState

Circuit-ingestion state for the invariant
`C (E_A(s) |alpha>_A tensor |x(s)>_D)`. Clifford gates update `C`, symbolic
Paulis update `E_A(s)`, and Pauli rotations/measurements are appended to the
pending FIFO for later exact lowering.
"""
mutable struct FrameFactoredState
    n::Int
    k::Int
    clifford::CliffordFrame
    active::ActiveState
    active_frame::ActivePauliFrame
    dormant::DormantState
    context::SymbolicContext
    pending_operations::Vector{PendingFactoredOperation}

    function FrameFactoredState(
        n::Integer,
        k::Integer,
        clifford::CliffordFrame,
        active::ActiveState,
        active_frame::ActivePauliFrame,
        dormant::DormantState,
        context::SymbolicContext,
        pending_operations::AbstractVector{<:PendingFactoredOperation} = PendingFactoredOperation[],
    )
        ni = _checked_nqubits(n)
        ki = _checked_nqubits(k)
        ki <= ni || throw(ArgumentError("active qubit count k=$ki exceeds n=$ni"))
        clifford.nqubits == ni ||
            throw(DimensionMismatch("CliffordFrame has $(clifford.nqubits) qubits; expected $ni"))
        active.k == ki ||
            throw(DimensionMismatch("ActiveState has $(active.k) qubits; expected $ki"))
        active_frame.k == ni ||
            throw(DimensionMismatch("ActivePauliFrame has $(active_frame.k) qubits; expected $ni"))
        dormant.d == ni - ki ||
            throw(DimensionMismatch("DormantState has $(dormant.d) qubits; expected $(ni - ki)"))
        active_frame.context === context ||
            throw(ArgumentError("ActivePauliFrame must share the FrameFactoredState SymbolicContext"))
        dormant.context === context ||
            throw(ArgumentError("DormantState must share the FrameFactoredState SymbolicContext"))
        stored_pending = PendingFactoredOperation[]
        for operation in pending_operations
            operation.pauli.pauli.nqubits == ni ||
                throw(DimensionMismatch("pending operation acts on $(operation.pauli.pauli.nqubits) qubits; expected $ni"))
            _bump_next_condition!(context, _max_condition(operation))
            push!(stored_pending, copy(operation))
        end
        return new(ni, ki, clifford, active, active_frame, dormant, context, stored_pending)
    end
end

function FrameFactoredState(n::Integer, k::Integer = 0, context::SymbolicContext = SymbolicContext())
    ni = _checked_nqubits(n)
    ki = _checked_nqubits(k)
    ki <= ni || throw(ArgumentError("active qubit count k=$ki exceeds n=$ni"))
    return FrameFactoredState(
        ni,
        ki,
        CliffordFrame(ni),
        ActiveState(ki),
        ActivePauliFrame(ni, context),
        DormantState(ni - ki, context),
        context,
    )
end

nqubits(state::FrameFactoredState)::Int = state.n
nactive(state::FrameFactoredState)::Int = state.k
ndormant(state::FrameFactoredState)::Int = state.n - state.k
symbolic_context(state::FrameFactoredState) = state.context
next_condition(state::FrameFactoredState)::Int = next_condition(state.context)
pending_operations(state::FrameFactoredState) = PendingFactoredOperation[copy(operation) for operation in state.pending_operations]
pending_rotations(state::FrameFactoredState) =
    PendingPauliRotation[copy(operation) for operation in state.pending_operations if operation isa PendingPauliRotation]
pending_measurements(state::FrameFactoredState) =
    PendingPauliMeasurement[copy(operation) for operation in state.pending_operations if operation isa PendingPauliMeasurement]

function Base.copy(state::FrameFactoredState)
    return FrameFactoredState(
        state.n,
        state.k,
        copy(state.clifford),
        ActiveState(state.active.k, copy(state.active.alpha)),
        ActivePauliFrame(state.active_frame.k, copy.(state.active_frame.terms), state.context),
        DormantState(state.dormant.bits, state.context),
        state.context,
        pending_operations(state),
    )
end

function _max_condition(sp::SymbolicPauliString)
    return _max_condition(sp.sign)
end

function _max_condition(operation::PendingPauliRotation)
    return _max_condition(operation.pauli)
end

function _max_condition(operation::PendingPauliMeasurement)
    max_condition = _max_condition(operation.pauli)
    operation.record_condition !== nothing &&
        (max_condition = max(max_condition, operation.record_condition))
    return max_condition
end

function _project_pauli_body(p::PauliString, qstart::Int, qcount::Int)
    out = PauliString(qcount)
    y_count = 0
    # Projection keeps the I/X/Y/Z body on a contiguous virtual subsystem. The
    # phase is rebuilt so that projected Y letters remain Hermitian Paulis.
    for q in 0:(qcount - 1)
        src = qstart + q
        xb = xbit(p, src)
        zb = zbit(p, src)
        xb && _set_xbit!(out, q)
        zb && _set_zbit!(out, q)
        xb && zb && (y_count += 1)
    end
    _set_phase!(out, y_count)
    return out
end

function _has_nonidentity_body(p::PauliString)
    nw = _nwords(p.nqubits)
    @inbounds for w in 1:nw
        (p.data[w] != 0 || p.data[nw + w] != 0) && return true
    end
    return false
end

function _check_physical_pauli(state::FrameFactoredState, p::PauliString)
    p.nqubits == state.n ||
        throw(DimensionMismatch("Pauli string acts on $(p.nqubits) qubits; frame-factored state has $(state.n) qubits"))
    return nothing
end

function _conjugate_by_active_frame(state::FrameFactoredState, p::PauliString)
    return conjugate_by(state.active_frame, p)
end

function _prepare_pending_pauli(state::FrameFactoredState, pauli::PauliString)
    _check_physical_pauli(state, pauli)
    pre = preimage(state.clifford, pauli)
    return _conjugate_by_active_frame(state, pre)
end

"""
    apply_pauli!(state, P_s)

Apply the symbolic physical Pauli `P^s` to the frame-factored state. This
computes `C' * P * C` and appends that full-register virtual Pauli to the
symbolic Pauli frame. The dormant part is not pushed into `|x(s)>_D` here:
future pending rotations and measurements are prepared by conjugating their
full-register Pauli bodies through this symbolic frame.
"""
function apply_pauli!(state::FrameFactoredState, cp::ConditionalPauliString)
    _check_physical_pauli(state, cp.pauli)
    _bump_next_condition!(state.context, cp.condition)

    pre = preimage(state.clifford, cp.pauli)
    _has_nonidentity_body(pre) && add_pauli!(state.active_frame, pre, cp.condition)
    return state
end

function apply_pauli!(state::FrameFactoredState, pauli::PauliString, condition::Integer)
    return apply_pauli!(state, ConditionalPauliString(pauli, condition))
end

function apply_pauli!(state::FrameFactoredState, pauli::PauliString, condition::SymbolicBool)
    _check_physical_pauli(state, pauli)
    # A constant-one symbolic condition means the Pauli is always applied. The
    # frame stores only conditional terms, so encode it as a p=1 Bernoulli term.
    if constant_term(condition)
        apply_pauli!(state, pauli, fresh_bernoulli_condition!(state.context, 1.0))
    end
    for condition_id in condition_ids(condition)
        apply_pauli!(state, pauli, condition_id)
    end
    return state
end

"""
    apply_pauli_rotation!(state, pauli, theta)

Apply the physical Pauli rotation `exp(-im * theta * pauli)` up to the current
frame-factored boundary. The method computes

    P'  = C' * pauli * C
    P'' = E' * P' * E

and appends the pending intermediate rotation `exp(-im * theta * P'')` to
`state.pending_operations`. Rewriting that pending rotation into a new active
state and dormant symbolic basis is intentionally left to the later lowering
step.
"""
function apply_pauli_rotation!(state::FrameFactoredState, pauli::PauliString, theta::Real)
    rotation = PendingPauliRotation(theta, _prepare_pending_pauli(state, pauli))
    push!(state.pending_operations, rotation)
    return rotation
end

"""
    apply_pauli_measurement!(state, pauli)

Prepare a physical Pauli measurement at the same frame-factored boundary as
pending rotations. The method computes

    P'  = C' * pauli * C
    P'' = E' * P' * E

and appends `PendingPauliMeasurement(P'')` to `state.pending_operations`.
Handling the pending measurement outcome and any active/dormant rewrite is left
to the later lowering step.
"""
function apply_pauli_measurement!(state::FrameFactoredState, pauli::PauliString)
    measurement = PendingPauliMeasurement(_prepare_pending_pauli(state, pauli))
    push!(state.pending_operations, measurement)
    return measurement
end

function apply_pauli_measurement!(
    state::FrameFactoredState,
    pauli::PauliString,
    sign::SymbolicBool,
    ;
    record::Union{Nothing,Integer} = nothing,
    record_condition::Union{Nothing,Integer} = nothing,
)
    prepared = _prepare_pending_pauli(state, pauli)
    measurement = PendingPauliMeasurement(
        SymbolicPauliString(prepared.pauli, xor(prepared.sign, sign)),
        record,
        record_condition,
    )
    push!(state.pending_operations, measurement)
    return measurement
end

function left_H!(state::FrameFactoredState, q::Integer)
    left_H!(state.clifford, q)
    return state
end

function left_S!(state::FrameFactoredState, q::Integer)
    left_S!(state.clifford, q)
    return state
end

function left_SDG!(state::FrameFactoredState, q::Integer)
    left_SDG!(state.clifford, q)
    return state
end

function left_X!(state::FrameFactoredState, q::Integer)
    left_X!(state.clifford, q)
    return state
end

function left_Z!(state::FrameFactoredState, q::Integer)
    left_Z!(state.clifford, q)
    return state
end

function left_CX!(state::FrameFactoredState, c::Integer, t::Integer)
    left_CX!(state.clifford, c, t)
    return state
end

function left_CZ!(state::FrameFactoredState, a::Integer, b::Integer)
    left_CZ!(state.clifford, a, b)
    return state
end

function left_SWAP!(state::FrameFactoredState, a::Integer, b::Integer)
    left_SWAP!(state.clifford, a, b)
    return state
end
