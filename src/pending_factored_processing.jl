"""
    PendingFactoredState

Post-circuit state used for draining pending Pauli rotations and measurements.
At this stage all circuit operations have already been converted into
`pending_operations`, so the concrete `CliffordFrame` and active Pauli frame are
discarded. The remaining invariant is

    |alpha>_A tensor |x(s)>_D

together with a FIFO queue of pending factored Pauli operations acting on the
same `n = k + ndormant` virtual qubits. During planning, `active` stores the
initial active state for the eventual update program, while `k` tracks the
current symbolic active-qubit count and `max_k` records the maximum active
width reached by the pending-processing plan.
"""
mutable struct PendingFactoredState
    n::Int
    k::Int
    max_k::Int
    active::ActiveState
    dormant::DormantState
    context::SymbolicContext
    pending_operations::Vector{PendingFactoredOperation}
    instructions::Vector{FactoredInstruction}
    next_record::Int

    function PendingFactoredState(
        n::Integer,
        k::Integer,
        active::ActiveState,
        dormant::DormantState,
        context::SymbolicContext,
        pending_operations::AbstractVector{<:PendingFactoredOperation} = PendingFactoredOperation[],
        instructions::AbstractVector{<:FactoredInstruction} = FactoredInstruction[],
        next_record::Integer = 1,
        max_k::Integer = max(active.k, _checked_nqubits(k)),
    )
        ni = _checked_nqubits(n)
        ki = _checked_nqubits(k)
        max_ki = _checked_nqubits(max_k)
        record = Int(next_record)
        record > 0 || throw(ArgumentError("next measurement record id must be positive"))
        ki <= ni || throw(ArgumentError("active qubit count k=$ki exceeds n=$ni"))
        max_ki <= ni || throw(ArgumentError("maximum active qubit count max_k=$max_ki exceeds n=$ni"))
        ki <= max_ki ||
            throw(ArgumentError("maximum active qubit count max_k=$max_ki is smaller than current k=$ki"))
        active.k <= ni ||
            throw(DimensionMismatch("initial ActiveState has $(active.k) qubits; expected at most $ni"))
        active.k <= max_ki ||
            throw(DimensionMismatch("initial ActiveState has $(active.k) qubits; expected at most max_k=$max_ki"))
        dormant.d == ni - ki ||
            throw(DimensionMismatch("DormantState has $(dormant.d) qubits; expected $(ni - ki)"))
        dormant.context === context ||
            throw(ArgumentError("DormantState must share the PendingFactoredState SymbolicContext"))

        stored_pending = PendingFactoredOperation[]
        for operation in pending_operations
            operation.pauli.pauli.nqubits == ni ||
                throw(DimensionMismatch("pending operation acts on $(operation.pauli.pauli.nqubits) qubits; expected $ni"))
            _bump_next_condition!(context, _max_condition(operation))
            if operation isa PendingPauliMeasurement && operation.record !== nothing
                record = max(record, operation.record + 1)
            end
            push!(stored_pending, copy(operation))
        end
        stored_instructions = FactoredInstruction[]
        for instruction in instructions
            _bump_next_condition!(context, _max_condition(instruction))
            push!(stored_instructions, copy(instruction))
        end

        return new(
            ni,
            ki,
            max_ki,
            ActiveState(active.k, copy(active.alpha)),
            DormantState(dormant.bits, context),
            context,
            stored_pending,
            stored_instructions,
            record,
        )
    end
end

function PendingFactoredState(n::Integer, k::Integer = 0, context::SymbolicContext = SymbolicContext())
    ni = _checked_nqubits(n)
    ki = _checked_nqubits(k)
    ki <= ni || throw(ArgumentError("active qubit count k=$ki exceeds n=$ni"))
    return PendingFactoredState(ni, ki, ActiveState(ki), DormantState(ni - ki, context), context)
end

function PendingFactoredState(state::FrameFactoredState)
    return PendingFactoredState(
        state.n,
        state.k,
        state.active,
        state.dormant,
        state.context,
        state.pending_operations,
    )
end

nqubits(state::PendingFactoredState)::Int = state.n
nactive(state::PendingFactoredState)::Int = state.k
ndormant(state::PendingFactoredState)::Int = state.n - state.k
max_active_qubits(state::PendingFactoredState)::Int = state.max_k
symbolic_context(state::PendingFactoredState) = state.context
next_condition(state::PendingFactoredState)::Int = next_condition(state.context)
Base.copy(state::PendingFactoredState) =
    PendingFactoredState(
        state.n,
        state.k,
        state.active,
        state.dormant,
        state.context,
        state.pending_operations,
        state.instructions,
        state.next_record,
        state.max_k,
    )

function _max_condition(instruction::FactoredInstruction)
    max_condition = 0
    for field in fieldnames(typeof(instruction))
        value = getfield(instruction, field)
        if value isa SymbolicBool
            max_condition = max(max_condition, _max_condition(value))
        elseif value isa Integer && field in (:branch, :record_condition)
            max_condition = max(max_condition, Int(value))
        end
    end
    return max_condition
end

"""
    has_pending_operations(state)

Return whether `state` has queued factored Pauli rotations or measurements
waiting to be lowered into `|alpha>_A tensor |x(s)>_D`.
"""
has_pending_operations(state::PendingFactoredState)::Bool = !isempty(state.pending_operations)

pending_operation_count(state::PendingFactoredState)::Int = length(state.pending_operations)
pending_operations(state::PendingFactoredState) = PendingFactoredOperation[copy(operation) for operation in state.pending_operations]
pending_rotations(state::PendingFactoredState) =
    PendingPauliRotation[copy(operation) for operation in state.pending_operations if operation isa PendingPauliRotation]
pending_measurements(state::PendingFactoredState) =
    PendingPauliMeasurement[copy(operation) for operation in state.pending_operations if operation isa PendingPauliMeasurement]
factored_instructions(state::PendingFactoredState) =
    FactoredInstruction[copy(instruction) for instruction in state.instructions]

"""
    factored_instruction_program(state)

Package the instructions already emitted by pending processing into an
executable `FactoredInstructionProgram`. This does not drain
`state.pending_operations`; call `plan_factored_updates!` when the queue should
be processed first.
"""
function factored_instruction_program(state::PendingFactoredState)
    return FactoredInstructionProgram(
        state.n,
        state.active.k,
        state.active,
        state.instructions,
        state.max_k,
        state.context,
    )
end

function _push_instruction!(state::PendingFactoredState, instruction::FactoredInstruction)
    # Instructions can introduce branch symbols or carry reserved measurement
    # record conditions. Keep the shared context ahead of every referenced id.
    _bump_next_condition!(state.context, _max_condition(instruction))
    push!(state.instructions, copy(instruction))
    return instruction
end

function _fresh_measurement_record!(state::PendingFactoredState)
    record = state.next_record
    state.next_record += 1
    return record
end

function _measurement_record!(state::PendingFactoredState, measurement::PendingPauliMeasurement)
    if measurement.record === nothing
        # Hidden measurements used for exact reset corrections need to assign
        # their symbolic outcome without entering Stim's public measurement
        # record stream. Plain unreported measurements keep the historical
        # behavior of allocating an internal record.
        measurement.record_condition !== nothing && return nothing
        return _fresh_measurement_record!(state)
    end
    # Parser-reserved Stim records keep their input record number. Internal
    # unreserved measurements allocated later must not collide with them.
    state.next_record = max(state.next_record, measurement.record + 1)
    return measurement.record
end

function _set_planning_active_count!(state::PendingFactoredState, k::Integer)
    ki = _checked_nqubits(k)
    ki <= state.n || throw(ArgumentError("active qubit count k=$ki exceeds n=$(state.n)"))
    state.k = ki
    state.max_k = max(state.max_k, state.k)
    # Pending processing tracks only the active width; dormant bits are kept in
    # the all-zero convention, so resizing recreates that symbolic basis.
    state.dormant = DormantState(state.n - state.k, state.context)
    return state
end

"""
    process_pending_operations!(state)

Drain all queued `PendingFactoredOperation`s into factored update instructions.
The queue entry is removed only after its operation-specific processor returns
successfully.
"""
function process_pending_operations!(state::PendingFactoredState)
    start = length(state.instructions) + 1
    while has_pending_operations(state)
        process_next_pending_operation!(state)
    end
    return FactoredInstruction[copy(state.instructions[idx]) for idx in start:length(state.instructions)]
end

"""
    plan_factored_updates!(state)

Drain all pending factored Pauli rotations and measurements, then return an
executable `FactoredInstructionProgram`. During this lowering pass the active
stabilizer basis is tracked symbolically: pending Paulis are rewritten into
the current active basis, while runtime instructions only perform the required
amplitude rotations, projections, branch assignments, and record writes.
"""
function plan_factored_updates!(state::PendingFactoredState)
    process_pending_operations!(state)
    return factored_instruction_program(state)
end

"""
    process_next_pending_operation!(state)

Process the first queued pending operation. Returns `nothing` when the queue is
empty; otherwise returns the instruction or instructions appended while
removing the operation from the queue.
"""
function process_next_pending_operation!(state::PendingFactoredState)
    has_pending_operations(state) || return nothing
    operation = state.pending_operations[1]
    start = length(state.instructions) + 1
    result = process_pending_operation!(state, operation)
    popfirst!(state.pending_operations)
    new_instructions = FactoredInstruction[copy(state.instructions[idx]) for idx in start:length(state.instructions)]
    isempty(new_instructions) && return result
    length(new_instructions) == 1 && return new_instructions[1]
    return new_instructions
end

function _check_pending_operation_state(state::PendingFactoredState, operation::PendingFactoredOperation)
    operation.pauli.pauli.nqubits == state.n ||
        throw(DimensionMismatch("pending operation acts on $(operation.pauli.pauli.nqubits) qubits; pending factored state has $(state.n) qubits"))
    _bump_next_condition!(state.context, _max_condition(operation))
    return nothing
end

process_pending_operation!(state::PendingFactoredState, rotation::PendingPauliRotation) =
    process_pending_rotation!(state, rotation)

process_pending_operation!(state::PendingFactoredState, measurement::PendingPauliMeasurement) =
    process_pending_measurement!(state, measurement)
