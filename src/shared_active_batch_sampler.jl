"""
    SharedActiveBatchExecutorState(program; batches, rng, values)

Packetized batched executor that stores each distinct active state once. A
packet column in `active` carries one dense active vector together with a packed
shot mask in `packet_masks`; active operations split packets only when the
shots in that packet need different signs or measurement branches.
"""
mutable struct SharedActiveBatchExecutorState{R,A<:AbstractMatrix{ComplexF64}}
    base::BatchFactoredExecutorState{R,A}
    active::Matrix{ComplexF64}
    active_scratch::Matrix{ComplexF64}
    packet_masks::Matrix{UInt64}
    mask_false::Vector{UInt64}
    mask_true::Vector{UInt64}
    npackets::Int
    materialized::Bool
end

function SharedActiveBatchExecutorState(
    program::FactoredInstructionProgram;
    batches::Integer = _default_batch_count(program.max_k),
    kernel_backend::Symbol = _default_batch_kernel_backend(),
    active_layout::Union{Nothing,Symbol} = nothing,
    rng = nothing,
    values = nothing,
)
    batch_count = _checked_batch_count(batches)
    backend = _normalize_batch_kernel_backend(kernel_backend)
    layout = _resolve_batch_active_layout(active_layout, backend)
    max_dim = _active_length(program.max_k)
    batch_words = _batch_word_count(batch_count)
    base = _new_batch_executor_state(program, batch_count, backend, layout, rng)
    runtime = SharedActiveBatchExecutorState(
        base,
        zeros(ComplexF64, max_dim, batch_count),
        zeros(ComplexF64, max_dim, batch_count),
        zeros(UInt64, batch_words, batch_count),
        zeros(UInt64, batch_words),
        zeros(UInt64, batch_words),
        0,
        false,
    )
    reset_executor!(runtime, program; shots = batch_count, values)
    return runtime
end

SharedActiveBatchExecutorState(
    program::FactoredInstructionProgram,
    batches::Integer;
    kernel_backend::Symbol = _default_batch_kernel_backend(),
    active_layout::Union{Nothing,Symbol} = nothing,
    rng = nothing,
    values = nothing,
) = SharedActiveBatchExecutorState(program; batches, kernel_backend, active_layout, rng, values)

active_packet_count(runtime::SharedActiveBatchExecutorState)::Int =
    runtime.materialized ? runtime.base.active_shots : runtime.npackets

"""
    sample_measurements_shared_active(program, shots; batches, kernel_backend, active_layout, rng, values)

Execute `shots` samples using the shared-active executor. Shots that currently
have identical active vectors are grouped into packets, so active rotations
and projections are applied once per packet until symbolic signs or measurement
branches force a split. When the packet count becomes too large, execution
materializes into the standard batch runtime.
"""
function sample_measurements_shared_active(
    program::FactoredInstructionProgram,
    shots::Integer;
    batches::Integer = _default_batch_count(program.max_k),
    kernel_backend::Symbol = _default_batch_kernel_backend(),
    active_layout::Union{Nothing,Symbol} = nothing,
    rng = nothing,
    values = nothing,
)
    runtime = SharedActiveBatchExecutorState(program; batches, kernel_backend, active_layout, rng)
    return sample_measurements!(runtime, program, shots; values)
end

"""
    sample_measurements!(runtime::SharedActiveBatchExecutorState, program, shots; values)

Reset and reuse a shared-active runtime for `shots` samples, returning a fresh
`shots x nrecords` `BitMatrix`.
"""
function sample_measurements!(
    runtime::SharedActiveBatchExecutorState,
    program::FactoredInstructionProgram,
    shots::Integer;
    values = nothing,
)
    total_shots = _checked_batch_sample_shots(shots)
    out = falses(total_shots, program.nrecords)
    total_shots == 0 && return out

    offset = 0
    while offset < total_shots
        active_shots = min(runtime.base.batches, total_shots - offset)
        reset_executor!(runtime, program; shots = active_shots, values)
        execute!(runtime, program)
        _copy_batch_measurements_to!(out, runtime.base, offset)
        offset += active_shots
    end
    return out
end

function reset_executor!(
    runtime::SharedActiveBatchExecutorState,
    program::FactoredInstructionProgram;
    shots::Integer = runtime.base.batches,
    values = nothing,
)
    base = runtime.base
    active_shots = _checked_active_batch_shots(base, shots)
    base.n = program.n
    base.k = program.initial_k
    base.active_shots = active_shots
    base.ndormant = program.n - program.initial_k
    _resize_batch_symbol_storage!(base, program.nsymbols)
    _resize_batch_measurement_storage!(base, program.nrecords)
    _assign_input_values!(base, values)
    if _shared_active_should_start_materialized(base.active_shots, program)
        _reserve_batch_active_capacity!(base, program.max_k)
        _set_batch_active_from_initial!(base, program.initial_active)
        runtime.npackets = base.active_shots
        runtime.materialized = true
    else
        _reserve_shared_active_capacity!(runtime, program.max_k)
        _set_shared_active_from_initial!(runtime, program.initial_active)
        runtime.materialized = false
    end
    return runtime
end

function execute!(
    runtime::SharedActiveBatchExecutorState,
    program::FactoredInstructionProgram,
)
    if runtime.materialized
        _check_batch_executor_program(runtime.base, program)
        runtime.base.active_shots == 0 && return runtime.base.measurements
        _sample_exogenous_symbols!(runtime.base, program)
        return _execute_factored_instructions!(runtime.base, program)
    end
    _check_shared_active_executor_program(runtime, program)
    runtime.base.active_shots == 0 && return runtime.base.measurements
    if _shared_active_should_start_materialized(runtime.base.active_shots, program)
        _materialize_shared_active_to_batch!(runtime)
        _sample_exogenous_symbols!(runtime.base, program)
        return _execute_factored_instructions!(runtime.base, program)
    end
    _sample_exogenous_symbols!(runtime.base, program)
    idx = 1
    while idx <= length(program.instructions)
        if _shared_active_should_materialize(runtime)
            _materialize_shared_active_to_batch!(runtime)
            while idx <= length(program.instructions)
                execute_instruction!(runtime.base, program.instructions[idx])
                idx += 1
            end
            return runtime.base.measurements
        end
        execute_instruction!(runtime, program.instructions[idx])
        idx += 1
    end
    return runtime.base.measurements
end

function execute!(
    runtime::SharedActiveBatchExecutorState,
    program::FactoredInstructionProgram,
    shots::Integer,
)
    runtime.base.active_shots = _checked_active_batch_shots(runtime.base, shots)
    return execute!(runtime, program)
end

function execute_instruction!(
    runtime::SharedActiveBatchExecutorState,
    instruction::ApplyPrecomputedActivePauliRotation,
)
    sign_bits = _eval_symbolic_bool!(runtime.base.eval_scratch, instruction.sign_plan, runtime.base)
    _shared_rotate_pauli!(runtime, instruction.rotation_kernel, sign_bits)
    return runtime
end

function execute_instruction!(
    runtime::SharedActiveBatchExecutorState,
    instruction::ApplyActiveBasisChange,
)
    _shared_apply_active_basis_change!(runtime, instruction.kind, instruction.qubit)
    return runtime
end

function execute_instruction!(
    runtime::SharedActiveBatchExecutorState,
    instruction::PromoteDormantRotation,
)
    sign_bits = _eval_symbolic_bool!(runtime.base.eval_scratch, instruction.sign_plan, runtime.base)
    _shared_promote_first_dormant_rotation!(runtime, instruction.theta, sign_bits)
    return runtime
end

function execute_instruction!(
    runtime::SharedActiveBatchExecutorState,
    instruction::RecordMeasurement,
)
    execute_instruction!(runtime.base, instruction)
    return runtime
end

function execute_instruction!(
    runtime::SharedActiveBatchExecutorState,
    instruction::IntroduceDormantMeasurementBranch,
)
    execute_instruction!(runtime.base, instruction)
    return runtime
end

function execute_instruction!(
    runtime::SharedActiveBatchExecutorState,
    instruction::MeasureActiveLastZ,
)
    return _measure_shared_active_last_z!(
        runtime,
        instruction.branch,
        instruction.outcome_plan,
        instruction.record,
        instruction.record_condition,
    )
end

function execute_instruction!(
    runtime::SharedActiveBatchExecutorState,
    instruction::MeasurePrecomputedActivePauli,
)
    return _measure_shared_precomputed_active_pauli!(
        runtime,
        instruction.kernel,
        instruction.branch,
        instruction.outcome_plan,
        instruction.record,
        instruction.record_condition,
    )
end

function _check_shared_active_executor_program(
    runtime::SharedActiveBatchExecutorState,
    program::FactoredInstructionProgram,
)
    base = runtime.base
    base.n == program.n ||
        throw(DimensionMismatch("shared batch executor has $(base.n) qubits; program has $(program.n)"))
    base.k + base.ndormant == base.n ||
        throw(DimensionMismatch("shared batch executor active+dormant width does not equal n"))
    size(runtime.active, 2) == base.batches ||
        throw(DimensionMismatch("shared active packet column count does not match batch size"))
    size(runtime.active_scratch) == size(runtime.active) ||
        throw(DimensionMismatch("shared active scratch shape does not match active shape"))
    size(runtime.packet_masks, 1) == _batch_word_count(base.batches) ||
        throw(DimensionMismatch("shared packet mask word count does not match batch size"))
    size(runtime.packet_masks, 2) == base.batches ||
        throw(DimensionMismatch("shared packet mask column count does not match batch size"))
    return nothing
end

function _reserve_shared_active_capacity!(
    runtime::SharedActiveBatchExecutorState,
    max_k::Integer,
)
    max_ki = _checked_active_k(max_k)
    max_dim = _active_length(max_ki)
    batches = runtime.base.batches
    if size(runtime.active, 1) < max_dim || size(runtime.active, 2) != batches
        runtime.active = zeros(ComplexF64, max_dim, batches)
    end
    if size(runtime.active_scratch, 1) < max_dim || size(runtime.active_scratch, 2) != batches
        runtime.active_scratch = zeros(ComplexF64, max_dim, batches)
    end
    batch_words = _batch_word_count(batches)
    if size(runtime.packet_masks, 1) != batch_words || size(runtime.packet_masks, 2) != batches
        runtime.packet_masks = zeros(UInt64, batch_words, batches)
    end
    if length(runtime.mask_false) != batch_words
        runtime.mask_false = zeros(UInt64, batch_words)
    end
    if length(runtime.mask_true) != batch_words
        runtime.mask_true = zeros(UInt64, batch_words)
    end
    return runtime
end

function _set_shared_active_from_initial!(
    runtime::SharedActiveBatchExecutorState,
    initial::ActiveState,
)
    base = runtime.base
    initial.k == base.k ||
        throw(DimensionMismatch("initial ActiveState has $(initial.k) qubits; expected $(base.k)"))
    dim = _check_active_storage(initial)
    size(runtime.active, 1) >= dim ||
        throw(DimensionMismatch("shared active storage has too few rows"))
    fill!(runtime.packet_masks, UInt64(0))
    runtime.npackets = base.active_shots == 0 ? 0 : 1
    runtime.materialized = false
    runtime.npackets == 0 && return runtime
    @inbounds @simd for idx in 1:dim
        runtime.active[idx, 1] = initial.alpha[idx]
    end
    nwords = _runtime_batch_word_count(base)
    @inbounds for word in 1:nwords
        runtime.packet_masks[word, 1] = _batch_live_word_mask(base, word)
    end
    return runtime
end

function _shared_active_should_materialize(runtime::SharedActiveBatchExecutorState)
    runtime.materialized && return true
    runtime.base.active_shots > 0 || return false
    if runtime.base.k >= 8
        return 4 * runtime.npackets >= runtime.base.active_shots
    end
    return 4 * runtime.npackets >= 3 * runtime.base.active_shots
end

function _shared_active_should_start_materialized(
    active_shots::Int,
    program::FactoredInstructionProgram,
)
    program.max_k >= 8 || return false
    active_shots > 0 || return false
    required_splits = max(1, ceil(Int, log2(active_shots)))
    return _shared_active_split_op_count(program) >= required_splits
end

function _shared_active_split_op_count(program::FactoredInstructionProgram)
    count = 0
    for instruction in program.instructions
        if instruction isa ApplyPrecomputedActivePauliRotation
            isempty(instruction.sign_plan.conditions) || (count += 1)
        elseif instruction isa PromoteDormantRotation
            isempty(instruction.sign_plan.conditions) || (count += 1)
        elseif instruction isa MeasureActiveLastZ || instruction isa MeasurePrecomputedActivePauli
            count += 1
        end
    end
    return count
end

function _materialize_shared_active_to_batch!(runtime::SharedActiveBatchExecutorState)
    runtime.materialized && return runtime
    base = runtime.base
    dim = _check_shared_active_storage(runtime)
    _reserve_batch_active_capacity!(base, base.k)
    nwords = _runtime_batch_word_count(base)
    @inbounds for packet in 1:runtime.npackets
        for word in 1:nwords
            mask = runtime.packet_masks[word, packet] & _batch_live_word_mask(base, word)
            while mask != 0
                bit = trailing_zeros(mask)
                shot = ((word - 1) << 6) + bit + 1
                @simd for idx in 1:dim
                    base.active[shot, idx] = runtime.active[idx, packet]
                end
                mask &= mask - UInt64(1)
            end
        end
    end
    runtime.materialized = true
    runtime.npackets = base.active_shots
    return runtime
end

function _check_shared_active_storage(runtime::SharedActiveBatchExecutorState)
    dim = _active_length(runtime.base.k)
    size(runtime.active, 1) >= dim ||
        throw(DimensionMismatch("shared active storage has too few rows for k=$(runtime.base.k)"))
    size(runtime.active_scratch, 1) >= dim ||
        throw(DimensionMismatch("shared active scratch has too few rows for k=$(runtime.base.k)"))
    return dim
end

function _clone_shared_packet!(
    runtime::SharedActiveBatchExecutorState,
    source_packet::Int,
    dim::Int,
)
    runtime.npackets < runtime.base.active_shots ||
        throw(ArgumentError("cannot split shared active packet beyond one packet per shot"))
    dest_packet = runtime.npackets + 1
    runtime.npackets = dest_packet
    @inbounds @simd for idx in 1:dim
        runtime.active[idx, dest_packet] = runtime.active[idx, source_packet]
    end
    return dest_packet
end

function _copy_shared_packet_mask!(
    runtime::SharedActiveBatchExecutorState,
    packet::Int,
    mask::Vector{UInt64},
)
    base = runtime.base
    nwords = _runtime_batch_word_count(base)
    @inbounds for word in 1:nwords
        runtime.packet_masks[word, packet] = mask[word] & _batch_live_word_mask(base, word)
    end
    @inbounds for word in (nwords + 1):size(runtime.packet_masks, 1)
        runtime.packet_masks[word, packet] = UInt64(0)
    end
    return runtime
end

function _shared_split_packet_by_bits!(
    runtime::SharedActiveBatchExecutorState,
    packet::Int,
    bits::Vector{UInt64},
    dim::Int,
)
    base = runtime.base
    nwords = _runtime_batch_word_count(base)
    has_false = false
    has_true = false
    @inbounds for word in 1:nwords
        mask = runtime.packet_masks[word, packet] & _batch_live_word_mask(base, word)
        true_mask = mask & bits[word]
        false_mask = mask & ~bits[word]
        runtime.mask_true[word] = true_mask
        runtime.mask_false[word] = false_mask
        has_true |= true_mask != 0
        has_false |= false_mask != 0
    end

    if has_false && has_true
        false_packet = packet
        _copy_shared_packet_mask!(runtime, false_packet, runtime.mask_false)
        true_packet = _clone_shared_packet!(runtime, packet, dim)
        _copy_shared_packet_mask!(runtime, true_packet, runtime.mask_true)
        return false_packet, true_packet
    elseif has_true
        return 0, packet
    else
        return packet, 0
    end
end

function _sample_shared_packet_branch_bits!(
    runtime::SharedActiveBatchExecutorState,
    packet::Int,
    probability::Float64,
    branch_bits::Vector{UInt64},
)
    base = runtime.base
    p = _check_probability(probability)
    nwords = _runtime_batch_word_count(base)
    if p <= 0.0
        return branch_bits
    elseif p >= 1.0
        @inbounds for word in 1:nwords
            branch_bits[word] |= runtime.packet_masks[word, packet] & _batch_live_word_mask(base, word)
        end
        return branch_bits
    end

    @inbounds for word in 1:nwords
        mask = runtime.packet_masks[word, packet] & _batch_live_word_mask(base, word)
        while mask != 0
            bit = trailing_zeros(mask)
            if _sample_bernoulli(base.rng, p)
                branch_bits[word] |= UInt64(1) << bit
            end
            mask &= mask - UInt64(1)
        end
    end
    return branch_bits
end

function _shared_rotate_pauli!(
    runtime::SharedActiveBatchExecutorState,
    kernel::PrecomputedActivePauliRotationKernel,
    sign_bits::Vector{UInt64},
)
    _check_active_rotation_kernel(runtime.base.k, kernel)
    dim = _check_shared_active_storage(runtime)
    initial_packets = runtime.npackets
    for packet in 1:initial_packets
        false_packet, true_packet = _shared_split_packet_by_bits!(runtime, packet, sign_bits, dim)
        false_packet != 0 && _shared_rotate_packet!(runtime, kernel, false, false_packet)
        true_packet != 0 && _shared_rotate_packet!(runtime, kernel, true, true_packet)
    end
    return runtime
end

function _shared_rotate_packet!(
    runtime::SharedActiveBatchExecutorState,
    kernel::PrecomputedActivePauliRotationKernel,
    sign::Bool,
    packet::Int,
)
    c = kernel.cos_theta
    if kernel.is_diagonal
        coefficients = sign ? kernel.diagonal_plus_coefficients : kernel.diagonal_minus_coefficients
        @inbounds @simd for idx in eachindex(coefficients)
            runtime.active[idx, packet] *= c + coefficients[idx]
        end
        return runtime
    end

    if kernel.action.zmask == 0
        coefficient =
            sign ? kernel.pair_left_plus_coefficients[1] : kernel.pair_left_minus_coefficients[1]
        coeff_im = imag(coefficient)
        @inbounds @simd for idx in eachindex(kernel.pair_left_indices)
            i0 = kernel.pair_left_indices[idx]
            i1 = kernel.pair_right_indices[idx]
            a0 = runtime.active[i0, packet]
            a1 = runtime.active[i1, packet]
            runtime.active[i0, packet] = ComplexF64(
                muladd(c, real(a0), -coeff_im * imag(a1)),
                muladd(c, imag(a0), coeff_im * real(a1)),
            )
            runtime.active[i1, packet] = ComplexF64(
                muladd(c, real(a1), -coeff_im * imag(a0)),
                muladd(c, imag(a1), coeff_im * real(a0)),
            )
        end
        return runtime
    end

    if _can_rotate_real_pair_flip(kernel)
        base_coeff = real(sign ? kernel.pair_left_plus_coefficients[1] : kernel.pair_left_minus_coefficients[1])
        zmask = kernel.action.zmask
        @inbounds @simd for idx in eachindex(kernel.pair_left_indices)
            i0 = kernel.pair_left_indices[idx]
            i1 = kernel.pair_right_indices[idx]
            coeff = _real_pair_flip_coeff(base_coeff, zmask, i0 - 1)
            a0 = runtime.active[i0, packet]
            a1 = runtime.active[i1, packet]
            runtime.active[i0, packet] = ComplexF64(
                muladd(c, real(a0), -coeff * real(a1)),
                muladd(c, imag(a0), -coeff * imag(a1)),
            )
            runtime.active[i1, packet] = ComplexF64(
                muladd(c, real(a1), coeff * real(a0)),
                muladd(c, imag(a1), coeff * imag(a0)),
            )
        end
        return runtime
    end

    left_coefficients = sign ? kernel.pair_left_plus_coefficients : kernel.pair_left_minus_coefficients
    right_coefficients = sign ? kernel.pair_right_plus_coefficients : kernel.pair_right_minus_coefficients
    @inbounds @simd for idx in eachindex(kernel.pair_left_indices)
        i0 = kernel.pair_left_indices[idx]
        i1 = kernel.pair_right_indices[idx]
        a0 = runtime.active[i0, packet]
        a1 = runtime.active[i1, packet]
        runtime.active[i0, packet] = c * a0 + right_coefficients[idx] * a1
        runtime.active[i1, packet] = c * a1 + left_coefficients[idx] * a0
    end
    return runtime
end

function _shared_apply_active_basis_change!(
    runtime::SharedActiveBatchExecutorState,
    kind::Symbol,
    q::Int,
)
    base = runtime.base
    0 <= q < base.k || throw(ArgumentError("active basis-change qubit is out of range"))
    dim = _check_shared_active_storage(runtime)
    mask = 1 << q
    if kind === :H
        invsqrt2 = inv(sqrt(2.0))
        @inbounds for packet in 1:runtime.npackets
            for basis in 0:(dim - 1)
                (basis & mask) == 0 || continue
                i0 = basis + 1
                i1 = (basis | mask) + 1
                a0 = runtime.active[i0, packet]
                a1 = runtime.active[i1, packet]
                runtime.active[i0, packet] = (a0 + a1) * invsqrt2
                runtime.active[i1, packet] = (a0 - a1) * invsqrt2
            end
        end
    elseif kind === :S
        @inbounds for packet in 1:runtime.npackets
            for basis in 0:(dim - 1)
                (basis & mask) != 0 && (runtime.active[basis + 1, packet] *= 1.0im)
            end
        end
    else
        throw(ArgumentError("unsupported active basis change $kind"))
    end
    return runtime
end

function _shared_promote_first_dormant_rotation!(
    runtime::SharedActiveBatchExecutorState,
    theta::Float64,
    sign_bits::Vector{UInt64},
)
    base = runtime.base
    base.ndormant > 0 ||
        throw(ArgumentError("cannot promote a dormant qubit when none remain"))
    dim = _check_shared_active_storage(runtime)
    promoted_dim = 2 * dim
    size(runtime.active, 1) >= promoted_dim ||
        throw(DimensionMismatch("shared active storage has too few rows for dormant promotion"))

    c = cos(theta)
    minus_i_s = ComplexF64(0.0, -sin(theta))
    plus_i_s = ComplexF64(0.0, sin(theta))
    initial_packets = runtime.npackets
    for packet in 1:initial_packets
        false_packet, true_packet = _shared_split_packet_by_bits!(runtime, packet, sign_bits, dim)
        false_packet != 0 &&
            _shared_promote_packet!(runtime, false_packet, dim, c, minus_i_s)
        true_packet != 0 &&
            _shared_promote_packet!(runtime, true_packet, dim, c, plus_i_s)
    end
    base.k += 1
    base.ndormant -= 1
    return runtime
end

function _shared_promote_packet!(
    runtime::SharedActiveBatchExecutorState,
    packet::Int,
    dim::Int,
    c::Float64,
    sin_coeff::ComplexF64,
)
    @inbounds @simd for basis in 1:dim
        promoted_basis = dim + basis
        amp = runtime.active[basis, packet]
        runtime.active[basis, packet] = c * amp
        runtime.active[promoted_basis, packet] = sin_coeff * amp
    end
    return runtime
end

function _measure_shared_active_last_z!(
    runtime::SharedActiveBatchExecutorState,
    branch_condition::Int,
    outcome_plan::SymbolicBoolEvaluationPlan,
    record::Int,
    record_condition::Union{Nothing,Int},
)
    base = runtime.base
    base.k > 0 || throw(ArgumentError("cannot measure the last active qubit when k == 0"))
    _fill_batch_bits!(base.eval_scratch, base, false)
    branch_bits = base.eval_scratch
    old_k = base.k
    new_k = old_k - 1
    dim = _active_length(new_k)
    initial_packets = runtime.npackets
    for packet in 1:initial_packets
        prob1 = _shared_active_last_z_probability_one(runtime, packet)
        _sample_shared_packet_branch_bits!(runtime, packet, prob1, branch_bits)
        false_packet, true_packet = _shared_split_packet_by_bits!(runtime, packet, branch_bits, 2 * dim)
        false_packet != 0 &&
            _shared_project_last_z_packet!(runtime, false_packet, false, 1.0 - prob1, dim)
        true_packet != 0 &&
            _shared_project_last_z_packet!(runtime, true_packet, true, prob1, dim)
    end
    base.k = new_k
    base.ndormant += 1
    _assign_symbol!(base, branch_condition, branch_bits)
    outcome_bits = _eval_symbolic_bool!(base.eval_scratch, outcome_plan, base)
    _write_measurement_record!(base, record, outcome_bits, record_condition)
    return runtime
end

function _shared_active_last_z_probability_one(
    runtime::SharedActiveBatchExecutorState,
    packet::Int,
)
    runtime.base.k > 0 || throw(ArgumentError("cannot measure last active qubit when k == 0"))
    half_dim = _active_length(runtime.base.k - 1)
    prob = 0.0
    @inbounds for basis in 1:half_dim
        prob += abs2(runtime.active[half_dim + basis, packet])
    end
    return min(max(prob, 0.0), 1.0)
end

function _shared_project_last_z_packet!(
    runtime::SharedActiveBatchExecutorState,
    packet::Int,
    branch::Bool,
    probability::Float64,
    dim::Int,
)
    probability > 0.0 ||
        throw(ArgumentError("sampled an impossible active measurement branch"))
    invnorm = inv(sqrt(probability))
    branch_offset = branch ? dim : 0
    @inbounds @simd for basis in 1:dim
        runtime.active[basis, packet] = runtime.active[branch_offset + basis, packet] * invnorm
    end
    return runtime
end

function _measure_shared_precomputed_active_pauli!(
    runtime::SharedActiveBatchExecutorState,
    kernel::PrecomputedActivePauliMeasurementKernel,
    branch_condition::Int,
    outcome_plan::SymbolicBoolEvaluationPlan,
    record::Int,
    record_condition::Union{Nothing,Int},
)
    base = runtime.base
    base.k > 0 || throw(ArgumentError("cannot measure an active Pauli when k == 0"))
    _check_active_measurement_kernel(base.k, kernel)
    if kernel.is_diagonal
        return _measure_shared_diagonal_active_pauli!(
            runtime,
            kernel,
            branch_condition,
            outcome_plan,
            record,
            record_condition,
        )
    end
    _fill_batch_bits!(base.eval_scratch, base, false)
    branch_bits = base.eval_scratch
    dim = _active_length(base.k)
    initial_packets = runtime.npackets
    for packet in 1:initial_packets
        prob_true = _shared_active_measurement_branch_probability(runtime, kernel, true, packet)
        prob_false, prob_true = _active_measurement_complementary_branch_probabilities(prob_true)
        _sample_shared_packet_branch_bits!(runtime, packet, prob_true, branch_bits)
        false_packet, true_packet = _shared_split_packet_by_bits!(runtime, packet, branch_bits, dim)
        false_packet != 0 &&
            _shared_project_active_pauli_measurement!(runtime, kernel, false, prob_false, false_packet)
        true_packet != 0 &&
            _shared_project_active_pauli_measurement!(runtime, kernel, true, prob_true, true_packet)
    end
    _assign_symbol!(base, branch_condition, branch_bits)
    base.k -= 1
    base.ndormant += 1
    outcome_bits = _eval_symbolic_bool!(base.eval_scratch, outcome_plan, base)
    _write_measurement_record!(base, record, outcome_bits, record_condition)
    return runtime
end

function _measure_shared_diagonal_active_pauli!(
    runtime::SharedActiveBatchExecutorState,
    kernel::PrecomputedActivePauliMeasurementKernel,
    branch_condition::Int,
    outcome_plan::SymbolicBoolEvaluationPlan,
    record::Int,
    record_condition::Union{Nothing,Int},
)
    base = runtime.base
    _fill_batch_bits!(base.eval_scratch, base, false)
    branch_bits = base.eval_scratch
    dim = _active_length(base.k)
    initial_packets = runtime.npackets
    for packet in 1:initial_packets
        prob_true = _shared_diagonal_measurement_branch_probability(runtime, kernel, true, packet)
        prob_false, prob_true = _active_measurement_complementary_branch_probabilities(prob_true)
        _sample_shared_packet_branch_bits!(runtime, packet, prob_true, branch_bits)
        false_packet, true_packet = _shared_split_packet_by_bits!(runtime, packet, branch_bits, dim)
        false_packet != 0 &&
            _shared_project_diagonal_active_pauli_measurement!(runtime, kernel, false, prob_false, false_packet)
        true_packet != 0 &&
            _shared_project_diagonal_active_pauli_measurement!(runtime, kernel, true, prob_true, true_packet)
    end
    _assign_symbol!(base, branch_condition, branch_bits)
    base.k -= 1
    base.ndormant += 1
    outcome_bits = _eval_symbolic_bool!(base.eval_scratch, outcome_plan, base)
    _write_measurement_record!(base, record, outcome_bits, record_condition)
    return runtime
end

function _shared_diagonal_measurement_branch_probability(
    runtime::SharedActiveBatchExecutorState,
    kernel::PrecomputedActivePauliMeasurementKernel,
    branch::Bool,
    packet::Int,
)
    sources = branch ? kernel.source0_true : kernel.source0_false
    prob = 0.0
    @inbounds @simd for idx in eachindex(sources)
        prob += abs2(runtime.active[sources[idx], packet])
    end
    return min(max(prob, 0.0), 1.0)
end

function _shared_active_measurement_branch_probability(
    runtime::SharedActiveBatchExecutorState,
    kernel::PrecomputedActivePauliMeasurementKernel,
    branch::Bool,
    packet::Int,
)
    sources0, sources1, coeffs0, coeffs1 = _measurement_kernel_branch(kernel, branch)
    prob = 0.0
    @inbounds @simd for idx in eachindex(sources0)
        amp = coeffs0[idx] * runtime.active[sources0[idx], packet]
        src1 = sources1[idx]
        src1 != 0 && (amp += coeffs1[idx] * runtime.active[src1, packet])
        prob += abs2(amp)
    end
    return min(max(prob, 0.0), 1.0)
end

function _shared_project_active_pauli_measurement!(
    runtime::SharedActiveBatchExecutorState,
    kernel::PrecomputedActivePauliMeasurementKernel,
    branch::Bool,
    probability::Float64,
    packet::Int,
)
    probability > 0.0 ||
        throw(ArgumentError("sampled an impossible active measurement branch"))
    sources0, sources1, coeffs0, coeffs1 = _measurement_kernel_branch(kernel, branch)
    invnorm = inv(sqrt(probability))
    @inbounds @simd for idx in eachindex(sources0)
        amp = coeffs0[idx] * runtime.active[sources0[idx], packet]
        src1 = sources1[idx]
        src1 != 0 && (amp += coeffs1[idx] * runtime.active[src1, packet])
        runtime.active_scratch[idx, packet] = amp * invnorm
    end
    leading = size(runtime.active, 1)
    copyto!(
        runtime.active,
        1 + (packet - 1) * leading,
        runtime.active_scratch,
        1 + (packet - 1) * leading,
        length(sources0),
    )
    return runtime
end

function _shared_project_diagonal_active_pauli_measurement!(
    runtime::SharedActiveBatchExecutorState,
    kernel::PrecomputedActivePauliMeasurementKernel,
    branch::Bool,
    probability::Float64,
    packet::Int,
)
    probability > 0.0 ||
        throw(ArgumentError("sampled an impossible active measurement branch"))
    sources = branch ? kernel.source0_true : kernel.source0_false
    invnorm = inv(sqrt(probability))
    # Diagonal measurement sources are never below their output index, so the
    # packet can be projected in place in ascending index order.
    @inbounds @simd for idx in eachindex(sources)
        runtime.active[idx, packet] = runtime.active[sources[idx], packet] * invnorm
    end
    return runtime
end
