"""
    FactoredExecutorState(program; rng, values)

Mutable runtime for executing one `FactoredInstructionProgram` shot. `values`
stores concrete assignments for symbolic condition ids, and `measurements`
stores 1-based measurement records emitted by factored instructions.
When `rng === nothing`, Julia's default global `rand()` source is used.
"""
mutable struct FactoredExecutorState{R}
    n::Int
    k::Int
    ndormant::Int
    active::ActiveState
    active_scratch::Vector{ComplexF64}
    values::Vector{Bool}
    assigned::Vector{Bool}
    value_words::Vector{UInt64}
    assigned_words::Vector{UInt64}
    measurements::Vector{Bool}
    rng::R
end

"""
    PresampledExogenous

Packed exogenous random-bit block for repeated single-shot execution of one
`FactoredInstructionProgram`. The logical layout is `nshots × nsymbols`, with
all entries initialized to false and rare/random events xored into
`value_words` during construction. Runtime assignment uses the selected shot
row as a sparse true-bit source instead of copying the whole row.
"""
struct PresampledExogenous
    nshots::Int
    nsymbols::Int
    nwords::Int
    exogenous_assigned_words::Vector{UInt64}
    value_words::Vector{UInt64}
end

function FactoredExecutorState(
    program::FactoredInstructionProgram;
    rng = nothing,
    values = nothing,
)
    runtime = FactoredExecutorState(
        program.n,
        program.initial_k,
        program.n - program.initial_k,
        ActiveState(program.initial_active.k, copy(program.initial_active.alpha)),
        zeros(ComplexF64, _active_length(program.initial_k)),
        fill(false, program.nsymbols),
        fill(false, program.nsymbols),
        zeros(UInt64, _symbol_word_count(program.nsymbols)),
        zeros(UInt64, _symbol_word_count(program.nsymbols)),
        fill(false, program.nrecords),
        rng,
    )
    _reserve_active_capacity!(runtime, program.max_k)
    _assign_input_values!(runtime, values)
    return runtime
end

"""
    sample_measurements(program; rng, values)

Execute `program` once and return its measurement record bits. Fixed
probability symbolic variables are sampled before instruction execution;
measurement branch variables are sampled lazily by their instructions.
"""
function sample_measurements(
    program::FactoredInstructionProgram;
    rng = nothing,
    values = nothing,
)
    runtime = FactoredExecutorState(program; rng, values)
    execute!(runtime, program)
    return copy(runtime.measurements)
end

"""
    sample_measurements!(runtime, program; values)

Reset and reuse `runtime` for one shot of `program`, returning the internal
measurement record vector. Callers that need to retain records should copy the
result before the next reuse.
"""
function sample_measurements!(
    runtime::FactoredExecutorState,
    program::FactoredInstructionProgram;
    values = nothing,
)
    reset_executor!(runtime, program; values)
    execute!(runtime, program)
    return runtime.measurements
end

"""
    sample_measurements!(runtime, program, samples, shot; values)

Execute one shot using exogenous symbols already stored in `samples`. The
selected row supplies fixed-probability noise and measurement-error symbols;
measurement branch variables are still sampled lazily during instruction
execution.
"""
function sample_measurements!(
    runtime::FactoredExecutorState,
    program::FactoredInstructionProgram,
    samples::PresampledExogenous,
    shot::Integer;
    values = nothing,
)
    reset_executor!(runtime, program; values)
    execute!(runtime, program, samples, shot)
    return runtime.measurements
end

function reset_executor!(
    runtime::FactoredExecutorState,
    program::FactoredInstructionProgram;
    values = nothing,
)
    runtime.n = program.n
    runtime.k = program.initial_k
    runtime.ndormant = program.n - program.initial_k
    _reserve_active_capacity!(runtime, program.max_k)
    _set_active_from_initial!(runtime.active, program.initial_active)

    nsymbols = program.nsymbols
    resize!(runtime.values, nsymbols)
    fill!(runtime.values, false)
    resize!(runtime.assigned, nsymbols)
    fill!(runtime.assigned, false)
    nwords = _symbol_word_count(nsymbols)
    resize!(runtime.value_words, nwords)
    fill!(runtime.value_words, UInt64(0))
    resize!(runtime.assigned_words, nwords)
    fill!(runtime.assigned_words, UInt64(0))

    resize!(runtime.measurements, program.nrecords)
    fill!(runtime.measurements, false)
    _assign_input_values!(runtime, values)
    return runtime
end

function execute!(runtime::FactoredExecutorState, program::FactoredInstructionProgram)
    _check_executor_program(runtime, program)
    _sample_exogenous_symbols!(runtime, program)
    return _execute_factored_instructions!(runtime, program)
end

function execute!(
    runtime::FactoredExecutorState,
    program::FactoredInstructionProgram,
    samples::PresampledExogenous,
    shot::Integer,
)
    _check_executor_program(runtime, program)
    program.nsymbols == samples.nsymbols ||
        throw(DimensionMismatch("presampled exogenous table has $(samples.nsymbols) symbols; program has $(program.nsymbols)"))
    _assign_presampled_exogenous!(runtime, samples, shot)
    return _execute_factored_instructions!(runtime, program)
end

function _execute_factored_instructions!(
    runtime::FactoredExecutorState,
    program::FactoredInstructionProgram,
)
    for instruction in program.instructions
        execute_instruction!(runtime, instruction)
    end
    return runtime.measurements
end

function execute_instruction!(runtime::FactoredExecutorState, instruction::ApplyPrecomputedActivePauliRotation)
    sign = _eval_symbolic_bool(instruction.sign_plan, runtime)
    rotate_pauli!(runtime.active, instruction.rotation_kernel, sign)
    return runtime
end

function execute_instruction!(runtime::FactoredExecutorState, instruction::ApplyActiveBasisChange)
    _apply_active_basis_change!(runtime.active, instruction.kind, instruction.qubit)
    return runtime
end

function execute_instruction!(runtime::FactoredExecutorState, instruction::PromoteDormantRotation)
    sign = _eval_symbolic_bool(instruction.sign_plan, runtime)
    theta = sign ? -instruction.theta : instruction.theta
    _promote_first_dormant_rotation!(runtime, theta)
    return runtime
end

function execute_instruction!(runtime::FactoredExecutorState, instruction::RecordMeasurement)
    outcome = _eval_symbolic_bool(instruction.outcome_plan, runtime)
    _write_measurement_record!(runtime, instruction.record, outcome, instruction.record_condition)
    return runtime
end

function execute_instruction!(runtime::FactoredExecutorState, instruction::MeasureActiveLastZ)
    return _measure_active_last_z!(
        runtime,
        instruction.branch,
        instruction.outcome_plan,
        instruction.record,
        instruction.record_condition,
    )
end

function execute_instruction!(runtime::FactoredExecutorState, instruction::MeasurePrecomputedActivePauli)
    return _measure_precomputed_active_pauli!(
        runtime,
        instruction.kernel,
        instruction.branch,
        instruction.outcome_plan,
        instruction.record,
        instruction.record_condition,
    )
end

function _measure_active_last_z!(
    runtime::FactoredExecutorState,
    branch_condition::Int,
    outcome_plan::SymbolicBoolEvaluationPlan,
    record::Union{Nothing,Int},
    record_condition::Union{Nothing,Int},
)
    runtime.k > 0 || throw(ArgumentError("cannot measure the last active qubit when k == 0"))
    prob1 = _active_last_z_probability_one(runtime.active)
    branch = _sample_bernoulli(runtime.rng, prob1)
    _assign_symbol!(runtime, branch_condition, branch)
    _project_active_last_z!(runtime.active, branch, prob1)
    runtime.k = runtime.active.k
    runtime.ndormant += 1

    outcome = _eval_symbolic_bool(outcome_plan, runtime)
    _write_measurement_record!(runtime, record, outcome, record_condition)
    return runtime
end

function _measure_precomputed_active_pauli!(
    runtime::FactoredExecutorState,
    kernel::PrecomputedActivePauliMeasurementKernel,
    branch_condition::Int,
    outcome_plan::SymbolicBoolEvaluationPlan,
    record::Union{Nothing,Int},
    record_condition::Union{Nothing,Int},
)
    runtime.k > 0 || throw(ArgumentError("cannot measure an active Pauli when k == 0"))
    _check_active_measurement_kernel(runtime.k, kernel)
    if kernel.is_diagonal
        return _measure_diagonal_active_pauli!(
            runtime,
            kernel,
            branch_condition,
            outcome_plan,
            record,
            record_condition,
        )
    end
    prob_true = _active_measurement_branch_probability(runtime.active, kernel, true)
    prob_false, prob_true = _active_measurement_complementary_branch_probabilities(prob_true)
    prob1 = prob_true
    branch = _sample_bernoulli(runtime.rng, prob1)
    probability = branch ? prob_true : prob_false
    _assign_symbol!(runtime, branch_condition, branch)
    _project_active_pauli_measurement!(runtime.active, runtime.active_scratch, kernel, branch, probability)
    runtime.k = runtime.active.k
    runtime.ndormant += 1

    outcome = _eval_symbolic_bool(outcome_plan, runtime)
    _write_measurement_record!(runtime, record, outcome, record_condition)
    return runtime
end

function _measure_diagonal_active_pauli!(
    runtime::FactoredExecutorState,
    kernel::PrecomputedActivePauliMeasurementKernel,
    branch_condition::Int,
    outcome_plan::SymbolicBoolEvaluationPlan,
    record::Union{Nothing,Int},
    record_condition::Union{Nothing,Int},
)
    prob_true = _active_diagonal_measurement_branch_probability(runtime.active, kernel, true)
    prob_false, prob_true = _active_measurement_complementary_branch_probabilities(prob_true)
    prob1 = prob_true
    branch = _sample_bernoulli(runtime.rng, prob1)
    probability = branch ? prob_true : prob_false
    _assign_symbol!(runtime, branch_condition, branch)
    _project_diagonal_active_pauli_measurement!(runtime.active, kernel, branch, probability)
    runtime.k = runtime.active.k
    runtime.ndormant += 1

    outcome = _eval_symbolic_bool(outcome_plan, runtime)
    _write_measurement_record!(runtime, record, outcome, record_condition)
    return runtime
end

@inline function _active_measurement_complementary_branch_probabilities(prob_true::Float64)
    pt = min(max(prob_true, 0.0), 1.0)
    return 1.0 - pt, pt
end

function _active_diagonal_measurement_branch_probability(
    active::ActiveState,
    kernel::PrecomputedActivePauliMeasurementKernel,
    branch::Bool,
)
    sources = branch ? kernel.source0_true : kernel.source0_false
    prob = 0.0
    @inbounds @simd for idx in eachindex(sources)
        prob += abs2(active.alpha[sources[idx]])
    end
    return min(max(prob, 0.0), 1.0)
end

function _active_measurement_branch_probability(
    active::ActiveState,
    kernel::PrecomputedActivePauliMeasurementKernel,
    branch::Bool,
)
    sources0, sources1, coeffs0, coeffs1 = _measurement_kernel_branch(kernel, branch)
    prob = 0.0
    @inbounds @simd for idx in eachindex(sources0)
        amp = coeffs0[idx] * active.alpha[sources0[idx]]
        src1 = sources1[idx]
        src1 != 0 && (amp += coeffs1[idx] * active.alpha[src1])
        prob += abs2(amp)
    end
    return min(max(prob, 0.0), 1.0)
end

function _project_active_pauli_measurement!(
    active::ActiveState,
    scratch::Vector{ComplexF64},
    kernel::PrecomputedActivePauliMeasurementKernel,
    branch::Bool,
    probability::Float64,
)
    probability > 0.0 ||
        throw(ArgumentError("sampled an impossible active measurement branch"))
    sources0, sources1, coeffs0, coeffs1 = _measurement_kernel_branch(kernel, branch)
    out_dim = length(sources0)
    length(scratch) >= out_dim ||
        throw(DimensionMismatch("active scratch has too few entries for measurement projection"))
    invnorm = inv(sqrt(probability))
    @inbounds @simd for idx in 1:out_dim
        amp = coeffs0[idx] * active.alpha[sources0[idx]]
        src1 = sources1[idx]
        src1 != 0 && (amp += coeffs1[idx] * active.alpha[src1])
        scratch[idx] = amp * invnorm
    end
    copyto!(active.alpha, 1, scratch, 1, out_dim)
    active.k -= 1
    return active
end

function _project_diagonal_active_pauli_measurement!(
    active::ActiveState,
    kernel::PrecomputedActivePauliMeasurementKernel,
    branch::Bool,
    probability::Float64,
)
    probability > 0.0 ||
        throw(ArgumentError("sampled an impossible active measurement branch"))
    sources = branch ? kernel.source0_true : kernel.source0_false
    out_dim = length(sources)
    invnorm = inv(sqrt(probability))
    @inbounds @simd for idx in 1:out_dim
        active.alpha[idx] = active.alpha[sources[idx]] * invnorm
    end
    active.k -= 1
    return active
end

function _measurement_kernel_branch(
    kernel::PrecomputedActivePauliMeasurementKernel,
    branch::Bool,
)
    return branch ?
        (kernel.source0_true, kernel.source1_true, kernel.coeff0_true, kernel.coeff1_true) :
        (kernel.source0_false, kernel.source1_false, kernel.coeff0_false, kernel.coeff1_false)
end

function execute_instruction!(runtime::FactoredExecutorState, instruction::IntroduceDormantMeasurementBranch)
    branch = _sample_bernoulli(runtime.rng, 0.5)
    _assign_symbol!(runtime, instruction.branch, branch)
    outcome = _eval_symbolic_bool(instruction.outcome_plan, runtime)
    _write_measurement_record!(runtime, instruction.record, outcome, instruction.record_condition)
    return runtime
end

function _check_executor_program(runtime::FactoredExecutorState, program::FactoredInstructionProgram)
    runtime.n == program.n ||
        throw(DimensionMismatch("executor has $(runtime.n) qubits; program has $(program.n)"))
    runtime.k == runtime.active.k ||
        throw(DimensionMismatch("executor k=$(runtime.k) but active state has k=$(runtime.active.k)"))
    runtime.k + runtime.ndormant == runtime.n ||
        throw(DimensionMismatch("executor active+dormant width does not equal n"))
    return nothing
end

function _assign_input_values!(runtime::FactoredExecutorState, values)
    values === nothing && return runtime
    for (condition, value) in values
        _assign_symbol!(runtime, condition, Bool(value))
    end
    return runtime
end

function _sample_exogenous_symbols!(runtime::FactoredExecutorState, program::FactoredInstructionProgram)
    # Categorical distributions are sampled as joint rows before independent
    # Bernoulli symbols. The context rejects overlapping categorical/Bernoulli
    # ownership when the symbols are created.
    for distribution in program.sampled_categorical_distributions
        _sample_categorical_distribution!(runtime, distribution)
    end
    for group in program.sampled_rare_categorical_groups
        _sample_rare_categorical_group!(runtime, group)
    end
    for idx in eachindex(program.sampled_bernoulli_conditions, program.sampled_bernoulli_probabilities)
        condition = program.sampled_bernoulli_conditions[idx]
        _is_assigned(runtime, condition) && continue
        _assign_symbol!(
            runtime,
            condition,
            _sample_bernoulli(runtime.rng, program.sampled_bernoulli_probabilities[idx]),
        )
    end
    for group in program.sampled_low_probability_bernoulli_groups
        _sample_low_probability_bernoulli_group!(runtime, group)
    end
    return runtime
end

"""
    presample_exogenous(program, nshots; rng)

Generate a packed `shots × symbols` block of exogenous random bits for
`program`. The block starts all false; Bernoulli/categorical true bits are xored
into it, with low-probability Bernoulli symbols and rare categorical rows
sampled by geometric gaps.
"""
function presample_exogenous(
    program::FactoredInstructionProgram,
    nshots::Integer;
    rng = nothing,
)
    shots = _checked_presample_shots(nshots)
    nwords = _symbol_word_count(program.nsymbols)
    value_words = zeros(UInt64, shots * nwords)
    exogenous_assigned_words = _exogenous_assigned_words(program)

    for distribution in program.sampled_categorical_distributions
        _presample_categorical_distribution!(value_words, nwords, rng, distribution, shots)
    end
    for group in program.sampled_rare_categorical_groups
        _presample_rare_categorical_group!(value_words, nwords, rng, group, shots)
    end
    for idx in eachindex(program.sampled_bernoulli_conditions, program.sampled_bernoulli_probabilities)
        _presample_bernoulli_condition!(
            value_words,
            nwords,
            rng,
            program.sampled_bernoulli_conditions[idx],
            program.sampled_bernoulli_probabilities[idx],
            shots,
        )
    end
    for group in program.sampled_low_probability_bernoulli_groups
        _presample_low_probability_bernoulli_group!(value_words, nwords, rng, group, shots)
    end

    return PresampledExogenous(
        shots,
        program.nsymbols,
        nwords,
        exogenous_assigned_words,
        value_words,
    )
end

@inline function _checked_presample_shots(nshots::Integer)::Int
    shots = Int(nshots)
    shots > 0 || throw(ArgumentError("presampled shot count must be positive"))
    return shots
end

function _exogenous_assigned_words(program::FactoredInstructionProgram)
    words = zeros(UInt64, _symbol_word_count(program.nsymbols))
    for distribution in program.sampled_categorical_distributions
        _mark_exogenous_conditions!(words, distribution.conditions)
    end
    for group in program.sampled_rare_categorical_groups
        for set in group.conditions
            _mark_exogenous_conditions!(words, set)
        end
    end
    _mark_exogenous_conditions!(words, program.sampled_bernoulli_conditions)
    for group in program.sampled_low_probability_bernoulli_groups
        _mark_exogenous_conditions!(words, group.conditions)
    end
    return words
end

function _mark_exogenous_conditions!(words::Vector{UInt64}, conditions::Vector{Int})
    for condition in conditions
        words[_symbol_word_index(condition)] |= _symbol_bit_mask(condition)
    end
    return words
end

function _presample_categorical_distribution!(
    value_words::Vector{UInt64},
    nwords::Int,
    rng,
    distribution::SymbolicCategoricalDistribution,
    nshots::Int,
)
    for shot in 1:nshots
        row = _sample_categorical_row(rng, distribution.probabilities)
        assignment = distribution.assignments[row]
        @inbounds for bit_idx in eachindex(distribution.conditions, assignment)
            assignment[bit_idx] &&
                _xor_presampled_condition!(value_words, nwords, shot, distribution.conditions[bit_idx])
        end
    end
    return value_words
end

function _presample_rare_categorical_group!(
    value_words::Vector{UInt64},
    nwords::Int,
    rng,
    group::RareCategoricalSampleGroup,
    nshots::Int,
)
    nsets = length(group.conditions)
    total_draws = nshots * nsets
    group.event_probability <= 0.0 && return value_words
    draw = 1
    while true
        gap = _sample_geometric_gap(rng, group.event_probability)
        gap > total_draws - draw && return value_words
        draw += gap
        shot = ((draw - 1) ÷ nsets) + 1
        set_idx = ((draw - 1) % nsets) + 1
        row = group.event_rows[
            _sample_categorical_row(rng, group.event_probabilities)
        ]
        conditions = group.conditions[set_idx]
        assignment = group.assignments[row]
        @inbounds for bit_idx in eachindex(conditions, assignment)
            assignment[bit_idx] &&
                _xor_presampled_condition!(value_words, nwords, shot, conditions[bit_idx])
        end
        draw += 1
    end
end

function _presample_bernoulli_condition!(
    value_words::Vector{UInt64},
    nwords::Int,
    rng,
    condition::Int,
    probability::Float64,
    nshots::Int,
)
    p = _check_probability(probability)
    p <= 0.0 && return value_words
    if p >= 1.0
        for shot in 1:nshots
            _xor_presampled_condition!(value_words, nwords, shot, condition)
        end
        return value_words
    elseif p < LOW_PROBABILITY_SAMPLE_THRESHOLD
        draw = 1
        while true
            gap = _sample_geometric_gap(rng, p)
            gap > nshots - draw && return value_words
            draw += gap
            _xor_presampled_condition!(value_words, nwords, draw, condition)
            draw += 1
        end
    end

    for shot in 1:nshots
        _sample_bernoulli(rng, p) &&
            _xor_presampled_condition!(value_words, nwords, shot, condition)
    end
    return value_words
end

function _presample_low_probability_bernoulli_group!(
    value_words::Vector{UInt64},
    nwords::Int,
    rng,
    group::BernoulliSampleGroup,
    nshots::Int,
)
    nconditions = length(group.conditions)
    total_draws = nshots * nconditions
    group.probability <= 0.0 && return value_words
    draw = 1
    while true
        gap = _sample_geometric_gap(rng, group.probability)
        gap > total_draws - draw && return value_words
        draw += gap
        shot = ((draw - 1) ÷ nconditions) + 1
        condition_idx = ((draw - 1) % nconditions) + 1
        _xor_presampled_condition!(value_words, nwords, shot, group.conditions[condition_idx])
        draw += 1
    end
end

@inline function _presampled_word_index(nwords::Int, shot::Int, word::Int)::Int
    return (shot - 1) * nwords + word
end

function _xor_presampled_condition!(
    value_words::Vector{UInt64},
    nwords::Int,
    shot::Int,
    condition::Int,
)
    word = _symbol_word_index(condition)
    value_words[_presampled_word_index(nwords, shot, word)] ⊻= _symbol_bit_mask(condition)
    return value_words
end

function _assign_presampled_exogenous!(
    runtime::FactoredExecutorState,
    samples::PresampledExogenous,
    shot::Integer,
)
    shot_idx = Int(shot)
    1 <= shot_idx <= samples.nshots ||
        throw(ArgumentError("presampled shot index $shot_idx is outside 1:$(samples.nshots)"))

    base = (shot_idx - 1) * samples.nwords
    _check_presampled_input_conflicts!(runtime, samples, base)
    @inbounds for word in eachindex(samples.exogenous_assigned_words)
        runtime.assigned_words[word] |= samples.exogenous_assigned_words[word]
    end
    _apply_presampled_true_bits_to_runtime!(runtime, samples, base)
    return runtime
end

function _check_presampled_input_conflicts!(
    runtime::FactoredExecutorState,
    samples::PresampledExogenous,
    base::Int,
)
    @inbounds for word in eachindex(samples.exogenous_assigned_words)
        overlap = runtime.assigned_words[word] & samples.exogenous_assigned_words[word]
        overlap == 0 && continue
        row = samples.value_words[base + word]
        while overlap != 0
            bit = trailing_zeros(overlap)
            condition = ((word - 1) << 6) + bit + 1
            expected = (row & (UInt64(1) << bit)) != 0
            runtime.values[condition] == expected ||
                throw(ArgumentError("condition s$condition was assigned inconsistent concrete values"))
            overlap &= overlap - UInt64(1)
        end
    end
    return runtime
end

function _apply_presampled_true_bits_to_runtime!(
    runtime::FactoredExecutorState,
    samples::PresampledExogenous,
    base::Int,
)
    @inbounds for word in 1:samples.nwords
        bits = samples.value_words[base + word]
        bits == 0 && continue
        runtime.value_words[word] |= bits
        while bits != 0
            bit = trailing_zeros(bits)
            condition = ((word - 1) << 6) + bit + 1
            condition <= length(runtime.values) && (runtime.values[condition] = true)
            bits &= bits - UInt64(1)
        end
    end
    return runtime
end

function _sample_categorical_distribution!(
    runtime::FactoredExecutorState,
    distribution::SymbolicCategoricalDistribution,
)
    return _sample_categorical_distribution!(
        runtime,
        distribution.conditions,
        distribution.assignments,
        distribution.probabilities,
    )
end

function _sample_categorical_distribution!(
    runtime::FactoredExecutorState,
    conditions::Vector{Int},
    assignments::Vector{Vector{Bool}},
    probabilities::Vector{Float64},
)
    all_assigned = true
    any_assigned = false
    for condition in conditions
        assigned = _is_assigned(runtime, condition)
        all_assigned &= assigned
        any_assigned |= assigned
    end
    all_assigned && return runtime
    any_assigned &&
        throw(ArgumentError("categorical symbolic distribution was only partially preassigned"))

    row = _sample_categorical_row(runtime.rng, probabilities)
    assignment = assignments[row]
    for (condition, value) in zip(conditions, assignment)
        _assign_symbol!(runtime, condition, value)
    end
    return runtime
end

function _sample_categorical_row(rng, probabilities::Vector{Float64})::Int
    r = _rand_float(rng)
    cumulative = 0.0
    for idx in eachindex(probabilities)
        cumulative += probabilities[idx]
        r <= cumulative && return idx
    end
    return length(probabilities)
end

function _sample_rare_categorical_group!(
    runtime::FactoredExecutorState,
    group::RareCategoricalSampleGroup,
)
    if _any_categorical_group_assigned(runtime, group.conditions)
        for conditions in group.conditions
            _sample_categorical_distribution!(
                runtime,
                conditions,
                group.assignments,
                group.probabilities,
            )
        end
        return runtime
    end

    _assign_categorical_group_false!(runtime, group.conditions)
    group.event_probability <= 0.0 && return runtime
    idx = 1
    while true
        gap = _sample_geometric_gap(runtime.rng, group.event_probability)
        gap > length(group.conditions) - idx && return runtime
        idx += gap
        row = group.event_rows[
            _sample_categorical_row(runtime.rng, group.event_probabilities)
        ]
        conditions = group.conditions[idx]
        assignment = group.assignments[row]
        @inbounds for bit_idx in eachindex(conditions, assignment)
            assignment[bit_idx] && _set_assigned_symbol_true_unchecked!(runtime, conditions[bit_idx])
        end
        idx += 1
    end
end

function _sample_low_probability_bernoulli_group!(
    runtime::FactoredExecutorState,
    group::BernoulliSampleGroup,
)
    if _any_assigned(runtime, group.conditions)
        for condition in group.conditions
            _is_assigned(runtime, condition) && continue
            _assign_symbol!(runtime, condition, _sample_bernoulli(runtime.rng, group.probability))
        end
        return runtime
    end

    _assign_conditions_false!(runtime, group.conditions)
    group.probability <= 0.0 && return runtime
    idx = 1
    while true
        gap = _sample_geometric_gap(runtime.rng, group.probability)
        gap > length(group.conditions) - idx && return runtime
        idx += gap
        _set_assigned_symbol_true_unchecked!(runtime, group.conditions[idx])
        idx += 1
    end
end

function _sample_bernoulli(rng, probability::Real)::Bool
    p = _check_probability(probability)
    p <= 0.0 && return false
    p >= 1.0 && return true
    return _rand_float(rng) < p
end

_rand_float(rng)::Float64 = rng === nothing ? rand() : rand(rng)

function _sample_geometric_gap(rng, probability::Float64)::Int
    0.0 < probability < 1.0 ||
        throw(ArgumentError("geometric gap probability must be in (0, 1); got $probability"))
    u = max(_rand_float(rng), floatmin(Float64))
    gap = floor(log(u) / log1p(-probability))
    (!isfinite(gap) || gap >= typemax(Int)) && return typemax(Int)
    return Int(gap)
end

function _any_categorical_group_assigned(runtime::FactoredExecutorState, condition_sets::Vector{Vector{Int}})
    for conditions in condition_sets
        _any_assigned(runtime, conditions) && return true
    end
    return false
end

function _any_assigned(runtime::FactoredExecutorState, conditions::Vector{Int})
    for condition in conditions
        _is_assigned(runtime, condition) && return true
    end
    return false
end

function _assign_categorical_group_false!(
    runtime::FactoredExecutorState,
    condition_sets::Vector{Vector{Int}},
)
    for conditions in condition_sets
        _assign_conditions_false!(runtime, conditions)
    end
    return runtime
end

function _assign_conditions_false!(runtime::FactoredExecutorState, conditions::Vector{Int})
    @inbounds for condition in conditions
        _set_symbol_assignment_unchecked!(runtime, condition, false)
    end
    return runtime
end

function _set_symbol_assignment_unchecked!(
    runtime::FactoredExecutorState,
    condition::Int,
    value::Bool,
)
    @inbounds begin
        runtime.assigned[condition] = true
        word = _symbol_word_index(condition)
        mask = _symbol_bit_mask(condition)
        if value
            runtime.values[condition] = true
            runtime.value_words[word] |= mask
        end
        runtime.assigned_words[word] |= mask
    end
    return runtime
end

function _set_assigned_symbol_true_unchecked!(runtime::FactoredExecutorState, condition::Int)
    @inbounds begin
        runtime.values[condition] = true
        runtime.value_words[_symbol_word_index(condition)] |= _symbol_bit_mask(condition)
    end
    return runtime
end

function _check_symbol_slot(runtime::FactoredExecutorState, condition::Integer)
    c = Int(condition)
    c > 0 || throw(ArgumentError("condition id must be positive"))
    c <= length(runtime.values) ||
        throw(ArgumentError("condition s$c exceeds executor symbol table length $(length(runtime.values))"))
    return c
end

function _is_assigned(runtime::FactoredExecutorState, condition::Integer)
    c = _check_symbol_slot(runtime, condition)
    @inbounds return runtime.assigned[c]
end

function _assign_symbol!(runtime::FactoredExecutorState, condition::Integer, value::Bool)
    c = _check_symbol_slot(runtime, condition)
    @inbounds if runtime.assigned[c]
        runtime.values[c] == value ||
            throw(ArgumentError("condition s$c was assigned inconsistent concrete values"))
        return runtime
    end
    _set_symbol_assignment_unchecked!(runtime, c, value)
    return runtime
end

function _assign_symbol!(runtime::FactoredExecutorState, condition::Nothing, value::Bool)
    return runtime
end

function _eval_symbolic_bool(expr::SymbolicBool, runtime::FactoredExecutorState)
    return _eval_symbolic_bool(SymbolicBoolEvaluationPlan(expr), runtime)
end

function _eval_symbolic_bool(plan::SymbolicBoolEvaluationPlan, runtime::FactoredExecutorState)
    isempty(plan.word_indices) && return _eval_symbolic_bool_scalar(plan, runtime)
    return _eval_symbolic_bool_packed(plan, runtime)
end

function _eval_symbolic_bool_scalar(plan::SymbolicBoolEvaluationPlan, runtime::FactoredExecutorState)
    out = plan.constant
    for condition in plan.conditions
        _symbol_assigned(runtime, condition) ||
            throw(ArgumentError("symbolic condition s$condition has no concrete value"))
        @inbounds out = xor(out, runtime.values[condition])
    end
    return out
end

@inline function _symbol_assigned(runtime::FactoredExecutorState, condition::Int)::Bool
    condition <= length(runtime.assigned) || return false
    word = _symbol_word_index(condition)
    word <= length(runtime.assigned_words) || return false
    return (runtime.assigned_words[word] & _symbol_bit_mask(condition)) != 0
end

function _eval_symbolic_bool_packed(plan::SymbolicBoolEvaluationPlan, runtime::FactoredExecutorState)
    @inbounds max_word = plan.word_indices[end]
    max_word <= length(runtime.assigned_words) ||
        throw(ArgumentError("symbolic condition expression has no concrete value"))
    out = plan.constant
    assigned_words = runtime.assigned_words
    value_words = runtime.value_words
    missing = UInt64(0)
    @inbounds for idx in eachindex(plan.word_indices, plan.word_masks)
        word = plan.word_indices[idx]
        mask = plan.word_masks[idx]
        missing |= mask & ~assigned_words[word]
        out = xor(out, isodd(count_ones(value_words[word] & mask)))
    end
    missing == UInt64(0) ||
        throw(ArgumentError("symbolic condition expression has no concrete value"))
    return out
end

function _write_measurement_record!(
    runtime::FactoredExecutorState,
    record::Nothing,
    outcome::Bool,
    record_condition::Union{Nothing,Int},
)
    _assign_symbol!(runtime, record_condition, outcome)
    return runtime
end

function _write_measurement_record!(
    runtime::FactoredExecutorState,
    record::Integer,
    outcome::Bool,
    record_condition::Union{Nothing,Int},
)
    ri = Int(record)
    ri > 0 || throw(ArgumentError("measurement record id must be positive"))
    if ri > length(runtime.measurements)
        resize!(runtime.measurements, ri)
    end
    runtime.measurements[ri] = outcome
    _assign_symbol!(runtime, record_condition, outcome)
    return runtime
end

function _promote_first_dormant_rotation!(runtime::FactoredExecutorState, theta::Float64)
    runtime.ndormant > 0 ||
        throw(ArgumentError("cannot promote a dormant qubit when none remain"))
    old = runtime.active
    dim = _check_active_storage(old)
    promoted_dim = 2 * dim
    if length(old.alpha) < promoted_dim
        resize!(old.alpha, promoted_dim)
    end
    if length(runtime.active_scratch) < promoted_dim
        resize!(runtime.active_scratch, promoted_dim)
    end
    c = cos(theta)
    minus_i_s = ComplexF64(0.0, -sin(theta))
    @inbounds for basis in 0:(dim - 1)
        amp = old.alpha[basis + 1]
        old.alpha[basis + 1] = c * amp
        old.alpha[dim + basis + 1] = minus_i_s * amp
    end
    runtime.k += 1
    runtime.ndormant -= 1
    runtime.active.k = runtime.k
    return runtime
end

function _apply_active_basis_change!(active::ActiveState, kind::Symbol, q::Int)
    0 <= q < active.k || throw(ArgumentError("active basis-change qubit is out of range"))
    dim = _check_active_storage(active)
    mask = 1 << q
    if kind === :H
        invsqrt2 = inv(sqrt(2.0))
        @inbounds for base in 0:(dim - 1)
            (base & mask) == 0 || continue
            i0 = base + 1
            i1 = (base | mask) + 1
            a0 = active.alpha[i0]
            a1 = active.alpha[i1]
            active.alpha[i0] = (a0 + a1) * invsqrt2
            active.alpha[i1] = (a0 - a1) * invsqrt2
        end
    elseif kind === :S
        @inbounds for basis in 0:(dim - 1)
            (basis & mask) != 0 && (active.alpha[basis + 1] *= 1.0im)
        end
    else
        throw(ArgumentError("unsupported active basis change $kind"))
    end
    return active
end

function _active_last_z_probability_one(active::ActiveState)
    active.k > 0 || throw(ArgumentError("cannot measure last active qubit when k == 0"))
    mask = 1 << (active.k - 1)
    prob = 0.0
    dim = _check_active_storage(active)
    @inbounds for basis in 0:(dim - 1)
        (basis & mask) != 0 && (prob += abs2(active.alpha[basis + 1]))
    end
    return min(max(prob, 0.0), 1.0)
end

function _project_active_last_z!(active::ActiveState, branch::Bool, prob1::Float64)
    k = active.k
    k > 0 || throw(ArgumentError("cannot project last active qubit when k == 0"))
    new_k = k - 1
    dim = 1 << new_k
    probability = branch ? prob1 : 1.0 - prob1
    probability > 0.0 ||
        throw(ArgumentError("sampled an impossible active measurement branch"))
    invnorm = inv(sqrt(probability))
    branch_offset = branch ? dim : 0
    @inbounds for basis in 0:(dim - 1)
        active.alpha[basis + 1] = active.alpha[branch_offset + basis + 1] * invnorm
    end
    active.k = new_k
    return active
end

function _set_active_from_initial!(active::ActiveState, initial::ActiveState)
    active.k = initial.k
    dim = _check_active_storage(initial)
    if length(active.alpha) < dim
        resize!(active.alpha, dim)
    end
    copyto!(active.alpha, 1, initial.alpha, 1, dim)
    return active
end

function _reserve_active_capacity!(runtime::FactoredExecutorState, max_k::Integer)
    max_ki = _checked_active_k(max_k)
    max_dim = _active_length(max_ki)
    _reserve_active_capacity!(runtime.active, max_ki)
    if length(runtime.active_scratch) < max_dim
        resize!(runtime.active_scratch, max_dim)
    end
    return runtime
end

function _reserve_active_capacity!(active::ActiveState, max_k::Integer)
    max_ki = _checked_active_k(max_k)
    max_dim = _active_length(max_ki)
    if length(active.alpha) < max_dim
        resize!(active.alpha, max_dim)
    end
    return active
end
