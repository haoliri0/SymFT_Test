"""
    FactoredInstruction

Abstract runtime instruction emitted by pending processing. Concrete
instructions act only on the active vector, symbolic condition table, dormant
branch symbols, and measurement record stream; none of them require a dense
physical `2^n` state vector.
"""
abstract type FactoredInstruction end

const PACKED_SYMBOLIC_EVAL_THRESHOLD = 32

@inline function _symbol_word_index(condition::Integer)::Int
    c = Int(condition)
    c > 0 || throw(ArgumentError("condition id must be positive"))
    return ((c - 1) >>> 6) + 1
end

@inline function _symbol_bit_mask(condition::Integer)::UInt64
    c = Int(condition)
    c > 0 || throw(ArgumentError("condition id must be positive"))
    return UInt64(1) << ((c - 1) & 63)
end

@inline _symbol_word_count(nsymbols::Integer)::Int = cld(max(Int(nsymbols), 0), 64)

"""
    SymbolicBoolEvaluationPlan(expr)

Precomputed runtime form of a symbolic XOR expression. Short expressions keep
their scalar condition list; longer expressions also cache packed word masks so
evaluation can use parity of `values & mask` chunks.
"""
struct SymbolicBoolEvaluationPlan
    constant::Bool
    conditions::Vector{Int}
    word_indices::Vector{Int}
    word_masks::Vector{UInt64}

    function SymbolicBoolEvaluationPlan(expr::SymbolicBool)
        word_indices = Int[]
        word_masks = UInt64[]
        if length(expr.conditions) > PACKED_SYMBOLIC_EVAL_THRESHOLD
            for condition in expr.conditions
                word = _symbol_word_index(condition)
                mask = _symbol_bit_mask(condition)
                if isempty(word_indices) || word_indices[end] != word
                    push!(word_indices, word)
                    push!(word_masks, mask)
                else
                    word_masks[end] |= mask
                end
            end
        end
        return new(expr.constant, copy(expr.conditions), word_indices, word_masks)
    end
end

Base.copy(plan::SymbolicBoolEvaluationPlan) =
    SymbolicBoolEvaluationPlan(
        SymbolicBool(plan.constant, plan.conditions),
    )
Base.:(==)(lhs::SymbolicBoolEvaluationPlan, rhs::SymbolicBoolEvaluationPlan) =
    lhs.constant == rhs.constant &&
    lhs.conditions == rhs.conditions &&
    lhs.word_indices == rhs.word_indices &&
    lhs.word_masks == rhs.word_masks

"""
    ApplyPrecomputedActivePauliRotation(pauli, theta, sign)

Runtime instruction for an active Pauli rotation. It uses a precomputed
signed-permutation plan for `P`, so each shot updates the active vector in
place by touching diagonal entries or disjoint amplitude pairs without building
`P*α`.
"""
struct ApplyPrecomputedActivePauliRotation <: FactoredInstruction
    pauli::PauliString
    action::ActivePauliAction
    rotation_kernel::PrecomputedActivePauliRotationKernel
    theta::Float64
    sign::SymbolicBool
    sign_plan::SymbolicBoolEvaluationPlan

    function ApplyPrecomputedActivePauliRotation(pauli::PauliString, theta::Real, sign::SymbolicBool)
        stored_pauli = copy(pauli)
        action = ActivePauliAction(stored_pauli)
        theta64 = Float64(theta)
        stored_sign = copy(sign)
        return new(
            stored_pauli,
            action,
            PrecomputedActivePauliRotationKernel(action, theta64),
            theta64,
            stored_sign,
            SymbolicBoolEvaluationPlan(stored_sign),
        )
    end
end

Base.copy(instruction::ApplyPrecomputedActivePauliRotation) =
    ApplyPrecomputedActivePauliRotation(instruction.pauli, instruction.theta, instruction.sign)
Base.:(==)(lhs::ApplyPrecomputedActivePauliRotation, rhs::ApplyPrecomputedActivePauliRotation) =
    lhs.pauli == rhs.pauli &&
    lhs.action == rhs.action &&
    lhs.rotation_kernel == rhs.rotation_kernel &&
    lhs.theta == rhs.theta &&
    lhs.sign == rhs.sign &&
    lhs.sign_plan == rhs.sign_plan

"""
    PromoteDormantRotation(theta, sign)

Runtime instruction emitted when a dormant X/Y rotation promotes the first
dormant qubit into the active subsystem.
"""
struct PromoteDormantRotation <: FactoredInstruction
    theta::Float64
    sign::SymbolicBool
    sign_plan::SymbolicBoolEvaluationPlan

    function PromoteDormantRotation(theta::Real, sign::SymbolicBool)
        stored_sign = copy(sign)
        return new(Float64(theta), stored_sign, SymbolicBoolEvaluationPlan(stored_sign))
    end
end

Base.copy(instruction::PromoteDormantRotation) =
    PromoteDormantRotation(instruction.theta, instruction.sign)
Base.:(==)(lhs::PromoteDormantRotation, rhs::PromoteDormantRotation) =
    lhs.theta == rhs.theta &&
    lhs.sign == rhs.sign &&
    lhs.sign_plan == rhs.sign_plan

"""
    RecordMeasurement(outcome, record, record_condition)

Record a deterministic symbolic measurement outcome. When present,
`record_condition` is the symbolic id reserved for the same Stim measurement
record, allowing later `rec[-k]` feedback to refer to this outcome.
"""
struct RecordMeasurement <: FactoredInstruction
    outcome::SymbolicBool
    outcome_plan::SymbolicBoolEvaluationPlan
    record::Union{Nothing,Int}
    record_condition::Union{Nothing,Int}

    function RecordMeasurement(
        outcome::SymbolicBool,
        record::Union{Nothing,Integer},
        record_condition::Union{Nothing,Integer} = nothing,
    )
        ri = _checked_instruction_record(record)
        condition = _checked_record_condition(record_condition)
        stored_outcome = copy(outcome)
        return new(stored_outcome, SymbolicBoolEvaluationPlan(stored_outcome), ri, condition)
    end
end

Base.copy(instruction::RecordMeasurement) =
    RecordMeasurement(instruction.outcome, instruction.record, instruction.record_condition)
Base.:(==)(lhs::RecordMeasurement, rhs::RecordMeasurement) =
    lhs.outcome == rhs.outcome &&
    lhs.outcome_plan == rhs.outcome_plan &&
    lhs.record == rhs.record &&
    lhs.record_condition == rhs.record_condition

"""
    MeasureActiveLastZ(branch, outcome, record, record_condition)

Runtime active measurement after planning has normalized the measured Pauli to
`Z` on the last active qubit.
"""
struct MeasureActiveLastZ <: FactoredInstruction
    branch::Int
    outcome::SymbolicBool
    outcome_plan::SymbolicBoolEvaluationPlan
    record::Union{Nothing,Int}
    record_condition::Union{Nothing,Int}

    function MeasureActiveLastZ(
        branch::Integer,
        outcome::SymbolicBool,
        record::Union{Nothing,Integer},
        record_condition::Union{Nothing,Integer} = nothing,
    )
        bi = Int(branch)
        bi > 0 || throw(ArgumentError("branch symbol id must be positive"))
        ri = _checked_instruction_record(record)
        condition = _checked_record_condition(record_condition)
        stored_outcome = copy(outcome)
        return new(bi, stored_outcome, SymbolicBoolEvaluationPlan(stored_outcome), ri, condition)
    end
end

Base.copy(instruction::MeasureActiveLastZ) =
    MeasureActiveLastZ(instruction.branch, instruction.outcome, instruction.record, instruction.record_condition)
Base.:(==)(lhs::MeasureActiveLastZ, rhs::MeasureActiveLastZ) =
    lhs.branch == rhs.branch &&
    lhs.outcome == rhs.outcome &&
    lhs.outcome_plan == rhs.outcome_plan &&
    lhs.record == rhs.record &&
    lhs.record_condition == rhs.record_condition

"""
    MeasurePrecomputedActivePauli(pauli, branch, outcome, record, record_condition)

Runtime active Pauli measurement with a precomputed projection kernel. The
pending planner has already updated the stabilizer-coordinate tableau for the
post-measurement basis; execution only samples a branch and applies the
corresponding amplitude projection.
"""
struct MeasurePrecomputedActivePauli <: FactoredInstruction
    pauli::PauliString
    kernel::PrecomputedActivePauliMeasurementKernel
    branch::Int
    outcome::SymbolicBool
    outcome_plan::SymbolicBoolEvaluationPlan
    record::Union{Nothing,Int}
    record_condition::Union{Nothing,Int}

    function MeasurePrecomputedActivePauli(
        pauli::PauliString,
        branch::Integer,
        outcome::SymbolicBool,
        record::Union{Nothing,Integer},
        record_condition::Union{Nothing,Integer} = nothing,
    )
        bi = Int(branch)
        bi > 0 || throw(ArgumentError("branch symbol id must be positive"))
        ri = _checked_instruction_record(record)
        condition = _checked_record_condition(record_condition)
        stored_outcome = copy(outcome)
        stored_pauli = copy(pauli)
        return new(
            stored_pauli,
            PrecomputedActivePauliMeasurementKernel(stored_pauli),
            bi,
            stored_outcome,
            SymbolicBoolEvaluationPlan(stored_outcome),
            ri,
            condition,
        )
    end
end

Base.copy(instruction::MeasurePrecomputedActivePauli) =
    MeasurePrecomputedActivePauli(
        instruction.pauli,
        instruction.branch,
        instruction.outcome,
        instruction.record,
        instruction.record_condition,
    )
Base.:(==)(lhs::MeasurePrecomputedActivePauli, rhs::MeasurePrecomputedActivePauli) =
    lhs.pauli == rhs.pauli &&
    lhs.kernel == rhs.kernel &&
    lhs.branch == rhs.branch &&
    lhs.outcome == rhs.outcome &&
    lhs.outcome_plan == rhs.outcome_plan &&
    lhs.record == rhs.record &&
    lhs.record_condition == rhs.record_condition

"""
    IntroduceDormantMeasurementBranch(branch, outcome, record, record_condition)

Runtime dormant X/Y measurement branch. The dormant basis remains in the
all-zero convention; the branch has already been pushed through pending Paulis.
"""
struct IntroduceDormantMeasurementBranch <: FactoredInstruction
    branch::Int
    outcome::SymbolicBool
    outcome_plan::SymbolicBoolEvaluationPlan
    record::Union{Nothing,Int}
    record_condition::Union{Nothing,Int}

    function IntroduceDormantMeasurementBranch(
        branch::Integer,
        outcome::SymbolicBool,
        record::Union{Nothing,Integer},
        record_condition::Union{Nothing,Integer} = nothing,
    )
        bi = Int(branch)
        bi > 0 || throw(ArgumentError("branch symbol id must be positive"))
        ri = _checked_instruction_record(record)
        condition = _checked_record_condition(record_condition)
        stored_outcome = copy(outcome)
        return new(bi, stored_outcome, SymbolicBoolEvaluationPlan(stored_outcome), ri, condition)
    end
end

Base.copy(instruction::IntroduceDormantMeasurementBranch) =
    IntroduceDormantMeasurementBranch(instruction.branch, instruction.outcome, instruction.record, instruction.record_condition)
Base.:(==)(lhs::IntroduceDormantMeasurementBranch, rhs::IntroduceDormantMeasurementBranch) =
    lhs.branch == rhs.branch &&
    lhs.outcome == rhs.outcome &&
    lhs.outcome_plan == rhs.outcome_plan &&
    lhs.record == rhs.record &&
    lhs.record_condition == rhs.record_condition

function _checked_instruction_record(record::Nothing)
    return nothing
end

function _checked_instruction_record(record::Integer)
    ri = Int(record)
    ri > 0 || throw(ArgumentError("measurement record id must be positive"))
    return ri
end

function _checked_record_condition(record_condition::Nothing)
    return nothing
end

function _checked_record_condition(record_condition::Integer)
    condition = Int(record_condition)
    condition > 0 || throw(ArgumentError("measurement record symbolic condition id must be positive"))
    return condition
end

const FactoredInstructionUnion = Union{
    ApplyPrecomputedActivePauliRotation,
    PromoteDormantRotation,
    RecordMeasurement,
    MeasureActiveLastZ,
    MeasurePrecomputedActivePauli,
    IntroduceDormantMeasurementBranch,
}

const LOW_PROBABILITY_SAMPLE_THRESHOLD = 0.02

"""
    BernoulliSampleGroup(probability, conditions)

Pre-sampling metadata for independent exogenous Bernoulli symbols with the
same probability. Low-probability groups are sampled by geometric gaps.
"""
struct BernoulliSampleGroup
    probability::Float64
    conditions::Vector{Int}

    function BernoulliSampleGroup(probability::Real, conditions::AbstractVector{<:Integer})
        p = _check_probability(probability)
        stored_conditions = Int[condition for condition in conditions]
        all(>(0), stored_conditions) ||
            throw(ArgumentError("Bernoulli sample group condition ids must be positive"))
        return new(p, stored_conditions)
    end
end

Base.copy(group::BernoulliSampleGroup) =
    BernoulliSampleGroup(group.probability, group.conditions)
Base.:(==)(lhs::BernoulliSampleGroup, rhs::BernoulliSampleGroup) =
    lhs.probability == rhs.probability && lhs.conditions == rhs.conditions

"""
    RareCategoricalSampleGroup

Pre-sampling metadata for categorical exogenous symbols whose all-false row has
probability above `1 - LOW_PROBABILITY_SAMPLE_THRESHOLD`. The geometric sampler
chooses which categorical draws leave the all-false row, then samples the rare
row conditioned on that event.
"""
struct RareCategoricalSampleGroup
    event_probability::Float64
    conditions::Vector{Vector{Int}}
    assignments::Vector{Vector{Bool}}
    probabilities::Vector{Float64}
    event_rows::Vector{Int}
    event_probabilities::Vector{Float64}

    function RareCategoricalSampleGroup(
        event_probability::Real,
        conditions::AbstractVector{<:AbstractVector{<:Integer}},
        assignments::AbstractVector{<:AbstractVector{Bool}},
        probabilities::AbstractVector{<:Real},
        event_rows::AbstractVector{<:Integer},
        event_probabilities::AbstractVector{<:Real},
    )
        event_p = _check_probability(event_probability)
        stored_conditions = Vector{Int}[Int[condition for condition in set] for set in conditions]
        stored_assignments = Vector{Bool}[Bool[bit for bit in assignment] for assignment in assignments]
        stored_probabilities = Float64[probability for probability in probabilities]
        stored_event_rows = Int[row for row in event_rows]
        stored_event_probabilities = Float64[probability for probability in event_probabilities]
        return new(
            event_p,
            stored_conditions,
            stored_assignments,
            stored_probabilities,
            stored_event_rows,
            stored_event_probabilities,
        )
    end
end

Base.copy(group::RareCategoricalSampleGroup) =
    RareCategoricalSampleGroup(
        group.event_probability,
        group.conditions,
        group.assignments,
        group.probabilities,
        group.event_rows,
        group.event_probabilities,
    )
Base.:(==)(lhs::RareCategoricalSampleGroup, rhs::RareCategoricalSampleGroup) =
    lhs.event_probability == rhs.event_probability &&
    lhs.conditions == rhs.conditions &&
    lhs.assignments == rhs.assignments &&
    lhs.probabilities == rhs.probabilities &&
    lhs.event_rows == rhs.event_rows &&
    lhs.event_probabilities == rhs.event_probabilities

"""
    FactoredInstructionProgram(n, initial_k, initial_active, instructions, max_k, context)

Executable instruction stream produced after draining a `PendingFactoredState`.
It carries the initial active state and the maximum active width required by
the plan, so a runtime can allocate once for repeated execution. The symbolic
context is copied into a compact sampling plan for exogenous Bernoulli and
categorical variables; measurement branch symbols are still assigned lazily by
measurement instructions.
"""
struct FactoredInstructionProgram
    n::Int
    initial_k::Int
    max_k::Int
    initial_active::ActiveState
    instructions::Vector{FactoredInstructionUnion}
    context::SymbolicContext
    nsymbols::Int
    nrecords::Int
    sampled_categorical_distributions::Vector{SymbolicCategoricalDistribution}
    sampled_rare_categorical_groups::Vector{RareCategoricalSampleGroup}
    sampled_bernoulli_conditions::Vector{Int}
    sampled_bernoulli_probabilities::Vector{Float64}
    sampled_low_probability_bernoulli_groups::Vector{BernoulliSampleGroup}

    function FactoredInstructionProgram(
        n::Integer,
        initial_k::Integer,
        initial_active::ActiveState,
        instructions::AbstractVector{<:FactoredInstruction},
        max_k::Integer = initial_k,
        context::SymbolicContext = SymbolicContext(),
    )
        ni = _checked_nqubits(n)
        ki = _checked_nqubits(initial_k)
        max_ki = _checked_nqubits(max_k)
        ki <= ni || throw(ArgumentError("initial active qubit count k=$ki exceeds n=$ni"))
        max_ki <= ni || throw(ArgumentError("maximum active qubit count max_k=$max_ki exceeds n=$ni"))
        ki <= max_ki ||
            throw(ArgumentError("maximum active qubit count max_k=$max_ki is smaller than initial k=$ki"))
        initial_active.k == ki ||
            throw(DimensionMismatch("initial ActiveState has $(initial_active.k) qubits; expected $ki"))
        stored_context = copy(context)
        stored = FactoredInstructionUnion[]
        record_count = 0
        for instruction in instructions
            _bump_next_condition!(stored_context, _max_condition(instruction))
            if :record in fieldnames(typeof(instruction))
                record = getfield(instruction, :record)
                record !== nothing && (record_count = max(record_count, record))
            end
            push!(stored, copy(instruction)::FactoredInstructionUnion)
        end
        symbol_count = max(0, stored_context.next_condition - 1)
        sampled_categorical_distributions, sampled_rare_categorical_groups =
            _build_categorical_sample_plan(stored_context.categorical_distributions)
        sampled_bernoulli_conditions = Int[]
        sampled_bernoulli_probabilities = Float64[]
        sampled_low_probability_bernoulli_groups = BernoulliSampleGroup[]
        for (condition, probability) in sort!(collect(stored_context.bernoulli_probabilities); by = first)
            if probability < LOW_PROBABILITY_SAMPLE_THRESHOLD
                _push_bernoulli_sample_group!(
                    sampled_low_probability_bernoulli_groups,
                    probability,
                    condition,
                )
            else
                push!(sampled_bernoulli_conditions, condition)
                push!(sampled_bernoulli_probabilities, probability)
            end
        end
        return new(
            ni,
            ki,
            max_ki,
            ActiveState(initial_active.k, copy(initial_active.alpha)),
            stored,
            stored_context,
            symbol_count,
            record_count,
            sampled_categorical_distributions,
            sampled_rare_categorical_groups,
            sampled_bernoulli_conditions,
            sampled_bernoulli_probabilities,
            sampled_low_probability_bernoulli_groups,
        )
    end
end

max_active_qubits(program::FactoredInstructionProgram)::Int = program.max_k

Base.copy(program::FactoredInstructionProgram) =
    FactoredInstructionProgram(
        program.n,
        program.initial_k,
        program.initial_active,
        program.instructions,
        program.max_k,
        program.context,
    )
Base.:(==)(lhs::FactoredInstructionProgram, rhs::FactoredInstructionProgram) =
    lhs.n == rhs.n &&
    lhs.initial_k == rhs.initial_k &&
    lhs.max_k == rhs.max_k &&
    lhs.initial_active.alpha == rhs.initial_active.alpha &&
    lhs.instructions == rhs.instructions &&
    lhs.context == rhs.context &&
    lhs.nsymbols == rhs.nsymbols &&
    lhs.nrecords == rhs.nrecords &&
    lhs.sampled_categorical_distributions == rhs.sampled_categorical_distributions &&
    lhs.sampled_rare_categorical_groups == rhs.sampled_rare_categorical_groups &&
    lhs.sampled_bernoulli_conditions == rhs.sampled_bernoulli_conditions &&
    lhs.sampled_bernoulli_probabilities == rhs.sampled_bernoulli_probabilities &&
    lhs.sampled_low_probability_bernoulli_groups == rhs.sampled_low_probability_bernoulli_groups

function _push_bernoulli_sample_group!(
    groups::Vector{BernoulliSampleGroup},
    probability::Float64,
    condition::Int,
)
    for idx in eachindex(groups)
        if groups[idx].probability == probability
            push!(groups[idx].conditions, condition)
            return groups
        end
    end
    push!(groups, BernoulliSampleGroup(probability, [condition]))
    return groups
end

function _build_categorical_sample_plan(
    distributions::Vector{SymbolicCategoricalDistribution},
)
    scalar = SymbolicCategoricalDistribution[]
    rare_groups = RareCategoricalSampleGroup[]
    for distribution in distributions
        info = _rare_categorical_sample_info(distribution)
        if info === nothing
            push!(scalar, copy(distribution))
        else
            _push_rare_categorical_sample_group!(rare_groups, distribution, info)
        end
    end
    return scalar, rare_groups
end

function _rare_categorical_sample_info(distribution::SymbolicCategoricalDistribution)
    false_row = 0
    for row in eachindex(distribution.assignments)
        if all(!, distribution.assignments[row])
            false_row = row
            break
        end
    end
    false_row == 0 && return nothing

    event_probability = 1.0 - distribution.probabilities[false_row]
    event_probability < LOW_PROBABILITY_SAMPLE_THRESHOLD || return nothing

    event_rows = Int[]
    event_probabilities = Float64[]
    if event_probability > 0.0
        inv_event_probability = inv(event_probability)
        for row in eachindex(distribution.assignments)
            row == false_row && continue
            probability = distribution.probabilities[row]
            probability <= 0.0 && continue
            push!(event_rows, row)
            push!(event_probabilities, probability * inv_event_probability)
        end
    end
    return (event_probability = event_probability,
        event_rows = event_rows,
        event_probabilities = event_probabilities)
end

function _push_rare_categorical_sample_group!(
    groups::Vector{RareCategoricalSampleGroup},
    distribution::SymbolicCategoricalDistribution,
    info,
)
    for idx in eachindex(groups)
        group = groups[idx]
        if group.event_probability == info.event_probability &&
                group.assignments == distribution.assignments &&
                group.probabilities == distribution.probabilities &&
                group.event_rows == info.event_rows &&
                group.event_probabilities == info.event_probabilities
            push!(group.conditions, copy(distribution.conditions))
            return groups
        end
    end
    push!(
        groups,
        RareCategoricalSampleGroup(
            info.event_probability,
            [distribution.conditions],
            distribution.assignments,
            distribution.probabilities,
            info.event_rows,
            info.event_probabilities,
        ),
    )
    return groups
end
