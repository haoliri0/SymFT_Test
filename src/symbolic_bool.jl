"""
    SymbolicBool(constant, conditions)

Affine XOR expression over positive integer condition ids. Repeated ids cancel
modulo two, so `SymbolicBool(true, [1, 1, 3])` canonicalizes to `1 xor s3`.
"""
struct SymbolicBool
    constant::Bool
    conditions::Vector{Int}

    function SymbolicBool(constant::Bool, conditions::AbstractVector{<:Integer})
        return new(constant, _normalize_condition_xor(conditions))
    end
end

SymbolicBool(constant::Bool) = SymbolicBool(constant, Int[])

"""
    SymbolicCategoricalDistribution(conditions, assignments, probabilities)

Metadata for correlated symbolic bits. Each row of `assignments` gives one
joint value of `conditions`, with the matching probability in `probabilities`.
This is used for depolarizing channels where several Pauli bits are sampled
together instead of independently.
"""
struct SymbolicCategoricalDistribution
    conditions::Vector{Int}
    assignments::Vector{Vector{Bool}}
    probabilities::Vector{Float64}

    function SymbolicCategoricalDistribution(
        conditions::AbstractVector{<:Integer},
        assignments::AbstractVector{<:AbstractVector{Bool}},
        probabilities::AbstractVector{<:Real},
    )
        stored_conditions = Int.(conditions)
        isempty(stored_conditions) &&
            throw(ArgumentError("categorical symbolic distribution needs at least one condition"))
        all(>(0), stored_conditions) ||
            throw(ArgumentError("condition ids must be positive"))
        length(unique(stored_conditions)) == length(stored_conditions) ||
            throw(ArgumentError("categorical symbolic distribution condition ids must be distinct"))
        length(assignments) == length(probabilities) ||
            throw(ArgumentError("categorical assignments and probabilities must have the same length"))
        isempty(assignments) &&
            throw(ArgumentError("categorical symbolic distribution needs at least one assignment"))

        stored_assignments = Vector{Bool}[]
        for assignment in assignments
            length(assignment) == length(stored_conditions) ||
                throw(ArgumentError("categorical assignment length $(length(assignment)) does not match $(length(stored_conditions)) conditions"))
            push!(stored_assignments, Bool[bit for bit in assignment])
        end

        stored_probabilities = Float64.(probabilities)
        for probability in stored_probabilities
            _check_probability(probability)
        end
        isapprox(sum(stored_probabilities), 1.0; atol = 1e-12, rtol = 1e-12) ||
            throw(ArgumentError("categorical symbolic distribution probabilities must sum to 1"))

        return new(stored_conditions, stored_assignments, stored_probabilities)
    end
end

Base.copy(distribution::SymbolicCategoricalDistribution) =
    SymbolicCategoricalDistribution(
        distribution.conditions,
        distribution.assignments,
        distribution.probabilities,
    )
Base.:(==)(lhs::SymbolicCategoricalDistribution, rhs::SymbolicCategoricalDistribution) =
    lhs.conditions == rhs.conditions &&
    lhs.assignments == rhs.assignments &&
    lhs.probabilities == rhs.probabilities

mutable struct SymbolicContext
    next_condition::Int
    bernoulli_probabilities::Dict{Int,Float64}
    categorical_distributions::Vector{SymbolicCategoricalDistribution}
    condition_to_categorical::Dict{Int,Int}

    function SymbolicContext(next_condition::Integer = 1)
        next = Int(next_condition)
        next > 0 || throw(ArgumentError("next condition id must be positive"))
        return new(
            next,
            Dict{Int,Float64}(),
            SymbolicCategoricalDistribution[],
            Dict{Int,Int}(),
        )
    end
end

Base.copy(context::SymbolicContext) = begin
    out = SymbolicContext(context.next_condition)
    out.bernoulli_probabilities = copy(context.bernoulli_probabilities)
    out.categorical_distributions =
        SymbolicCategoricalDistribution[copy(distribution) for distribution in context.categorical_distributions]
    out.condition_to_categorical = copy(context.condition_to_categorical)
    out
end

Base.:(==)(lhs::SymbolicContext, rhs::SymbolicContext) =
    lhs.next_condition == rhs.next_condition &&
    lhs.bernoulli_probabilities == rhs.bernoulli_probabilities &&
    lhs.categorical_distributions == rhs.categorical_distributions &&
    lhs.condition_to_categorical == rhs.condition_to_categorical

next_condition(context::SymbolicContext)::Int = context.next_condition
bernoulli_probabilities(context::SymbolicContext) = copy(context.bernoulli_probabilities)
categorical_distributions(context::SymbolicContext) =
    SymbolicCategoricalDistribution[copy(distribution) for distribution in context.categorical_distributions]

function bernoulli_probability(context::SymbolicContext, condition::Integer)
    c = Int(condition)
    return get(context.bernoulli_probabilities, c, nothing)
end

function categorical_distribution(context::SymbolicContext, condition::Integer)
    c = Int(condition)
    idx = get(context.condition_to_categorical, c, 0)
    idx == 0 && return nothing
    return copy(context.categorical_distributions[idx])
end

"""
    symbolic_bool(condition)

Return the symbolic boolean variable with positive integer id `condition`.
Symbolic booleans are stored canonically as affine XOR expressions over these
ids plus a constant bit.
"""
symbolic_bool(condition::Integer) = SymbolicBool(false, [condition])

constant_term(expr::SymbolicBool)::Bool = expr.constant
condition_ids(expr::SymbolicBool) = copy(expr.conditions)
Base.copy(expr::SymbolicBool) = SymbolicBool(expr.constant, expr.conditions)
Base.:(==)(a::SymbolicBool, b::SymbolicBool) =
    a.constant == b.constant && a.conditions == b.conditions
Base.hash(expr::SymbolicBool, h::UInt) = hash(expr.conditions, hash(expr.constant, h))

function _normalize_condition_xor(conditions::AbstractVector{<:Integer})
    out = Int[]
    for condition in conditions
        _toggle_condition!(out, Int(condition))
    end
    return out
end

function _toggle_condition!(conditions::Vector{Int}, condition::Int)
    condition > 0 || throw(ArgumentError("condition id must be positive"))
    idx = searchsortedfirst(conditions, condition)
    # `conditions` is kept sorted, and duplicate ids cancel because symbolic
    # booleans are XOR expressions.
    if idx <= length(conditions) && conditions[idx] == condition
        deleteat!(conditions, idx)
    else
        insert!(conditions, idx, condition)
    end
    return conditions
end

function _max_condition(expr::SymbolicBool)
    isempty(expr.conditions) && return 0
    return expr.conditions[end]
end

function _check_probability(probability::Real)
    p = Float64(probability)
    0.0 <= p <= 1.0 || throw(ArgumentError("probability must be between 0 and 1; got $p"))
    return p
end

function _bump_next_condition!(context::SymbolicContext, condition::Integer)
    c = Int(condition)
    c >= 0 || throw(ArgumentError("condition id must be nonnegative"))
    context.next_condition = max(context.next_condition, c + 1)
    return context
end

function _bump_next_condition!(context::SymbolicContext, expr::SymbolicBool)
    return _bump_next_condition!(context, _max_condition(expr))
end

function fresh_condition!(context::SymbolicContext)
    condition = context.next_condition
    context.next_condition += 1
    return condition
end

fresh_symbolic_bool!(context::SymbolicContext) = symbolic_bool(fresh_condition!(context))

function register_bernoulli_condition!(context::SymbolicContext, condition::Integer, probability::Real)
    c = Int(condition)
    c > 0 || throw(ArgumentError("condition id must be positive"))
    haskey(context.condition_to_categorical, c) &&
        throw(ArgumentError("condition s$c is already part of a categorical distribution"))
    context.bernoulli_probabilities[c] = _check_probability(probability)
    _bump_next_condition!(context, c)
    return c
end

function fresh_bernoulli_condition!(context::SymbolicContext, probability::Real)
    p = _check_probability(probability)
    condition = fresh_condition!(context)
    register_bernoulli_condition!(context, condition, p)
    return condition
end

fresh_bernoulli_bool!(context::SymbolicContext, probability::Real) =
    symbolic_bool(fresh_bernoulli_condition!(context, probability))

function fresh_categorical_conditions!(
    context::SymbolicContext,
    assignments::AbstractVector{<:AbstractVector{Bool}},
    probabilities::AbstractVector{<:Real},
)
    isempty(assignments) &&
        throw(ArgumentError("categorical symbolic distribution needs at least one assignment"))
    nbits = length(assignments[1])
    nbits > 0 ||
        throw(ArgumentError("categorical symbolic distribution needs at least one bit"))
    # Validate the proposed table before consuming fresh condition ids.
    SymbolicCategoricalDistribution(collect(1:nbits), assignments, probabilities)
    conditions = Int[fresh_condition!(context) for _ in 1:nbits]
    distribution = SymbolicCategoricalDistribution(conditions, assignments, probabilities)
    push!(context.categorical_distributions, distribution)
    group = length(context.categorical_distributions)
    for condition in conditions
        context.condition_to_categorical[condition] = group
    end
    return conditions
end

function fresh_categorical_bools!(
    context::SymbolicContext,
    assignments::AbstractVector{<:AbstractVector{Bool}},
    probabilities::AbstractVector{<:Real},
)
    return SymbolicBool[symbolic_bool(condition) for condition in
        fresh_categorical_conditions!(context, assignments, probabilities)]
end

Base.:!(expr::SymbolicBool) = SymbolicBool(!expr.constant, expr.conditions)

function Base.xor(lhs::SymbolicBool, rhs::SymbolicBool)
    conditions = copy(lhs.conditions)
    for condition in rhs.conditions
        _toggle_condition!(conditions, condition)
    end
    return SymbolicBool(xor(lhs.constant, rhs.constant), conditions)
end

Base.xor(lhs::SymbolicBool, rhs::Bool) = SymbolicBool(xor(lhs.constant, rhs), lhs.conditions)
Base.xor(lhs::Bool, rhs::SymbolicBool) = xor(rhs, lhs)

function _as_symbolic_bool(value::SymbolicBool)
    return copy(value)
end

function _as_symbolic_bool(value::Bool)
    return SymbolicBool(value)
end

function _as_symbolic_bool(condition::Integer)
    return symbolic_bool(condition)
end

function Base.show(io::IO, expr::SymbolicBool)
    pieces = String[]
    expr.constant && push!(pieces, "1")
    for condition in expr.conditions
        push!(pieces, "s$condition")
    end
    if isempty(pieces)
        print(io, "0")
    else
        print(io, join(pieces, " xor "))
    end
end
