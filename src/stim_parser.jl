"""
    StimDetector(records, coords, line)

Detector annotation lowered to absolute Stim measurement-record indices.
`records` are 1-based positions in the public Stim measurement record stream.
"""
struct StimDetector
    records::Vector{Int}
    coords::Vector{Float64}
    line::Int

    function StimDetector(
        records::AbstractVector{<:Integer},
        coords::AbstractVector{<:Real},
        line::Integer,
    )
        stored_records = Int[Int(record) for record in records]
        all(>(0), stored_records) ||
            throw(ArgumentError("detector record ids must be positive"))
        return new(stored_records, Float64[coord for coord in coords], Int(line))
    end
end

Base.copy(detector::StimDetector) =
    StimDetector(detector.records, detector.coords, detector.line)
Base.:(==)(lhs::StimDetector, rhs::StimDetector) =
    lhs.records == rhs.records && lhs.coords == rhs.coords && lhs.line == rhs.line

"""
    StimObservableInclude(index, records, line)

Observable annotation lowered to absolute Stim measurement-record indices.
Each include xors the parity of `records` into observable `index`.
"""
struct StimObservableInclude
    index::Int
    records::Vector{Int}
    line::Int

    function StimObservableInclude(
        index::Integer,
        records::AbstractVector{<:Integer},
        line::Integer,
    )
        idx = Int(index)
        idx >= 0 || throw(ArgumentError("observable index must be nonnegative"))
        stored_records = Int[Int(record) for record in records]
        all(>(0), stored_records) ||
            throw(ArgumentError("observable record ids must be positive"))
        return new(idx, stored_records, Int(line))
    end
end

Base.copy(observable::StimObservableInclude) =
    StimObservableInclude(observable.index, observable.records, observable.line)
Base.:(==)(lhs::StimObservableInclude, rhs::StimObservableInclude) =
    lhs.index == rhs.index && lhs.records == rhs.records && lhs.line == rhs.line

"""
    StimParseResult

Result of parsing a local Stim file into the frame-factored ingestion API.
`measurement_records` contains symbolic aliases for public Stim measurement
records in Stim record order. Detector and observable annotations store
absolute indices into that same public record stream.
"""
struct StimParseResult
    state::FrameFactoredState
    measurement_records::Vector{SymbolicBool}
    detectors::Vector{StimDetector}
    observables::Vector{StimObservableInclude}
end

stim_state(result::StimParseResult) = result.state
stim_measurement_records(result::StimParseResult) =
    SymbolicBool[copy(record) for record in result.measurement_records]
stim_detectors(result::StimParseResult) =
    StimDetector[copy(detector) for detector in result.detectors]
stim_observable_includes(result::StimParseResult) =
    StimObservableInclude[copy(observable) for observable in result.observables]
stim_observables(result::StimParseResult) = stim_observable_includes(result)

"""
    parse_stim_file(path)
    parse_stim_text(text)
    parse_stim_lines(lines)

Parse the Stim operations used by the local `.stim` fixtures into the
frame-factored API. The parser supports the gate/annotation surface present in
`d3.stim` and `d5.stim`: Clifford `CX`/`CZ`/`H`/`S`/`S_DAG` style gates,
Pauli noise, depolarizing noise, Pauli measurements, MPP, exact resets,
measurement resets, detectors, observables, coordinate shifts,
record-controlled Pauli feedback, `REPEAT` blocks, and local `T`/`T_DAG`
extensions as `exp(-im * theta * Z)` rotations.

Measurement-record feedback is represented with reserved symbolic record
conditions attached to the corresponding pending measurements. Exact resets
use hidden measurement conditions that do not enter Stim's public measurement
record stream.
"""

struct _StimInstruction
    op::String
    tag::Union{Nothing,String}
    parens::Vector{Float64}
    targets::Vector{String}
    line::Int
end

struct _StimParsedLine
    instruction::_StimInstruction
    block_start::Bool
end

struct _StimRepeatBlock
    count::Int
    body::Vector{Any}
    line::Int
end

mutable struct _StimParseAccumulator
    state::FrameFactoredState
    records::Vector{SymbolicBool}
    detectors::Vector{StimDetector}
    observables::Vector{StimObservableInclude}
    coord_shift::Vector{Float64}
end

function _strip_stim_comment(line::AbstractString, line_number::Integer)
    in_brackets = false
    escaped = false
    for idx in eachindex(line)
        ch = line[idx]
        if in_brackets
            if escaped
                escaped = false
            elseif ch == '\\'
                escaped = true
            elseif ch == ']'
                in_brackets = false
            end
        elseif ch == '['
            in_brackets = true
        elseif ch == '#'
            before = idx == firstindex(line) ? "" : line[firstindex(line):prevind(line, idx)]
            return String(strip(before))
        end
    end
    escaped && throw(ArgumentError("unterminated Stim escape on line $line_number"))
    in_brackets && throw(ArgumentError("unterminated Stim bracketed target or tag on line $line_number"))
    return String(strip(line))
end

function _parse_stim_tag(head::AbstractString, idx::Int, line_number::Integer)
    out = IOBuffer()
    idx = nextind(head, idx)
    while idx <= lastindex(head)
        ch = head[idx]
        if ch == ']'
            return String(take!(out)), nextind(head, idx)
        elseif ch == '\\'
            next = nextind(head, idx)
            next <= lastindex(head) ||
                throw(ArgumentError("unterminated Stim tag escape on line $line_number"))
            esc = head[next]
            if esc == 'B'
                print(out, '\\')
            elseif esc == 'C'
                print(out, ']')
            elseif esc == 'r'
                print(out, '\r')
            elseif esc == 'n'
                print(out, '\n')
            else
                throw(ArgumentError("unsupported Stim tag escape \\$esc on line $line_number"))
            end
            idx = nextind(head, next)
        else
            print(out, ch)
            idx = nextind(head, idx)
        end
    end
    throw(ArgumentError("unterminated Stim tag on line $line_number"))
end

function _parse_stim_arguments(head::AbstractString, idx::Int, line_number::Integer)
    depth = 0
    start = idx
    while idx <= lastindex(head)
        ch = head[idx]
        if ch == '('
            depth += 1
        elseif ch == ')'
            depth -= 1
            depth < 0 && throw(ArgumentError("unbalanced parentheses on Stim line $line_number"))
            if depth == 0
                body = idx == nextind(head, start) ? "" : head[nextind(head, start):prevind(head, idx)]
                args = Float64[]
                if !isempty(strip(body))
                    for arg in split(body, ",")
                        push!(args, parse(Float64, strip(arg)))
                    end
                end
                return args, nextind(head, idx)
            end
        end
        idx = nextind(head, idx)
    end
    throw(ArgumentError("unbalanced parentheses on Stim line $line_number"))
end

function _parse_stim_instruction(body::AbstractString, line_number::Integer)
    m = match(r"^([A-Za-z][A-Za-z0-9_]*)", body)
    m === nothing && throw(ArgumentError("invalid Stim instruction on line $line_number"))
    name = m.captures[1]
    op = uppercase(name)
    idx = firstindex(body) + ncodeunits(name)
    tag = nothing
    parens = Float64[]

    if idx <= lastindex(body) && body[idx] == '['
        tag, idx = _parse_stim_tag(body, idx, line_number)
    end
    if idx <= lastindex(body) && body[idx] == '('
        parens, idx = _parse_stim_arguments(body, idx, line_number)
    end
    if idx <= lastindex(body) && !isspace(body[idx])
        throw(ArgumentError("invalid Stim instruction head on line $line_number"))
    end

    targets = String[]
    if idx <= lastindex(body)
        rest = strip(body[idx:lastindex(body)])
        !isempty(rest) && append!(targets, String.(split(rest)))
    end
    return _StimInstruction(op, tag, parens, targets, Int(line_number))
end

function _parse_stim_line(line::AbstractString, line_number::Integer)
    body = _strip_stim_comment(line, line_number)
    isempty(body) && return nothing
    body == "}" && return :block_end

    block_start = false
    if endswith(body, "{")
        block_start = true
        body = String(strip(body[firstindex(body):prevind(body, lastindex(body))]))
        isempty(body) && throw(ArgumentError("missing block instruction on line $line_number"))
    end
    return _StimParsedLine(_parse_stim_instruction(body, line_number), block_start)
end

function _stim_repeat_count(inst::_StimInstruction)
    inst.op == "REPEAT" ||
        throw(ArgumentError("unsupported Stim block instruction `$(inst.op)` on line $(inst.line)"))
    inst.tag === nothing || throw(ArgumentError("REPEAT on line $(inst.line) does not accept a tag"))
    isempty(inst.parens) || throw(ArgumentError("REPEAT on line $(inst.line) does not accept parens arguments"))
    length(inst.targets) == 1 ||
        throw(ArgumentError("REPEAT on line $(inst.line) expects one repeat count target"))
    occursin(r"^\d+$", inst.targets[1]) ||
        throw(ArgumentError("invalid REPEAT count `$(inst.targets[1])` on line $(inst.line)"))
    count = parse(Int, inst.targets[1])
    count > 0 || throw(ArgumentError("Stim REPEAT blocks must have positive repeat count"))
    return count
end

function _parse_stim_nodes(
    lines::AbstractVector{<:AbstractString},
    start::Integer = firstindex(lines),
    in_block::Bool = false,
)
    nodes = Any[]
    pos = Int(start)
    last = lastindex(lines)
    while pos <= last
        parsed = _parse_stim_line(lines[pos], pos)
        if parsed === nothing
            pos += 1
            continue
        elseif parsed === :block_end
            in_block || throw(ArgumentError("unmatched Stim block terminator on line $pos"))
            return nodes, pos + 1
        end

        if parsed.block_start
            count = _stim_repeat_count(parsed.instruction)
            body, next_pos = _parse_stim_nodes(lines, pos + 1, true)
            push!(nodes, _StimRepeatBlock(count, body, parsed.instruction.line))
            pos = next_pos
        else
            parsed.instruction.op == "REPEAT" &&
                throw(ArgumentError("REPEAT on line $(parsed.instruction.line) must start a block"))
            push!(nodes, parsed.instruction)
            pos += 1
        end
    end
    in_block && throw(ArgumentError("unterminated Stim block"))
    return nodes, pos
end

function _stim_paren_probability(inst::_StimInstruction, default::Float64 = 0.0)
    isempty(inst.parens) && return default
    length(inst.parens) == 1 ||
        throw(ArgumentError("$(inst.op) on line $(inst.line) expects at most one probability argument"))
    return _check_probability(inst.parens[1])
end

function _stim_required_probability(inst::_StimInstruction)
    length(inst.parens) == 1 ||
        throw(ArgumentError("$(inst.op) on line $(inst.line) expects one probability argument"))
    return _check_probability(inst.parens[1])
end

function _stim_qubit_target(target::AbstractString, line::Integer)
    inverted = startswith(target, "!")
    body = inverted ? target[2:end] : target
    occursin('*', body) &&
        throw(ArgumentError("expected a qubit target on line $line; got `$target`"))
    startswith(body, "rec[") &&
        throw(ArgumentError("expected a qubit target on line $line; got record target `$target`"))
    startswith(body, "sweep[") &&
        throw(ArgumentError("expected a qubit target on line $line; got sweep target `$target`"))
    m = match(r"^\d+$", body)
    m === nothing && throw(ArgumentError("invalid qubit target `$target` on line $line"))
    return parse(Int, body), inverted
end

function _stim_qubit_targets(inst::_StimInstruction)
    out = Int[]
    for target in inst.targets
        q, inverted = _stim_qubit_target(target, inst.line)
        inverted && throw(ArgumentError("$(inst.op) on line $(inst.line) does not accept inverted targets"))
        push!(out, q)
    end
    return out
end

function _stim_measurement_targets(inst::_StimInstruction)
    out = Tuple{Int,Bool}[]
    for target in inst.targets
        push!(out, _stim_qubit_target(target, inst.line))
    end
    return out
end

function _stim_record_index(target::AbstractString, nrecords::Integer, line::Integer)
    m = match(r"^rec\[(-\d+)\]$", target)
    m === nothing && throw(ArgumentError("invalid measurement record target `$target` on line $line"))
    offset = parse(Int, m.captures[1])
    idx = Int(nrecords) + offset + 1
    1 <= idx <= nrecords ||
        throw(BoundsError(1:Int(nrecords), idx))
    return idx
end

function _stim_record_indices(inst::_StimInstruction, records::Vector{SymbolicBool})
    out = Int[]
    for target in inst.targets
        push!(out, _stim_record_index(target, length(records), inst.line))
    end
    return out
end

function _stim_observable_index(inst::_StimInstruction)
    length(inst.parens) == 1 ||
        throw(ArgumentError("OBSERVABLE_INCLUDE on line $(inst.line) expects one observable index"))
    value = inst.parens[1]
    isinteger(value) && value >= 0 ||
        throw(ArgumentError("OBSERVABLE_INCLUDE on line $(inst.line) expects a nonnegative integer index"))
    return round(Int, value)
end

function _stim_mpp_targets(inst::_StimInstruction)
    groups = String[]
    current = ""
    for target in inst.targets
        if target == "*"
            !isempty(current) && !endswith(current, "*") ||
                throw(ArgumentError("misplaced MPP combiner on line $(inst.line)"))
            current *= "*"
        elseif isempty(current)
            current = target
        elseif endswith(current, "*")
            current *= target
        else
            push!(groups, current)
            current = target
        end
    end
    isempty(current) && return groups
    endswith(current, "*") &&
        throw(ArgumentError("dangling MPP combiner on line $(inst.line)"))
    push!(groups, current)
    return groups
end

function _stim_target_max_qubit(target::AbstractString, line::Integer)
    target == "*" && return -1
    body = startswith(target, "!") ? target[2:end] : target
    (startswith(body, "rec[") || startswith(body, "sweep[")) && return -1
    max_q = -1
    for raw_factor in split(body, "*")
        factor = String(raw_factor)
        isempty(factor) && throw(ArgumentError("empty Pauli-product factor on line $line"))
        startswith(factor, "!") && (factor = factor[2:end])
        if occursin(r"^[XYZ]\d+$", factor)
            max_q = max(max_q, parse(Int, factor[2:end]))
        elseif occursin(r"^\d+$", factor)
            max_q = max(max_q, parse(Int, factor))
        else
            throw(ArgumentError("invalid Stim target `$target` on line $line"))
        end
    end
    return max_q
end

function _stim_all_qubits(nodes::AbstractVector)
    max_q = -1
    for node in nodes
        if node isa _StimRepeatBlock
            max_q = max(max_q, _stim_all_qubits(node.body) - 1)
            continue
        end
        inst = node::_StimInstruction
        inst.op in ("DETECTOR", "OBSERVABLE_INCLUDE", "TICK", "SHIFT_COORDS") && continue
        for target in inst.targets
            max_q = max(max_q, _stim_target_max_qubit(target, inst.line))
        end
    end
    return max_q + 1
end

function _pauli_on_target(n::Integer, axis::AbstractString, q::Integer)
    axis == "X" && return pauli_x(n, q)
    axis == "Y" && return pauli_y(n, q)
    axis == "Z" && return pauli_z(n, q)
    throw(ArgumentError("unsupported Pauli axis `$axis`"))
end

function _stim_mpp_pauli(n::Integer, target::AbstractString, line::Integer)
    p = pauli_identity(n)
    inverted = false
    for raw_factor in split(target, "*")
        factor = String(raw_factor)
        if startswith(factor, "!")
            inverted = !inverted
            factor = factor[2:end]
        end
        m = match(r"^([XYZ])(\d+)$", factor)
        m === nothing && throw(ArgumentError("invalid MPP factor `$raw_factor` on line $line"))
        p = p * _pauli_on_target(n, m.captures[1], parse(Int, m.captures[2]))
    end
    return p, inverted
end

function _reserve_measurement_record!(records::Vector{SymbolicBool}, context::SymbolicContext)
    condition = fresh_condition!(context)
    # Reserve an alias now so later `rec[-k]` feedback can be converted into a
    # symbolic conditional Pauli before the pending measurement is lowered.
    push!(records, symbolic_bool(condition))
    return length(records), condition
end

function _measurement_sign_with_readout_error!(
    context::SymbolicContext,
    inverted::Bool,
    probability::Float64,
)
    sign = SymbolicBool(inverted)
    probability != 0.0 && (sign = xor(sign, fresh_bernoulli_bool!(context, probability)))
    return sign
end

function _coords_with_shift(coords::Vector{Float64}, shift::Vector{Float64})
    isempty(shift) && return copy(coords)
    out = Vector{Float64}(undef, max(length(coords), length(shift)))
    @inbounds for idx in eachindex(out)
        base = idx <= length(coords) ? coords[idx] : 0.0
        offset = idx <= length(shift) ? shift[idx] : 0.0
        out[idx] = base + offset
    end
    return out
end

function _apply_shift_coords!(acc::_StimParseAccumulator, inst::_StimInstruction)
    inst.tag === nothing || throw(ArgumentError("SHIFT_COORDS on line $(inst.line) does not accept a tag"))
    isempty(inst.targets) || throw(ArgumentError("SHIFT_COORDS on line $(inst.line) does not accept targets"))
    if length(acc.coord_shift) < length(inst.parens)
        old_len = length(acc.coord_shift)
        resize!(acc.coord_shift, length(inst.parens))
        fill!(view(acc.coord_shift, (old_len + 1):length(acc.coord_shift)), 0.0)
    end
    @inbounds for idx in eachindex(inst.parens)
        acc.coord_shift[idx] += inst.parens[idx]
    end
    return acc
end

function _apply_depolarize1!(state::FrameFactoredState, q::Integer, probability::Float64)
    # Stim DEPOLARIZE1 samples I, Z, X, Y with probabilities
    # 1-p, p/3, p/3, p/3. The two symbolic bits are ordered as (x, z).
    assignments = Bool[
        false false
        false true
        true false
        true true
    ]
    bits = fresh_categorical_bools!(
        state.context,
        [vec(assignments[row, :]) for row in axes(assignments, 1)],
        [1.0 - probability, probability / 3, probability / 3, probability / 3],
    )
    apply_pauli!(state, pauli_x(state.n, q), bits[1])
    apply_pauli!(state, pauli_z(state.n, q), bits[2])
    return bits
end

function _apply_depolarize2!(
    state::FrameFactoredState,
    q1::Integer,
    q2::Integer,
    probability::Float64,
)
    assignments = Vector{Bool}[]
    push!(assignments, Bool[false, false, false, false])
    # Four correlated bits are ordered as (x1, z1, x2, z2). Every non-identity
    # two-qubit Pauli receives probability p/15.
    for x1 in (false, true), z1 in (false, true), x2 in (false, true), z2 in (false, true)
        assignment = Bool[x1, z1, x2, z2]
        any(assignment) || continue
        push!(assignments, assignment)
    end
    bits = fresh_categorical_bools!(
        state.context,
        assignments,
        vcat([1.0 - probability], fill(probability / 15, 15)),
    )
    apply_pauli!(state, pauli_x(state.n, q1), bits[1])
    apply_pauli!(state, pauli_z(state.n, q1), bits[2])
    apply_pauli!(state, pauli_x(state.n, q2), bits[3])
    apply_pauli!(state, pauli_z(state.n, q2), bits[4])
    return bits
end

function _apply_record_feedback!(
    state::FrameFactoredState,
    records::Vector{SymbolicBool},
    inst::_StimInstruction,
)
    length(inst.targets) % 2 == 0 ||
        throw(ArgumentError("record-controlled $(inst.op) on line $(inst.line) requires paired targets"))
    for idx in 1:2:length(inst.targets)
        record_target = inst.targets[idx]
        q, inverted = _stim_qubit_target(inst.targets[idx + 1], inst.line)
        inverted && throw(ArgumentError("record-controlled $(inst.op) on line $(inst.line) does not accept inverted qubit targets"))
        record = records[_stim_record_index(record_target, length(records), inst.line)]
        pauli = inst.op in ("CX", "CNOT") ? pauli_x(state.n, q) : pauli_z(state.n, q)
        apply_pauli!(state, pauli, record)
    end
    return state
end

function _apply_reset!(
    state::FrameFactoredState,
    measurement_pauli::PauliString,
    correction_pauli::PauliString,
)
    condition = fresh_condition!(state.context)
    apply_pauli_measurement!(
        state,
        measurement_pauli,
        SymbolicBool(false);
        record_condition = condition,
    )
    apply_pauli!(state, correction_pauli, symbolic_bool(condition))
    return state
end

function _apply_measurement_reset!(
    state::FrameFactoredState,
    records::Vector{SymbolicBool},
    measurement_pauli::PauliString,
    correction_pauli::PauliString,
    inverted::Bool,
    probability::Float64,
)
    sign = _measurement_sign_with_readout_error!(state.context, inverted, probability)
    record, record_condition = _reserve_measurement_record!(records, state.context)
    apply_pauli_measurement!(
        state,
        measurement_pauli,
        sign;
        record,
        record_condition,
    )
    # The public record may include target inversion or readout error. Reset
    # correction must use the physical measurement branch, i.e. record xor sign.
    apply_pauli!(state, correction_pauli, xor(symbolic_bool(record_condition), sign))
    return record, record_condition
end

function _apply_stim_instruction!(
    acc::_StimParseAccumulator,
    inst::_StimInstruction,
)
    state = acc.state
    records = acc.records
    op = inst.op

    if op in ("QUBIT_COORDS", "TICK")
        return acc
    elseif op == "SHIFT_COORDS"
        return _apply_shift_coords!(acc, inst)
    elseif op == "DETECTOR"
        push!(
            acc.detectors,
            StimDetector(_stim_record_indices(inst, records), _coords_with_shift(inst.parens, acc.coord_shift), inst.line),
        )
        return acc
    elseif op == "OBSERVABLE_INCLUDE"
        push!(
            acc.observables,
            StimObservableInclude(_stim_observable_index(inst), _stim_record_indices(inst, records), inst.line),
        )
        return acc
    elseif op in ("CX", "CNOT", "CZ") && !isempty(inst.targets) && startswith(inst.targets[1], "rec[")
        # Record-controlled feedback becomes a symbolic physical Pauli, which
        # is then stored in the active Pauli frame after Clifford preimage.
        _apply_record_feedback!(state, records, inst)
        return acc
    elseif op in ("CX", "CNOT")
        qs = _stim_qubit_targets(inst)
        iseven(length(qs)) || throw(ArgumentError("$op on line $(inst.line) requires paired targets"))
        for idx in 1:2:length(qs)
            left_CX!(state, qs[idx], qs[idx + 1])
        end
    elseif op == "CZ"
        qs = _stim_qubit_targets(inst)
        iseven(length(qs)) || throw(ArgumentError("CZ on line $(inst.line) requires paired targets"))
        for idx in 1:2:length(qs)
            left_CZ!(state, qs[idx], qs[idx + 1])
        end
    elseif op == "SWAP"
        qs = _stim_qubit_targets(inst)
        iseven(length(qs)) || throw(ArgumentError("SWAP on line $(inst.line) requires paired targets"))
        for idx in 1:2:length(qs)
            left_SWAP!(state, qs[idx], qs[idx + 1])
        end
    elseif op == "H"
        for q in _stim_qubit_targets(inst)
            left_H!(state, q)
        end
    elseif op in ("S", "SQRT_Z")
        for q in _stim_qubit_targets(inst)
            left_S!(state, q)
        end
    elseif op in ("S_DAG", "SQRT_Z_DAG")
        for q in _stim_qubit_targets(inst)
            left_SDG!(state, q)
        end
    elseif op in ("X", "Y", "Z")
        for q in _stim_qubit_targets(inst)
            if op == "X"
                left_X!(state, q)
            elseif op == "Z"
                left_Z!(state, q)
            else
                left_X!(state, q)
                left_Z!(state, q)
            end
        end
    elseif op == "T"
        for q in _stim_qubit_targets(inst)
            # Local extension: T = exp(-i*pi/8*Z) up to shot-global phase.
            apply_pauli_rotation!(state, pauli_z(state.n, q), pi / 8)
        end
    elseif op == "T_DAG"
        for q in _stim_qubit_targets(inst)
            # Local extension matching the default exp(-i*theta*P) convention.
            apply_pauli_rotation!(state, pauli_z(state.n, q), -pi / 8)
        end
    elseif op in ("M", "MZ")
        probability = _stim_paren_probability(inst)
        for (q, inverted) in _stim_measurement_targets(inst)
            sign = _measurement_sign_with_readout_error!(state.context, inverted, probability)
            record, record_condition = _reserve_measurement_record!(records, state.context)
            apply_pauli_measurement!(
                state,
                pauli_z(state.n, q),
                sign;
                record,
                record_condition,
            )
        end
    elseif op == "MX"
        probability = _stim_paren_probability(inst)
        for (q, inverted) in _stim_measurement_targets(inst)
            sign = _measurement_sign_with_readout_error!(state.context, inverted, probability)
            record, record_condition = _reserve_measurement_record!(records, state.context)
            apply_pauli_measurement!(
                state,
                pauli_x(state.n, q),
                sign;
                record,
                record_condition,
            )
        end
    elseif op == "MY"
        probability = _stim_paren_probability(inst)
        for (q, inverted) in _stim_measurement_targets(inst)
            sign = _measurement_sign_with_readout_error!(state.context, inverted, probability)
            record, record_condition = _reserve_measurement_record!(records, state.context)
            apply_pauli_measurement!(
                state,
                pauli_y(state.n, q),
                sign;
                record,
                record_condition,
            )
        end
    elseif op in ("MR", "MRZ")
        probability = _stim_paren_probability(inst)
        for (q, inverted) in _stim_measurement_targets(inst)
            _apply_measurement_reset!(
                state,
                records,
                pauli_z(state.n, q),
                pauli_x(state.n, q),
                inverted,
                probability,
            )
        end
    elseif op == "MRX"
        probability = _stim_paren_probability(inst)
        for (q, inverted) in _stim_measurement_targets(inst)
            _apply_measurement_reset!(
                state,
                records,
                pauli_x(state.n, q),
                pauli_z(state.n, q),
                inverted,
                probability,
            )
        end
    elseif op == "MRY"
        probability = _stim_paren_probability(inst)
        for (q, inverted) in _stim_measurement_targets(inst)
            _apply_measurement_reset!(
                state,
                records,
                pauli_y(state.n, q),
                pauli_x(state.n, q),
                inverted,
                probability,
            )
        end
    elseif op in ("R", "RZ")
        isempty(inst.parens) || throw(ArgumentError("$op on line $(inst.line) takes no probability argument"))
        for q in _stim_qubit_targets(inst)
            _apply_reset!(state, pauli_z(state.n, q), pauli_x(state.n, q))
        end
    elseif op == "RX"
        isempty(inst.parens) || throw(ArgumentError("RX on line $(inst.line) takes no probability argument"))
        for q in _stim_qubit_targets(inst)
            _apply_reset!(state, pauli_x(state.n, q), pauli_z(state.n, q))
        end
    elseif op == "RY"
        isempty(inst.parens) || throw(ArgumentError("RY on line $(inst.line) takes no probability argument"))
        for q in _stim_qubit_targets(inst)
            _apply_reset!(state, pauli_y(state.n, q), pauli_x(state.n, q))
        end
    elseif op == "MPP"
        probability = _stim_paren_probability(inst)
        for target in _stim_mpp_targets(inst)
            pauli, inverted = _stim_mpp_pauli(state.n, target, inst.line)
            sign = _measurement_sign_with_readout_error!(state.context, inverted, probability)
            record, record_condition = _reserve_measurement_record!(records, state.context)
            apply_pauli_measurement!(state, pauli, sign; record, record_condition)
        end
    elseif op == "X_ERROR"
        probability = _stim_required_probability(inst)
        for q in _stim_qubit_targets(inst)
            apply_pauli!(state, pauli_x(state.n, q), fresh_bernoulli_bool!(state.context, probability))
        end
    elseif op == "Y_ERROR"
        probability = _stim_required_probability(inst)
        for q in _stim_qubit_targets(inst)
            apply_pauli!(state, pauli_y(state.n, q), fresh_bernoulli_bool!(state.context, probability))
        end
    elseif op == "Z_ERROR"
        probability = _stim_required_probability(inst)
        for q in _stim_qubit_targets(inst)
            apply_pauli!(state, pauli_z(state.n, q), fresh_bernoulli_bool!(state.context, probability))
        end
    elseif op == "DEPOLARIZE1"
        probability = _stim_required_probability(inst)
        for q in _stim_qubit_targets(inst)
            _apply_depolarize1!(state, q, probability)
        end
    elseif op == "DEPOLARIZE2"
        probability = _stim_required_probability(inst)
        qs = _stim_qubit_targets(inst)
        iseven(length(qs)) || throw(ArgumentError("DEPOLARIZE2 on line $(inst.line) requires paired targets"))
        for idx in 1:2:length(qs)
            _apply_depolarize2!(state, qs[idx], qs[idx + 1], probability)
        end
    else
        throw(ArgumentError("unsupported Stim operation `$op` on line $(inst.line)"))
    end

    return acc
end

function _apply_stim_nodes!(acc::_StimParseAccumulator, nodes::AbstractVector)
    for node in nodes
        if node isa _StimRepeatBlock
            for _ in 1:node.count
                _apply_stim_nodes!(acc, node.body)
            end
        else
            _apply_stim_instruction!(acc, node::_StimInstruction)
        end
    end
    return acc
end

function parse_stim_lines(lines::AbstractVector{<:AbstractString})
    nodes, next_pos = _parse_stim_nodes(lines)
    next_pos == lastindex(lines) + 1 ||
        throw(AssertionError("Stim parser stopped before consuming all lines"))
    n = _stim_all_qubits(nodes)
    acc = _StimParseAccumulator(
        FrameFactoredState(n, 0),
        SymbolicBool[],
        StimDetector[],
        StimObservableInclude[],
        Float64[],
    )
    _apply_stim_nodes!(acc, nodes)
    return StimParseResult(acc.state, acc.records, acc.detectors, acc.observables)
end

function parse_stim_text(text::AbstractString)
    return parse_stim_lines(collect(eachline(IOBuffer(text))))
end

function parse_stim_file(path::AbstractString)
    return parse_stim_lines(collect(eachline(path)))
end
