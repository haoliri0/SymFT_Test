#!/usr/bin/env julia

using Profile
using SymFT

const ROOT = normpath(joinpath(@__DIR__, ".."))

function _usage()
    return """
    Usage:
      julia --project=. benchmarks/profile_stim_sampling.jl [fixture] [shots] [presampled] [stack_profile]

    Arguments:
      fixture        Stim file path, relative to repo root, or "all".
                     Default: d5.stim
      shots          Positive integer shot count. Default: 3000
      presampled     true/false, whether to use presampled exogenous symbols.
                     Default: true
      stack_profile  true/false, whether to print Julia Profile stack samples.
                     Default: false

    Examples:
      julia --project=. benchmarks/profile_stim_sampling.jl d5.stim 3000 true
      julia --project=. benchmarks/profile_stim_sampling.jl all 1000 false true
    """
end

function _parse_bool(raw::AbstractString)
    lowered = lowercase(strip(raw))
    lowered in ("1", "true", "yes", "y", "on", "presampled", "profile") && return true
    lowered in ("0", "false", "no", "n", "off") && return false
    throw(ArgumentError("expected boolean argument, got $raw"))
end

function _arg_or_env(idx::Int, env::AbstractString, default::AbstractString)
    return length(ARGS) >= idx ? ARGS[idx] : get(ENV, env, default)
end

function _fixtures()
    raw = _arg_or_env(1, "SYMFT_PROFILE_FIXTURE", "d5.stim")
    if lowercase(raw) == "all"
        return ("d3.stim", "d5.stim")
    end
    return (raw,)
end

function _shots()
    raw = _arg_or_env(2, "SYMFT_PROFILE_SHOTS", "3000")
    shots = parse(Int, raw)
    shots > 0 || throw(ArgumentError("shot count must be positive"))
    return shots
end

function _presampled()
    raw = _arg_or_env(3, "SYMFT_PROFILE_PRESAMPLED", "true")
    return _parse_bool(raw)
end

function _stack_profile()
    raw = _arg_or_env(4, "SYMFT_PROFILE_STACK", "false")
    return _parse_bool(raw)
end

function _fixture_path(fixture::AbstractString)
    isabspath(fixture) && return fixture
    return joinpath(ROOT, fixture)
end

function _build_program(fixture::AbstractString)
    path = _fixture_path(fixture)
    isfile(path) || throw(ArgumentError("fixture file does not exist: $path"))
    parsed = parse_stim_file(path)
    pending = PendingFactoredState(stim_state(parsed))
    return plan_factored_updates!(pending)
end

function _consume_measurements(records::Vector{Bool})
    checksum = 0
    @inbounds for bit in records
        checksum += Int(bit)
    end
    return checksum
end

function _sample_one!(
    runtime::FactoredExecutorState,
    program::FactoredInstructionProgram,
    samples::Nothing,
    shot::Int,
)
    return sample_measurements!(runtime, program)
end

function _sample_one!(
    runtime::FactoredExecutorState,
    program::FactoredInstructionProgram,
    samples::PresampledExogenous,
    shot::Int,
)
    return sample_measurements!(runtime, program, samples, shot)
end

function _run_loop!(
    runtime::FactoredExecutorState,
    program::FactoredInstructionProgram,
    samples,
    shots::Int,
)
    checksum = 0
    @inbounds for shot in 1:shots
        checksum += _consume_measurements(_sample_one!(runtime, program, samples, shot))
    end
    return checksum
end

mutable struct ProfileBuckets
    names::Vector{String}
    times::Vector{UInt64}
    counts::Vector{Int}
end

ProfileBuckets(names::Vector{String}) =
    ProfileBuckets(names, zeros(UInt64, length(names)), zeros(Int, length(names)))

function _add_bucket!(buckets::ProfileBuckets, idx::Int, dt::UInt64)
    buckets.times[idx] += dt
    buckets.counts[idx] += 1
    return buckets
end

function _bucket_index(instruction::FactoredInstruction)
    instruction isa ApplyPrecomputedActivePauliRotation && return 3
    instruction isa PromoteDormantRotation && return 4
    instruction isa RecordMeasurement && return 5
    instruction isa MeasureActiveLastZ && return 6
    instruction isa MeasurePrecomputedActivePauli && return 7
    instruction isa IntroduceDormantMeasurementBranch && return 8
    throw(ArgumentError("unsupported instruction type $(typeof(instruction))"))
end

function _attributed_run!(
    runtime::FactoredExecutorState,
    program::FactoredInstructionProgram,
    samples,
    shots::Int,
)
    names = String[
        "reset",
        samples === nothing ? "sample_exogenous" : "assign_presampled",
        "ApplyPrecomputedActivePauliRotation",
        "PromoteDormantRotation",
        "RecordMeasurement",
        "MeasureActiveLastZ",
        "MeasurePrecomputedActivePauli",
        "IntroduceDormantMeasurementBranch",
        "consume_measurements",
    ]
    buckets = ProfileBuckets(names)
    checksum = 0
    total_start = time_ns()

    @inbounds for shot in 1:shots
        t = time_ns()
        reset_executor!(runtime, program)
        _add_bucket!(buckets, 1, time_ns() - t)

        if samples === nothing
            t = time_ns()
            SymFT._sample_exogenous_symbols!(runtime, program)
            _add_bucket!(buckets, 2, time_ns() - t)
        else
            t = time_ns()
            SymFT._assign_presampled_exogenous!(runtime, samples, shot)
            _add_bucket!(buckets, 2, time_ns() - t)
        end

        for instruction in program.instructions
            idx = _bucket_index(instruction)
            t = time_ns()
            execute_instruction!(runtime, instruction)
            _add_bucket!(buckets, idx, time_ns() - t)
        end

        t = time_ns()
        checksum += _consume_measurements(runtime.measurements)
        _add_bucket!(buckets, 9, time_ns() - t)
    end

    return buckets, time_ns() - total_start, checksum
end

function _instruction_mix(program::FactoredInstructionProgram)
    counts = Dict{String,Int}()
    for instruction in program.instructions
        name = string(nameof(typeof(instruction)))
        counts[name] = get(counts, name, 0) + 1
    end
    return sort(collect(counts); by = last, rev = true)
end

function _print_instruction_mix(program::FactoredInstructionProgram)
    println("Instruction mix:")
    for (name, count) in _instruction_mix(program)
        println("  ", rpad(name, 38), lpad(count, 8))
    end
end

function _print_buckets(buckets::ProfileBuckets, total_ns::UInt64)
    println()
    println("Attributed runtime buckets:")
    println(join((
        rpad("bucket", 34),
        lpad("seconds", 12),
        lpad("% total", 9),
        lpad("calls", 12),
        lpad("ns/call", 12),
    ), " "))
    order = sort(collect(eachindex(buckets.names)); by = idx -> buckets.times[idx], rev = true)
    for idx in order
        elapsed = Float64(buckets.times[idx]) / 1e9
        pct = 100 * Float64(buckets.times[idx]) / max(Float64(total_ns), 1.0)
        calls = buckets.counts[idx]
        per_call = calls == 0 ? 0.0 : Float64(buckets.times[idx]) / calls
        println(join((
            rpad(buckets.names[idx], 34),
            lpad(round(elapsed; digits = 6), 12),
            lpad(round(pct; digits = 2), 9),
            lpad(calls, 12),
            lpad(round(per_call; digits = 1), 12),
        ), " "))
    end
end

function _profile_stack!(
    runtime::FactoredExecutorState,
    program::FactoredInstructionProgram,
    samples,
    shots::Int,
)
    println()
    println("Julia Profile flat stack samples:")
    Profile.clear()
    @profile _run_loop!(runtime, program, samples, shots)
    Profile.print(format = :flat, sortedby = :count, maxdepth = 40, mincount = 10)
end

function _summary_size(samples::Nothing)
    return 0
end

function _summary_size(samples::PresampledExogenous)
    return Base.summarysize(samples)
end

function _run_fixture(fixture::AbstractString, shots::Int, use_presampled::Bool, stack_profile::Bool)
    program = _build_program(fixture)
    runtime = FactoredExecutorState(program)

    for _ in 1:20
        sample_measurements!(runtime, program)
    end

    prep_start = time_ns()
    samples = use_presampled ? presample_exogenous(program, shots) : nothing
    prep_elapsed = (time_ns() - prep_start) / 1e9

    # Warm the exact path being profiled after optional presampling.
    warm_shots = min(shots, 20)
    _run_loop!(runtime, program, samples, warm_shots)
    GC.gc()

    bytes = @allocated _run_loop!(runtime, program, samples, shots)
    GC.gc()

    buckets, attributed_ns, checksum = _attributed_run!(runtime, program, samples, shots)
    elapsed = Float64(attributed_ns) / 1e9
    rate = shots / max(elapsed, eps(Float64))

    println()
    println("Fixture: ", fixture)
    println("n=", program.n,
        " initial_k=", program.initial_k,
        " max_k=", max_active_qubits(program),
        " instructions=", length(program.instructions),
        " records=", program.nrecords,
        " symbols=", program.nsymbols)
    println("shots=", shots,
        " presampled=", use_presampled,
        " prep_s=", round(prep_elapsed; digits = 6),
        " presample_bytes=", _summary_size(samples))
    println("attributed_elapsed_s=", round(elapsed; digits = 6),
        " attributed_shots_per_s=", round(rate; digits = 2),
        " allocation_bytes_per_shot=", round(bytes / shots; digits = 2),
        " checksum=", checksum)

    _print_instruction_mix(program)
    _print_buckets(buckets, attributed_ns)

    if stack_profile
        GC.gc()
        _profile_stack!(runtime, program, samples, shots)
    end
end

function main()
    if !isempty(ARGS) && ARGS[1] in ("-h", "--help", "help")
        print(_usage())
        return
    end

    fixtures = _fixtures()
    shots = _shots()
    use_presampled = _presampled()
    stack_profile = _stack_profile()

    println("SymFT single-shot sampler profile")
    println("Julia ", VERSION)
    for fixture in fixtures
        _run_fixture(fixture, shots, use_presampled, stack_profile)
    end
end

main()
