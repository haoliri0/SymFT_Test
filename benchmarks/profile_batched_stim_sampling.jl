#!/usr/bin/env julia

using Profile
using SymFT

const ROOT = normpath(joinpath(@__DIR__, ".."))

function _usage()
    return """
    Usage:
      julia --project=. benchmarks/profile_batched_stim_sampling.jl [fixture] [shots] [batch_size] [stack_profile] [kernel_backend]

    Arguments:
      fixture        Stim file path, relative to repo root, or "all".
                     Default: d5.stim
      shots          Positive integer shot count. Default: 3000
      batch_size     Positive integer batch size, or "auto" for the sampler
                     default chosen from max_k. Default: auto
      stack_profile  true/false, whether to print Julia Profile stack samples.
                     Default: false
      kernel_backend structarray/turbo, c/avx2, or scalar. Default: structarray

    Examples:
      julia --project=. benchmarks/profile_batched_stim_sampling.jl d5.stim 3000 64
      julia --project=. benchmarks/profile_batched_stim_sampling.jl all 1000 128 true
    """
end

function _parse_bool(raw::AbstractString)
    lowered = lowercase(strip(raw))
    lowered in ("1", "true", "yes", "y", "on", "profile") && return true
    lowered in ("0", "false", "no", "n", "off") && return false
    throw(ArgumentError("expected boolean argument, got $raw"))
end

function _arg_or_env(idx::Int, env::AbstractString, default::AbstractString)
    return length(ARGS) >= idx ? ARGS[idx] : get(ENV, env, default)
end

function _fixtures()
    raw = _arg_or_env(1, "SYMFT_BATCH_PROFILE_FIXTURE", "d5.stim")
    if lowercase(raw) == "all"
        return ("d3.stim", "d5.stim")
    end
    return (raw,)
end

function _shots()
    raw = _arg_or_env(2, "SYMFT_BATCH_PROFILE_SHOTS", "3000")
    shots = parse(Int, raw)
    shots > 0 || throw(ArgumentError("shot count must be positive"))
    return shots
end

function _batch_size()
    raw = _arg_or_env(3, "SYMFT_BATCH_SIZE", "auto")
    lowered = lowercase(strip(raw))
    lowered in ("auto", "default") && return nothing
    batch_size = parse(Int, raw)
    batch_size > 0 || throw(ArgumentError("batch size must be positive"))
    return batch_size
end

function _stack_profile()
    raw = _arg_or_env(4, "SYMFT_BATCH_PROFILE_STACK", "false")
    return _parse_bool(raw)
end

function _kernel_backend()
    raw = _arg_or_env(5, "SYMFT_BATCH_KERNEL_BACKEND", "structarray")
    return Symbol(lowercase(strip(raw)))
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

@inline function _low_bits_mask(nbits::Int)::UInt64
    nbits <= 0 && return UInt64(0)
    nbits >= 64 && return typemax(UInt64)
    return (UInt64(1) << nbits) - UInt64(1)
end

@inline function _live_word_mask(active_shots::Int, word::Int)::UInt64
    return _low_bits_mask(active_shots - ((word - 1) << 6))
end

function _consume_measurements(runtime::BatchFactoredExecutorState)
    checksum = 0
    nwords = cld(runtime.active_shots, 64)
    @inbounds for record in 1:size(runtime.measurements, 2)
        for word in 1:nwords
            checksum += count_ones(runtime.measurements[word, record] & _live_word_mask(runtime.active_shots, word))
        end
    end
    return checksum
end

function _run_loop!(
    runtime::BatchFactoredExecutorState,
    program::FactoredInstructionProgram,
    shots::Int,
)
    checksum = 0
    offset = 0
    while offset < shots
        block = min(runtime.batches, shots - offset)
        reset_executor!(runtime, program; shots = block)
        execute!(runtime, program)
        checksum += _consume_measurements(runtime)
        offset += block
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
    runtime::BatchFactoredExecutorState,
    program::FactoredInstructionProgram,
    shots::Int,
)
    names = String[
        "reset_batch",
        "sample_exogenous_batch",
        "ApplyPrecomputedActivePauliRotation",
        "PromoteDormantRotation",
        "RecordMeasurement",
        "MeasureActiveLastZ",
        "MeasurePrecomputedActivePauli",
        "IntroduceDormantMeasurementBranch",
        "consume_packed_measurements",
    ]
    buckets = ProfileBuckets(names)
    checksum = 0
    blocks = 0
    total_start = time_ns()

    offset = 0
    while offset < shots
        block = min(runtime.batches, shots - offset)
        blocks += 1

        t = time_ns()
        reset_executor!(runtime, program; shots = block)
        _add_bucket!(buckets, 1, time_ns() - t)

        t = time_ns()
        SymFT._sample_exogenous_symbols!(runtime, program)
        _add_bucket!(buckets, 2, time_ns() - t)

        for instruction in program.instructions
            idx = _bucket_index(instruction)
            t = time_ns()
            execute_instruction!(runtime, instruction)
            _add_bucket!(buckets, idx, time_ns() - t)
        end

        t = time_ns()
        checksum += _consume_measurements(runtime)
        _add_bucket!(buckets, 9, time_ns() - t)

        offset += block
    end

    return buckets, time_ns() - total_start, checksum, blocks
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

function _print_buckets(buckets::ProfileBuckets, total_ns::UInt64, shots::Int)
    println()
    println("Attributed runtime buckets:")
    println(join((
        rpad("bucket", 34),
        lpad("seconds", 12),
        lpad("% total", 9),
        lpad("calls", 12),
        lpad("ns/call", 12),
        lpad("ns/shot", 12),
    ), " "))
    order = sort(collect(eachindex(buckets.names)); by = idx -> buckets.times[idx], rev = true)
    for idx in order
        elapsed = Float64(buckets.times[idx]) / 1e9
        pct = 100 * Float64(buckets.times[idx]) / max(Float64(total_ns), 1.0)
        calls = buckets.counts[idx]
        per_call = calls == 0 ? 0.0 : Float64(buckets.times[idx]) / calls
        per_shot = Float64(buckets.times[idx]) / shots
        println(join((
            rpad(buckets.names[idx], 34),
            lpad(round(elapsed; digits = 6), 12),
            lpad(round(pct; digits = 2), 9),
            lpad(calls, 12),
            lpad(round(per_call; digits = 1), 12),
            lpad(round(per_shot; digits = 1), 12),
        ), " "))
    end
end

function _profile_stack!(
    runtime::BatchFactoredExecutorState,
    program::FactoredInstructionProgram,
    shots::Int,
)
    println()
    println("Julia Profile flat stack samples:")
    Profile.clear()
    @profile _run_loop!(runtime, program, shots)
    Profile.print(format = :flat, sortedby = :count, maxdepth = 40, mincount = 10)
end

function _run_fixture(
    fixture::AbstractString,
    shots::Int,
    batch_size,
    stack_profile::Bool,
    kernel_backend::Symbol,
)
    program = _build_program(fixture)
    effective_batch_size = isnothing(batch_size) ? SymFT._default_batch_count(program.max_k) : batch_size
    runtime = BatchFactoredExecutorState(program; batches = effective_batch_size, kernel_backend)

    warm_shots = min(shots, 20 * effective_batch_size)
    _run_loop!(runtime, program, warm_shots)
    GC.gc()

    bytes = @allocated _run_loop!(runtime, program, shots)
    GC.gc()

    buckets, attributed_ns, checksum, blocks = _attributed_run!(runtime, program, shots)
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
        " batch_size=", effective_batch_size,
        " kernel_backend=", runtime.kernel_backend,
        " blocks=", blocks,
        " runtime_bytes=", Base.summarysize(runtime))
    println("attributed_elapsed_s=", round(elapsed; digits = 6),
        " attributed_shots_per_s=", round(rate; digits = 2),
        " allocation_bytes_per_shot=", round(bytes / shots; digits = 2),
        " checksum=", checksum)

    _print_instruction_mix(program)
    _print_buckets(buckets, attributed_ns, shots)

    if stack_profile
        GC.gc()
        _profile_stack!(runtime, program, shots)
    end
end

function main()
    if !isempty(ARGS) && ARGS[1] in ("-h", "--help", "help")
        print(_usage())
        return
    end

    fixtures = _fixtures()
    shots = _shots()
    batch_size = _batch_size()
    stack_profile = _stack_profile()
    kernel_backend = _kernel_backend()

    println("SymFT batched sampler profile")
    println("Julia ", VERSION)
    for fixture in fixtures
        _run_fixture(fixture, shots, batch_size, stack_profile, kernel_backend)
    end
end

main()
