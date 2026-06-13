#!/usr/bin/env julia

using SymFT

const ROOT = normpath(joinpath(@__DIR__, ".."))
const FIXTURES = ("d3.stim", "d5.stim")

function _bench_shots()
    raw = !isempty(ARGS) ? ARGS[1] : get(ENV, "SYMFT_BENCH_SHOTS", "100")
    shots = parse(Int, raw)
    shots > 0 || throw(ArgumentError("shot count must be positive"))
    return shots
end

function _bench_warmup()
    warmup = parse(Int, get(ENV, "SYMFT_BENCH_WARMUP", "5"))
    warmup >= 0 || throw(ArgumentError("warmup count must be nonnegative"))
    return warmup
end

function _bench_presampled()
    raw = length(ARGS) >= 2 ? ARGS[2] : get(ENV, "SYMFT_PRESAMPLE_EXOGENOUS", "false")
    return lowercase(raw) in ("1", "true", "yes", "presampled")
end

function _build_program(path::AbstractString)
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

function _run_fixture(fixture::AbstractString, shots::Int, warmup::Int, presampled::Bool)
    path = joinpath(ROOT, fixture)
    program = _build_program(path)
    runtime = FactoredExecutorState(program)

    for _ in 1:warmup
        sample_measurements!(runtime, program)
    end
    prep_t0 = time_ns()
    samples = presampled ? presample_exogenous(program, shots) : nothing
    prep_elapsed = (time_ns() - prep_t0) / 1e9
    GC.gc()

    checksum = 0
    t0 = time_ns()
    if samples === nothing
        for _ in 1:shots
            checksum += _consume_measurements(sample_measurements!(runtime, program))
        end
    else
        for shot in 1:shots
            checksum += _consume_measurements(sample_measurements!(runtime, program, samples, shot))
        end
    end
    elapsed = (time_ns() - t0) / 1e9
    rate = shots / max(elapsed, eps(Float64))

    return (
        fixture = fixture,
        n = program.n,
        initial_k = program.initial_k,
        max_k = max_active_qubits(program),
        instructions = length(program.instructions),
        shots = shots,
        prep_elapsed = prep_elapsed,
        elapsed = elapsed,
        rate = rate,
        checksum = checksum,
    )
end

function _print_header(shots::Int, warmup::Int, presampled::Bool)
    println("SymFT single-shot FactoredInstructionProgram sampler")
    println("shots=$shots warmup=$warmup presampled=$presampled")
    println(join((
        rpad("fixture", 10),
        lpad("n", 5),
        lpad("k0", 5),
        lpad("max_k", 7),
        lpad("instr", 8),
        lpad("prep_s", 10),
        lpad("elapsed_s", 12),
        lpad("shots/s", 18),
        lpad("checksum", 11),
    ), " "))
end

function _print_result(result)
    println(join((
        rpad(result.fixture, 10),
        lpad(result.n, 5),
        lpad(result.initial_k, 5),
        lpad(result.max_k, 7),
        lpad(result.instructions, 8),
        lpad(round(result.prep_elapsed; digits = 4), 10),
        lpad(round(result.elapsed; digits = 4), 12),
        lpad(string(round(result.rate; digits = 2), " shots/s"), 18),
        lpad(result.checksum, 11),
    ), " "))
end

function main()
    shots = _bench_shots()
    warmup = _bench_warmup()
    presampled = _bench_presampled()
    _print_header(shots, warmup, presampled)
    for fixture in FIXTURES
        _print_result(_run_fixture(fixture, shots, warmup, presampled))
    end
end

main()
