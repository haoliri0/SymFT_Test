#!/usr/bin/env julia

using SymFT

const ROOT = normpath(joinpath(@__DIR__, ".."))

function _usage()
    return """
    Usage:
      julia --project=. benchmarks/estimate_stim_rates.jl [fixture] [shots] [batch_size] [observable]

    Defaults:
      fixture      d3.stim
      shots        100000000
      batch_size   8192
      observable   0
    """
end

function _arg_or_env(idx::Int, env::AbstractString, default::AbstractString)
    return length(ARGS) >= idx ? ARGS[idx] : get(ENV, env, default)
end

function _fixture_path(raw::AbstractString)
    isabspath(raw) && return raw
    return joinpath(ROOT, raw)
end

function _positive_int(raw::AbstractString, name::AbstractString)
    value = parse(Int, raw)
    value > 0 || throw(ArgumentError("$name must be positive"))
    return value
end

function _nonnegative_int(raw::AbstractString, name::AbstractString)
    value = parse(Int, raw)
    value >= 0 || throw(ArgumentError("$name must be nonnegative"))
    return value
end

function main()
    if !isempty(ARGS) && ARGS[1] in ("-h", "--help", "help")
        print(_usage())
        return
    end

    fixture = _fixture_path(_arg_or_env(1, "SYMFT_RATE_FIXTURE", "d3.stim"))
    shots = _positive_int(_arg_or_env(2, "SYMFT_RATE_SHOTS", "100000000"), "shots")
    batch_size = _positive_int(_arg_or_env(3, "SYMFT_BATCH_SIZE", "8192"), "batch_size")
    observable = _nonnegative_int(_arg_or_env(4, "SYMFT_RATE_OBSERVABLE", "0"), "observable")

    parsed = parse_stim_file(fixture)
    summary = estimate_stim_logical_error_rate(
        parsed,
        shots;
        batches = batch_size,
        observable,
    )

    println("fixture=", fixture)
    println("shots=", summary.shots)
    println("batch_size=", summary.batch_size)
    println("observable=", summary.observable)
    println("discarded=", summary.discarded)
    println("accepted=", summary.accepted)
    println("logical_errors=", summary.logical_errors)
    println("discard_rate=", discard_rate(summary))
    println("logical_error_rate=", logical_error_rate(summary))
    println("elapsed_s=", summary.elapsed)
    println("shots_per_s=", summary.shots / max(summary.elapsed, eps(Float64)))
end

main()
