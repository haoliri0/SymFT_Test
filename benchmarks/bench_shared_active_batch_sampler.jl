#!/usr/bin/env julia

using SymFT

const ROOT = normpath(joinpath(@__DIR__, ".."))

function _usage()
    return """
    Usage:
      julia --project=. benchmarks/bench_shared_active_batch_sampler.jl [fixture] [shots] [batch_size] [kernel_backend]

    Arguments:
      fixture     Stim file path, relative to repo root. Default: d5.stim
      shots       Positive integer shot count. Default: 3000
      batch_size  Positive integer batch size, or "auto" for the sampler
                  default chosen from max_k. Default: auto
      kernel_backend structarray/turbo, c/avx2, or scalar. Default: structarray
    """
end

function _arg_or_env(idx::Int, env::AbstractString, default::AbstractString)
    return length(ARGS) >= idx ? ARGS[idx] : get(ENV, env, default)
end

function _fixture()
    raw = _arg_or_env(1, "SYMFT_SHARED_BATCH_FIXTURE", "d5.stim")
    raw in ("-h", "--help", "help") && (print(_usage()); exit())
    return isabspath(raw) ? raw : joinpath(ROOT, raw)
end

function _shots()
    shots = parse(Int, _arg_or_env(2, "SYMFT_SHARED_BATCH_SHOTS", "3000"))
    shots > 0 || throw(ArgumentError("shot count must be positive"))
    return shots
end

function _batch_size()
    raw = _arg_or_env(3, "SYMFT_SHARED_BATCH_SIZE", "auto")
    lowered = lowercase(strip(raw))
    lowered in ("auto", "default") && return nothing
    batch_size = parse(Int, raw)
    batch_size > 0 || throw(ArgumentError("batch size must be positive"))
    return batch_size
end

function _kernel_backend()
    raw = _arg_or_env(4, "SYMFT_SHARED_BATCH_KERNEL_BACKEND", "structarray")
    return Symbol(lowercase(strip(raw)))
end

function _build_program(path::AbstractString)
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

function _consume_measurements(measurements::Matrix{UInt64}, active_shots::Int)
    checksum = 0
    nwords = cld(active_shots, 64)
    @inbounds for record in 1:size(measurements, 2)
        for word in 1:nwords
            checksum += count_ones(measurements[word, record] & _live_word_mask(active_shots, word))
        end
    end
    return checksum
end

function _run_standard!(
    runtime::BatchFactoredExecutorState,
    program::FactoredInstructionProgram,
    shots::Int,
)
    checksum = 0
    blocks = 0
    packet_sum = 0
    packet_max = 0
    offset = 0
    while offset < shots
        block = min(runtime.batches, shots - offset)
        blocks += 1
        reset_executor!(runtime, program; shots = block)
        execute!(runtime, program)
        checksum += _consume_measurements(runtime.measurements, runtime.active_shots)
        packet_sum += block
        packet_max = max(packet_max, block)
        offset += block
    end
    return checksum, blocks, packet_sum, packet_max
end

function _run_shared!(
    runtime::SharedActiveBatchExecutorState,
    program::FactoredInstructionProgram,
    shots::Int,
)
    checksum = 0
    blocks = 0
    packet_sum = 0
    packet_max = 0
    offset = 0
    while offset < shots
        block = min(runtime.base.batches, shots - offset)
        blocks += 1
        reset_executor!(runtime, program; shots = block)
        execute!(runtime, program)
        packets = active_packet_count(runtime)
        packet_sum += packets
        packet_max = max(packet_max, packets)
        checksum += _consume_measurements(runtime.base.measurements, runtime.base.active_shots)
        offset += block
    end
    return checksum, blocks, packet_sum, packet_max
end

function _time_run!(runner, runtime, program, shots)
    GC.gc()
    elapsed = @elapsed checksum, blocks, packet_sum, packet_max = runner(runtime, program, shots)
    return (
        elapsed = elapsed,
        rate = shots / max(elapsed, eps(Float64)),
        checksum = checksum,
        blocks = blocks,
        avg_packets = packet_sum / blocks,
        max_packets = packet_max,
    )
end

function _print_result(name::AbstractString, result)
    println(join((
        rpad(name, 10),
        lpad(round(result.elapsed; digits = 5), 12),
        lpad(round(result.rate; digits = 2), 14),
        lpad(result.blocks, 8),
        lpad(round(result.avg_packets; digits = 2), 12),
        lpad(result.max_packets, 12),
        lpad(result.checksum, 12),
    ), " "))
end

function main()
    fixture = _fixture()
    shots = _shots()
    batch_size = _batch_size()
    kernel_backend = _kernel_backend()
    program = _build_program(fixture)
    effective_batch_size = isnothing(batch_size) ? SymFT._default_batch_count(program.max_k) : batch_size

    standard = BatchFactoredExecutorState(program; batches = effective_batch_size, kernel_backend)
    shared = SharedActiveBatchExecutorState(program; batches = effective_batch_size, kernel_backend)

    # Warm both paths once to keep the printed timings focused on execution.
    _run_standard!(standard, program, min(shots, effective_batch_size))
    _run_shared!(shared, program, min(shots, effective_batch_size))

    println("SymFT shared-active batch sampler benchmark")
    println("fixture=", fixture,
        " shots=", shots,
        " batch_size=", effective_batch_size,
        " kernel_backend=", standard.kernel_backend,
        " n=", program.n,
        " max_k=", max_active_qubits(program),
        " records=", program.nrecords)
    println(join((
        rpad("mode", 10),
        lpad("elapsed_s", 12),
        lpad("shots/s", 14),
        lpad("blocks", 8),
        lpad("avg_packets", 12),
        lpad("max_packets", 12),
        lpad("checksum", 12),
    ), " "))

    _print_result("standard", _time_run!(_run_standard!, standard, program, shots))
    _print_result("shared", _time_run!(_run_shared!, shared, program, shots))
end

main()
