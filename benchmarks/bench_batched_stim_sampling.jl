#!/usr/bin/env julia

using SymFT

const ROOT = normpath(joinpath(@__DIR__, ".."))
const FIXTURES = ("d3.stim", "d5.stim")

function _bench_shots()
    raw = !isempty(ARGS) ? ARGS[1] : get(ENV, "SYMFT_BATCH_BENCH_SHOTS", "1000")
    shots = parse(Int, raw)
    shots > 0 || throw(ArgumentError("shot count must be positive"))
    return shots
end

function _bench_batch_size()
    raw = length(ARGS) >= 2 ? ARGS[2] : get(ENV, "SYMFT_BATCH_SIZE", "auto")
    lowered = lowercase(strip(raw))
    lowered in ("auto", "default") && return nothing
    batch_size = parse(Int, raw)
    batch_size > 0 || throw(ArgumentError("batch size must be positive"))
    return batch_size
end

function _bench_warmup_blocks()
    raw = length(ARGS) >= 3 ? ARGS[3] : get(ENV, "SYMFT_BATCH_BENCH_WARMUP", "5")
    warmup = parse(Int, raw)
    warmup >= 0 || throw(ArgumentError("warmup block count must be nonnegative"))
    return warmup
end

function _bench_kernel_backend()
    raw = length(ARGS) >= 4 ? ARGS[4] : get(ENV, "SYMFT_BATCH_KERNEL_BACKEND", "structarray")
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

function _run_batch_blocks!(
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

function _run_fixture(
    fixture::AbstractString,
    shots::Int,
    batch_size,
    warmup_blocks::Int,
    kernel_backend::Symbol,
)
    path = joinpath(ROOT, fixture)
    program = _build_program(path)
    effective_batch_size = isnothing(batch_size) ? SymFT._default_batch_count(program.max_k) : batch_size
    runtime = BatchFactoredExecutorState(program; batches = effective_batch_size, kernel_backend)

    for _ in 1:warmup_blocks
        reset_executor!(runtime, program)
        execute!(runtime, program)
        _consume_measurements(runtime)
    end
    GC.gc()

    t0 = time_ns()
    checksum = _run_batch_blocks!(runtime, program, shots)
    elapsed = (time_ns() - t0) / 1e9
    rate = shots / max(elapsed, eps(Float64))

    return (
        fixture = fixture,
        n = program.n,
        initial_k = program.initial_k,
        max_k = max_active_qubits(program),
        instructions = length(program.instructions),
        records = program.nrecords,
        symbols = program.nsymbols,
        kernel_backend = runtime.kernel_backend,
        shots = shots,
        batch_size = effective_batch_size,
        blocks = cld(shots, effective_batch_size),
        elapsed = elapsed,
        rate = rate,
        runtime_bytes = Base.summarysize(runtime),
        checksum = checksum,
    )
end

function _print_header(shots::Int, batch_size, warmup_blocks::Int, kernel_backend::Symbol)
    println("SymFT batched FactoredInstructionProgram sampler")
    batch_label = isnothing(batch_size) ? "auto" : string(batch_size)
    println("shots=$shots batch_size=$batch_label warmup_blocks=$warmup_blocks kernel_backend=$kernel_backend")
    println(join((
        rpad("fixture", 10),
        lpad("n", 5),
        lpad("k0", 5),
        lpad("max_k", 7),
        lpad("instr", 8),
        lpad("records", 8),
        lpad("symbols", 8),
        lpad("backend", 12),
        lpad("blocks", 8),
        lpad("runtime_B", 12),
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
        lpad(result.records, 8),
        lpad(result.symbols, 8),
        lpad(result.kernel_backend, 12),
        lpad(result.blocks, 8),
        lpad(result.runtime_bytes, 12),
        lpad(round(result.elapsed; digits = 4), 12),
        lpad(string(round(result.rate; digits = 2), " shots/s"), 18),
        lpad(result.checksum, 11),
    ), " "))
end

function main()
    shots = _bench_shots()
    batch_size = _bench_batch_size()
    warmup_blocks = _bench_warmup_blocks()
    kernel_backend = _bench_kernel_backend()
    _print_header(shots, batch_size, warmup_blocks, kernel_backend)
    for fixture in FIXTURES
        _print_result(_run_fixture(fixture, shots, batch_size, warmup_blocks, kernel_backend))
    end
end

main()
