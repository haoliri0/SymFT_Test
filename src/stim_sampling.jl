"""
    StimSampleSummary

Streaming detector/observable sampling summary for a parsed Stim circuit.
`logical_errors` counts observable flips among accepted shots only.
"""
struct StimSampleSummary
    shots::Int
    discarded::Int
    accepted::Int
    logical_errors::Int
    observable::Int
    batch_size::Int
    elapsed::Float64
end

function StimSampleSummary(
    shots::Integer,
    discarded::Integer,
    logical_errors::Integer,
    observable::Integer,
    batch_size::Integer,
    elapsed::Real,
)
    total = _checked_batch_sample_shots(shots)
    discarded_i = Int(discarded)
    logical_i = Int(logical_errors)
    0 <= discarded_i <= total ||
        throw(ArgumentError("discarded count must be between 0 and total shots"))
    accepted = total - discarded_i
    0 <= logical_i <= accepted ||
        throw(ArgumentError("logical error count must be between 0 and accepted shots"))
    obs = Int(observable)
    obs >= 0 || throw(ArgumentError("observable index must be nonnegative"))
    batch = _checked_batch_count(batch_size)
    return StimSampleSummary(total, discarded_i, accepted, logical_i, obs, batch, Float64(elapsed))
end

discard_rate(summary::StimSampleSummary) =
    summary.shots == 0 ? NaN : summary.discarded / summary.shots

logical_error_rate(summary::StimSampleSummary) =
    summary.accepted == 0 ? NaN : summary.logical_errors / summary.accepted

function _stim_program(parsed::StimParseResult)
    pending = PendingFactoredState(stim_state(parsed))
    return plan_factored_updates!(pending)
end

function _checked_observable_index(observable::Integer)
    obs = Int(observable)
    obs >= 0 || throw(ArgumentError("observable index must be nonnegative"))
    return obs
end

function _clear_packed_batch_bits!(bits::Vector{UInt64})
    fill!(bits, UInt64(0))
    return bits
end

function _xor_stim_records!(
    out::Vector{UInt64},
    runtime::BatchFactoredExecutorState,
    records::Vector{Int},
)
    _clear_packed_batch_bits!(out)
    nwords = _runtime_batch_word_count(runtime)
    @inbounds for record in records
        1 <= record <= size(runtime.measurements, 2) ||
            throw(BoundsError(1:size(runtime.measurements, 2), record))
        for word in 1:nwords
            out[word] ⊻= runtime.measurements[word, record]
        end
    end
    @inbounds for word in 1:nwords
        out[word] &= _batch_live_word_mask(runtime, word)
    end
    return out
end

function _accumulate_stim_batch_summary!(
    discard_bits::Vector{UInt64},
    logical_bits::Vector{UInt64},
    scratch::Vector{UInt64},
    runtime::BatchFactoredExecutorState,
    detectors::Vector{StimDetector},
    observables::Vector{StimObservableInclude},
    observable::Int,
)
    _clear_packed_batch_bits!(discard_bits)
    _clear_packed_batch_bits!(logical_bits)
    nwords = _runtime_batch_word_count(runtime)

    for detector in detectors
        _xor_stim_records!(scratch, runtime, detector.records)
        @inbounds for word in 1:nwords
            discard_bits[word] |= scratch[word]
        end
    end

    for include in observables
        include.index == observable || continue
        _xor_stim_records!(scratch, runtime, include.records)
        @inbounds for word in 1:nwords
            logical_bits[word] ⊻= scratch[word]
        end
    end

    discarded = 0
    logical_errors = 0
    @inbounds for word in 1:nwords
        live = _batch_live_word_mask(runtime, word)
        discarded_word = discard_bits[word] & live
        accepted_word = (~discarded_word) & live
        discarded += count_ones(discarded_word)
        logical_errors += count_ones(logical_bits[word] & accepted_word)
    end
    return discarded, logical_errors
end

function _estimate_stim_logical_error_rate!(
    runtime::BatchFactoredExecutorState,
    program::FactoredInstructionProgram,
    parsed::StimParseResult,
    shots::Integer,
    observable::Integer,
)
    total_shots = _checked_batch_sample_shots(shots)
    obs = _checked_observable_index(observable)
    detectors = stim_detectors(parsed)
    observables = stim_observable_includes(parsed)
    batch_words = size(runtime.measurements, 1)
    discard_bits = zeros(UInt64, batch_words)
    logical_bits = zeros(UInt64, batch_words)
    scratch = zeros(UInt64, batch_words)

    discarded = 0
    logical_errors = 0
    offset = 0
    t0 = time_ns()
    while offset < total_shots
        block = min(runtime.batches, total_shots - offset)
        reset_executor!(runtime, program; shots = block)
        execute!(runtime, program)
        d, l = _accumulate_stim_batch_summary!(
            discard_bits,
            logical_bits,
            scratch,
            runtime,
            detectors,
            observables,
            obs,
        )
        discarded += d
        logical_errors += l
        offset += block
    end
    elapsed = (time_ns() - t0) / 1e9
    return StimSampleSummary(total_shots, discarded, logical_errors, obs, runtime.batches, elapsed)
end

"""
    estimate_stim_logical_error_rate(parsed, shots; batches, kernel_backend, observable, rng)
    estimate_stim_logical_error_rate(path, shots; batches, kernel_backend, observable, rng)

Use the batch sampler to stream detector postselection and observable parity
counts. The discard rate is the fraction of shots with any detector firing.
The logical error rate is the requested observable's parity conditioned on
acceptance.
"""
function estimate_stim_logical_error_rate(
    parsed::StimParseResult,
    shots::Integer;
    batches = nothing,
    kernel_backend::Symbol = _default_batch_kernel_backend(),
    active_layout::Union{Nothing,Symbol} = nothing,
    observable::Integer = 0,
    rng = nothing,
)
    program = _stim_program(parsed)
    batch_count = batches === nothing ? _default_batch_count(program.max_k) : _checked_batch_count(batches)
    runtime = BatchFactoredExecutorState(program; batches = batch_count, kernel_backend, active_layout, rng)
    return _estimate_stim_logical_error_rate!(runtime, program, parsed, shots, observable)
end

function estimate_stim_logical_error_rate(
    path::AbstractString,
    shots::Integer;
    batches = nothing,
    kernel_backend::Symbol = _default_batch_kernel_backend(),
    active_layout::Union{Nothing,Symbol} = nothing,
    observable::Integer = 0,
    rng = nothing,
)
    parsed = parse_stim_file(path)
    return estimate_stim_logical_error_rate(
        parsed,
        shots;
        batches,
        kernel_backend,
        active_layout,
        observable,
        rng,
    )
end
