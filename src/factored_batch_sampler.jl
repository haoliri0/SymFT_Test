import Libdl

"""
    BatchFactoredExecutorState(program; batches, kernel_backend, active_layout, rng, values)

Mutable runtime for executing one `FactoredInstructionProgram` in blocks of
shots. The active subsystem is stored as one dense active vector per shot,
with shape `(batches, 1 << program.max_k)`. Symbol values and measurement
records are packed across shots, so symbolic expressions can be evaluated once
per batch while active amplitudes remain independent per shot. `kernel_backend`
selects the active update kernels: `:structarray`/`:turbo` uses Julia
StructArrays plus LoopVectorization, `:c`/`:avx2` uses the interleaved C AVX2
kernels, and `:scalar` uses the generic Julia fallback.
"""
mutable struct BatchFactoredExecutorState{R,A<:AbstractMatrix{ComplexF64}}
    n::Int
    k::Int
    batches::Int
    active_shots::Int
    ndormant::Int
    active::A
    active_scratch::A
    values::BitMatrix
    assigned::BitVector
    value_words::Matrix{UInt64}
    assigned_words::Vector{UInt64}
    measurements::Matrix{UInt64}
    eval_scratch::Vector{UInt64}
    rotation_coefficients::Vector{Float64}
    branch_prob_false::Vector{Float64}
    branch_prob_true::Vector{Float64}
    kernel_backend::Symbol
    rng::R
end

const _DEFAULT_BATCH_ACTIVE_AMPLITUDES = 1 << 17
const _DEFAULT_BATCH_SHOTS = 2048
const _DEFAULT_BATCH_KERNEL_BACKEND = :structarray
const _BATCH_AVX2_MIN_SHOTS = 128
const _BATCH_AVX2_HANDLE = Ref{Any}(nothing)
const _BATCH_AVX2_SYMBOL = Ref{Ptr{Cvoid}}(C_NULL)
const _BATCH_AVX2_UNIFORM_XMASK_SYMBOL = Ref{Ptr{Cvoid}}(C_NULL)
const _BATCH_AVX2_REAL_PAIR_FLIP_SYMBOL = Ref{Ptr{Cvoid}}(C_NULL)
const _BATCH_AVX2_REAL_PAIR_FLIP_XMASK_SYMBOL = Ref{Ptr{Cvoid}}(C_NULL)
const _BATCH_AVX2_DIAGONAL_MEASURE_PROB_SYMBOL = Ref{Ptr{Cvoid}}(C_NULL)
const _BATCH_AVX2_DIAGONAL_MEASURE_PROJECT_SYMBOL = Ref{Ptr{Cvoid}}(C_NULL)
const _BATCH_AVX2_NONDIAGONAL_MEASURE_PROB_SYMBOL = Ref{Ptr{Cvoid}}(C_NULL)
const _BATCH_AVX2_NONDIAGONAL_MEASURE_PROJECT_SYMBOL = Ref{Ptr{Cvoid}}(C_NULL)
const _BATCH_AVX2_PROMOTE_SYMBOL = Ref{Ptr{Cvoid}}(C_NULL)
const _BATCH_AVX2_TRIED = Ref(false)

function _default_batch_count(max_k::Integer)::Int
    max_ki = _checked_active_k(max_k)
    dim = _active_length(max_ki)
    return max(1, min(_DEFAULT_BATCH_SHOTS, _DEFAULT_BATCH_ACTIVE_AMPLITUDES ÷ dim))
end

function _default_batch_kernel_backend()::Symbol
    raw = Symbol(lowercase(strip(get(ENV, "SYMFT_BATCH_KERNEL_BACKEND", string(_DEFAULT_BATCH_KERNEL_BACKEND)))))
    return _normalize_batch_kernel_backend(raw)
end

function _normalize_batch_kernel_backend(kernel_backend::Symbol)::Symbol
    if kernel_backend in (:structarray, :struct, :soa, :loopvectorization, :turbo, :julia_turbo)
        return :structarray
    elseif kernel_backend in (:c, :avx2, :c_avx2)
        return :c
    elseif kernel_backend in (:scalar, :julia, :fallback)
        return :scalar
    elseif kernel_backend === :auto
        return _DEFAULT_BATCH_KERNEL_BACKEND
    end
    throw(ArgumentError("unsupported batch active kernel backend $kernel_backend"))
end

function _normalize_batch_active_layout(active_layout::Symbol)::Symbol
    if active_layout in (:struct, :structarray, :soa)
        return :struct
    elseif active_layout in (:complex, :matrix, :interleaved)
        return :complex
    end
    throw(ArgumentError("unsupported batch active layout $active_layout"))
end

function _default_batch_active_layout(kernel_backend::Symbol)::Symbol
    normalized = _normalize_batch_kernel_backend(kernel_backend)
    normalized === :structarray && return :struct
    return :complex
end

function _resolve_batch_active_layout(
    active_layout::Union{Nothing,Symbol},
    kernel_backend::Symbol,
)::Symbol
    backend = _normalize_batch_kernel_backend(kernel_backend)
    layout = isnothing(active_layout) ?
        _default_batch_active_layout(backend) :
        _normalize_batch_active_layout(active_layout)
    if backend === :structarray && layout !== :struct
        throw(ArgumentError("StructArray/LoopVectorization batch kernel requires active_layout = :struct"))
    elseif backend === :c && layout !== :complex
        throw(ArgumentError("C AVX2 batch kernel requires active_layout = :complex"))
    end
    return layout
end

# Shared backing-state allocator for the standard batch executor and the
# materialized fallback inside the shared-active executor.
function _new_batch_executor_state(
    program::FactoredInstructionProgram,
    batch_count::Int,
    backend::Symbol,
    layout::Symbol,
    rng,
)
    max_dim = _active_length(program.max_k)
    batch_words = _batch_word_count(batch_count)
    active = _new_batch_active_storage(layout, batch_count, max_dim)
    active_scratch = _new_batch_active_storage(layout, batch_count, max_dim)
    return BatchFactoredExecutorState(
        program.n,
        program.initial_k,
        batch_count,
        batch_count,
        program.n - program.initial_k,
        active,
        active_scratch,
        falses(batch_count, program.nsymbols),
        falses(program.nsymbols),
        zeros(UInt64, batch_words, program.nsymbols),
        zeros(UInt64, _symbol_word_count(program.nsymbols)),
        zeros(UInt64, batch_words, program.nrecords),
        zeros(UInt64, batch_words),
        zeros(Float64, batch_count),
        zeros(Float64, batch_count),
        zeros(Float64, batch_count),
        backend,
        rng,
    )
end

function _batch_avx2_cpu_supported()::Bool
    Sys.ARCH === :x86_64 || return false
    get(ENV, "SYMFT_DISABLE_AVX2_BATCH_ROT", "") in ("1", "true", "yes", "on") &&
        return false
    if Sys.islinux() && isfile("/proc/cpuinfo")
        try
            return occursin(" avx2 ", read("/proc/cpuinfo", String))
        catch
            return false
        end
    end
    return get(ENV, "SYMFT_ASSUME_AVX2", "") in ("1", "true", "yes", "on")
end

function _batch_avx2_symbol()::Ptr{Cvoid}
    _BATCH_AVX2_SYMBOL[] != C_NULL && return _BATCH_AVX2_SYMBOL[]
    _BATCH_AVX2_TRIED[] && return C_NULL
    _BATCH_AVX2_TRIED[] = true
    _batch_avx2_cpu_supported() || return C_NULL

    source = joinpath(@__DIR__, "active_rotation_pair_avx2.c")
    lib = joinpath(tempdir(), "libsymft_batch_rotation_avx2.so")
    try
        if !isfile(lib) || stat(lib).mtime < stat(source).mtime
            run(`gcc -O3 -mavx2 -mfma -fPIC -shared -o $lib $source`)
        end
        handle = Libdl.dlopen(lib)
        symbol = Libdl.dlsym(handle, :symft_batch_rotate_uniform_imag_pairs_colmajor_avx2)
        xmask_symbol = Libdl.dlsym(handle, :symft_batch_rotate_uniform_imag_xmask_colmajor_avx2)
        real_pair_flip_symbol = Libdl.dlsym(handle, :symft_batch_rotate_real_pair_flip_colmajor_avx2)
        real_pair_flip_xmask_symbol =
            Libdl.dlsym(handle, :symft_batch_rotate_real_pair_flip_xmask_colmajor_avx2)
        diagonal_measure_prob_symbol =
            Libdl.dlsym(handle, :symft_batch_diagonal_measure_true_prob_colmajor_avx2)
        diagonal_measure_project_symbol =
            Libdl.dlsym(handle, :symft_batch_project_diagonal_measure_colmajor_avx2)
        nondiagonal_measure_prob_symbol =
            Libdl.dlsym(handle, :symft_batch_nondiagonal_measure_true_prob_colmajor_avx2)
        nondiagonal_measure_project_symbol =
            Libdl.dlsym(handle, :symft_batch_project_nondiagonal_measure_colmajor_avx2)
        promote_symbol =
            Libdl.dlsym(handle, :symft_batch_promote_first_dormant_rotation_colmajor_avx2)
        _BATCH_AVX2_HANDLE[] = handle
        _BATCH_AVX2_SYMBOL[] = symbol
        _BATCH_AVX2_UNIFORM_XMASK_SYMBOL[] = xmask_symbol
        _BATCH_AVX2_REAL_PAIR_FLIP_SYMBOL[] = real_pair_flip_symbol
        _BATCH_AVX2_REAL_PAIR_FLIP_XMASK_SYMBOL[] = real_pair_flip_xmask_symbol
        _BATCH_AVX2_DIAGONAL_MEASURE_PROB_SYMBOL[] = diagonal_measure_prob_symbol
        _BATCH_AVX2_DIAGONAL_MEASURE_PROJECT_SYMBOL[] = diagonal_measure_project_symbol
        _BATCH_AVX2_NONDIAGONAL_MEASURE_PROB_SYMBOL[] = nondiagonal_measure_prob_symbol
        _BATCH_AVX2_NONDIAGONAL_MEASURE_PROJECT_SYMBOL[] = nondiagonal_measure_project_symbol
        _BATCH_AVX2_PROMOTE_SYMBOL[] = promote_symbol
        return symbol
    catch
        _BATCH_AVX2_HANDLE[] = nothing
        _BATCH_AVX2_SYMBOL[] = C_NULL
        _BATCH_AVX2_UNIFORM_XMASK_SYMBOL[] = C_NULL
        _BATCH_AVX2_REAL_PAIR_FLIP_SYMBOL[] = C_NULL
        _BATCH_AVX2_REAL_PAIR_FLIP_XMASK_SYMBOL[] = C_NULL
        _BATCH_AVX2_DIAGONAL_MEASURE_PROB_SYMBOL[] = C_NULL
        _BATCH_AVX2_DIAGONAL_MEASURE_PROJECT_SYMBOL[] = C_NULL
        _BATCH_AVX2_NONDIAGONAL_MEASURE_PROB_SYMBOL[] = C_NULL
        _BATCH_AVX2_NONDIAGONAL_MEASURE_PROJECT_SYMBOL[] = C_NULL
        _BATCH_AVX2_PROMOTE_SYMBOL[] = C_NULL
        return C_NULL
    end
end

function _batch_avx2_uniform_xmask_symbol()::Ptr{Cvoid}
    _batch_avx2_symbol()
    return _BATCH_AVX2_UNIFORM_XMASK_SYMBOL[]
end

function _batch_avx2_real_pair_flip_symbol()::Ptr{Cvoid}
    _batch_avx2_symbol()
    return _BATCH_AVX2_REAL_PAIR_FLIP_SYMBOL[]
end

function _batch_avx2_real_pair_flip_xmask_symbol()::Ptr{Cvoid}
    _batch_avx2_symbol()
    return _BATCH_AVX2_REAL_PAIR_FLIP_XMASK_SYMBOL[]
end

function _batch_avx2_diagonal_measure_prob_symbol()::Ptr{Cvoid}
    _batch_avx2_symbol()
    return _BATCH_AVX2_DIAGONAL_MEASURE_PROB_SYMBOL[]
end

function _batch_avx2_diagonal_measure_project_symbol()::Ptr{Cvoid}
    _batch_avx2_symbol()
    return _BATCH_AVX2_DIAGONAL_MEASURE_PROJECT_SYMBOL[]
end

function _batch_avx2_nondiagonal_measure_prob_symbol()::Ptr{Cvoid}
    _batch_avx2_symbol()
    return _BATCH_AVX2_NONDIAGONAL_MEASURE_PROB_SYMBOL[]
end

function _batch_avx2_nondiagonal_measure_project_symbol()::Ptr{Cvoid}
    _batch_avx2_symbol()
    return _BATCH_AVX2_NONDIAGONAL_MEASURE_PROJECT_SYMBOL[]
end

function _batch_avx2_promote_symbol()::Ptr{Cvoid}
    _batch_avx2_symbol()
    return _BATCH_AVX2_PROMOTE_SYMBOL[]
end

function BatchFactoredExecutorState(
    program::FactoredInstructionProgram;
    batches::Integer = _default_batch_count(program.max_k),
    kernel_backend::Symbol = _default_batch_kernel_backend(),
    active_layout::Union{Nothing,Symbol} = nothing,
    rng = nothing,
    values = nothing,
)
    batch_count = _checked_batch_count(batches)
    backend = _normalize_batch_kernel_backend(kernel_backend)
    layout = _resolve_batch_active_layout(active_layout, backend)
    runtime = _new_batch_executor_state(program, batch_count, backend, layout, rng)
    reset_executor!(runtime, program; shots = batch_count, values)
    return runtime
end

BatchFactoredExecutorState(
    program::FactoredInstructionProgram,
    batches::Integer;
    kernel_backend::Symbol = _default_batch_kernel_backend(),
    active_layout::Union{Nothing,Symbol} = nothing,
    rng = nothing,
    values = nothing,
) = BatchFactoredExecutorState(program; batches, kernel_backend, active_layout, rng, values)

"""
    sample_measurements(program, shots; batches, kernel_backend, active_layout, rng, values)

Execute `shots` samples of `program` in batches and return a
`shots × nrecords` `BitMatrix`.
"""
function sample_measurements(
    program::FactoredInstructionProgram,
    shots::Integer;
    batches::Integer = _default_batch_count(program.max_k),
    kernel_backend::Symbol = _default_batch_kernel_backend(),
    active_layout::Union{Nothing,Symbol} = nothing,
    rng = nothing,
    values = nothing,
)
    runtime = BatchFactoredExecutorState(program; batches, kernel_backend, active_layout, rng)
    return sample_measurements!(runtime, program, shots; values)
end

"""
    sample_measurements!(runtime, program, shots; values)

Reset and reuse `runtime` to execute `shots` samples of `program`, returning a
fresh `shots × nrecords` `BitMatrix`.
"""
function sample_measurements!(
    runtime::BatchFactoredExecutorState,
    program::FactoredInstructionProgram,
    shots::Integer;
    values = nothing,
)
    total_shots = _checked_batch_sample_shots(shots)
    out = falses(total_shots, program.nrecords)
    total_shots == 0 && return out

    offset = 0
    while offset < total_shots
        active_shots = min(runtime.batches, total_shots - offset)
        reset_executor!(runtime, program; shots = active_shots, values)
        execute!(runtime, program)
        _copy_batch_measurements_to!(out, runtime, offset)
        offset += active_shots
    end
    return out
end

function reset_executor!(
    runtime::BatchFactoredExecutorState,
    program::FactoredInstructionProgram;
    shots::Integer = runtime.batches,
    values = nothing,
)
    active_shots = _checked_active_batch_shots(runtime, shots)
    runtime.n = program.n
    runtime.k = program.initial_k
    runtime.active_shots = active_shots
    runtime.ndormant = program.n - program.initial_k
    _reserve_batch_active_capacity!(runtime, program.max_k)
    _resize_batch_symbol_storage!(runtime, program.nsymbols)
    _resize_batch_measurement_storage!(runtime, program.nrecords)
    _set_batch_active_from_initial!(runtime, program.initial_active)
    _assign_input_values!(runtime, values)
    return runtime
end

"""
    execute!(runtime, program)
    execute!(runtime, program, shots)

Run `program.instructions` in an initialized batch runtime. Exogenous symbols
from the program's sampling plan are assigned before instruction execution;
measurement branch symbols are assigned by measurement instructions. The
optional `shots` overload changes the active shot count for the next batch.
"""
function execute!(
    runtime::BatchFactoredExecutorState,
    program::FactoredInstructionProgram,
)
    _check_batch_executor_program(runtime, program)
    runtime.active_shots == 0 && return runtime.measurements
    _sample_exogenous_symbols!(runtime, program)
    return _execute_factored_instructions!(runtime, program)
end

function execute!(
    runtime::BatchFactoredExecutorState,
    program::FactoredInstructionProgram,
    shots::Integer,
)
    runtime.active_shots = _checked_active_batch_shots(runtime, shots)
    return execute!(runtime, program)
end

function _execute_factored_instructions!(
    runtime::BatchFactoredExecutorState,
    program::FactoredInstructionProgram,
)
    for instruction in program.instructions
        execute_instruction!(runtime, instruction)
    end
    return runtime.measurements
end

function execute_instruction!(runtime::BatchFactoredExecutorState, instruction::ApplyPrecomputedActivePauliRotation)
    sign_bits = _eval_symbolic_bool!(runtime.eval_scratch, instruction.sign_plan, runtime)
    _batch_rotate_pauli!(runtime, instruction.rotation_kernel, sign_bits)
    return runtime
end

function execute_instruction!(runtime::BatchFactoredExecutorState, instruction::ApplyActiveBasisChange)
    _batch_apply_active_basis_change!(runtime, instruction.kind, instruction.qubit)
    return runtime
end

function execute_instruction!(runtime::BatchFactoredExecutorState, instruction::PromoteDormantRotation)
    sign_bits = _eval_symbolic_bool!(runtime.eval_scratch, instruction.sign_plan, runtime)
    _batch_promote_first_dormant_rotation!(runtime, instruction.theta, sign_bits)
    return runtime
end

function execute_instruction!(runtime::BatchFactoredExecutorState, instruction::RecordMeasurement)
    outcome_bits = _eval_symbolic_bool!(runtime.eval_scratch, instruction.outcome_plan, runtime)
    _write_measurement_record!(runtime, instruction.record, outcome_bits, instruction.record_condition)
    return runtime
end

function execute_instruction!(runtime::BatchFactoredExecutorState, instruction::MeasureActiveLastZ)
    return _measure_active_last_z!(
        runtime,
        instruction.branch,
        instruction.outcome_plan,
        instruction.record,
        instruction.record_condition,
    )
end

function execute_instruction!(runtime::BatchFactoredExecutorState, instruction::MeasurePrecomputedActivePauli)
    return _measure_precomputed_active_pauli!(
        runtime,
        instruction.kernel,
        instruction.branch,
        instruction.outcome_plan,
        instruction.record,
        instruction.record_condition,
    )
end

function _measure_active_last_z!(
    runtime::BatchFactoredExecutorState,
    branch_condition::Int,
    outcome_plan::SymbolicBoolEvaluationPlan,
    record::Union{Nothing,Int},
    record_condition::Union{Nothing,Int},
)
    runtime.k > 0 || throw(ArgumentError("cannot measure the last active qubit when k == 0"))
    _fill_batch_bits!(runtime.eval_scratch, runtime, false)
    branch_bits = runtime.eval_scratch
    old_k = runtime.k
    new_k = old_k - 1
    dim = _active_length(new_k)

    prob_true = _clear_batch_probabilities!(runtime, runtime.branch_prob_true)
    @inbounds for basis in 1:dim
        src = dim + basis
        @simd for shot in 1:runtime.active_shots
            prob_true[shot] += abs2(runtime.active[shot, src])
        end
    end

    invnorms = runtime.branch_prob_false
    @inbounds for shot in 1:runtime.active_shots
        prob1 = min(max(prob_true[shot], 0.0), 1.0)
        branch = _sample_bernoulli(runtime.rng, prob1)
        branch && _set_batch_bit!(branch_bits, shot)
        probability = branch ? prob1 : 1.0 - prob1
        probability > 0.0 ||
            throw(ArgumentError("sampled an impossible active measurement branch"))
        invnorms[shot] = inv(sqrt(probability))
    end

    @inbounds for basis in 1:dim
        false_src = basis
        true_src = dim + basis
        @simd for shot in 1:runtime.active_shots
            src = _batch_bit(branch_bits, shot) ? true_src : false_src
            runtime.active[shot, basis] = runtime.active[shot, src] * invnorms[shot]
        end
    end
    runtime.k = new_k
    runtime.ndormant += 1

    _assign_symbol!(runtime, branch_condition, branch_bits)
    outcome_bits = _eval_symbolic_bool!(runtime.eval_scratch, outcome_plan, runtime)
    _write_measurement_record!(runtime, record, outcome_bits, record_condition)
    return runtime
end

function _measure_precomputed_active_pauli!(
    runtime::BatchFactoredExecutorState,
    kernel::PrecomputedActivePauliMeasurementKernel,
    branch_condition::Int,
    outcome_plan::SymbolicBoolEvaluationPlan,
    record::Union{Nothing,Int},
    record_condition::Union{Nothing,Int},
)
    runtime.k > 0 || throw(ArgumentError("cannot measure an active Pauli when k == 0"))
    _check_active_measurement_kernel(runtime.k, kernel)
    if kernel.is_diagonal
        return _measure_diagonal_active_pauli!(
            runtime,
            kernel,
            branch_condition,
            outcome_plan,
            record,
            record_condition,
        )
    end
    _fill_batch_bits!(runtime.eval_scratch, runtime, false)
    branch_bits = runtime.eval_scratch
    prob_true = _clear_batch_probabilities!(runtime, runtime.branch_prob_true)
    if runtime.active_shots < _BATCH_AVX2_MIN_SHOTS ||
            !(
                _batch_nondiagonal_measure_true_prob_struct_turbo!(runtime, kernel, prob_true) ||
                _batch_nondiagonal_measure_true_prob_avx2!(runtime, kernel, prob_true)
            )
        _accumulate_batch_active_measurement_branch_probabilities!(
            runtime,
            kernel,
            true,
            prob_true,
        )
    end
    invnorms = _sample_batch_active_measurement_branches_from_true!(
        runtime,
        branch_bits,
        prob_true,
    )
    if runtime.active_shots < _BATCH_AVX2_MIN_SHOTS ||
            !(
                _batch_project_nondiagonal_measurement_struct_turbo!(
                    runtime,
                    kernel,
                    branch_bits,
                    invnorms,
                ) ||
                _batch_project_nondiagonal_measurement_avx2!(runtime, kernel, branch_bits, invnorms)
            )
        _project_batch_active_pauli_measurement!(runtime, kernel, branch_bits, invnorms)
    end
    _copy_batch_active_scratch!(runtime, length(kernel.source0_false))
    _assign_symbol!(runtime, branch_condition, branch_bits)
    runtime.k -= 1
    runtime.ndormant += 1

    outcome_bits = _eval_symbolic_bool!(runtime.eval_scratch, outcome_plan, runtime)
    _write_measurement_record!(runtime, record, outcome_bits, record_condition)
    return runtime
end

function _measure_diagonal_active_pauli!(
    runtime::BatchFactoredExecutorState,
    kernel::PrecomputedActivePauliMeasurementKernel,
    branch_condition::Int,
    outcome_plan::SymbolicBoolEvaluationPlan,
    record::Union{Nothing,Int},
    record_condition::Union{Nothing,Int},
)
    _fill_batch_bits!(runtime.eval_scratch, runtime, false)
    branch_bits = runtime.eval_scratch
    source_false = kernel.source0_false
    source_true = kernel.source0_true
    out_dim = length(source_false)
    prob_true = runtime.branch_prob_true
    if runtime.active_shots < _BATCH_AVX2_MIN_SHOTS ||
            !(
                _batch_diagonal_measure_true_prob_struct_turbo!(runtime, source_true, out_dim, prob_true) ||
                _batch_diagonal_measure_true_prob_avx2!(runtime, source_true, out_dim, prob_true)
            )
        _clear_batch_probabilities!(runtime, prob_true)
        @inbounds for idx in eachindex(source_false, source_true)
            st = source_true[idx]
            @simd for shot in 1:runtime.active_shots
                prob_true[shot] += abs2(runtime.active[shot, st])
            end
        end
    end
    invnorms = _sample_batch_active_measurement_branches_from_true!(
        runtime,
        branch_bits,
        prob_true,
    )
    if runtime.active_shots < _BATCH_AVX2_MIN_SHOTS ||
            !(
                _batch_project_diagonal_measurement_struct_turbo!(
                    runtime,
                    source_false,
                    source_true,
                    out_dim,
                    branch_bits,
                    invnorms,
                ) ||
                _batch_project_diagonal_measurement_avx2!(
                    runtime,
                    source_false,
                    source_true,
                    out_dim,
                    branch_bits,
                    invnorms,
                )
            )
        # Diagonal measurement sources are never below their output index, so
        # the active prefix can be projected in place in ascending index order.
        @inbounds for idx in 1:out_dim
            sf = source_false[idx]
            st = source_true[idx]
            @simd for shot in 1:runtime.active_shots
                src = _batch_bit(branch_bits, shot) ? st : sf
                runtime.active[shot, idx] = runtime.active[shot, src] * invnorms[shot]
            end
        end
    end
    _assign_symbol!(runtime, branch_condition, branch_bits)
    runtime.k -= 1
    runtime.ndormant += 1

    outcome_bits = _eval_symbolic_bool!(runtime.eval_scratch, outcome_plan, runtime)
    _write_measurement_record!(runtime, record, outcome_bits, record_condition)
    return runtime
end

function _batch_diagonal_measure_true_prob_avx2!(
    runtime::BatchFactoredExecutorState,
    source_true::Vector{Int},
    out_dim::Int,
    prob_true::Vector{Float64},
)::Bool
    runtime.kernel_backend === :c || return false
    runtime.active isa Matrix{ComplexF64} || return false
    symbol = _batch_avx2_diagonal_measure_prob_symbol()
    symbol == C_NULL && return false
    length(prob_true) >= runtime.active_shots ||
        throw(DimensionMismatch("probability scratch is too short for the active batch"))
    ccall(
        symbol,
        Cvoid,
        (Ptr{Cdouble}, Int64, Int64, Ptr{Int64}, Int64, Ptr{Cdouble}),
        runtime.active,
        size(runtime.active, 1),
        runtime.active_shots,
        source_true,
        out_dim,
        prob_true,
    )
    return true
end

function _batch_project_diagonal_measurement_avx2!(
    runtime::BatchFactoredExecutorState,
    source_false::Vector{Int},
    source_true::Vector{Int},
    out_dim::Int,
    branch_bits::Vector{UInt64},
    invnorms::Vector{Float64},
)::Bool
    runtime.kernel_backend === :c || return false
    runtime.active isa Matrix{ComplexF64} || return false
    symbol = _batch_avx2_diagonal_measure_project_symbol()
    symbol == C_NULL && return false
    length(invnorms) >= runtime.active_shots ||
        throw(DimensionMismatch("inverse-norm scratch is too short for the active batch"))
    length(branch_bits) >= _runtime_batch_word_count(runtime) ||
        throw(DimensionMismatch("branch bit vector is too short for the active batch"))
    ccall(
        symbol,
        Cvoid,
        (Ptr{Cdouble}, Ptr{Cdouble}, Int64, Int64, Ptr{Int64}, Ptr{Int64}, Int64, Ptr{UInt64}, Ptr{Cdouble}),
        runtime.active,
        runtime.active,
        size(runtime.active, 1),
        runtime.active_shots,
        source_false,
        source_true,
        out_dim,
        branch_bits,
        invnorms,
    )
    return true
end

function _batch_nondiagonal_measure_true_prob_avx2!(
    runtime::BatchFactoredExecutorState,
    kernel::PrecomputedActivePauliMeasurementKernel,
    prob_true::Vector{Float64},
)::Bool
    runtime.kernel_backend === :c || return false
    runtime.active isa Matrix{ComplexF64} || return false
    symbol = _batch_avx2_nondiagonal_measure_prob_symbol()
    symbol == C_NULL && return false
    length(prob_true) >= runtime.active_shots ||
        throw(DimensionMismatch("probability scratch is too short for the active batch"))
    out_dim = length(kernel.source0_false)
    ccall(
        symbol,
        Cvoid,
        (Ptr{Cdouble}, Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Int64, Ptr{Cdouble}),
        runtime.active,
        size(runtime.active, 1),
        runtime.active_shots,
        kernel.source0_false,
        kernel.source1_false,
        kernel.coeff1_false,
        out_dim,
        prob_true,
    )
    return true
end

function _batch_project_nondiagonal_measurement_avx2!(
    runtime::BatchFactoredExecutorState,
    kernel::PrecomputedActivePauliMeasurementKernel,
    branch_bits::Vector{UInt64},
    invnorms::Vector{Float64},
)::Bool
    runtime.kernel_backend === :c || return false
    runtime.active isa Matrix{ComplexF64} || return false
    symbol = _batch_avx2_nondiagonal_measure_project_symbol()
    symbol == C_NULL && return false
    length(invnorms) >= runtime.active_shots ||
        throw(DimensionMismatch("inverse-norm scratch is too short for the active batch"))
    length(branch_bits) >= _runtime_batch_word_count(runtime) ||
        throw(DimensionMismatch("branch bit vector is too short for the active batch"))
    out_dim = length(kernel.source0_false)
    ccall(
        symbol,
        Cvoid,
        (Ptr{Cdouble}, Ptr{Cdouble}, Int64, Int64, Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, Int64, Ptr{UInt64}, Ptr{Cdouble}),
        runtime.active,
        runtime.active_scratch,
        size(runtime.active, 1),
        runtime.active_shots,
        kernel.source0_false,
        kernel.source1_false,
        kernel.coeff1_false,
        out_dim,
        branch_bits,
        invnorms,
    )
    return true
end

function _clear_batch_probabilities!(
    runtime::BatchFactoredExecutorState,
    probabilities::Vector{Float64},
)
    length(probabilities) >= runtime.active_shots ||
        throw(DimensionMismatch("probability scratch is too short for the active batch"))
    @inbounds @simd for shot in 1:runtime.active_shots
        probabilities[shot] = 0.0
    end
    return probabilities
end

function _sample_batch_active_measurement_branches_from_true!(
    runtime::BatchFactoredExecutorState,
    branch_bits::Vector{UInt64},
    prob_true::Vector{Float64},
)
    @inbounds for shot in 1:runtime.active_shots
        prob_false, pt = _active_measurement_complementary_branch_probabilities(prob_true[shot])
        branch = _sample_bernoulli(runtime.rng, pt)
        branch && _set_batch_bit!(branch_bits, shot)
        probability = branch ? pt : prob_false
        probability > 0.0 ||
            throw(ArgumentError("sampled an impossible active measurement branch"))
        prob_true[shot] = inv(sqrt(probability))
    end
    return prob_true
end

function _accumulate_batch_active_measurement_branch_probabilities!(
    runtime::BatchFactoredExecutorState,
    kernel::PrecomputedActivePauliMeasurementKernel,
    branch::Bool,
    probabilities::Vector{Float64},
)
    sources0, sources1, coeffs0, coeffs1 = _measurement_kernel_branch(kernel, branch)
    @inbounds for idx in eachindex(sources0)
        src0 = sources0[idx]
        src1 = sources1[idx]
        coeff0 = coeffs0[idx]
        if src1 == 0
            @simd for shot in 1:runtime.active_shots
                amp = coeff0 * runtime.active[shot, src0]
                probabilities[shot] += abs2(amp)
            end
        else
            coeff1 = coeffs1[idx]
            @simd for shot in 1:runtime.active_shots
                amp = coeff0 * runtime.active[shot, src0] +
                    coeff1 * runtime.active[shot, src1]
                probabilities[shot] += abs2(amp)
            end
        end
    end
    return probabilities
end

function _project_batch_active_pauli_measurement!(
    runtime::BatchFactoredExecutorState,
    kernel::PrecomputedActivePauliMeasurementKernel,
    branch_bits::Vector{UInt64},
    invnorms::Vector{Float64},
)
    fs0, fs1, fc0, fc1 = _measurement_kernel_branch(kernel, false)
    ts0, ts1, tc0, tc1 = _measurement_kernel_branch(kernel, true)
    @inbounds for idx in eachindex(fs0, ts0)
        fsrc0 = fs0[idx]
        fsrc1 = fs1[idx]
        fcoeff0 = fc0[idx]
        fcoeff1 = fc1[idx]
        tsrc0 = ts0[idx]
        tsrc1 = ts1[idx]
        tcoeff0 = tc0[idx]
        tcoeff1 = tc1[idx]
        @simd for shot in 1:runtime.active_shots
            if _batch_bit(branch_bits, shot)
                amp = tcoeff0 * runtime.active[shot, tsrc0]
                tsrc1 != 0 && (amp += tcoeff1 * runtime.active[shot, tsrc1])
            else
                amp = fcoeff0 * runtime.active[shot, fsrc0]
                fsrc1 != 0 && (amp += fcoeff1 * runtime.active[shot, fsrc1])
            end
            runtime.active_scratch[shot, idx] = amp * invnorms[shot]
        end
    end
    return runtime
end

function execute_instruction!(
    runtime::BatchFactoredExecutorState,
    instruction::IntroduceDormantMeasurementBranch,
)
    branch_bits = runtime.eval_scratch
    _fill_batch_bits!(branch_bits, runtime, false)
    @inbounds for shot in 1:runtime.active_shots
        _sample_bernoulli(runtime.rng, 0.5) && _set_batch_bit!(branch_bits, shot)
    end
    _assign_symbol!(runtime, instruction.branch, branch_bits)
    outcome_bits = _eval_symbolic_bool!(runtime.eval_scratch, instruction.outcome_plan, runtime)
    _write_measurement_record!(runtime, instruction.record, outcome_bits, instruction.record_condition)
    return runtime
end

function _checked_batch_count(batches::Integer)::Int
    batch_count = Int(batches)
    batch_count > 0 || throw(ArgumentError("batch size must be positive"))
    return batch_count
end

function _checked_batch_sample_shots(shots::Integer)::Int
    shot_count = Int(shots)
    shot_count >= 0 || throw(ArgumentError("shot count must be nonnegative"))
    return shot_count
end

function _checked_active_batch_shots(runtime::BatchFactoredExecutorState, shots::Integer)::Int
    active_shots = Int(shots)
    0 <= active_shots <= runtime.batches ||
        throw(ArgumentError("active shot count $active_shots is outside 0:$(runtime.batches)"))
    return active_shots
end

@inline function _batch_word_count(shots::Integer)::Int
    return cld(max(Int(shots), 0), 64)
end

@inline function _runtime_batch_word_count(runtime::BatchFactoredExecutorState)::Int
    return _batch_word_count(runtime.active_shots)
end

@inline function _batch_word_index(shot::Integer)::Int
    s = Int(shot)
    s > 0 || throw(ArgumentError("shot index must be positive"))
    return ((s - 1) >>> 6) + 1
end

@inline function _batch_bit_mask(shot::Integer)::UInt64
    s = Int(shot)
    s > 0 || throw(ArgumentError("shot index must be positive"))
    return UInt64(1) << ((s - 1) & 63)
end

@inline function _low_bits_mask(nbits::Int)::UInt64
    nbits <= 0 && return UInt64(0)
    nbits >= 64 && return typemax(UInt64)
    return (UInt64(1) << nbits) - UInt64(1)
end

@inline function _batch_live_word_mask(runtime::BatchFactoredExecutorState, word::Int)::UInt64
    remaining = runtime.active_shots - ((word - 1) << 6)
    return _low_bits_mask(remaining)
end

@inline function _batch_bit(bits::Vector{UInt64}, shot::Int)::Bool
    return (bits[_batch_word_index(shot)] & _batch_bit_mask(shot)) != 0
end

@inline function _set_batch_bit!(bits::Vector{UInt64}, shot::Int)
    bits[_batch_word_index(shot)] |= _batch_bit_mask(shot)
    return bits
end

function _fill_batch_bits!(
    bits::Vector{UInt64},
    runtime::BatchFactoredExecutorState,
    value::Bool,
)
    nwords = _runtime_batch_word_count(runtime)
    length(bits) >= nwords ||
        throw(DimensionMismatch("bit scratch has length $(length(bits)); expected at least $nwords"))
    fill_word = value ? typemax(UInt64) : UInt64(0)
    @inbounds for word in 1:nwords
        bits[word] = fill_word & _batch_live_word_mask(runtime, word)
    end
    @inbounds for word in (nwords + 1):length(bits)
        bits[word] = UInt64(0)
    end
    return bits
end

function _check_batch_executor_program(
    runtime::BatchFactoredExecutorState,
    program::FactoredInstructionProgram,
)
    runtime.n == program.n ||
        throw(DimensionMismatch("batch executor has $(runtime.n) qubits; program has $(program.n)"))
    runtime.k + runtime.ndormant == runtime.n ||
        throw(DimensionMismatch("batch executor active+dormant width does not equal n"))
    size(runtime.active, 1) == runtime.batches ||
        throw(DimensionMismatch("active matrix row count does not match batch size"))
    size(runtime.active_scratch) == size(runtime.active) ||
        throw(DimensionMismatch("active scratch shape does not match active shape"))
    return nothing
end

function _reserve_batch_active_capacity!(
    runtime::BatchFactoredExecutorState,
    max_k::Integer,
)
    max_ki = _checked_active_k(max_k)
    max_dim = _active_length(max_ki)
    if size(runtime.active, 1) != runtime.batches || size(runtime.active, 2) < max_dim
        runtime.active = _new_batch_active_storage_like(runtime.active, runtime.batches, max_dim)
    end
    if size(runtime.active_scratch, 1) != runtime.batches || size(runtime.active_scratch, 2) < max_dim
        runtime.active_scratch =
            _new_batch_active_storage_like(runtime.active_scratch, runtime.batches, max_dim)
    end
    return runtime
end

function _new_batch_active_storage(layout::Symbol, batches::Int, max_dim::Int)
    if layout === :complex || layout === :matrix
        return zeros(ComplexF64, batches, max_dim)
    elseif layout === :struct || layout === :soa
        return StructArray{ComplexF64}((zeros(Float64, batches, max_dim), zeros(Float64, batches, max_dim)))
    end
    throw(ArgumentError("unsupported batch active layout $layout"))
end

function _new_batch_active_storage_like(active::AbstractMatrix{ComplexF64}, batches::Int, max_dim::Int)
    if active isa StructArray
        return _new_batch_active_storage(:struct, batches, max_dim)
    end
    return _new_batch_active_storage(:complex, batches, max_dim)
end

function _resize_batch_symbol_storage!(
    runtime::BatchFactoredExecutorState,
    nsymbols::Int,
)
    batch_words = _batch_word_count(runtime.batches)
    if size(runtime.values, 1) != runtime.batches || size(runtime.values, 2) != nsymbols
        runtime.values = falses(runtime.batches, nsymbols)
    else
        fill!(runtime.values, false)
    end
    if length(runtime.assigned) != nsymbols
        runtime.assigned = falses(nsymbols)
    else
        fill!(runtime.assigned, false)
    end
    if size(runtime.value_words, 1) != batch_words || size(runtime.value_words, 2) != nsymbols
        runtime.value_words = zeros(UInt64, batch_words, nsymbols)
    else
        fill!(runtime.value_words, UInt64(0))
    end
    assigned_word_count = _symbol_word_count(nsymbols)
    if length(runtime.assigned_words) != assigned_word_count
        runtime.assigned_words = zeros(UInt64, assigned_word_count)
    else
        fill!(runtime.assigned_words, UInt64(0))
    end
    if length(runtime.eval_scratch) != batch_words
        runtime.eval_scratch = zeros(UInt64, batch_words)
    else
        fill!(runtime.eval_scratch, UInt64(0))
    end
    return runtime
end

function _resize_batch_measurement_storage!(
    runtime::BatchFactoredExecutorState,
    nrecords::Int,
)
    batch_words = _batch_word_count(runtime.batches)
    if size(runtime.measurements, 1) != batch_words || size(runtime.measurements, 2) != nrecords
        runtime.measurements = zeros(UInt64, batch_words, nrecords)
    else
        fill!(runtime.measurements, UInt64(0))
    end
    return runtime
end

function _ensure_batch_measurement_storage!(
    runtime::BatchFactoredExecutorState,
    record::Int,
)
    record <= size(runtime.measurements, 2) && return runtime
    old = runtime.measurements
    runtime.measurements = zeros(UInt64, size(old, 1), record)
    copyto!(runtime.measurements, old)
    return runtime
end

function _set_batch_active_from_initial!(
    runtime::BatchFactoredExecutorState,
    initial::ActiveState,
)
    initial.k == runtime.k ||
        throw(DimensionMismatch("initial ActiveState has $(initial.k) qubits; expected $(runtime.k)"))
    dim = _check_active_storage(initial)
    size(runtime.active, 2) >= dim ||
        throw(DimensionMismatch("batch active storage has too few columns"))
    @inbounds for idx in 1:dim
        amp = initial.alpha[idx]
        @simd for shot in 1:runtime.active_shots
            runtime.active[shot, idx] = amp
        end
    end
    return runtime
end

function _assign_input_values!(runtime::BatchFactoredExecutorState, values)
    values === nothing && return runtime
    for (condition, value) in values
        _assign_symbol!(runtime, condition, Bool(value))
    end
    return runtime
end

function _sample_exogenous_symbols!(
    runtime::BatchFactoredExecutorState,
    program::FactoredInstructionProgram,
)
    for distribution in program.sampled_categorical_distributions
        _sample_categorical_distribution!(runtime, distribution)
    end
    for group in program.sampled_rare_categorical_groups
        _sample_rare_categorical_group!(runtime, group)
    end
    for idx in eachindex(program.sampled_bernoulli_conditions, program.sampled_bernoulli_probabilities)
        condition = program.sampled_bernoulli_conditions[idx]
        _is_assigned(runtime, condition) && continue
        _sample_bernoulli_condition!(
            runtime,
            condition,
            program.sampled_bernoulli_probabilities[idx],
        )
    end
    for group in program.sampled_low_probability_bernoulli_groups
        _sample_low_probability_bernoulli_group!(runtime, group)
    end
    return runtime
end

function _sample_categorical_distribution!(
    runtime::BatchFactoredExecutorState,
    distribution::SymbolicCategoricalDistribution,
)
    return _sample_categorical_distribution!(
        runtime,
        distribution.conditions,
        distribution.assignments,
        distribution.probabilities,
    )
end

function _sample_categorical_distribution!(
    runtime::BatchFactoredExecutorState,
    conditions::Vector{Int},
    assignments::Vector{Vector{Bool}},
    probabilities::Vector{Float64},
)
    all_assigned = true
    any_assigned = false
    for condition in conditions
        assigned = _is_assigned(runtime, condition)
        all_assigned &= assigned
        any_assigned |= assigned
    end
    all_assigned && return runtime
    any_assigned &&
        throw(ArgumentError("categorical symbolic distribution was only partially preassigned"))

    _assign_conditions_false!(runtime, conditions)
    @inbounds for shot in 1:runtime.active_shots
        row = _sample_categorical_row(runtime.rng, probabilities)
        assignment = assignments[row]
        for bit_idx in eachindex(conditions, assignment)
            assignment[bit_idx] &&
                _set_batch_symbol_true_unchecked!(runtime, conditions[bit_idx], shot)
        end
    end
    return runtime
end

function _sample_rare_categorical_group!(
    runtime::BatchFactoredExecutorState,
    group::RareCategoricalSampleGroup,
)
    if _any_categorical_group_assigned(runtime, group.conditions)
        for conditions in group.conditions
            _sample_categorical_distribution!(
                runtime,
                conditions,
                group.assignments,
                group.probabilities,
            )
        end
        return runtime
    end

    _assign_categorical_group_false!(runtime, group.conditions)
    group.event_probability <= 0.0 && return runtime
    nsets = length(group.conditions)
    total_draws = runtime.active_shots * nsets
    draw = 1
    while true
        gap = _sample_geometric_gap(runtime.rng, group.event_probability)
        gap > total_draws - draw && return runtime
        draw += gap
        shot = ((draw - 1) ÷ nsets) + 1
        set_idx = ((draw - 1) % nsets) + 1
        row = group.event_rows[
            _sample_categorical_row(runtime.rng, group.event_probabilities)
        ]
        conditions = group.conditions[set_idx]
        assignment = group.assignments[row]
        @inbounds for bit_idx in eachindex(conditions, assignment)
            assignment[bit_idx] &&
                _set_batch_symbol_true_unchecked!(runtime, conditions[bit_idx], shot)
        end
        draw += 1
    end
end

function _sample_bernoulli_condition!(
    runtime::BatchFactoredExecutorState,
    condition::Int,
    probability::Float64,
)
    p = _check_probability(probability)
    if p <= 0.0
        _assign_symbol!(runtime, condition, false)
        return runtime
    elseif p >= 1.0
        _assign_symbol!(runtime, condition, true)
        return runtime
    end

    _set_batch_symbol_false_unchecked!(runtime, condition)
    @inbounds for shot in 1:runtime.active_shots
        _sample_bernoulli(runtime.rng, p) &&
            _set_batch_symbol_true_unchecked!(runtime, condition, shot)
    end
    return runtime
end

function _sample_low_probability_bernoulli_group!(
    runtime::BatchFactoredExecutorState,
    group::BernoulliSampleGroup,
)
    if _any_assigned(runtime, group.conditions)
        for condition in group.conditions
            _is_assigned(runtime, condition) && continue
            _sample_bernoulli_condition!(runtime, condition, group.probability)
        end
        return runtime
    end

    _assign_conditions_false!(runtime, group.conditions)
    group.probability <= 0.0 && return runtime
    nconditions = length(group.conditions)
    total_draws = runtime.active_shots * nconditions
    draw = 1
    while true
        gap = _sample_geometric_gap(runtime.rng, group.probability)
        gap > total_draws - draw && return runtime
        draw += gap
        shot = ((draw - 1) ÷ nconditions) + 1
        condition_idx = ((draw - 1) % nconditions) + 1
        _set_batch_symbol_true_unchecked!(runtime, group.conditions[condition_idx], shot)
        draw += 1
    end
end

function _any_categorical_group_assigned(
    runtime::BatchFactoredExecutorState,
    condition_sets::Vector{Vector{Int}},
)
    for conditions in condition_sets
        _any_assigned(runtime, conditions) && return true
    end
    return false
end

function _any_assigned(runtime::BatchFactoredExecutorState, conditions::Vector{Int})
    for condition in conditions
        _is_assigned(runtime, condition) && return true
    end
    return false
end

function _assign_categorical_group_false!(
    runtime::BatchFactoredExecutorState,
    condition_sets::Vector{Vector{Int}},
)
    for conditions in condition_sets
        _assign_conditions_false!(runtime, conditions)
    end
    return runtime
end

function _assign_conditions_false!(runtime::BatchFactoredExecutorState, conditions::Vector{Int})
    @inbounds for condition in conditions
        _set_batch_symbol_false_unchecked!(runtime, condition)
    end
    return runtime
end

function _check_symbol_slot(runtime::BatchFactoredExecutorState, condition::Integer)
    c = Int(condition)
    c > 0 || throw(ArgumentError("condition id must be positive"))
    c <= size(runtime.values, 2) ||
        throw(ArgumentError("condition s$c exceeds batch executor symbol table length $(size(runtime.values, 2))"))
    return c
end

@inline function _symbol_assigned(runtime::BatchFactoredExecutorState, condition::Int)::Bool
    condition <= length(runtime.assigned) || return false
    word = _symbol_word_index(condition)
    word <= length(runtime.assigned_words) || return false
    return (runtime.assigned_words[word] & _symbol_bit_mask(condition)) != 0
end

function _is_assigned(runtime::BatchFactoredExecutorState, condition::Integer)
    c = _check_symbol_slot(runtime, condition)
    @inbounds return runtime.assigned[c]
end

function _mark_batch_symbol_assigned_unchecked!(
    runtime::BatchFactoredExecutorState,
    condition::Int,
)
    @inbounds begin
        runtime.assigned[condition] = true
        runtime.assigned_words[_symbol_word_index(condition)] |= _symbol_bit_mask(condition)
    end
    return runtime
end

function _set_batch_symbol_false_unchecked!(
    runtime::BatchFactoredExecutorState,
    condition::Int,
)
    # Batch reset already zeros value storage; false assignment only needs to
    # make the symbolic value visible to later expression evaluation.
    _mark_batch_symbol_assigned_unchecked!(runtime, condition)
    return runtime
end

function _set_batch_symbol_true_unchecked!(
    runtime::BatchFactoredExecutorState,
    condition::Int,
    shot::Int,
)
    @inbounds begin
        runtime.values[shot, condition] = true
        runtime.value_words[_batch_word_index(shot), condition] |= _batch_bit_mask(shot)
    end
    return runtime
end

function _assign_symbol!(runtime::BatchFactoredExecutorState, condition::Nothing, value)
    return runtime
end

function _assign_symbol!(
    runtime::BatchFactoredExecutorState,
    condition::Integer,
    value::Bool,
)
    c = _check_symbol_slot(runtime, condition)
    if runtime.assigned[c]
        _batch_symbol_matches_scalar(runtime, c, value) ||
            throw(ArgumentError("condition s$c was assigned inconsistent concrete values"))
        return runtime
    end
    _mark_batch_symbol_assigned_unchecked!(runtime, c)
    value || return runtime
    @inbounds begin
        for word in 1:size(runtime.value_words, 1)
            runtime.value_words[word, c] = _batch_live_word_mask(runtime, word)
        end
        for shot in 1:runtime.batches
            runtime.values[shot, c] = shot <= runtime.active_shots
        end
    end
    return runtime
end

function _assign_symbol!(
    runtime::BatchFactoredExecutorState,
    condition::Integer,
    bits::Vector{UInt64},
)
    c = _check_symbol_slot(runtime, condition)
    if runtime.assigned[c]
        _batch_symbol_matches_bits(runtime, c, bits) ||
            throw(ArgumentError("condition s$c was assigned inconsistent concrete values"))
        return runtime
    end
    _mark_batch_symbol_assigned_unchecked!(runtime, c)
    _copy_bits_to_batch_symbol_unchecked!(runtime, c, bits)
    return runtime
end

function _batch_symbol_matches_scalar(
    runtime::BatchFactoredExecutorState,
    condition::Int,
    value::Bool,
)
    nwords = _runtime_batch_word_count(runtime)
    @inbounds for word in 1:nwords
        expected = value ? _batch_live_word_mask(runtime, word) : UInt64(0)
        (runtime.value_words[word, condition] & _batch_live_word_mask(runtime, word)) == expected ||
            return false
    end
    return true
end

function _batch_symbol_matches_bits(
    runtime::BatchFactoredExecutorState,
    condition::Int,
    bits::Vector{UInt64},
)
    nwords = _runtime_batch_word_count(runtime)
    length(bits) >= nwords ||
        throw(DimensionMismatch("bit vector has length $(length(bits)); expected at least $nwords"))
    @inbounds for word in 1:nwords
        mask = _batch_live_word_mask(runtime, word)
        ((runtime.value_words[word, condition] ⊻ bits[word]) & mask) == 0 ||
            return false
    end
    return true
end

function _copy_bits_to_batch_symbol_unchecked!(
    runtime::BatchFactoredExecutorState,
    condition::Int,
    bits::Vector{UInt64},
)
    nwords = _runtime_batch_word_count(runtime)
    length(bits) >= nwords ||
        throw(DimensionMismatch("bit vector has length $(length(bits)); expected at least $nwords"))
    @inbounds begin
        for word in 1:size(runtime.value_words, 1)
            runtime.value_words[word, condition] =
                word <= nwords ? bits[word] & _batch_live_word_mask(runtime, word) : UInt64(0)
        end
        for shot in 1:runtime.batches
            runtime.values[shot, condition] =
                shot <= runtime.active_shots ? _batch_bit(bits, shot) : false
        end
    end
    return runtime
end

function _eval_symbolic_bool!(
    out::Vector{UInt64},
    plan::SymbolicBoolEvaluationPlan,
    runtime::BatchFactoredExecutorState,
)
    _fill_batch_bits!(out, runtime, plan.constant)
    isempty(plan.conditions) && return out
    _check_symbolic_bool_assigned!(runtime, plan)
    _xor_symbolic_conditions_unchecked!(out, runtime, plan.conditions)
    _mask_batch_bits!(out, runtime)
    return out
end

function _check_symbolic_bool_assigned!(
    runtime::BatchFactoredExecutorState,
    plan::SymbolicBoolEvaluationPlan,
)
    if !isempty(plan.word_indices)
        @inbounds max_word = plan.word_indices[end]
        max_word <= length(runtime.assigned_words) ||
            throw(ArgumentError("symbolic condition expression has no concrete value"))
        assigned_words = runtime.assigned_words
        missing = UInt64(0)
        @inbounds for idx in eachindex(plan.word_indices, plan.word_masks)
            word = plan.word_indices[idx]
            mask = plan.word_masks[idx]
            missing |= mask & ~assigned_words[word]
        end
        missing == UInt64(0) ||
            throw(ArgumentError("symbolic condition expression has no concrete value"))
        return nothing
    end
    for condition in plan.conditions
        _symbol_assigned(runtime, condition) ||
            throw(ArgumentError("symbolic condition s$condition has no concrete value"))
    end
    return nothing
end

function _xor_symbolic_conditions_unchecked!(
    out::Vector{UInt64},
    runtime::BatchFactoredExecutorState,
    conditions::Vector{Int},
)
    nwords = _runtime_batch_word_count(runtime)
    if nwords == 1
        @inbounds for condition in conditions
            out[1] ⊻= runtime.value_words[1, condition]
        end
        return out
    end
    @inbounds for condition in conditions
        for word in 1:nwords
            out[word] ⊻= runtime.value_words[word, condition]
        end
    end
    return out
end

function _mask_batch_bits!(bits::Vector{UInt64}, runtime::BatchFactoredExecutorState)
    nwords = _runtime_batch_word_count(runtime)
    @inbounds for word in 1:nwords
        bits[word] &= _batch_live_word_mask(runtime, word)
    end
    return bits
end

function _write_measurement_record!(
    runtime::BatchFactoredExecutorState,
    record::Nothing,
    outcome_bits::Vector{UInt64},
    record_condition::Union{Nothing,Int},
)
    _assign_symbol!(runtime, record_condition, outcome_bits)
    return runtime
end

function _write_measurement_record!(
    runtime::BatchFactoredExecutorState,
    record::Integer,
    outcome_bits::Vector{UInt64},
    record_condition::Union{Nothing,Int},
)
    ri = Int(record)
    ri > 0 || throw(ArgumentError("measurement record id must be positive"))
    _ensure_batch_measurement_storage!(runtime, ri)
    nwords = _runtime_batch_word_count(runtime)
    @inbounds for word in 1:size(runtime.measurements, 1)
        runtime.measurements[word, ri] =
            word <= nwords ? outcome_bits[word] & _batch_live_word_mask(runtime, word) : UInt64(0)
    end
    _assign_symbol!(runtime, record_condition, outcome_bits)
    return runtime
end

function _copy_batch_measurements_to!(
    out::AbstractMatrix{Bool},
    runtime::BatchFactoredExecutorState,
    offset::Int,
)
    nrecords = size(out, 2)
    @inbounds for record in 1:nrecords
        for shot in 1:runtime.active_shots
            out[offset + shot, record] =
                (runtime.measurements[_batch_word_index(shot), record] & _batch_bit_mask(shot)) != 0
        end
    end
    return out
end

function _check_batch_active_storage(runtime::BatchFactoredExecutorState)
    dim = _active_length(runtime.k)
    size(runtime.active, 2) >= dim ||
        throw(DimensionMismatch("batch active storage has too few columns for k=$(runtime.k)"))
    size(runtime.active_scratch, 2) >= dim ||
        throw(DimensionMismatch("batch active scratch has too few columns for k=$(runtime.k)"))
    return dim
end

function _copy_batch_active_scratch!(runtime::BatchFactoredExecutorState, dim::Int)
    leading = size(runtime.active, 1)
    @inbounds for basis in 1:dim
        copyto!(
            runtime.active,
            1 + (basis - 1) * leading,
            runtime.active_scratch,
            1 + (basis - 1) * leading,
            runtime.active_shots,
        )
    end
    return runtime
end

function _batch_apply_active_basis_change!(
    runtime::BatchFactoredExecutorState,
    kind::Symbol,
    q::Int,
)
    0 <= q < runtime.k || throw(ArgumentError("active basis-change qubit is out of range"))
    dim = _check_batch_active_storage(runtime)
    mask = 1 << q
    if kind === :H
        invsqrt2 = inv(sqrt(2.0))
        @inbounds for base in 0:(dim - 1)
            (base & mask) == 0 || continue
            i0 = base + 1
            i1 = (base | mask) + 1
            @simd for shot in 1:runtime.active_shots
                a0 = runtime.active[shot, i0]
                a1 = runtime.active[shot, i1]
                runtime.active[shot, i0] = (a0 + a1) * invsqrt2
                runtime.active[shot, i1] = (a0 - a1) * invsqrt2
            end
        end
    elseif kind === :S
        @inbounds for basis in 0:(dim - 1)
            (basis & mask) != 0 || continue
            idx = basis + 1
            @simd for shot in 1:runtime.active_shots
                runtime.active[shot, idx] *= 1.0im
            end
        end
    else
        throw(ArgumentError("unsupported active basis change $kind"))
    end
    return runtime
end

function _batch_rotate_pauli!(
    runtime::BatchFactoredExecutorState,
    action::ActivePauliAction,
    theta::Float64,
    sign_bits::Vector{UInt64},
)
    _check_active_action(runtime.k, action)
    dim = _check_batch_active_storage(runtime)
    c = cos(theta)
    minus_i_s = ComplexF64(0.0, -sin(theta))
    plus_i_s = ComplexF64(0.0, sin(theta))
    xmask = action.xmask
    zmask = action.zmask
    even_phase = action.even_phase
    odd_phase = action.odd_phase

    if xmask == 0
        @inbounds for basis in 0:(dim - 1)
            phase = isodd(count_ones(basis & zmask)) ? odd_phase : even_phase
            idx = basis + 1
            @simd for shot in 1:runtime.active_shots
                sin_coeff = _batch_bit(sign_bits, shot) ? plus_i_s : minus_i_s
                runtime.active[shot, idx] *= c + sin_coeff * phase
            end
        end
        return runtime
    end

    pair_bit = trailing_zeros(xmask)
    npairs = dim >>> 1
    if zmask == 0
        @inbounds for packed in 0:(npairs - 1)
            base = _insert_zero_bit(packed, pair_bit)
            paired = xor(base, xmask)
            i0 = base + 1
            i1 = paired + 1
            @simd for shot in 1:runtime.active_shots
                coefficient = (_batch_bit(sign_bits, shot) ? plus_i_s : minus_i_s) * even_phase
                a0 = runtime.active[shot, i0]
                a1 = runtime.active[shot, i1]
                runtime.active[shot, i0] = c * a0 + coefficient * a1
                runtime.active[shot, i1] = c * a1 + coefficient * a0
            end
        end
        return runtime
    end

    @inbounds for packed in 0:(npairs - 1)
        base = _insert_zero_bit(packed, pair_bit)
        paired = xor(base, xmask)
        i0 = base + 1
        i1 = paired + 1
        parity0 = isodd(count_ones(base & zmask))
        phase0 = parity0 ? odd_phase : even_phase
        phase1 = xor(parity0, action.xz_overlap_odd) ? odd_phase : even_phase
        @simd for shot in 1:runtime.active_shots
            sin_coeff = _batch_bit(sign_bits, shot) ? plus_i_s : minus_i_s
            a0 = runtime.active[shot, i0]
            a1 = runtime.active[shot, i1]
            runtime.active[shot, i0] = c * a0 + sin_coeff * phase1 * a1
            runtime.active[shot, i1] = c * a1 + sin_coeff * phase0 * a0
        end
    end
    return runtime
end

function _batch_rotate_uniform_phase_pairs!(
    runtime::BatchFactoredExecutorState,
    kernel::PrecomputedActivePauliRotationKernel,
    sign_bits::Vector{UInt64},
)
    c = kernel.cos_theta
    minus_coeff_im = imag(kernel.pair_left_minus_coefficients[1])
    plus_coeff_im = imag(kernel.pair_left_plus_coefficients[1])
    coefficients = _fill_batch_rotation_coefficients!(
        runtime.rotation_coefficients,
        runtime,
        sign_bits,
        minus_coeff_im,
        plus_coeff_im,
    )

    if runtime.active_shots >= _BATCH_AVX2_MIN_SHOTS &&
            _batch_rotate_uniform_phase_struct_turbo!(runtime, kernel, coefficients)
        return runtime
    end

    if runtime.active_shots >= _BATCH_AVX2_MIN_SHOTS &&
            _batch_rotate_uniform_phase_xmask_avx2!(runtime, kernel, coefficients)
        return runtime
    end

    @inbounds for idx in eachindex(kernel.pair_left_indices)
        i0 = kernel.pair_left_indices[idx]
        i1 = kernel.pair_right_indices[idx]
        @simd for shot in 1:runtime.active_shots
            coeff_im = coefficients[shot]
            a0 = runtime.active[shot, i0]
            a1 = runtime.active[shot, i1]
            runtime.active[shot, i0] = ComplexF64(
                muladd(c, real(a0), -coeff_im * imag(a1)),
                muladd(c, imag(a0), coeff_im * real(a1)),
            )
            runtime.active[shot, i1] = ComplexF64(
                muladd(c, real(a1), -coeff_im * imag(a0)),
                muladd(c, imag(a1), coeff_im * real(a0)),
            )
        end
    end
    return runtime
end

function _fill_batch_rotation_coefficients!(
    coefficients::Vector{Float64},
    runtime::BatchFactoredExecutorState,
    sign_bits::Vector{UInt64},
    minus_coeff::Float64,
    plus_coeff::Float64,
)
    length(coefficients) >= runtime.active_shots ||
        throw(DimensionMismatch("rotation coefficient scratch is too short for the active batch"))
    nwords = _runtime_batch_word_count(runtime)
    length(sign_bits) >= nwords ||
        throw(DimensionMismatch("sign bit vector has length $(length(sign_bits)); expected at least $nwords"))
    @inbounds for word in 1:nwords
        bits = sign_bits[word]
        base_shot = (word - 1) << 6
        live = min(64, runtime.active_shots - base_shot)
        @simd for bit in 0:(live - 1)
            shot = base_shot + bit + 1
            coefficients[shot] = ((bits >>> bit) & UInt64(1)) == UInt64(1) ? plus_coeff : minus_coeff
        end
    end
    return coefficients
end

function _batch_rotate_uniform_phase_xmask_avx2!(
    runtime::BatchFactoredExecutorState,
    kernel::PrecomputedActivePauliRotationKernel,
    coefficients::Vector{Float64},
)::Bool
    runtime.kernel_backend === :c || return false
    runtime.active isa Matrix{ComplexF64} || return false
    symbol = _batch_avx2_uniform_xmask_symbol()
    symbol == C_NULL && return false
    xmask = kernel.action.xmask
    ccall(
        symbol,
        Cvoid,
        (Ptr{Cdouble}, Int64, Int64, UInt64, UInt32, Int64, Cdouble, Ptr{Cdouble}),
        runtime.active,
        size(runtime.active, 1),
        runtime.active_shots,
        UInt64(xmask),
        UInt32(trailing_zeros(xmask)),
        length(kernel.pair_left_indices),
        kernel.cos_theta,
        coefficients,
    )
    return true
end

function _batch_rotate_pauli!(
    runtime::BatchFactoredExecutorState,
    kernel::PrecomputedActivePauliRotationKernel,
    sign_bits::Vector{UInt64},
)
    _check_active_rotation_kernel(runtime.k, kernel)
    _check_batch_active_storage(runtime)
    c = kernel.cos_theta

    if kernel.is_diagonal
        @inbounds for idx in eachindex(kernel.diagonal_plus_coefficients)
            plus_coefficient = kernel.diagonal_plus_coefficients[idx]
            minus_coefficient = kernel.diagonal_minus_coefficients[idx]
            @simd for shot in 1:runtime.active_shots
                coefficient = _batch_bit(sign_bits, shot) ? plus_coefficient : minus_coefficient
                runtime.active[shot, idx] *= c + coefficient
            end
        end
        return runtime
    end

    if kernel.action.zmask == 0
        return _batch_rotate_uniform_phase_pairs!(runtime, kernel, sign_bits)
    end

    if _can_rotate_real_pair_flip(kernel)
        return _batch_rotate_real_pair_flip!(runtime, kernel, sign_bits)
    end

    @inbounds for idx in eachindex(kernel.pair_left_indices)
        i0 = kernel.pair_left_indices[idx]
        i1 = kernel.pair_right_indices[idx]
        left_plus = kernel.pair_left_plus_coefficients[idx]
        right_plus = kernel.pair_right_plus_coefficients[idx]
        left_minus = kernel.pair_left_minus_coefficients[idx]
        right_minus = kernel.pair_right_minus_coefficients[idx]
        @simd for shot in 1:runtime.active_shots
            if _batch_bit(sign_bits, shot)
                left_coefficient = left_plus
                right_coefficient = right_plus
            else
                left_coefficient = left_minus
                right_coefficient = right_minus
            end
            a0 = runtime.active[shot, i0]
            a1 = runtime.active[shot, i1]
            runtime.active[shot, i0] = c * a0 + right_coefficient * a1
            runtime.active[shot, i1] = c * a1 + left_coefficient * a0
        end
    end
    return runtime
end

function _batch_rotate_real_pair_flip!(
    runtime::BatchFactoredExecutorState,
    kernel::PrecomputedActivePauliRotationKernel,
    sign_bits::Vector{UInt64},
)
    c = kernel.cos_theta
    coefficients = _fill_batch_rotation_coefficients!(
        runtime.rotation_coefficients,
        runtime,
        sign_bits,
        real(kernel.pair_left_minus_coefficients[1]),
        real(kernel.pair_left_plus_coefficients[1]),
    )
    zmask = kernel.action.zmask
    if runtime.active_shots >= _BATCH_AVX2_MIN_SHOTS &&
            _batch_rotate_real_pair_flip_struct_turbo!(runtime, kernel, coefficients, zmask)
        return runtime
    end

    if runtime.active_shots >= _BATCH_AVX2_MIN_SHOTS && (
            _batch_rotate_real_pair_flip_xmask_avx2!(runtime, kernel, coefficients, zmask) ||
            _batch_rotate_real_pair_flip_avx2!(runtime, kernel, coefficients, zmask)
        )
        return runtime
    end

    @inbounds for idx in eachindex(kernel.pair_left_indices)
        i0 = kernel.pair_left_indices[idx]
        i1 = kernel.pair_right_indices[idx]
        phase_sign = isodd(count_ones((i0 - 1) & zmask)) ? -1.0 : 1.0
        @simd for shot in 1:runtime.active_shots
            coeff = phase_sign * coefficients[shot]
            a0 = runtime.active[shot, i0]
            a1 = runtime.active[shot, i1]
            runtime.active[shot, i0] = ComplexF64(
                muladd(c, real(a0), -coeff * real(a1)),
                muladd(c, imag(a0), -coeff * imag(a1)),
            )
            runtime.active[shot, i1] = ComplexF64(
                muladd(c, real(a1), coeff * real(a0)),
                muladd(c, imag(a1), coeff * imag(a0)),
            )
        end
    end
    return runtime
end

function _batch_rotate_real_pair_flip_avx2!(
    runtime::BatchFactoredExecutorState,
    kernel::PrecomputedActivePauliRotationKernel,
    coefficients::Vector{Float64},
    zmask::Int,
)::Bool
    runtime.kernel_backend === :c || return false
    runtime.active isa Matrix{ComplexF64} || return false
    symbol = _batch_avx2_real_pair_flip_symbol()
    symbol == C_NULL && return false
    ccall(
        symbol,
        Cvoid,
        (Ptr{Cdouble}, Int64, Int64, Ptr{Int64}, Ptr{Int64}, Int64, Cdouble, Ptr{Cdouble}, UInt64),
        runtime.active,
        size(runtime.active, 1),
        runtime.active_shots,
        kernel.pair_left_indices,
        kernel.pair_right_indices,
        length(kernel.pair_left_indices),
        kernel.cos_theta,
        coefficients,
        UInt64(zmask),
    )
    return true
end

function _batch_rotate_real_pair_flip_xmask_avx2!(
    runtime::BatchFactoredExecutorState,
    kernel::PrecomputedActivePauliRotationKernel,
    coefficients::Vector{Float64},
    zmask::Int,
)::Bool
    runtime.kernel_backend === :c || return false
    runtime.active isa Matrix{ComplexF64} || return false
    symbol = _batch_avx2_real_pair_flip_xmask_symbol()
    symbol == C_NULL && return false
    xmask = kernel.action.xmask
    ccall(
        symbol,
        Cvoid,
        (Ptr{Cdouble}, Int64, Int64, UInt64, UInt32, Int64, Cdouble, Ptr{Cdouble}, UInt64),
        runtime.active,
        size(runtime.active, 1),
        runtime.active_shots,
        UInt64(xmask),
        UInt32(trailing_zeros(xmask)),
        length(kernel.pair_left_indices),
        kernel.cos_theta,
        coefficients,
        UInt64(zmask),
    )
    return true
end

function _batch_promote_first_dormant_rotation!(
    runtime::BatchFactoredExecutorState,
    theta::Float64,
    sign_bits::Vector{UInt64},
)
    runtime.ndormant > 0 ||
        throw(ArgumentError("cannot promote a dormant qubit when none remain"))
    dim = _check_batch_active_storage(runtime)
    promoted_dim = 2 * dim
    size(runtime.active, 2) >= promoted_dim ||
        throw(DimensionMismatch("batch active storage has too few columns for dormant promotion"))

    c = cos(theta)
    s = sin(theta)
    coefficients = _fill_batch_rotation_coefficients!(
        runtime.rotation_coefficients,
        runtime,
        sign_bits,
        -s,
        s,
    )
    if runtime.active_shots >= _BATCH_AVX2_MIN_SHOTS &&
            _batch_promote_first_dormant_rotation_struct_turbo!(runtime, dim, c, coefficients)
        runtime.k += 1
        runtime.ndormant -= 1
        return runtime
    end

    if runtime.active_shots >= _BATCH_AVX2_MIN_SHOTS &&
            _batch_promote_first_dormant_rotation_avx2!(runtime, dim, c, coefficients)
        runtime.k += 1
        runtime.ndormant -= 1
        return runtime
    end

    @inbounds for basis in 1:dim
        promoted_basis = dim + basis
        @simd for shot in 1:runtime.active_shots
            coeff = coefficients[shot]
            amp = runtime.active[shot, basis]
            runtime.active[shot, basis] = c * amp
            runtime.active[shot, promoted_basis] = ComplexF64(-coeff * imag(amp), coeff * real(amp))
        end
    end
    runtime.k += 1
    runtime.ndormant -= 1
    return runtime
end

function _batch_promote_first_dormant_rotation_avx2!(
    runtime::BatchFactoredExecutorState,
    dim::Int,
    c::Float64,
    coefficients::Vector{Float64},
)::Bool
    runtime.kernel_backend === :c || return false
    runtime.active isa Matrix{ComplexF64} || return false
    symbol = _batch_avx2_promote_symbol()
    symbol == C_NULL && return false
    ccall(
        symbol,
        Cvoid,
        (Ptr{Cdouble}, Int64, Int64, Int64, Cdouble, Ptr{Cdouble}),
        runtime.active,
        size(runtime.active, 1),
        runtime.active_shots,
        dim,
        c,
        coefficients,
    )
    return true
end
