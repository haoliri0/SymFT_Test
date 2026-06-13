#!/usr/bin/env julia

using Libdl
using Random
using SymFT

function _arg_int(idx::Int, env::AbstractString, default::Int)
    raw = length(ARGS) >= idx ? ARGS[idx] : get(ENV, env, string(default))
    value = parse(Int, raw)
    value > 0 || throw(ArgumentError("$env must be positive"))
    return value
end

function _pauli_from_masks(k::Int, xmask::Int, zmask::Int)
    p = SymFT.PauliString(k)
    y_count = 0
    for q in 0:(k - 1)
        xb = ((xmask >>> q) & 1) != 0
        zb = ((zmask >>> q) & 1) != 0
        xb && SymFT._set_xbit!(p, q)
        zb && SymFT._set_zbit!(p, q)
        xb && zb && (y_count += 1)
    end
    SymFT._set_phase!(p, y_count)
    return p
end

function _random_active_state(k::Int, rng::AbstractRNG)
    dim = SymFT._active_length(k)
    alpha = randn(rng, ComplexF64, dim)
    alpha ./= sqrt(sum(abs2, alpha))
    return SymFT.ActiveState(k, alpha)
end

function _build_c_pair_kernel()
    source = joinpath(@__DIR__, "active_rotation_pair_avx2.c")
    lib = joinpath(tempdir(), "libsymft_active_rotation_pair_avx2.so")
    if !isfile(lib) || stat(lib).mtime < stat(source).mtime
        run(`gcc -O3 -mavx2 -mfma -fPIC -shared -o $lib $source`)
    end
    return Libdl.dlopen(lib)
end

function _local_axis_phase_column!(
    active::Matrix{ComplexF64},
    k::Int,
    axis::Int,
    z::ComplexF64,
    shot::Int,
)
    dim = SymFT._active_length(k)
    step = 1 << axis
    period = step << 1
    @inbounds for base in 1:period:dim
        last = base + period - 1
        @simd for idx in (base + step):last
            active[shot, idx] *= z
        end
    end
    return active
end

function _local_axis_phase_vector!(
    alpha::Vector{ComplexF64},
    k::Int,
    axis::Int,
    z::ComplexF64,
)
    dim = SymFT._active_length(k)
    step = 1 << axis
    period = step << 1
    @inbounds for base in 1:period:dim
        last = base + period - 1
        @simd for idx in (base + step):last
            alpha[idx] *= z
        end
    end
    return alpha
end

function _apply_h_vector!(alpha::Vector{ComplexF64}, k::Int, axis::Int)
    mask = 1 << axis
    invsqrt2 = inv(sqrt(2.0))
    dim = SymFT._active_length(k)
    @inbounds for base in 0:(dim - 1)
        (base & mask) == 0 || continue
        i0 = base + 1
        i1 = (base | mask) + 1
        a0 = alpha[i0]
        a1 = alpha[i1]
        alpha[i0] = (a0 + a1) * invsqrt2
        alpha[i1] = (a0 - a1) * invsqrt2
    end
    return alpha
end

function _apply_s_vector!(alpha::Vector{ComplexF64}, k::Int, axis::Int)
    mask = 1 << axis
    dim = SymFT._active_length(k)
    @inbounds @simd for basis in 0:(dim - 1)
        (basis & mask) != 0 && (alpha[basis + 1] *= 1.0im)
    end
    return alpha
end

function _apply_cx_vector!(alpha::Vector{ComplexF64}, k::Int, control::Int, target::Int)
    cmask = 1 << control
    tmask = 1 << target
    dim = SymFT._active_length(k)
    @inbounds for basis in 0:(dim - 1)
        ((basis & cmask) != 0 && (basis & tmask) == 0) || continue
        i0 = basis + 1
        i1 = xor(basis, tmask) + 1
        alpha[i0], alpha[i1] = alpha[i1], alpha[i0]
    end
    return alpha
end

function _apply_cz_vector!(alpha::Vector{ComplexF64}, k::Int, a::Int, b::Int)
    amask = 1 << a
    bmask = 1 << b
    dim = SymFT._active_length(k)
    @inbounds @simd for basis in 0:(dim - 1)
        ((basis & amask) != 0 && (basis & bmask) != 0) && (alpha[basis + 1] = -alpha[basis + 1])
    end
    return alpha
end

function _apply_h_column!(active::Matrix{ComplexF64}, k::Int, axis::Int, shot::Int)
    mask = 1 << axis
    invsqrt2 = inv(sqrt(2.0))
    dim = SymFT._active_length(k)
    @inbounds for base in 0:(dim - 1)
        (base & mask) == 0 || continue
        i0 = base + 1
        i1 = (base | mask) + 1
        a0 = active[shot, i0]
        a1 = active[shot, i1]
        active[shot, i0] = (a0 + a1) * invsqrt2
        active[shot, i1] = (a0 - a1) * invsqrt2
    end
    return active
end

function _apply_s_column!(active::Matrix{ComplexF64}, k::Int, axis::Int, shot::Int)
    mask = 1 << axis
    dim = SymFT._active_length(k)
    @inbounds @simd for basis in 0:(dim - 1)
        (basis & mask) != 0 && (active[shot, basis + 1] *= 1.0im)
    end
    return active
end

function _apply_cx_column!(active::Matrix{ComplexF64}, k::Int, control::Int, target::Int, shot::Int)
    cmask = 1 << control
    tmask = 1 << target
    dim = SymFT._active_length(k)
    @inbounds for basis in 0:(dim - 1)
        ((basis & cmask) != 0 && (basis & tmask) == 0) || continue
        i0 = basis + 1
        i1 = xor(basis, tmask) + 1
        active[shot, i0], active[shot, i1] = active[shot, i1], active[shot, i0]
    end
    return active
end

function _apply_cz_column!(active::Matrix{ComplexF64}, k::Int, a::Int, b::Int, shot::Int)
    amask = 1 << a
    bmask = 1 << b
    dim = SymFT._active_length(k)
    @inbounds @simd for basis in 0:(dim - 1)
        ((basis & amask) != 0 && (basis & bmask) != 0) &&
            (active[shot, basis + 1] = -active[shot, basis + 1])
    end
    return active
end

function _mask_support(mask::Int, k::Int)
    return Int[q for q in 0:(k - 1) if ((mask >>> q) & 1) != 0]
end

function _basis_change_proxy_gates(k::Int, xmask::Int, zmask::Int, pivot::Int)
    current = _pauli_from_masks(k, xmask, zmask)
    gates = Tuple[]
    if xmask != 0
        SymFT.xbit(current, pivot) ||
            throw(ArgumentError("X basis-change pivot $pivot is not in X support"))
        for q in _mask_support(xmask, k)
            q == pivot && continue
            push!(gates, (:CX, pivot, q))
            current = SymFT._forward_conjugate_CX(current, pivot, q)
        end
        for q in 0:(k - 1)
            q == pivot && continue
            SymFT.zbit(current, q) || continue
            push!(gates, (:CZ, pivot, q))
            current = SymFT._forward_conjugate_CZ(current, pivot, q)
        end
        if SymFT.zbit(current, pivot)
            push!(gates, (:S, pivot))
            current = SymFT._forward_conjugate_S(current, pivot)
        end
        push!(gates, (:H, pivot))
        current = SymFT._forward_conjugate_H(current, pivot)
    else
        SymFT.zbit(current, pivot) ||
            throw(ArgumentError("Z basis-change pivot $pivot is not in Z support"))
        for q in _mask_support(zmask, k)
            q == pivot && continue
            push!(gates, (:CX, q, pivot))
            current = SymFT._forward_conjugate_CX(current, q, pivot)
        end
    end
    SymFT.xbit(current, pivot) &&
        throw(AssertionError("basis-change proxy did not produce a Z-axis Pauli"))
    SymFT.zbit(current, pivot) ||
        throw(AssertionError("basis-change proxy lost the selected Pauli"))
    return gates
end

function _apply_basis_change_proxy_gates_vector!(alpha::Vector{ComplexF64}, k::Int, gates)
    for gate in gates
        if gate[1] === :H
            _apply_h_vector!(alpha, k, gate[2])
        elseif gate[1] === :S
            _apply_s_vector!(alpha, k, gate[2])
        elseif gate[1] === :CX
            _apply_cx_vector!(alpha, k, gate[2], gate[3])
        elseif gate[1] === :CZ
            _apply_cz_vector!(alpha, k, gate[2], gate[3])
        else
            throw(ArgumentError("unsupported basis-change proxy gate $(gate[1])"))
        end
    end
    return alpha
end

function _apply_basis_change_proxy_gates_column!(active::Matrix{ComplexF64}, k::Int, gates, shot::Int)
    for gate in gates
        if gate[1] === :H
            _apply_h_column!(active, k, gate[2], shot)
        elseif gate[1] === :S
            _apply_s_column!(active, k, gate[2], shot)
        elseif gate[1] === :CX
            _apply_cx_column!(active, k, gate[2], gate[3], shot)
        elseif gate[1] === :CZ
            _apply_cz_column!(active, k, gate[2], gate[3], shot)
        else
            throw(ArgumentError("unsupported basis-change proxy gate $(gate[1])"))
        end
    end
    return active
end

function _bench_single_pair!(state::SymFT.ActiveState, kernel, reps::Int)
    t = time_ns()
    for _ in 1:reps
        SymFT.rotate_pauli!(state, kernel, false)
    end
    return time_ns() - t
end

function _bench_single_c_uniform_imag_pairs!(
    alpha::Vector{ComplexF64},
    kernel,
    handle,
    symbol::Symbol,
    reps::Int,
)
    ptr = Libdl.dlsym(handle, symbol)
    left = kernel.pair_left_indices
    right = kernel.pair_right_indices
    npairs = length(left)
    coeff_im = imag(kernel.pair_left_minus_coefficients[1])
    t = time_ns()
    for _ in 1:reps
        ccall(
            ptr,
            Cvoid,
            (Ptr{Cdouble}, Ptr{Int64}, Ptr{Int64}, Int64, Cdouble, Cdouble),
            alpha,
            left,
            right,
            npairs,
            kernel.cos_theta,
            coeff_im,
        )
    end
    return time_ns() - t
end

function _bench_single_c_uniform_imag_xmask!(
    alpha::Vector{ComplexF64},
    kernel,
    handle,
    reps::Int,
)
    ptr = Libdl.dlsym(handle, :symft_rotate_uniform_imag_xmask_scalar)
    coeff_im = imag(kernel.pair_left_minus_coefficients[1])
    xmask = UInt64(kernel.action.xmask)
    pair_bit = UInt32(trailing_zeros(kernel.action.xmask))
    k = UInt32(kernel.action.nqubits)
    t = time_ns()
    for _ in 1:reps
        ccall(
            ptr,
            Cvoid,
            (Ptr{Cdouble}, UInt32, UInt64, UInt32, Cdouble, Cdouble),
            alpha,
            k,
            xmask,
            pair_bit,
            kernel.cos_theta,
            coeff_im,
        )
    end
    return time_ns() - t
end

function _bench_single_local_axis!(alpha::Vector{ComplexF64}, k::Int, axis::Int, z::ComplexF64, reps::Int)
    t = time_ns()
    for _ in 1:reps
        _local_axis_phase_vector!(alpha, k, axis, z)
    end
    return time_ns() - t
end

function _bench_single_basis_change_phase!(
    alpha::Vector{ComplexF64},
    k::Int,
    gates,
    pivot::Int,
    z::ComplexF64,
    reps::Int,
)
    t = time_ns()
    for _ in 1:reps
        _apply_basis_change_proxy_gates_vector!(alpha, k, gates)
        _local_axis_phase_vector!(alpha, k, pivot, z)
    end
    return time_ns() - t
end

function _bench_batch_pair!(runtime::SymFT.BatchFactoredExecutorState, kernel, sign_bits::Vector{UInt64}, reps::Int)
    t = time_ns()
    for _ in 1:reps
        SymFT._batch_rotate_pauli!(runtime, kernel, sign_bits)
    end
    return time_ns() - t
end

function _batch_sign_coefficients(kernel, sign_bits::Vector{UInt64}, shots::Int)
    minus_coeff_im = imag(kernel.pair_left_minus_coefficients[1])
    plus_coeff_im = imag(kernel.pair_left_plus_coefficients[1])
    coeffs = Vector{Float64}(undef, shots)
    nwords = cld(shots, 64)
    @inbounds for word in 1:nwords
        bits = sign_bits[word]
        base_shot = (word - 1) << 6
        live = min(64, shots - base_shot)
        @simd for bit in 0:(live - 1)
            shot = base_shot + bit + 1
            coeffs[shot] = ((bits >>> bit) & UInt64(1)) == UInt64(1) ? plus_coeff_im : minus_coeff_im
        end
    end
    return coeffs
end

function _bench_batch_shot_major_uniform_imag!(
    active::Matrix{ComplexF64},
    kernel,
    sign_coefficients::Vector{Float64},
    reps::Int,
)
    c = kernel.cos_theta
    shots = size(active, 1)
    t = time_ns()
    for _ in 1:reps
        @inbounds for idx in eachindex(kernel.pair_left_indices)
            i0 = kernel.pair_left_indices[idx]
            i1 = kernel.pair_right_indices[idx]
            @simd for shot in 1:shots
                coeff_im = sign_coefficients[shot]
                a0 = active[shot, i0]
                a1 = active[shot, i1]
                active[shot, i0] = ComplexF64(
                    c * real(a0) - coeff_im * imag(a1),
                    c * imag(a0) + coeff_im * real(a1),
                )
                active[shot, i1] = ComplexF64(
                    c * real(a1) - coeff_im * imag(a0),
                    c * imag(a1) + coeff_im * real(a0),
                )
            end
        end
    end
    return time_ns() - t
end

function _bench_batch_c_shot_major_uniform_imag!(
    active::Matrix{ComplexF64},
    kernel,
    sign_coefficients::Vector{Float64},
    handle,
    reps::Int,
)
    ptr = Libdl.dlsym(handle, :symft_batch_rotate_uniform_imag_pairs_colmajor_avx2)
    left = kernel.pair_left_indices
    right = kernel.pair_right_indices
    npairs = length(left)
    leading_shots = size(active, 1)
    active_shots = length(sign_coefficients)
    t = time_ns()
    for _ in 1:reps
        ccall(
            ptr,
            Cvoid,
            (Ptr{Cdouble}, Int64, Int64, Ptr{Int64}, Ptr{Int64}, Int64, Cdouble, Ptr{Cdouble}),
            active,
            leading_shots,
            active_shots,
            left,
            right,
            npairs,
            kernel.cos_theta,
            sign_coefficients,
        )
    end
    return time_ns() - t
end

function _bench_batch_basis_change_phase!(
    active::Matrix{ComplexF64},
    k::Int,
    gates,
    pivot::Int,
    z::ComplexF64,
    shots::Int,
    reps::Int,
)
    t = time_ns()
    for _ in 1:reps
        @inbounds for shot in 1:shots
            _apply_basis_change_proxy_gates_column!(active, k, gates, shot)
            _local_axis_phase_column!(active, k, pivot, z, shot)
        end
    end
    return time_ns() - t
end

function _bench_batch_local_axis!(
    active::Matrix{ComplexF64},
    k::Int,
    axis::Int,
    z::ComplexF64,
    shots::Int,
    reps::Int,
)
    t = time_ns()
    for _ in 1:reps
        @inbounds for shot in 1:shots
            _local_axis_phase_column!(active, k, axis, z, shot)
        end
    end
    return time_ns() - t
end

function _format_row(name, scope, locator, elapsed_ns, reps, states, active_dim, touched_dim)
    ns_per_state = Float64(elapsed_ns) / (reps * states)
    ns_per_active_amp = ns_per_state / active_dim
    ns_per_touched_amp = ns_per_state / touched_dim
    println(join((
        rpad(name, 28),
        rpad(scope, 8),
        rpad(locator, 16),
        lpad(round(elapsed_ns / 1e9; digits = 6), 12),
        lpad(round(ns_per_state; digits = 2), 14),
        lpad(round(ns_per_active_amp; digits = 4), 15),
        lpad(round(ns_per_touched_amp; digits = 4), 16),
    ), " "))
end

function main()
    k = _arg_int(1, "SYMFT_ROT_KERNEL_K", 10)
    single_reps = _arg_int(2, "SYMFT_ROT_KERNEL_SINGLE_REPS", 20000)
    batch_reps = _arg_int(3, "SYMFT_ROT_KERNEL_BATCH_REPS", 600)
    batch_size = _arg_int(4, "SYMFT_ROT_KERNEL_BATCH_SIZE", 512)
    xmask = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : parse(Int, get(ENV, "SYMFT_ROT_KERNEL_XMASK", "515"))
    zmask = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : parse(Int, get(ENV, "SYMFT_ROT_KERNEL_ZMASK", "0"))

    dim = SymFT._active_length(k)
    theta = pi / 8
    z = cis(2theta)
    rng = MersenneTwister(0x5f3759df)
    pauli = _pauli_from_masks(k, xmask, zmask)
    kernel = SymFT.PrecomputedActivePauliRotationKernel(SymFT.ActivePauliAction(pauli), theta)
    axes_to_test = collect(0:(k - 1))

    program = SymFT.FactoredInstructionProgram(
        k,
        k,
        _random_active_state(k, rng),
        SymFT.FactoredInstruction[],
        k,
        SymFT.SymbolicContext(),
    )
    runtime = SymFT.BatchFactoredExecutorState(program; batches = batch_size)
    runtime.active_shots = batch_size
    @inbounds for shot in 1:batch_size
        state = _random_active_state(k, rng)
        @simd for idx in 1:dim
            runtime.active[shot, idx] = state.alpha[idx]
        end
    end
    sign_bits = rand(rng, UInt64, cld(batch_size, 64))

    println("SymFT active rotation kernel benchmark")
    println("k=$k dim=$dim xmask=$xmask zmask=$zmask single_reps=$single_reps batch_reps=$batch_reps batch_size=$batch_size")
    println("local_axis_phase is a Clifft-style OP_ARRAY_ROT proxy: diag(1,z) on one active axis.")
    println("basis_change_phase is an explicit dense basis-change proxy before that local phase; it is not the SymFT runtime path.")
    println(join((
        rpad("kernel", 28),
        rpad("scope", 8),
        rpad("axis/mask", 16),
        lpad("elapsed_s", 12),
        lpad("ns/state", 14),
        lpad("ns/active_amp", 15),
        lpad("ns/touched_amp", 16),
    ), " "))

    state = _random_active_state(k, rng)
    _bench_single_pair!(state, kernel, max(1, single_reps ÷ 50))
    elapsed = _bench_single_pair!(state, kernel, single_reps)
    _format_row("symft_any_pauli_pair", "single", "x=$(xmask),z=$(zmask)", elapsed, single_reps, 1, dim, dim)

    if zmask == 0
        handle = _build_c_pair_kernel()
        for (name, symbol) in (
            ("c_scalar_pair", :symft_rotate_uniform_imag_pairs_scalar),
            ("c_avx2_pair", :symft_rotate_uniform_imag_pairs_avx2),
        )
            alpha = _random_active_state(k, rng).alpha
            _bench_single_c_uniform_imag_pairs!(alpha, kernel, handle, symbol, max(1, single_reps ÷ 50))
            elapsed = _bench_single_c_uniform_imag_pairs!(alpha, kernel, handle, symbol, single_reps)
            _format_row(name, "single", "x=$(xmask),z=0", elapsed, single_reps, 1, dim, dim)
        end
        alpha = _random_active_state(k, rng).alpha
        _bench_single_c_uniform_imag_xmask!(alpha, kernel, handle, max(1, single_reps ÷ 50))
        elapsed = _bench_single_c_uniform_imag_xmask!(alpha, kernel, handle, single_reps)
        _format_row("c_scalar_xmask", "single", "x=$(xmask),z=0", elapsed, single_reps, 1, dim, dim)
    end

    for axis in axes_to_test
        alpha = _random_active_state(k, rng).alpha
        _bench_single_local_axis!(alpha, k, axis, z, max(1, single_reps ÷ 50))
        elapsed = _bench_single_local_axis!(alpha, k, axis, z, single_reps)
        _format_row("clifft_local_axis_phase", "single", "axis=$axis", elapsed, single_reps, 1, dim, dim >>> 1)
    end

    support = _mask_support(xmask != 0 ? xmask : zmask, k)
    basis_change_pivots = unique(Int[first(support), last(support)])
    for pivot in basis_change_pivots
        gates = _basis_change_proxy_gates(k, xmask, zmask, pivot)
        alpha = _random_active_state(k, rng).alpha
        _bench_single_basis_change_phase!(alpha, k, gates, pivot, z, max(1, single_reps ÷ 50))
        elapsed = _bench_single_basis_change_phase!(alpha, k, gates, pivot, z, single_reps)
        _format_row(
            "clifft_basis_change_phase",
            "single",
            "pivot=$pivot,g=$(length(gates))",
            elapsed,
            single_reps,
            1,
            dim,
            dim,
        )
    end

    _bench_batch_pair!(runtime, kernel, sign_bits, max(1, batch_reps ÷ 20))
    elapsed = _bench_batch_pair!(runtime, kernel, sign_bits, batch_reps)
    _format_row("symft_any_pauli_pair", "batch", "x=$(xmask),z=$(zmask)", elapsed, batch_reps, batch_size, dim, dim)

    if zmask == 0
        shot_major = Matrix{ComplexF64}(undef, batch_size, dim)
        @inbounds for shot in 1:batch_size
            @simd for basis in 1:dim
                shot_major[shot, basis] = runtime.active[shot, basis]
            end
        end
        sign_coefficients = _batch_sign_coefficients(kernel, sign_bits, batch_size)
        _bench_batch_shot_major_uniform_imag!(
            shot_major,
            kernel,
            sign_coefficients,
            max(1, batch_reps ÷ 20),
        )
        elapsed = _bench_batch_shot_major_uniform_imag!(
            shot_major,
            kernel,
            sign_coefficients,
            batch_reps,
        )
        _format_row("shot_major_pair", "batch", "x=$(xmask),z=0", elapsed, batch_reps, batch_size, dim, dim)

        c_shot_major = copy(shot_major)
        _bench_batch_c_shot_major_uniform_imag!(
            c_shot_major,
            kernel,
            sign_coefficients,
            handle,
            max(1, batch_reps ÷ 20),
        )
        elapsed = _bench_batch_c_shot_major_uniform_imag!(
            c_shot_major,
            kernel,
            sign_coefficients,
            handle,
            batch_reps,
        )
        _format_row("c_avx2_shot_major_pair", "batch", "x=$(xmask),z=0", elapsed, batch_reps, batch_size, dim, dim)
    end

    for axis in axes_to_test
        active = Matrix(copy(runtime.active))
        _bench_batch_local_axis!(active, k, axis, z, batch_size, max(1, batch_reps ÷ 20))
        elapsed = _bench_batch_local_axis!(active, k, axis, z, batch_size, batch_reps)
        _format_row("clifft_local_axis_phase", "batch", "axis=$axis", elapsed, batch_reps, batch_size, dim, dim >>> 1)
    end

    for pivot in basis_change_pivots
        gates = _basis_change_proxy_gates(k, xmask, zmask, pivot)
        active = Matrix(copy(runtime.active))
        _bench_batch_basis_change_phase!(active, k, gates, pivot, z, batch_size, max(1, batch_reps ÷ 20))
        elapsed = _bench_batch_basis_change_phase!(active, k, gates, pivot, z, batch_size, batch_reps)
        _format_row(
            "clifft_basis_change_phase",
            "batch",
            "pivot=$pivot,g=$(length(gates))",
            elapsed,
            batch_reps,
            batch_size,
            dim,
            dim,
        )
    end
end

main()
