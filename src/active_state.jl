"""
    ActiveState(k, alpha)

Dense state for the active virtual subsystem only. The invariant is
`alpha[1:(1 << k)]` stores the active vector; no physical `2^n` vector is ever
stored here. Constructors enforce exact buffer lengths, while sampler runtimes
may reserve larger buffers and operate on the active prefix.
"""
mutable struct ActiveState
    k::Int
    alpha::Vector{ComplexF64}

    function ActiveState(k::Integer, alpha::Vector{ComplexF64})
        ki = _checked_active_k(k)
        dim = _active_length(ki)
        length(alpha) == dim ||
            throw(ArgumentError("alpha has length $(length(alpha)); expected 1 << $ki = $dim"))
        return new(ki, copy(alpha))
    end
end

ActiveState() = ActiveState(0, ComplexF64[1.0 + 0.0im])

function ActiveState(k::Integer)
    ki = _checked_active_k(k)
    dim = _active_length(ki)
    alpha = zeros(ComplexF64, dim)
    alpha[1] = 1.0 + 0.0im
    return ActiveState(ki, alpha)
end

function ActiveState(k::Integer, alpha::AbstractVector{<:Number})
    packed = collect(ComplexF64, alpha)
    return ActiveState(k, packed)
end

function _checked_active_k(k::Integer)::Int
    ki = Int(k)
    ki >= 0 || throw(ArgumentError("active qubit count must be nonnegative"))
    # Active basis indices are machine integers. This limit is about index
    # arithmetic, not about the packed Pauli representation, which can exceed
    # 64 qubits.
    ki < Sys.WORD_SIZE - 1 ||
        throw(ArgumentError("active qubit count $ki is too large for packed computational-basis indices"))
    return ki
end

@inline _active_length(k::Int)::Int = 1 << k
@inline _active_dim(state::ActiveState)::Int = _active_length(state.k)

function _check_active_storage(state::ActiveState)
    dim = _active_dim(state)
    length(state.alpha) >= dim ||
        throw(DimensionMismatch("alpha has length $(length(state.alpha)); expected at least $dim"))
    return dim
end

@inline function _phase_factor(phase::Int)::ComplexF64
    p = phase & 3
    p == 0 && return 1.0 + 0.0im
    p == 1 && return 0.0 + 1.0im
    p == 2 && return -1.0 + 0.0im
    return 0.0 - 1.0im
end

"""
    ActivePauliAction(pauli)

Cached one-word action of a Pauli string on an active computational basis.
Active states are intentionally limited to machine-indexable widths, so the
runtime can use integer masks instead of re-reading packed Pauli words on every
shot.
"""
struct ActivePauliAction
    nqubits::Int
    xmask::Int
    zmask::Int
    even_phase::ComplexF64
    odd_phase::ComplexF64
    xz_overlap_odd::Bool

    function ActivePauliAction(p::PauliString)
        p.nqubits < Sys.WORD_SIZE - 1 ||
            throw(ArgumentError("Pauli string has too many qubits for active-state basis indexing"))
        _pauli_squares_to_identity(p) ||
            throw(ArgumentError("Pauli rotation requires P^2 == I"))
        xmask, zmask = _active_masks(p)
        even_phase = _phase_factor(phase_exponent(p))
        return new(p.nqubits, xmask, zmask, even_phase, -even_phase, isodd(count_ones(xmask & zmask)))
    end

    function ActivePauliAction(
        nqubits::Integer,
        xmask::Integer,
        zmask::Integer,
        even_phase::ComplexF64,
        odd_phase::ComplexF64,
        xz_overlap_odd::Bool,
    )
        n = _checked_active_k(nqubits)
        return new(n, Int(xmask), Int(zmask), even_phase, odd_phase, xz_overlap_odd)
    end
end

Base.copy(action::ActivePauliAction) =
    ActivePauliAction(
        action.nqubits,
        action.xmask,
        action.zmask,
        action.even_phase,
        action.odd_phase,
        action.xz_overlap_odd,
    )
Base.:(==)(lhs::ActivePauliAction, rhs::ActivePauliAction) =
    lhs.nqubits == rhs.nqubits &&
    lhs.xmask == rhs.xmask &&
    lhs.zmask == rhs.zmask &&
    lhs.even_phase == rhs.even_phase &&
    lhs.odd_phase == rhs.odd_phase &&
    lhs.xz_overlap_odd == rhs.xz_overlap_odd

"""
    ActivePauliRotationPlan(pauli)

Active-state rotation plan for `exp(-im * theta * pauli)`. The Pauli
action is a signed permutation: diagonal when `xmask == 0`, otherwise a set of
disjoint two-amplitude pairs. This plan precomputes the relevant indices and
phases once so every shot can apply tight scalar updates to the active prefix.
"""
struct ActivePauliRotationPlan
    action::ActivePauliAction
    is_diagonal::Bool
    variable_phases::Bool
    phase_odd::Vector{Bool}
    pair_left::Vector{Int}

    function ActivePauliRotationPlan(action::ActivePauliAction)
        dim = _active_length(action.nqubits)
        if action.xmask == 0
            variable_phases = action.zmask != 0
            phase_odd = Bool[]
            if variable_phases
                phase_odd = Vector{Bool}(undef, dim)
                @inbounds for basis in 0:(dim - 1)
                    phase_odd[basis + 1] = _active_action_phase_odd(action, basis)
                end
            end
            return new(copy(action), true, variable_phases, phase_odd, Int[])
        end

        npairs = dim >>> 1
        pair_left = Vector{Int}(undef, npairs)
        variable_phases = action.zmask != 0
        phase_odd = variable_phases ? Vector{Bool}(undef, npairs) : Bool[]
        pair_selector = action.xmask & -action.xmask
        idx = 1
        @inbounds for left in 0:(dim - 1)
            (left & pair_selector) == 0 || continue
            pair_left[idx] = left
            variable_phases && (phase_odd[idx] = _active_action_phase_odd(action, left))
            idx += 1
        end
        return new(copy(action), false, variable_phases, phase_odd, pair_left)
    end
end

ActivePauliRotationPlan(p::PauliString) = ActivePauliRotationPlan(ActivePauliAction(p))

Base.copy(plan::ActivePauliRotationPlan) =
    ActivePauliRotationPlan(copy(plan.action))
Base.:(==)(lhs::ActivePauliRotationPlan, rhs::ActivePauliRotationPlan) =
    lhs.action == rhs.action &&
    lhs.is_diagonal == rhs.is_diagonal &&
    lhs.variable_phases == rhs.variable_phases &&
    lhs.phase_odd == rhs.phase_odd &&
    lhs.pair_left == rhs.pair_left

@inline function _active_action_phase_odd(action::ActivePauliAction, basis::Int)::Bool
    return isodd(count_ones(basis & action.zmask))
end

@inline function _active_action_phase(action::ActivePauliAction, basis::Int)::ComplexF64
    return _active_action_phase_odd(action, basis) ? action.odd_phase : action.even_phase
end

"""
    PrecomputedActivePauliRotationKernel(action, theta)

Instruction-level kernel for `exp(-im * theta * P)`. Unlike
`ActivePauliRotationPlan`, this stores 1-based permutation indices and the
signed complex coefficients for both symbolic signs of `P`, trading one-time
program memory for less per-shot arithmetic in the sampler.
"""
struct PrecomputedActivePauliRotationKernel
    action::ActivePauliAction
    is_diagonal::Bool
    theta::Float64
    cos_theta::Float64
    diagonal_minus_coefficients::Vector{ComplexF64}
    diagonal_plus_coefficients::Vector{ComplexF64}
    pair_left_indices::Vector{Int}
    pair_right_indices::Vector{Int}
    pair_left_minus_coefficients::Vector{ComplexF64}
    pair_right_minus_coefficients::Vector{ComplexF64}
    pair_left_plus_coefficients::Vector{ComplexF64}
    pair_right_plus_coefficients::Vector{ComplexF64}

    function PrecomputedActivePauliRotationKernel(action::ActivePauliAction, theta::Real)
        theta64 = Float64(theta)
        c = cos(theta64)
        minus_i_s = ComplexF64(0.0, -sin(theta64))
        plus_i_s = ComplexF64(0.0, sin(theta64))
        dim = _active_length(action.nqubits)

        if action.xmask == 0
            diagonal_minus = Vector{ComplexF64}(undef, dim)
            diagonal_plus = Vector{ComplexF64}(undef, dim)
            @inbounds for basis in 0:(dim - 1)
                phase = _active_action_phase(action, basis)
                diagonal_minus[basis + 1] = minus_i_s * phase
                diagonal_plus[basis + 1] = plus_i_s * phase
            end
            return new(
                copy(action),
                true,
                theta64,
                c,
                diagonal_minus,
                diagonal_plus,
                Int[],
                Int[],
                ComplexF64[],
                ComplexF64[],
                ComplexF64[],
                ComplexF64[],
            )
        end

        npairs = dim >>> 1
        pair_left = Vector{Int}(undef, npairs)
        pair_right = Vector{Int}(undef, npairs)
        left_minus = Vector{ComplexF64}(undef, npairs)
        right_minus = Vector{ComplexF64}(undef, npairs)
        left_plus = Vector{ComplexF64}(undef, npairs)
        right_plus = Vector{ComplexF64}(undef, npairs)
        pair_selector = action.xmask & -action.xmask
        idx = 1
        @inbounds for left in 0:(dim - 1)
            (left & pair_selector) == 0 || continue
            right = xor(left, action.xmask)
            left_phase = _active_action_phase(action, left)
            right_phase = _active_action_phase(action, right)
            pair_left[idx] = left + 1
            pair_right[idx] = right + 1
            left_minus[idx] = minus_i_s * left_phase
            right_minus[idx] = minus_i_s * right_phase
            left_plus[idx] = plus_i_s * left_phase
            right_plus[idx] = plus_i_s * right_phase
            idx += 1
        end
        return new(
            copy(action),
            false,
            theta64,
            c,
            ComplexF64[],
            ComplexF64[],
            pair_left,
            pair_right,
            left_minus,
            right_minus,
            left_plus,
            right_plus,
        )
    end
end

Base.copy(kernel::PrecomputedActivePauliRotationKernel) =
    PrecomputedActivePauliRotationKernel(copy(kernel.action), kernel.theta)
Base.:(==)(lhs::PrecomputedActivePauliRotationKernel, rhs::PrecomputedActivePauliRotationKernel) =
    lhs.action == rhs.action &&
    lhs.is_diagonal == rhs.is_diagonal &&
    lhs.theta == rhs.theta &&
    lhs.cos_theta == rhs.cos_theta &&
    lhs.diagonal_minus_coefficients == rhs.diagonal_minus_coefficients &&
    lhs.diagonal_plus_coefficients == rhs.diagonal_plus_coefficients &&
    lhs.pair_left_indices == rhs.pair_left_indices &&
    lhs.pair_right_indices == rhs.pair_right_indices &&
    lhs.pair_left_minus_coefficients == rhs.pair_left_minus_coefficients &&
    lhs.pair_right_minus_coefficients == rhs.pair_right_minus_coefficients &&
    lhs.pair_left_plus_coefficients == rhs.pair_left_plus_coefficients &&
    lhs.pair_right_plus_coefficients == rhs.pair_right_plus_coefficients

"""
    PrecomputedActivePauliMeasurementKernel(pauli)

Projection kernel for measuring a Hermitian active Pauli directly in the
current amplitude coordinates. Non-diagonal Paulis combine disjoint pairs of
source amplitudes; diagonal Paulis filter one source amplitude per output.
"""
struct PrecomputedActivePauliMeasurementKernel
    action::ActivePauliAction
    pivot::Int
    is_diagonal::Bool
    source0_false::Vector{Int}
    source1_false::Vector{Int}
    coeff0_false::Vector{ComplexF64}
    coeff1_false::Vector{ComplexF64}
    coeff1_false_real::Vector{Float64}
    coeff1_false_imag::Vector{Float64}
    source0_true::Vector{Int}
    source1_true::Vector{Int}
    coeff0_true::Vector{ComplexF64}
    coeff1_true::Vector{ComplexF64}

    function PrecomputedActivePauliMeasurementKernel(action::ActivePauliAction)
        k = action.nqubits
        k > 0 || throw(ArgumentError("cannot build an active measurement kernel for k == 0"))
        action.xmask != 0 && return _precomputed_nondiagonal_measurement_kernel(action)
        action.zmask != 0 && return _precomputed_diagonal_measurement_kernel(action)
        throw(ArgumentError("cannot build an active measurement kernel for identity Pauli"))
    end

    function PrecomputedActivePauliMeasurementKernel(
        action::ActivePauliAction,
        pivot::Integer,
        is_diagonal::Bool,
        source0_false::Vector{Int},
        source1_false::Vector{Int},
        coeff0_false::Vector{ComplexF64},
        coeff1_false::Vector{ComplexF64},
        source0_true::Vector{Int},
        source1_true::Vector{Int},
        coeff0_true::Vector{ComplexF64},
        coeff1_true::Vector{ComplexF64},
    )
        coeff1_false_real = is_diagonal ? Float64[] : real.(coeff1_false)
        coeff1_false_imag = is_diagonal ? Float64[] : imag.(coeff1_false)
        return new(
            copy(action),
            Int(pivot),
            is_diagonal,
            source0_false,
            source1_false,
            coeff0_false,
            coeff1_false,
            coeff1_false_real,
            coeff1_false_imag,
            source0_true,
            source1_true,
            coeff0_true,
            coeff1_true,
        )
    end
end

PrecomputedActivePauliMeasurementKernel(p::PauliString) =
    PrecomputedActivePauliMeasurementKernel(ActivePauliAction(p))

Base.copy(kernel::PrecomputedActivePauliMeasurementKernel) =
    PrecomputedActivePauliMeasurementKernel(copy(kernel.action))
Base.:(==)(lhs::PrecomputedActivePauliMeasurementKernel, rhs::PrecomputedActivePauliMeasurementKernel) =
    lhs.action == rhs.action &&
    lhs.pivot == rhs.pivot &&
    lhs.is_diagonal == rhs.is_diagonal &&
    lhs.source0_false == rhs.source0_false &&
    lhs.source1_false == rhs.source1_false &&
    lhs.coeff0_false == rhs.coeff0_false &&
    lhs.coeff1_false == rhs.coeff1_false &&
    lhs.coeff1_false_real == rhs.coeff1_false_real &&
    lhs.coeff1_false_imag == rhs.coeff1_false_imag &&
    lhs.source0_true == rhs.source0_true &&
    lhs.source1_true == rhs.source1_true &&
    lhs.coeff0_true == rhs.coeff0_true &&
    lhs.coeff1_true == rhs.coeff1_true

function _precomputed_nondiagonal_measurement_kernel(action::ActivePauliAction)
    k = action.nqubits
    pivot = trailing_zeros(action.xmask)
    dim = _active_length(k)
    out_dim = dim >>> 1
    source0 = Vector{Int}(undef, out_dim)
    source1 = Vector{Int}(undef, out_dim)
    coeff0 = fill(ComplexF64(inv(sqrt(2.0)), 0.0), out_dim)
    coeff1_false = Vector{ComplexF64}(undef, out_dim)
    coeff1_true = Vector{ComplexF64}(undef, out_dim)
    idx = 1
    @inbounds for packed in 0:(out_dim - 1)
        x0 = _insert_zero_bit(packed, pivot)
        x1 = xor(x0, action.xmask)
        eta = _active_action_phase(action, x0)
        source0[idx] = x0 + 1
        source1[idx] = x1 + 1
        coeff = inv(eta) * inv(sqrt(2.0))
        coeff1_false[idx] = coeff
        coeff1_true[idx] = -coeff
        idx += 1
    end
    return PrecomputedActivePauliMeasurementKernel(
        copy(action),
        pivot,
        false,
        source0,
        source1,
        coeff0,
        coeff1_false,
        copy(source0),
        copy(source1),
        copy(coeff0),
        coeff1_true,
    )
end

function _precomputed_diagonal_measurement_kernel(action::ActivePauliAction)
    k = action.nqubits
    pivot = trailing_zeros(action.zmask)
    dim = _active_length(k)
    out_dim = dim >>> 1
    source_false = Vector{Int}(undef, out_dim)
    source_true = Vector{Int}(undef, out_dim)
    zeros_source = zeros(Int, out_dim)
    ones_coeff = fill(1.0 + 0.0im, out_dim)
    zeros_coeff = zeros(ComplexF64, out_dim)
    negative_phase = action.even_phase == -1.0 + 0.0im
    (action.even_phase == 1.0 + 0.0im || negative_phase) ||
        throw(ArgumentError("diagonal active measurement Pauli must have real eigenvalues"))
    phase_bit = negative_phase ? 1 : 0
    z_without_pivot = action.zmask & ~(1 << pivot)
    @inbounds for packed in 0:(out_dim - 1)
        x_without_pivot = _insert_zero_bit(packed, pivot)
        parity = count_ones(x_without_pivot & z_without_pivot) & 1
        false_pivot = xor(phase_bit, parity)
        true_pivot = xor(false_pivot, 1)
        source_false[packed + 1] = (x_without_pivot | (false_pivot << pivot)) + 1
        source_true[packed + 1] = (x_without_pivot | (true_pivot << pivot)) + 1
    end
    return PrecomputedActivePauliMeasurementKernel(
        copy(action),
        pivot,
        true,
        source_false,
        zeros_source,
        ones_coeff,
        zeros_coeff,
        source_true,
        zeros_source,
        ones_coeff,
        zeros_coeff,
    )
end

@inline function _insert_zero_bit(packed::Int, bit::Int)::Int
    low_mask = (1 << bit) - 1
    low = packed & low_mask
    high = packed & ~low_mask
    return low | (high << 1)
end

function _check_active_pauli(k::Int, p::PauliString)
    p.nqubits == k ||
        throw(DimensionMismatch("Pauli string acts on $(p.nqubits) qubits; active state has $k qubits"))
    return nothing
end

function _check_active_action(k::Int, action::ActivePauliAction)
    action.nqubits == k ||
        throw(DimensionMismatch("Pauli action acts on $(action.nqubits) qubits; active state has $k qubits"))
    return nothing
end

function _check_active_rotation_plan(k::Int, plan::ActivePauliRotationPlan)
    _check_active_action(k, plan.action)
    return nothing
end

function _check_active_rotation_kernel(k::Int, kernel::PrecomputedActivePauliRotationKernel)
    _check_active_action(k, kernel.action)
    return nothing
end

function _check_active_measurement_kernel(k::Int, kernel::PrecomputedActivePauliMeasurementKernel)
    _check_active_action(k, kernel.action)
    return nothing
end

@inline function _can_rotate_real_pair_flip(kernel::PrecomputedActivePauliRotationKernel)::Bool
    kernel.is_diagonal && return false
    kernel.action.zmask == 0 && return false
    kernel.action.xz_overlap_odd || return false
    return iszero(real(kernel.action.even_phase)) && !iszero(imag(kernel.action.even_phase))
end

@inline function _real_pair_flip_coeff(base_coeff::Float64, zmask::Int, basis::Int)::Float64
    return isodd(count_ones(basis & zmask)) ? -base_coeff : base_coeff
end

function _active_masks(p::PauliString)
    nw = _nwords(p.nqubits)
    nw == 0 && return 0, 0
    return Int(p.data[1]), Int(p.data[nw + 1])
end

function _pauli_squares_to_identity(p::PauliString)
    nw = _nwords(p.nqubits)
    y_count = 0
    @inbounds for w in 1:nw
        y_count += count_ones(p.data[w] & p.data[nw + w])
    end
    return iseven(phase_exponent(p) - y_count)
end

"""
    apply_pauli!(out, p, alpha)

Write `out .= p * alpha` for a `k`-qubit packed Pauli string acting on the
computational basis. Active qubit `q` is bit position `q` in the zero-based
basis index.
"""
function apply_pauli!(out::Vector{ComplexF64}, p::PauliString, alpha::Vector{ComplexF64})
    p.nqubits < Sys.WORD_SIZE - 1 ||
        throw(ArgumentError("Pauli string has too many qubits for active-state basis indexing"))
    dim = _active_length(p.nqubits)
    length(alpha) == dim ||
        throw(DimensionMismatch("alpha has length $(length(alpha)); expected $dim"))
    length(out) == dim ||
        throw(DimensionMismatch("output has length $(length(out)); expected $dim"))
    out !== alpha ||
        throw(ArgumentError("apply_pauli! output must be distinct from input alpha"))

    xmask, zmask = _active_masks(p)
    even_phase = _phase_factor(phase_exponent(p))
    odd_phase = -even_phase

    # X bits permute basis indices by xor; Z bits contribute the parity phase
    # from the source computational basis state.
    @inbounds for src in 0:(dim - 1)
        dst = xor(src, xmask) + 1
        phase = isodd(count_ones(src & zmask)) ? odd_phase : even_phase
        out[dst] = phase * alpha[src + 1]
    end
    return out
end

function apply_pauli!(state::ActiveState, p::PauliString)
    _check_active_pauli(state.k, p)
    dim = _check_active_storage(state)
    xmask, zmask = _active_masks(p)
    even_phase = _phase_factor(phase_exponent(p))
    odd_phase = -even_phase

    if xmask == 0
        @inbounds for src in 0:(dim - 1)
            phase = isodd(count_ones(src & zmask)) ? odd_phase : even_phase
            state.alpha[src + 1] *= phase
        end
        return state
    end

    pair_selector = xmask & -xmask
    xz_overlap_odd = isodd(count_ones(xmask & zmask))
    @inbounds for base in 0:(dim - 1)
        (base & pair_selector) == 0 || continue
        paired = xor(base, xmask)
        i0 = base + 1
        i1 = paired + 1
        a0 = state.alpha[i0]
        a1 = state.alpha[i1]
        parity0 = isodd(count_ones(base & zmask))
        phase0 = parity0 ? odd_phase : even_phase
        phase1 = xor(parity0, xz_overlap_odd) ? odd_phase : even_phase
        state.alpha[i0] = phase1 * a1
        state.alpha[i1] = phase0 * a0
    end
    return state
end

function _apply_pauli_prefix!(
    out::Vector{ComplexF64},
    p::PauliString,
    alpha::Vector{ComplexF64},
    dim::Int,
)
    xmask, zmask = _active_masks(p)
    even_phase = _phase_factor(phase_exponent(p))
    odd_phase = -even_phase

    @inbounds for src in 0:(dim - 1)
        dst = xor(src, xmask) + 1
        phase = isodd(count_ones(src & zmask)) ? odd_phase : even_phase
        out[dst] = phase * alpha[src + 1]
    end
    return out
end

"""
    rotate_pauli!(state, p, theta)

Apply the exact active-subsystem Pauli rotation
`exp(-im * theta * p)` in place:

    alpha .= cos(theta) .* alpha .- im * sin(theta) .* (p * alpha)

The runtime path uses a cached `ActivePauliAction` and updates coupled
computational-basis amplitudes directly, avoiding a scratch-vector Pauli apply.
"""
function rotate_pauli!(state::ActiveState, p::PauliString, theta::Real)
    _check_active_pauli(state.k, p)
    return rotate_pauli!(state, ActivePauliAction(p), theta)
end

function rotate_pauli!(state::ActiveState, action::ActivePauliAction, theta::Real)
    _check_active_action(state.k, action)
    dim = _check_active_storage(state)

    c = cos(theta)
    minus_i_s = ComplexF64(0.0, -sin(theta))
    xmask = action.xmask
    zmask = action.zmask
    even_phase = action.even_phase
    odd_phase = action.odd_phase

    if xmask == 0
        @inbounds for basis in 0:(dim - 1)
            phase = isodd(count_ones(basis & zmask)) ? odd_phase : even_phase
            state.alpha[basis + 1] *= c + minus_i_s * phase
        end
        return state
    end

    pair_selector = xmask & -xmask
    @inbounds for base in 0:(dim - 1)
        (base & pair_selector) == 0 || continue
        paired = xor(base, xmask)
        i0 = base + 1
        i1 = paired + 1
        a0 = state.alpha[i0]
        a1 = state.alpha[i1]
        parity0 = isodd(count_ones(base & zmask))
        phase0 = parity0 ? odd_phase : even_phase
        phase1 = xor(parity0, action.xz_overlap_odd) ? odd_phase : even_phase
        state.alpha[i0] = c * a0 + minus_i_s * phase1 * a1
        state.alpha[i1] = c * a1 + minus_i_s * phase0 * a0
    end
    return state
end

function rotate_pauli!(
    state::ActiveState,
    plan::ActivePauliRotationPlan,
    theta::Real,
)
    c = cos(theta)
    minus_i_s = ComplexF64(0.0, -sin(theta))
    return rotate_pauli!(state, plan, c, minus_i_s)
end

function rotate_pauli!(
    state::ActiveState,
    plan::ActivePauliRotationPlan,
    c::Float64,
    sin_coeff::ComplexF64,
)
    _check_active_rotation_plan(state.k, plan)
    _check_active_storage(state)
    even_coeff = sin_coeff * plan.action.even_phase
    odd_coeff = sin_coeff * plan.action.odd_phase

    if plan.is_diagonal
        if plan.variable_phases
            @inbounds for idx in eachindex(plan.phase_odd)
                state.alpha[idx] *= c + (plan.phase_odd[idx] ? odd_coeff : even_coeff)
            end
        else
            factor = c + even_coeff
            @inbounds for idx in 1:_active_length(plan.action.nqubits)
                state.alpha[idx] *= factor
            end
        end
        return state
    end

    if plan.variable_phases
        @inbounds for idx in eachindex(plan.pair_left)
            left = plan.pair_left[idx]
            right = xor(left, plan.action.xmask)
            i0 = left + 1
            i1 = right + 1
            a0 = state.alpha[i0]
            a1 = state.alpha[i1]
            left_odd = plan.phase_odd[idx]
            left_coeff = left_odd ? odd_coeff : even_coeff
            right_coeff = xor(left_odd, plan.action.xz_overlap_odd) ? odd_coeff : even_coeff
            state.alpha[i0] = c * a0 + right_coeff * a1
            state.alpha[i1] = c * a1 + left_coeff * a0
        end
    else
        @inbounds for left in plan.pair_left
            right = xor(left, plan.action.xmask)
            i0 = left + 1
            i1 = right + 1
            a0 = state.alpha[i0]
            a1 = state.alpha[i1]
            state.alpha[i0] = c * a0 + even_coeff * a1
            state.alpha[i1] = c * a1 + even_coeff * a0
        end
    end
    return state
end

function rotate_pauli!(
    state::ActiveState,
    kernel::PrecomputedActivePauliRotationKernel,
    sign::Bool,
)
    _check_active_rotation_kernel(state.k, kernel)
    _check_active_storage(state)
    c = kernel.cos_theta
    alpha = state.alpha

    if kernel.is_diagonal
        coefficients = sign ? kernel.diagonal_plus_coefficients : kernel.diagonal_minus_coefficients
        @inbounds for idx in eachindex(coefficients)
            alpha[idx] *= c + coefficients[idx]
        end
        return state
    end

    if kernel.action.zmask == 0
        coefficient =
            sign ? kernel.pair_left_plus_coefficients[1] : kernel.pair_left_minus_coefficients[1]
        coeff_im = imag(coefficient)
        @inbounds @simd for idx in eachindex(kernel.pair_left_indices)
            i0 = kernel.pair_left_indices[idx]
            i1 = kernel.pair_right_indices[idx]
            a0 = alpha[i0]
            a1 = alpha[i1]
            alpha[i0] = ComplexF64(
                muladd(c, real(a0), -coeff_im * imag(a1)),
                muladd(c, imag(a0), coeff_im * real(a1)),
            )
            alpha[i1] = ComplexF64(
                muladd(c, real(a1), -coeff_im * imag(a0)),
                muladd(c, imag(a1), coeff_im * real(a0)),
            )
        end
        return state
    end

    if _can_rotate_real_pair_flip(kernel)
        base_coeff =
            real(sign ? kernel.pair_left_plus_coefficients[1] : kernel.pair_left_minus_coefficients[1])
        zmask = kernel.action.zmask
        @inbounds @simd for idx in eachindex(kernel.pair_left_indices)
            i0 = kernel.pair_left_indices[idx]
            i1 = kernel.pair_right_indices[idx]
            coeff = _real_pair_flip_coeff(base_coeff, zmask, i0 - 1)
            a0 = alpha[i0]
            a1 = alpha[i1]
            alpha[i0] = ComplexF64(
                muladd(c, real(a0), -coeff * real(a1)),
                muladd(c, imag(a0), -coeff * imag(a1)),
            )
            alpha[i1] = ComplexF64(
                muladd(c, real(a1), coeff * real(a0)),
                muladd(c, imag(a1), coeff * imag(a0)),
            )
        end
        return state
    end

    left_coefficients = sign ? kernel.pair_left_plus_coefficients : kernel.pair_left_minus_coefficients
    right_coefficients = sign ? kernel.pair_right_plus_coefficients : kernel.pair_right_minus_coefficients
    @inbounds for idx in eachindex(kernel.pair_left_indices)
        i0 = kernel.pair_left_indices[idx]
        i1 = kernel.pair_right_indices[idx]
        a0 = alpha[i0]
        a1 = alpha[i1]
        alpha[i0] = c * a0 + right_coefficients[idx] * a1
        alpha[i1] = c * a1 + left_coefficients[idx] * a0
    end
    return state
end

pauli_rotation!(state::ActiveState, p::PauliString, theta::Real) = rotate_pauli!(state, p, theta)
