function _batch_uses_structarray_turbo(runtime::BatchFactoredExecutorState)::Bool
    return runtime.kernel_backend === :structarray && runtime.active isa StructArray
end

function _batch_diagonal_measure_true_prob_struct_turbo!(
    runtime::BatchFactoredExecutorState,
    source_true::Vector{Int},
    out_dim::Int,
    prob_true::Vector{Float64},
)::Bool
    _batch_uses_structarray_turbo(runtime) || return false
    active_shots = runtime.active_shots
    length(prob_true) >= active_shots ||
        throw(DimensionMismatch("probability scratch is too short for the active batch"))
    active_shots > 0 && out_dim > 0 || return true
    re = runtime.active.re
    im = runtime.active.im
    @turbo for shot in 1:active_shots
        prob_true[shot] = 0.0
    end
    @turbo for idx in 1:out_dim
        src = source_true[idx]
        for shot in 1:active_shots
            r = re[shot, src]
            i = im[shot, src]
            prob_true[shot] += r * r + i * i
        end
    end
    return true
end

function _batch_project_diagonal_measurement_struct_turbo!(
    runtime::BatchFactoredExecutorState,
    source_false::Vector{Int},
    source_true::Vector{Int},
    out_dim::Int,
    branch_bits::Vector{UInt64},
    invnorms::Vector{Float64},
)::Bool
    _batch_uses_structarray_turbo(runtime) || return false
    active_shots = runtime.active_shots
    length(invnorms) >= active_shots ||
        throw(DimensionMismatch("inverse-norm scratch is too short for the active batch"))
    length(branch_bits) >= _runtime_batch_word_count(runtime) ||
        throw(DimensionMismatch("branch bit vector is too short for the active batch"))
    active_shots > 0 && out_dim > 0 || return true
    re = runtime.active.re
    im = runtime.active.im
    # This projection is intentionally ordered by output index. Diagonal
    # sources can be above their destination in the active prefix, so a
    # multi-level reordered loop would need a scratch copy and is slower here.
    @inbounds for idx in 1:out_dim
        sf = source_false[idx]
        st = source_true[idx]
        @turbo for shot in 1:active_shots
            shot0 = shot - 1
            branch =
                ((branch_bits[(shot0 >>> 6) + 1] >>> (shot0 & 63)) & UInt64(1)) == UInt64(1)
            src = ifelse(branch, st, sf)
            norm = invnorms[shot]
            re[shot, idx] = re[shot, src] * norm
            im[shot, idx] = im[shot, src] * norm
        end
    end
    return true
end

function _batch_nondiagonal_measure_true_prob_struct_turbo!(
    runtime::BatchFactoredExecutorState,
    kernel::PrecomputedActivePauliMeasurementKernel,
    prob_true::Vector{Float64},
)::Bool
    _batch_uses_structarray_turbo(runtime) || return false
    active_shots = runtime.active_shots
    length(prob_true) >= active_shots ||
        throw(DimensionMismatch("probability scratch is too short for the active batch"))
    out_dim = length(kernel.source0_false)
    active_shots > 0 && out_dim > 0 || return true

    re = runtime.active.re
    im = runtime.active.im
    sources0 = kernel.source0_false
    sources1 = kernel.source1_false
    coeff1_re = kernel.coeff1_false_real
    coeff1_im = kernel.coeff1_false_imag
    invsqrt2 = inv(sqrt(2.0))
    @turbo for shot in 1:active_shots
        prob_true[shot] = 0.0
    end
    @turbo for idx in 1:out_dim
        src0 = sources0[idx]
        src1 = sources1[idx]
        c1r = coeff1_re[idx]
        c1i = coeff1_im[idx]
        for shot in 1:active_shots
            r0 = re[shot, src0]
            i0 = im[shot, src0]
            r1 = re[shot, src1]
            i1 = im[shot, src1]
            prod_r = c1r * r1 - c1i * i1
            prod_i = c1r * i1 + c1i * r1
            amp_r = invsqrt2 * r0 - prod_r
            amp_i = invsqrt2 * i0 - prod_i
            prob_true[shot] += amp_r * amp_r + amp_i * amp_i
        end
    end
    return true
end

function _batch_project_nondiagonal_measurement_struct_turbo!(
    runtime::BatchFactoredExecutorState,
    kernel::PrecomputedActivePauliMeasurementKernel,
    branch_bits::Vector{UInt64},
    invnorms::Vector{Float64},
)::Bool
    _batch_uses_structarray_turbo(runtime) || return false
    scratch = runtime.active_scratch
    scratch isa StructArray ||
        throw(DimensionMismatch("StructArray active storage requires StructArray scratch storage"))
    active_shots = runtime.active_shots
    length(invnorms) >= active_shots ||
        throw(DimensionMismatch("inverse-norm scratch is too short for the active batch"))
    length(branch_bits) >= _runtime_batch_word_count(runtime) ||
        throw(DimensionMismatch("branch bit vector is too short for the active batch"))
    out_dim = length(kernel.source0_false)
    active_shots > 0 && out_dim > 0 || return true

    re = runtime.active.re
    im = runtime.active.im
    scratch_re = scratch.re
    scratch_im = scratch.im
    fs0 = kernel.source0_false
    fs1 = kernel.source1_false
    coeff1_re = kernel.coeff1_false_real
    coeff1_im = kernel.coeff1_false_imag
    invsqrt2 = inv(sqrt(2.0))
    @turbo for idx in 1:out_dim
        fsrc0 = fs0[idx]
        fsrc1 = fs1[idx]
        c1r = coeff1_re[idx]
        c1i = coeff1_im[idx]
        for shot in 1:active_shots
            shot0 = shot - 1
            branch =
                ((branch_bits[(shot0 >>> 6) + 1] >>> (shot0 & 63)) & UInt64(1)) == UInt64(1)
            branch_sign = ifelse(branch, -1.0, 1.0)
            r0 = re[shot, fsrc0]
            i0 = im[shot, fsrc0]
            r1 = re[shot, fsrc1]
            i1 = im[shot, fsrc1]
            prod_r = c1r * r1 - c1i * i1
            prod_i = c1r * i1 + c1i * r1
            norm = invnorms[shot]
            scratch_re[shot, idx] = (invsqrt2 * r0 + branch_sign * prod_r) * norm
            scratch_im[shot, idx] = (invsqrt2 * i0 + branch_sign * prod_i) * norm
        end
    end
    return true
end

function _batch_rotate_uniform_phase_struct_turbo!(
    runtime::BatchFactoredExecutorState,
    kernel::PrecomputedActivePauliRotationKernel,
    coefficients::Vector{Float64},
)::Bool
    _batch_uses_structarray_turbo(runtime) || return false
    active_shots = runtime.active_shots
    npairs = length(kernel.pair_left_indices)
    active_shots > 0 && npairs > 0 || return true

    re = runtime.active.re
    im = runtime.active.im
    left = kernel.pair_left_indices
    right = kernel.pair_right_indices
    c = kernel.cos_theta
    @turbo for idx in 1:npairs
        for shot in 1:active_shots
            i0 = left[idx]
            i1 = right[idx]
            coeff = coefficients[shot]
            r0 = re[shot, i0]
            im0 = im[shot, i0]
            r1 = re[shot, i1]
            im1 = im[shot, i1]
            re[shot, i0] = muladd(c, r0, -coeff * im1)
            im[shot, i0] = muladd(c, im0, coeff * r1)
            re[shot, i1] = muladd(c, r1, -coeff * im0)
            im[shot, i1] = muladd(c, im1, coeff * r0)
        end
    end
    return true
end

function _batch_rotate_real_pair_flip_struct_turbo!(
    runtime::BatchFactoredExecutorState,
    kernel::PrecomputedActivePauliRotationKernel,
    coefficients::Vector{Float64},
    zmask::Int,
)::Bool
    _batch_uses_structarray_turbo(runtime) || return false
    active_shots = runtime.active_shots
    npairs = length(kernel.pair_left_indices)
    active_shots > 0 && npairs > 0 || return true

    re = runtime.active.re
    im = runtime.active.im
    left = kernel.pair_left_indices
    right = kernel.pair_right_indices
    c = kernel.cos_theta
    @turbo for idx in 1:npairs
        for shot in 1:active_shots
            i0 = left[idx]
            i1 = right[idx]
            phase_sign = ifelse(isodd(count_ones((i0 - 1) & zmask)), -1.0, 1.0)
            coeff = phase_sign * coefficients[shot]
            r0 = re[shot, i0]
            im0 = im[shot, i0]
            r1 = re[shot, i1]
            im1 = im[shot, i1]
            re[shot, i0] = muladd(c, r0, -coeff * r1)
            im[shot, i0] = muladd(c, im0, -coeff * im1)
            re[shot, i1] = muladd(c, r1, coeff * r0)
            im[shot, i1] = muladd(c, im1, coeff * im0)
        end
    end
    return true
end

function _batch_promote_first_dormant_rotation_struct_turbo!(
    runtime::BatchFactoredExecutorState,
    dim::Int,
    c::Float64,
    coefficients::Vector{Float64},
)::Bool
    _batch_uses_structarray_turbo(runtime) || return false
    active_shots = runtime.active_shots
    active_shots > 0 && dim > 0 || return true
    re = runtime.active.re
    im = runtime.active.im
    @turbo for basis in 1:dim
        for shot in 1:active_shots
            promoted_basis = dim + basis
            coeff = coefficients[shot]
            r = re[shot, basis]
            i = im[shot, basis]
            re[shot, basis] = c * r
            im[shot, basis] = c * i
            re[shot, promoted_basis] = -coeff * i
            im[shot, promoted_basis] = coeff * r
        end
    end
    return true
end
