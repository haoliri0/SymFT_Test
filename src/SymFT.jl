"""
    SymFT

Frame-factored Clifford+T sampler for Stim-like noisy circuits. The simulator
keeps a concrete Clifford frame, a dense active vector of length `2^k`, and
symbolic dormant stabilizer signs; it does not allocate a full physical
`2^n` state vector in production samplers.

Typical workflow:

1. Parse a circuit with `parse_stim_file`, `parse_stim_text`, or
   `parse_stim_lines`.
2. Convert the parsed frame state to `PendingFactoredState`.
3. Lower pending Pauli rotations and measurements with
   `plan_factored_updates!`.
4. Sample the resulting `FactoredInstructionProgram` with the single-shot,
   batched, or shared-active executor.
"""
module SymFT

using LoopVectorization: @turbo
using StructArrays: StructArray

# Public surface is intentionally flat for now: tests and experiments import
# the core packed Pauli, frame, pending-planning, and Stim parsing APIs from
# this module while the implementation remains split by representation layer.
export PauliString,
    ActiveState,
    ActivePauliFrame,
    BatchFactoredExecutorState,
    CliffordFrame,
    ConditionalPauliString,
    DormantState,
    FactoredInstruction,
    FactoredInstructionProgram,
    FactoredExecutorState,
    FrameFactoredState,
    PresampledExogenous,
    SharedActiveBatchExecutorState,
    StimDetector,
    StimObservableInclude,
    StimParseResult,
    StimSampleSummary,
    ApplyPrecomputedActivePauliRotation,
    ApplyActiveBasisChange,
    IntroduceDormantMeasurementBranch,
    MeasureActiveLastZ,
    MeasurePrecomputedActivePauli,
    PendingFactoredState,
    PendingPauliMeasurement,
    PendingFactoredOperation,
    PendingPauliRotation,
    PromoteDormantRotation,
    RecordMeasurement,
    SymbolicBool,
    SymbolicCategoricalDistribution,
    SymbolicContext,
    SymbolicPauliString,
    nqubits,
    nactive,
    ndormant,
    max_active_qubits,
    phase_exponent,
    xbit,
    zbit,
    pauli_identity,
    pauli_x,
    pauli_y,
    pauli_z,
    pauli_string,
    phase_shift!,
    pauli_anticommutes,
    add_pauli!,
    assign_dormant_symbol!,
    active_packet_count,
    condition_ids,
    condition_id,
    constant_term,
    dormant_bit,
    dormant_bits,
    flip_dormant_bit!,
    sign_conditions,
    symbolic_sign,
    set_dormant_bit!,
    fresh_condition!,
    fresh_symbolic_bool!,
    next_condition,
    symbolic_context,
    symbolic_bool,
    xor_dormant_bit!,
    conjugate_by,
    apply_pauli!,
    rotate_pauli!,
    pauli_rotation!,
    apply_pauli_measurement!,
    apply_pauli_rotation!,
    pending_measurements,
    bernoulli_probabilities,
    bernoulli_probability,
    categorical_distribution,
    categorical_distributions,
    factored_instructions,
    factored_instruction_program,
    has_pending_operations,
    plan_factored_updates!,
    pending_operation_count,
    pending_operations,
    pending_rotations,
    parse_stim_file,
    parse_stim_lines,
    parse_stim_text,
    presample_exogenous,
    estimate_stim_logical_error_rate,
    discard_rate,
    logical_error_rate,
    sample_measurements,
    sample_measurements_shared_active,
    sample_measurements!,
    reset_executor!,
    process_next_pending_operation!,
    process_pending_measurement!,
    process_pending_operation!,
    process_pending_operations!,
    process_pending_rotation!,
    execute!,
    execute_instruction!,
    preimage,
    preimage!,
    left_H!,
    left_S!,
    left_SDG!,
    left_X!,
    left_Z!,
    left_CX!,
    left_CZ!,
    left_SWAP!,
    right_H!,
    right_S!,
    right_SDG!,
    right_X!,
    right_Z!,
    right_CX!,
    right_CZ!,
    right_SWAP!,
    stim_detectors,
    stim_measurement_records,
    stim_observable_includes,
    stim_observables,
    stim_state,
    fresh_bernoulli_bool!,
    fresh_bernoulli_condition!,
    fresh_categorical_bools!,
    fresh_categorical_conditions!

include("pauli_string.jl")
include("symbolic_bool.jl")
include("active_pauli_frame.jl")
include("dormant_state.jl")
include("active_state.jl")
include("factored_update_program.jl")
include("clifford_frame.jl")
include("frame_factored_state.jl")
include("stim_parser.jl")
# Pending processing is split into the shared queue/planner shell, Clifford
# rewrite helpers, and operation-specific rotation/measurement lowering.
include("pending_factored_processing.jl")
include("pending_factored_rewrites.jl")
include("pending_factored_rotations.jl")
include("pending_factored_measurements.jl")
include("factored_single_shot_sampler.jl")
include("factored_batch_sampler.jl")
include("batch_structarray_turbo_kernels.jl")
include("shared_active_batch_sampler.jl")
include("stim_sampling.jl")

end
