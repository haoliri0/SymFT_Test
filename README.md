# SymFT.jl

SymFT is a correctness-first, frame-factored Clifford+T sampler for noisy
Stim-style circuits with Pauli measurements, Pauli noise, `T`/`T_DAG` gates,
detectors, observables, and measurement-record-controlled Pauli feedback.

The production samplers never allocate a full physical `2^n` state vector.
The only dense quantum vector is the active subsystem vector `alpha` of length
`2^k`, where `k` is the number of active virtual qubits needed by the current
factored update plan.

## State Model

The simulator maintains the state up to shot-global phase in factored form:

```text
|psi> ~= C ( E_A(s) |alpha>_A tensor |0>_D )
```

where:

- `C` is the concrete Clifford frame.
- `A` is the active virtual subsystem, represented by the dense vector
  `alpha`.
- `D` is the dormant subsystem, always in zeros.
- `E_A(s)` is the symbolic Pauli frame on all qubits.

Stim qubits are parsed as 0-based qubits. Julia arrays are still 1-based, so
code that indexes arrays by qubit uses `q + 1` explicitly.

## Method Highlight

Pauli rotations and Pauli measurements are not handled by applying dense
Clifford basis changes to the active vector. SymFT separates planning from
sampling:

- During parsing, `T`, `T_DAG`, Pauli measurements, Pauli noise, and feedback
  become pending factored Pauli operations after conjugation through the current
  concrete Clifford and symbolic Pauli frames.
- During pending processing, the planner tracks the stabilizer-coordinate basis
  of the active subsystem. When a reduction step changes that coordinate basis,
  it rewrites the current and queued pending Paulis into the new coordinates.
  The dense active vector `alpha` is not updated by those coordinate rewrites.
- Active Pauli rotations emit precomputed pair-update instructions. Sampling
  applies the exact in-place amplitude pair update for `exp(-im * theta * P)`,
  with shot-dependent symbolic signs supplied by packed condition values.
- Active Pauli measurements emit precomputed probability/projection kernels.
  Sampling computes the branch probability from `alpha`, samples the branch,
  projects and renormalizes `alpha`, and writes the measurement record. The
  post-measurement stabilizer-coordinate update was already accounted for in
  pending processing.
- Batch active storage is shot-major in the performance-critical sense: the
  logical shape is `shots x active_dim`, and Julia's column-major layout makes
  the shot dimension contiguous for each active basis column. Batch rotation
  and measurement kernels exploit this by SIMD/vectorizing across many shots
  for the same active basis index or amplitude pair.
- The C AVX2 kernels follow the same Julia memory layout. Although C arrays are
  normally described in row-major terms, these kernels are passed the raw Julia
  column-major buffer and index it with an explicit `leading_shots` stride. In
  C pointer-order terms the layout is `active[basis][shot]`: `shot` is the
  contiguous inner dimension, and `basis` advances by `leading_shots`.
- Dormant non-diagonal support is handled by symbolic branch introduction,
  dormant promotion, and sign pushback through the pending FIFO.

In particular, there is no active Clifford localization runtime path for Pauli
rotations or measurements, and these runtime updates do not rely on applying a
Pauli operator to `alpha` as an intermediate state.

## Workflow

0. Initialise the package:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

In Julia, load the package first:

```julia
using SymFT
```

1. Parse the circuit.

   ```julia
   parsed = parse_stim_file("d5.stim")
   ```

   The parser builds a `FrameFactoredState`. Clifford operations update the
   concrete frame. Noise, feedback, T gates, T_DAG gates, and Pauli
   measurements become symbolic pending operations in the factored frame.

2. Build the pending planner state.

   ```julia
   pending = PendingFactoredState(stim_state(parsed))
   ```

   Pending processing starts from the parsed frame state and drains a FIFO of
   factored Pauli rotations and measurements.

3. Lower pending operations.

   ```julia
   program = plan_factored_updates!(pending)
   ```

   The planner tracks the active stabilizer basis while it processes pending
   operations. It rewrites pending Paulis into the current active basis and
   emits a compact `FactoredInstructionProgram`. Runtime instructions only do
   amplitude rotations, projections, symbolic branch assignments, and
   measurement record writes.

4. Sample the instruction program.

   Single shot:

   ```julia
   record = sample_measurements(program)
   ```

   Standard batch:

   ```julia
   records = sample_measurements(program, 100_000; kernel_backend = :structarray)
   ```

   Shared-active batch:

   ```julia
   records = sample_measurements_shared_active(program, 100_000)
   ```

   Batch outputs are `shots x nrecords` `BitMatrix` values.

5. Stream detector and observable statistics.

   ```julia
   summary = estimate_stim_logical_error_rate(parsed, 100_000)
   discard_rate(summary)
   logical_error_rate(summary)
   ```

   Detector postselection and observable parity are accumulated from packed
   batch measurement records without materializing unnecessary per-shot data.

## Samplers

`FactoredExecutorState` executes one shot at a time. It is useful for exact
debugging, deterministic RNG tests, and comparing against batch behavior.

`BatchFactoredExecutorState` executes blocks of shots. Symbol assignments and
measurement records are packed across shots. The active vectors are independent
per shot.

`SharedActiveBatchExecutorState` keeps identical active states in packets. It
updates one active vector per packet until signs or measurement branches split
the packet. If packets saturate the batch, it materializes into the standard
batch representation.

## Batch Kernel Backends

The batch sampler supports three active-vector kernel backends:

- `:structarray` or `:turbo`: stores real and imaginary parts separately using
  `StructArray` and uses LoopVectorization kernels.
- `:c` or `:avx2`: stores interleaved `ComplexF64` matrices and uses the C
  AVX2 kernels when available.
- `:scalar`: uses the generic Julia fallback kernels.

The default backend is `:structarray`. It can be overridden with the
`kernel_backend` keyword or with `SYMFT_BATCH_KERNEL_BACKEND`.

Default batch size is chosen from the maximum active width:

```text
min(2048, 2^17 / 2^max_k)
```

This keeps the active memory footprint bounded while allowing large batches
when the active subsystem is small.

## Exactness Notes

- `T` and `T_DAG` are handled exactly as Pauli rotations in the factored frame.
- Measurements sample the correct active branch probabilities and project the
  active vector in place or through runtime projection buffers as required.
- Symbolic exogenous noise can be sampled per shot or presampled into packed
  word tables.
- Detector and observable processing uses the emitted measurement records.

Approximate behavior should only be introduced behind an explicitly named
approximate mode.

## Useful Commands

Run the test suite:

```bash
julia --project=. test/runtests.jl
```

Benchmark the standard batch sampler:

```bash
julia --project=. benchmarks/bench_batched_stim_sampling.jl 100000 auto 3 structarray
```

Profile the batch sampler:

```bash
julia --project=. benchmarks/profile_batched_stim_sampling.jl d5.stim 100000 auto true structarray
```

Benchmark the shared-active sampler:

```bash
julia --project=. benchmarks/bench_shared_active_batch_sampler.jl d5.stim 100000 auto structarray
```
