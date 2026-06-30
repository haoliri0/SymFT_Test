# Postselection Design

This note describes the C++ detector postselection design for the single-shot
and batch samplers.

The important boundary is:

- Frontend code parses circuit files and lowers them to a
  `FactoredInstructionProgram` with detector instructions.
- Prepared sampler code consumes that internal program, prepares reusable
  runtime/scratch/presampling storage, and samples requested shots.
- Benchmark/tool code chooses options, constructs a prepared sampler, calls
  `sample(...)`, and reports timings/results.
- Sampler code owns detector instruction insertion, presampled value generation,
  detector postselection execution, and result accumulation. It builds batch
  last-use tables, decides how detector instructions are handled in
  postselection mode, and maintains active state, symbols, measurement records,
  detector records, and expression columns consistently.

## Core Idea

Stim detectors are parity checks over measurement records, but they also have a
source order in the Stim circuit. The sampler turns them into classical
`RecordDetector` instructions in the planned instruction stream, not checks
grouped by the maximum record they reference.

The parser records where each `DETECTOR` appeared in the flattened Stim stream.
Lowering maps that source position to a pending-operation prefix, and the
factored planner maps the pending prefix to an executable instruction
checkpoint where detector instructions are inserted:

```text
checkpoint 0      before executable instruction 0
checkpoint i + 1  immediately after executable instruction i
```

Each detector instruction carries a `SymbolicBool` outcome, built by xoring the
symbolic expressions for the measurement records it references. During program
construction, this becomes a `SymbolicBoolEvaluationPlan`, just like a
measurement outcome.

In no-postselection mode, executing `RecordDetector` evaluates the outcome and
records it in detector bit storage.

In postselection mode, executing `RecordDetector` evaluates the same outcome and
then rejects the shot or marks the batch lane dead if the value is `1`.

For an ordinary single-shot sampler, rejection means immediately returning
`false` from the postselected execution function.

For a batch sampler, each SIMD lane is a shot. Rejection means marking the lane
dead. Dead lanes are then compacted out at selected points so later active-state
work does not continue to spend time on rejected shots.

## Shared Presampled Expression Layer

Both samplers use the shared packed expression module:

- `cpp/src/sampler/presampled_expression.hpp`
- `cpp/src/sampler/presampled_expression.cpp`

The main types are:

```cpp
PresampledExpressionPlan
PresampledExpressionBlock
```

The plan splits each instruction's symbolic expression into:

- an exogenous part, depending only on presampled exogenous symbols, and
- a residual part, depending on symbols produced during active execution.

For a chunk of shots, `evaluate_presampled_expression_block` evaluates the
exogenous parts across all shots as packed bit vectors. If expression `E2` is
similar to an earlier expression `E1`, the plan stores a parent-delta relation:

```text
E2 = E1 xor delta_constant xor xor(delta_conditions)
```

This avoids recomputing every expression from scratch. The expression block has
layout:

```text
expression_words[expression_index * shot_words + shot_word]
```

This layout is useful to both samplers:

- Single-shot execution reads one bit from a packed expression row.
- Batch execution slices packed expression rows into the current batch page and
  then treats them like compactable batch columns.

This layer is deliberately not in `single_shot.*` anymore because it is shared
infrastructure.

The prepared single-shot and batch circuit samplers own the prepared storage for
this layer. The sampled bits themselves are not reused: each call to
`sample(...)` refreshes presampled exogenous values for each chunk before those
values are consumed by shot or batch execution.

## Detector Instructions

Detector insertion lives in:

```text
cpp/src/frontend/stim.hpp
cpp/src/frontend/stim_sampling.cpp
```

The public Stim planning function is:

```cpp
plan_stim_factored_program(parsed)
```

It consumes the parsed Stim frontend state:

- the planned quantum/measurement `FactoredInstructionProgram`,
- the lowered `CircuitDetector`/`StimDetector` values returned by the frontend,
- the symbolic measurement-record expressions returned by lowering.

It returns a new `FactoredInstructionProgram` whose instruction stream contains
`RecordDetector` instructions at the detector source-order positions.

There is no separate single-shot detector postselection plan. The detector
instruction is the sampling instruction.

### Batch Metadata

Batch postselection still needs compaction metadata, but this is sampler-owned
state, not a caller-provided plan. The caller passes:

```cpp
BatchDetectorPostselectionOptions
```

The relevant option fields are:

- `mask_dead_shots_min_fraction_denominator`
- `retained_record_uses`

`retained_record_uses` is an optional pointer to record groups that must remain
available after execution, such as observable includes counted by a caller.

The last-use tables are batch-specific. They tell compaction which dynamic
symbol columns and measurement-record columns can still be read later.
Detector instructions read dynamic symbol columns through their
`outcome_plan`; they do not need measurement-record columns at runtime.

The sampler derives:

- `condition_last_use_by_index`
- `record_last_use_by_index`

It stores them in `BatchDetectorPostselectionScratch` and reuses them while the
same program and retained-record option are used for later batch pages.
Callers that want to prewarm reusable batch state can call
`prepare_batch_detector_postselection_scratch(scratch, runtime, program,
options)` before entering the hot sampling loop.

## Single-Shot Postselection

Single-shot postselection is implemented by:

```cpp
execute_postselected_in_place(...)
```

in `cpp/src/sampler/single_shot_sampler.cpp`.

There are two overloads:

- one uses scalar `PresampledExogenous`
- one uses `PresampledExpressionPlan` and `PresampledExpressionBlock`

The optimized streaming benchmark uses the expression overload.

The algorithm is:

```text
prepare runtime for the shot
for instruction:
    if instruction is RecordDetector:
        value = evaluate instruction.outcome_plan
        record detector value
        if postselection is enabled and value == 1:
            return false
    else:
        execute instruction
return true
```

Equivalently, the detector instruction is a classical instruction in the same
stream as rotations and measurements. Postselection only changes what happens
after the detector value is evaluated.

For the expression overload, instruction symbolic values come from:

```text
presampled expression bit for this shot
xor residual symbolic expression evaluated from runtime symbols
```

This applies to detector instructions as well as measurement instructions.

If a detector fires, the sampler stops the shot immediately. No suffix of the
circuit is executed for rejected shots. This is why single-shot postselection is
naturally efficient: it avoids all later active-state work for failed shots.

In no-postselection mode, detector instructions still execute and write
detector bits to `runtime.detector_words`.

## Batch Postselection

Batch postselection is implemented by:

```cpp
execute_batch_postselected_in_place(...)
```

in `cpp/src/sampler/batch_runtime.cpp`.

The public function delegates to the combined expression-backed implementation.
There are no separate raw-exogenous, suffix-fallback, tail-fill, or benchmark-only
postselection paths.

### Batch State

The prepared batch runtime stores each shot's dense active vector contiguously:

```text
active_re[shot * active_stride + basis]
active_im[shot * active_stride + basis]
```

The bit-packed dynamic symbol, measurement-record, and detector-record columns
use fixed `batch_words`, based on the batch capacity. The dense active-state
storage uses `active_stride`, based on the maximum active dimension for the
program.

This split matters:

- bit columns stay simple and fixed-width;
- active rotations and promotions can skip dead shots without touching their
  dense vectors;
- compaction is still available when later instructions need a survivor prefix.

### Expression Workspace

At the start of a batch page, batch postselection slices the shared
`PresampledExpressionBlock` into scratch columns:

```text
scratch.expression_words[expression_index * batch_words + word]
```

These expression columns are compacted together with the batch when lanes die.
At instruction execution time, `BatchExpressionEvaluator` reads the expression
column for the instruction and xors any residual runtime symbolic expression.

### Detector Handling

When the batch sampler reaches a `RecordDetector` instruction:

```text
value_bits = evaluate instruction.outcome_plan over live lanes
if postselection is enabled:
    mark lanes with value_bits == 1 as dead
```

Newly rejected lanes are accumulated into:

```text
scratch.dead_bits
scratch.dead_count
```

Symbolic detector values are computed through the same packed symbolic-expression
path as measurement outcomes. Record-parity detectors update the dead-lane mask
directly from measurement-record columns instead of materializing a temporary
detector record column.

### Deferred Compaction

The sampler does not compact after every detector. It marks lanes dead and
compacts before later work when it is necessary.

Compaction is forced before instructions that cannot safely execute over dead
lanes, such as active measurements and dormant measurement branches.

For pure-over-dead instructions, such as rotations, promotions, record-only
symbolic writes, and detector instructions, the shot-major path keeps the
dead-lane mask. Dense rotations and promotions skip dead shots; cheap classical
instructions may still execute over the full page.

For lower-level basis-major execution, pure-over-dead compaction remains
thresholded. The public option is:

```cpp
BatchDetectorPostselectionOptions::mask_dead_shots_min_fraction_denominator
```

The default is `2`.

### What Compaction Moves

When compaction runs, it computes:

```text
keep_bits = live lanes not marked dead
live_sources[dst] = old lane source for new live lane dst
```

Then it compacts all state needed by later execution:

- dense active amplitudes;
- live dynamic symbol columns;
- live measurement-record columns;
- live detector-record columns;
- live presampled expression columns.

Symbol columns are compacted only when they are assigned and still have a future
use. Measurement columns are compacted only if they still have a retained future
use, such as observable counting. Detector instructions use symbolic condition
columns, not measurement-record columns, so detectors do not force measurement
records to remain live. Expression columns are compacted only if their block
expression can still be read by a future instruction.

This is why the batch plan needs condition, record, and expression last-use
metadata.

### End of Batch Page

After all instructions, any remaining dead lanes are compacted one last time.
The function returns:

```cpp
BatchDetectorPostselectionResult {
    discarded,
    accepted
}
```

At this point `runtime.active_shots` is the number of surviving shots, and
measurement and detector records are packed in survivor-prefix order.

## Prepared Sampler API

The high-level C++ entry points are:

```cpp
auto single = prepare_single_shot_sampler_from_stim_file(path, options);
auto batch = prepare_batch_sampler_from_stim_file(path, options);

auto result = single.sample(shots);
auto result = batch.sample(shots);
```

Construction is the preprocessing phase. A frontend helper such as
`prepare_single_shot_sampler_from_stim_file` parses a Stim circuit, asks the
Stim frontend to build a `FactoredInstructionProgram` with detector
instructions via `plan_stim_factored_program`, and collects observable record
uses for result counting. The returned prepared sampler itself is frontend
neutral: it owns the internal program, reusable executor state, postselection
scratch, and packed exogenous/expression storage.

Sampling is the consumptive phase. Each `sample(...)` call refreshes
presampled exogenous values chunk by chunk, executes the planned instruction
stream, handles postselection inside detector instruction execution, and
accumulates accepted/discarded/logical counts.

The benchmark is not the postselection implementation. It constructs one of
these prepared sampler objects, calls `sample(...)`, and prints the returned
counts/timings. It does not group detectors by record, compute detector
checkpoints, compute detector values, prepare exogenous storage, compute batch
last-use tables, or count logical observables itself.

For batch postselection, `PreparedCircuitBatchSampler` calls:

```cpp
execute_batch_postselected_in_place(
    runtime,
    program,
    expression_plan,
    expression_block,
    first_sample_shot,
    postselection_scratch,
    postselection_options)
```

The compaction policy, detector checking, expression evaluation, and survivor
ordering are all sampler-owned behavior.

## Why Batch Postselection Needed More Design Than Single Shot

Single-shot postselection has an easy win: when a detector fires, stop executing
that shot.

Batch postselection cannot simply stop the whole batch when one lane fails. It
must keep surviving lanes and avoid wasting active-state work on dead lanes.
This creates a tension:

- compact too often, and lane movement dominates;
- compact too rarely, and active rotations/promotions run over rejected lanes.

The current design balances this with:

- packed detector instruction evaluation;
- dead-lane masks;
- cost-aware deferred compaction;
- last-use-aware compaction of only needed columns;
- shot-major active storage;
- dead-shot skipping for active rotations and promotions;
- packed expression evaluation with parent-delta reuse.

The key performance improvement came from treating postselection as part of the
batch sampler state transition, not as a benchmark-side filter. Once dead lanes
are represented inside the batch runtime, the sampler can compact all dependent
state consistently and continue with a dense live prefix.

## Correctness Invariants

The postselection implementation preserves these invariants:

- Detector instructions occur at the detector's planned source-order point.
- A detector instruction's value is the same symbolic parity as the
  corresponding Stim detector.
- A fired detector rejects the shot permanently in postselection mode.
- In no-postselection mode, detector values are recorded in detector bit
  storage.
- For batch postselection, survivor order is stable under prefix compaction.
- Active amplitudes, dynamic symbols, measurement records, detector records, and
  expression columns are compacted together.
- Future symbolic evaluations see the same values they would have seen if
  rejected shots had never been present.
- Measurement records for accepted survivors remain available for observable
  counting when passed as retained record uses.

## Current Limitations

- The batch compaction policy is tuned for the current active kernels and may
  need revisiting if active-state layout or kernels change.
