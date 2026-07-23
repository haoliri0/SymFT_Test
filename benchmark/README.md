# SymFT benchmark circuits

This directory contains circuit inputs for the paper's near-Clifford and
pure-Clifford result sections.  The corpus is intentionally limited to circuit
generation and metadata; it does not contain timing results or hardware claims.

The machine-readable inventory is [`circuit/manifest.json`](circuit/manifest.json).
All circuits use physical error rate `p = 1e-3`.

## Important: two circuit dialects

The `.stim` suffix describes the text format, not universal compatibility with
Stim.

- `pure_surface_*` are genuine Stim circuits.  They contain only Clifford
  operations, Pauli noise, measurements, detectors, and observables.  Use these
  for SymFT/Stim/Clifft/Tsim comparisons.
- `msc_*`, `msc_proxy_*`, and `coherent_surface_*` use the extended
  SymFT/Clifft Stim-like dialect.  They contain `T`, `T_DAG`, or `R_Z`, so **Stim
  cannot parse or sample them**.  Use these for near-Clifford simulators only.

SymFT and Clifft specify rotation angles in half-turns.  Therefore
`R_Z(0.02)` means an angle of `0.02*pi` radians.

## Circuit matrix

| Family | Configurations | Intended comparison | Status |
| --- | --- | --- | --- |
| Magic-state cultivation | `d=3`, `d=5` | SymFT, SOFT, Clifft, Tsim; CPU/GPU | Canonical Clifft inject+cultivate circuits |
| MSC performance proxy | `d=7` | SymFT/other extended-dialect tools; FP64 only | Unverified performance workload |
| Coherent surface code | `(d,r)=(3,1),(3,3),(5,1),(5,5)` | SymFT, Clifft, Tsim; not Stim | Stim scaffold with coherent `R_Z` noise transform |
| Pure-Clifford surface code | `(d,r)=(7,7),(9,9)` | SymFT, Stim, Clifft, Tsim | Generated and validated by Stim |

The same circuit should be used for CPU and GPU runs.  Precision is a backend
setting, not a circuit property.  The proposed result matrix should include
both FP32 and FP64 for `d=3`, but only FP64 for the `d=5` and `d=7` MSC cases;
FP32 is not expected to provide adequate numerical accuracy there.

## Magic-state-cultivation provenance

`msc_d3_inject_cultivate_p1e-3.stim` and
`msc_d5_inject_cultivate_p1e-3.stim` are byte-for-byte copies of Clifft's
published `cultivation_d3.stim` and `cultivation_d5.stim` at commit
`db7dc9f13a2c2854690e92390c779048a1ac1400`.  Their SHA-256 hashes are pinned in
the generator.  Clifft's construction is based on Gidney, Shutty, and Jones'
magic-state-cultivation code at commit
`871e68ff6df2f75190b1bfd6351459d1b5a037e3`.

### Why the distance-7 file is a proxy

Gidney's released generator has specialized constructions only for `d=3` and
`d=5`.  In particular, the `d=5` construction includes hand-written
`d=3 -> d=5` growth, double-cat verification, feed-forward corrections, and
detector-healing rules.  There is no parameter-only route to a correct `d=7`
cultivation circuit.

`msc_proxy_d7_unverified_p1e-3.stim` is therefore deliberately named as a
proxy.  It was constructed as follows:

1. Start from the canonical `d=5` unitary inject+cultivate construction.
2. Re-express the logical-Y flow for Gidney's generic growth primitive.
3. Apply generic color-code growth from `d=5` to `d=7`.
4. Apply five distance-7 superdense syndrome cycles.
5. Apply the same S-to-T conversion/noise conventions used by Clifft.

This gives a 76-qubit, 310-measurement, 299-detector workload with an MSC-like
active-space profile.  It does **not** contain a newly derived and validated
distance-7 double-cat gadget.  Do not use its detector statistics, acceptance
rate, or logical observable for correctness or logical-error claims.  It is
only suitable for the explicitly qualified performance experiment described
in the paper draft.

## Coherent-noise construction

Each coherent-noise file begins with Stim's
`surface_code:rotated_memory_z` circuit using the same four noise settings as
the pure-Clifford files:

```text
after_clifford_depolarization = 1e-3
after_reset_flip_probability = 1e-3
before_measure_flip_probability = 1e-3
before_round_data_depolarization = 1e-3
```

Following Clifft's benchmark, every generated `DEPOLARIZE1(...)` and
`DEPOLARIZE2(...)` instruction is replaced by `R_Z(0.02)` on the same targets.
Reset and measurement flip errors remain stochastic.  The `d=5,r=5` case is
already extremely expensive for exact near-Clifford simulation, so no
coherent `d=7` case is included.

## Regeneration and checks

Run from the repository root using the local environment:

```bash
.venv/bin/python benchmark/generate_circuits.py
```

This deterministically regenerates all pure-Clifford and coherent-noise files,
checks the pinned canonical MSC inputs, and refreshes the manifest.  The MSC
files are vendored because Stim cannot generate or parse their non-Clifford
instructions.

For a read-only freshness check suitable for CI:

```bash
.venv/bin/python benchmark/generate_circuits.py --check
```

The generator parses every pure-Clifford output with Stim and constructs its
detector error model.  Extended-dialect files must additionally be parsed or
compiled with SymFT/Clifft; passing the pure Stim validation does not validate
the non-Clifford semantics.

To verify SymFT parsing and planning without allocating an active-state vector
or sampling any shots, build and run the dedicated planner tool:

```bash
cmake -S . -B build -DSYMFT_CPP_ENABLE_CUDA=OFF
cmake --build build --target symft_plan
build/cpp/symft_plan benchmark/circuit/coherent_surface_d5_r5_p1e-3_rz0p02.stim
```

`symft_plan` reports parse time, planning time, maximum active width, pending
operation counts before and after optimization, rotation fusions, measurement
left-swaps, and peak resident memory. Use this tool—not `symft_bench` with zero
shots—for large-width preprocessing checks, because benchmark executors
allocate active state storage independently of the requested shot count.

## Single-threaded Tsim rate harness

`tsim_rate_bench.py` benchmarks Tsim's public detector sampler on its CPU
backend while forcing Eigen, BLAS, OpenMP, and related library thread counts to
one. It reports parsing, sampler construction, and fixed-shape JAX warmup
separately from the measured repetitions. The `sample_s_*` timings include
Tsim's full Boolean detector/observable output materialization and the
host-side reduction to the same discarded, accepted, and logical-error counts
reported by `symft_rate_bench`.

For the paper's distance-three cultivation configuration:

```bash
.venv/bin/python benchmark/tsim_rate_bench.py \
    --circuit benchmark/circuit/msc_d3_inject_cultivate_p1e-3.stim \
    --shots 128 \
    --batch-size 128 \
    --warmups 1 \
    --repeats 7 \
    --postselect-detectors \
    --cpu-affinity 0
```

Omit `--cpu-affinity` when logical CPU 0 is unavailable, or select a CPU from
the process's allowed affinity set. `sample_shots_per_s` follows
`symft_rate_bench` and uses the mean repetition time;
`sample_shots_per_s_median` is the seven-repetition paper metric. The reported
process-CPU/wall-time ratio should remain near one for a valid single-thread
run.

Tsim 0.1.4 does not expose in-sampler detector postselection. Consequently,
`--postselect-detectors` performs the rejection/count reduction on the
materialized output inside the timed region; it does not let Tsim skip rejected
shots. The script labels this as `postselection_implementation
host_output_reduction`.

## Benchmarking notes

- Keep compilation/preprocessing time separate from steady-state sampling
  throughput, and compile once before repeated sampling.
- Use identical detector postselection and output contracts across tools.
- Report attempted shots/s and, when postselecting, survivor rate separately.
- Record simulator commit, compiler flags, thread affinity, CPU/GPU model, and
  FP32/FP64 mode with every result.
- Treat the `d=7` MSC proxy label and correctness limitation as part of every
  table or figure caption that includes it.

## References and license

- [Clifft paper benchmark circuits](https://github.com/unitaryfoundation/clifft-paper/tree/main/qec_bench/circuits)
- [Magic state cultivation](https://github.com/Strilanc/magic-state-cultivation)
- [Stim circuit generation API](https://github.com/quantumlib/Stim)

The two canonical MSC inputs and the construction used for the proxy derive
from the Apache-2.0-licensed Clifft paper repository.  A copy of that license is
provided in [`LICENSE-Clifft-paper`](LICENSE-Clifft-paper).
