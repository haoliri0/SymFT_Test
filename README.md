# SymFT

SymFT is an exact, high-throughput simulator for noisy adaptive Clifford-dominated circuits.
It supports Pauli rotations, stochastic Pauli noise, mid-circuit Pauli measurements, measurement-record-controlled Pauli feedback, detectors, observables, and postselection through a Stim-style circuit frontend.

SymFT is the second-generation successor to SOFT. 
It replaces per-shot sparse generalized-stabilizer evolution with symbolic Clifford–Pauli frame factorization shared across shots and adaptive stabilizer-coordinate planning.
The implementation provides C++20 CPU and CUDA GPU backends and a typed Python package.

- [Draft paper: *SymFT: Universal Fault-Tolerant Quantum Circuit Simulation via Symbolic Clifford–Pauli Frames and Stabilizer Coordinates*]()
- [Detailed Python interface documentation](python/README.md)

## How SymFT works

SymFT combines symbolic Clifford–Pauli frame factorization with adaptive stabilizer-coordinate planning.

### 1. Symbolic Clifford–Pauli frame factorization

For each noise assignment and measurement record, SymFT factorizes the branch operator, up to global phase, into a concrete Clifford frame, a symbolic Pauli frame, and an ordered sequence of pulled-back Pauli rotations and measurement projectors.
The residual Clifford and Pauli frames are unitary, so they do not affect the branch probability.

Clifford gates, Pauli noise, and Pauli feedback therefore do not have to be replayed for every shot.
Clifford conjugation is resolved once, while noise and feedback remain as affine symbolic signs on the pulled-back rotations and projectors.

### 2. Adaptive stabilizer coordinates

A one-time planner tracks a shared stabilizer–destabilizer tableau that defines the stabilizer-coordinate basis.
Some coordinates are active and the rest are dormant.
A dynamically sized dense active-state vector stores one coefficient for each active-basis state, so its size is controlled by the active width rather than directly by the number of physical qubits.

Dormant components are handled through tableau updates and symbolic Pauli corrections.
Active components become specialized diagonal or paired-amplitude instructions.
A planned instruction stores only compact Pauli masks, phase, pivot, and angle;
basis-dependent coefficients and measurement source indices are derived during execution instead of being retained as per-instruction exponential tables.
Consecutive active rotations execute shot by shot so each shot's dense active state remains cache-resident while the instruction sequence is applied.
Non-diagonal active measurements use runtime-dispatched AVX2/AVX-512 kernels that derive coefficient signs from the compact Pauli masks directly in SIMD registers, without allocating a coefficient table.
A non-Clifford rotation can promote a dormant coordinate, while an active measurement can move a coordinate to the dormant set.
SymFT executes multi-coordinate Pauli operations directly and does not add a runtime Clifford-localization pass over the dense active-state vector.

### 3. Compile once, sample many times

The sampler executes only the emitted instruction stream.
It samples independent symbols, evaluates causal affine expressions, updates the dense active-state vector, projects active measurements, records outcomes, and accumulates detector and observable parities.
It does not revisit the original circuit, update Pauli frames, conjugate Pauli strings, or reconstruct the planning tableau.

The method avoids a full physical state vector. 
Its dominant exponential cost is controlled by the peak active width rather than the number of physical qubits, although the worst case remains exponential.

## Results reported in the draft

The draft reports attempted-shot throughput, including noise generation,
quantum sampling, detector and observable evaluation, result readback, and
detector postselection where applicable. Parsing and one-time preprocessing are
excluded. CPU results use one pinned physical core; GPU results use a saturated
NVIDIA GeForce RTX 4090. Each reported rate is the median of seven repetitions.

| Workload | SymFT | Comparison | Speedup |
| --- | ---: | ---: | ---: |
| 118-qubit pure-Clifford QEC, CPU FP64 | 2.032 million shots/s | Stim: 748,700 shots/s | 2.715× |
| Distance-3 cultivation, CPU FP64 | 1.881 million shots/s | Clifft FP64: 454,600 shots/s | 4.138× |
| Distance-5 cultivation, CPU FP64 | 167,300 shots/s | Clifft FP64: 56,480 shots/s | 2.961× |
| Distance-3 cultivation, CPU | 1.881 million shots/s | Tsim native: 41.0 shots/s | 45,876× |
| Distance-3 cultivation, GPU FP32 | 81.72 million shots/s | Tsim native: 23,200 shots/s | 3,522× |
| Distance-3 cultivation, GPU FP64 | 68.59 million shots/s | SOFT FP64: 336,600 shots/s | 203.8× |
| Distance-5 cultivation, GPU FP64 | 2.777 million shots/s | SOFT FP64: 7,172 shots/s | 387.3× |

These are public-interface comparisons, not kernel-only comparisons. Output
contracts differ between implementations, and the Tsim native lane uses lower
precision than SymFT's CPU FP64 lane. Tsim did not complete distance-5
preprocessing within the draft's cutoff. See the paper for the full protocol,
hardware details, precision lanes, output contracts, and outcome checks.

## Supported circuit model

The frontend accepts a substantial Stim-style subset:

- Clifford gates, including single-qubit axis variants, controlled Pauli
  gates, `SWAP`, `ISWAP`, and `SQRT_XX`/`SQRT_YY`/`SQRT_ZZ` variants;
- non-Clifford gates and rotations including `T`, `T_DAG`, `R_X`/`R_Y`/`R_Z`,
  `R_XX`/`R_YY`/`R_ZZ`, `R_PAULI`, `U`/`U3`, and `SPP`/`SPP_DAG`;
- `M`/`MX`/`MY`, reset and measure-reset operations, `MPP`,
  `MXX`/`MYY`/`MZZ`, inverted measurement targets, and `MPAD`;
- `X_ERROR`/`Y_ERROR`/`Z_ERROR`, depolarizing and Pauli channels up to three
  qubits, correlated-error chains, heralded erasure, and heralded Pauli noise;
- measurement-record-controlled Pauli feedback;
- `REPEAT`, coordinates, `DETECTOR`, and `OBSERVABLE_INCLUDE`.

This is not a drop-in parser for the complete Stim language. In particular,
sweep-controlled operations are rejected. Unsupported instructions or invalid
target and parameter combinations raise an error with source-line context.
Following Clifft, rotation angles are specified in half-turns. Multiply an
angle parameter by $\pi$ to obtain radians.

## Python installation

Requirements:

- Python 3.9 or newer;
- NumPy 1.20 or newer;
- a C++20 compiler.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./python
```

The Python build compiles the C++ sources directly; CMake is not required for
the extension.

## Python quick start

Construct a circuit from text and return per-shot measurement records:

```python
import symft

circuit = symft.Circuit("""
H 0
T 0
M 0
OBSERVABLE_INCLUDE(0) rec[-1]
""")

samples = circuit.sample(shots=1_000, seed=123, batch=True)
print(samples.shape)  # (1000, 1)
print(samples.dtype)  # bool
```

Load a `.stim` file:

```python
circuit = symft.Circuit(path="benchmark/circuit/msc_d3_inject_cultivate_p1e-3.stim")
# Equivalent:
circuit = symft.read_stim_file("benchmark/circuit/msc_d3_inject_cultivate_p1e-3.stim")
```

Compile a reusable sampler when the same circuit will be sampled repeatedly:

```python
sampler = circuit.compile_sampler(batch=True)

first = sampler.sample(shots=10_000, seed=1)
second = sampler.sample(shots=10_000, seed=2)

print(sampler.max_active_qubits)
```

`sample` returns a two-dimensional `numpy.bool_` array with shape
`(shots, num_measurements)`. With `bit_packed=True`, it instead returns
`numpy.uint8` with shape `(shots, ceil(num_measurements / 8))`; record zero is
the least-significant bit of byte zero.

## Detectors and logical-error counts

Use `sample_detectors` when every detector bit is needed:

```python
detectors = circuit.sample_detectors(shots=1_000, seed=1)
print(detectors.shape)  # (1000, circuit.num_detectors)
```

Use `sample_counts` for high-throughput aggregation without materializing all
per-shot records:

```python
result = circuit.sample_counts(
    shots=1_000_000,
    seed=1,
    observable=0,
    batch=True,
    threads=4,
    postselect_detectors=True,
)

print(result["discard_rate"])
print(result["logical_error_rate"])
print(result["active_threads"])
print(result["timing"])
```

A shot is discarded if any detector fires. The selected
`OBSERVABLE_INCLUDE` records are combined by parity, and an accepted shot with
observable parity one is counted as a logical error. Consequently,
`accepted + discarded == shots`, `discard_rate == discarded / shots`, and
`logical_error_rate == logical_errors / accepted`. A rate is `nan` when its
denominator is zero.

For repeated counts jobs, reuse the prepared program and worker storage:

```python
sampler = circuit.compile_counts_sampler(
    batch=True,
    observable=0,
    postselect_detectors=True,
    threads=4,
)

run_a = sampler.sample(shots=1_000_000, stream_id=10)
run_b = sampler.sample(shots=1_000_000, stream_id=11)
print(sampler.info)
```

An explicit `seed` or `stream_id` makes a given execution path reproducible.
Single-shot and batch paths use different random-number partitioning, so they
should be compared statistically rather than expected to produce identical
bit streams.

The complete API, metadata schema, batching controls, error behavior, and type
signatures are documented in the [Python interface guide](python/README.md).
The installed package also includes `py.typed` and `.pyi` files.

## Execution backends

The C++ implementation provides:

- a single-shot CPU sampler;
- a batch CPU sampler with packed symbol, measurement, and detector columns;
- multithreaded chunk execution for batch counts;
- exact early detector postselection and live-shot compaction;
- scalar kernels and CMake-built AVX2/AVX-512 kernels with runtime dispatch;
- an optional CUDA counts backend.

The batch sampler stores each shot's dense active-state vector contiguously:
`active_re[shot * active_stride + basis]` and the corresponding imaginary
array. The automatic batch size is limited by the prepared program's peak
active width to control the active-state-vector cache footprint.

Inspect the active backends from Python:

```python
print(symft.active_simd_backend())
print(symft.active_batch_backend())
print(symft.cuda_enabled())
print(symft.active_cuda_backend())
```

Backend names depend on the build and host. `cuda_enabled()` reports whether
CUDA support was compiled into the extension; it does not guarantee that a
compatible device is available at runtime.

### Optional CUDA Python build

With a CUDA toolkit and `nvcc` available:

```bash
SYMFT_PY_ENABLE_CUDA=1 python -m pip install -e ./python
```

Then select the CUDA counts sampler with `cuda=True`:

```python
result = circuit.sample_counts(shots=1_000_000, cuda=True)
sampler = circuit.compile_counts_sampler(cuda=True)
```

Set `CUDA_HOME` or `CUDA_PATH` when the toolkit is not discoverable. Optional
build controls include `SYMFT_PY_CUDA_ARCH`, `SYMFT_PY_CUDA_NVCC_FLAGS`, and
`SYMFT_PY_CUDA_REAL_DOUBLE=1`. See the
[Python interface guide](python/README.md#cuda-counts-backend) for CUDA
execution modes.

## C++ build

CMake 3.20 or newer and a C++20 compiler are required:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSYMFT_CPP_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run the simple circuit summary tool:

```bash
./build/cpp/symft_cli benchmark/circuit/msc_d3_inject_cultivate_p1e-3.stim 1000
```

Run the configurable single-shot/batch rate harness:

```bash
./build/cpp/symft_rate_bench \
  --sampler both \
  --circuit benchmark/circuit/msc_d3_inject_cultivate_p1e-3.stim \
  --shots 1000000 \
  --batch-size auto \
  --threads 1
```

The library target is `symft_cpp`, and its public source-tree headers are under
`cpp/src`. The prepared counts API can be used directly:

```cpp
#include "frontend/stim_prepared_sampler.hpp"

#include <cstdint>
#include <iostream>

int main() {
    symft::CircuitSamplingOptions options;
    options.observable = 0;
    options.threads = 4;
    options.postselect_detectors = true;

    auto sampler =
        symft::prepare_batch_sampler_from_stim_file(
            "benchmark/circuit/msc_d3_inject_cultivate_p1e-3.stim", options);
    auto run = sampler.sample(1'000'000, std::uint64_t{1});

    std::cout << run.counts.discarded << '\n';
    std::cout << run.counts.logical_errors << '\n';
}
```

Enable the C++ CUDA target with:

```bash
cmake -S . -B build-cuda \
  -DSYMFT_CPP_ENABLE_CUDA=ON \
  -DSYMFT_CPP_CUDA_REAL_DOUBLE=OFF
cmake --build build-cuda -j
```

## Development

Run the C++ tests:

```bash
cmake -S . -B build -DSYMFT_CPP_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run the Python tests:

```bash
cd python
python setup.py build_ext --inplace
PYTHONPATH=src python -m unittest discover -s tests -v
```

Repository layout:

```text
cpp/src/core/       Pauli algebra, symbolic expressions, and frames
cpp/src/circuit/    Circuit IR and lowering
cpp/src/factored/   Stabilizer-coordinate state and planner
cpp/src/sampler/    Single-shot, batch, and prepared samplers
cpp/src/frontend/   Stim-style parser and sampling frontend
cpp/src/simd/       Scalar and CPU SIMD kernels
cpp/src/cuda/       Optional CUDA sampler
python/src/symft/   Native Python binding, type hints, and package API
benchmark/          Stim fixtures and benchmark inputs
cpp/tests/          C++ correctness tests
python/tests/       Python interface tests
```
