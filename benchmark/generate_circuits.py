#!/usr/bin/env python3
"""Generate and validate the SymFT paper benchmark circuit corpus.

Stim generates the pure-Clifford surface-code circuits.  The coherent-noise
variants are produced by replacing Stim's one- and two-qubit depolarization
instructions with SymFT/Clifft ``R_Z`` instructions over the same targets.

The magic-state-cultivation circuits are vendored inputs, not Stim circuits:
Stim cannot parse their ``T``/``T_DAG`` instructions.  They are copied into a
custom output directory when ``--output-dir`` is used, and are otherwise
checked in place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import stim


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "circuit"

PHYSICAL_ERROR_RATE = 1e-3
COHERENT_RZ_HALF_TURNS = 0.02

# The two canonical MSC files are byte-for-byte copies of the corresponding
# Clifft paper circuits at commit db7dc9f13a2c2854690e92390c779048a1ac1400.
CANONICAL_MSC_SHA256 = {
    "msc_d3_inject_cultivate_p1e-3.stim": (
        "90a7d841e003e5ee38137cd9a3eb6529bb552e49c424bc6b0932a27d97cdb41f"
    ),
    "msc_d5_inject_cultivate_p1e-3.stim": (
        "c2b4566917bd9bf27a5705284dac02700ef0dcc7c03c91066670db376d633a6d"
    ),
}

VENDORED_NEAR_CLIFFORD = (
    "msc_d3_inject_cultivate_p1e-3.stim",
    "msc_d5_inject_cultivate_p1e-3.stim",
    "msc_proxy_d7_unverified_p1e-3.stim",
)

PURE_SURFACE_CONFIGS = (
    (7, 7),
    (9, 9),
)

# This is Clifft's coherent-noise matrix.  d=5, r=5 is already a very hard
# exact workload (its published peak active width is 24), so a coherent d=7
# circuit is intentionally not included.
COHERENT_SURFACE_CONFIGS = (
    (3, 1),
    (3, 3),
    (5, 1),
    (5, 5),
)


@dataclass(frozen=True)
class CircuitMetadata:
    file: str
    family: str
    dialect: str
    distance: int
    rounds: int | None
    physical_error_rate: float
    coherent_rz_half_turns: float | None
    correctness_status: str
    qubits: int
    measurements: int
    detectors: int
    observables: int
    lines: int
    bytes: int
    sha256: str


def _surface_code(distance: int, rounds: int) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=PHYSICAL_ERROR_RATE,
        after_reset_flip_probability=PHYSICAL_ERROR_RATE,
        before_measure_flip_probability=PHYSICAL_ERROR_RATE,
        before_round_data_depolarization=PHYSICAL_ERROR_RATE,
    )


def _coherent_noise_text(circuit: stim.Circuit) -> str:
    """Replace stochastic gate depolarization by coherent Z rotations.

    Angles follow the SymFT/Clifft convention and are measured in half-turns,
    so R_Z(0.02) means a rotation angle of 0.02*pi radians.
    """
    lines: list[str] = []
    for line in str(circuit).splitlines():
        stripped = line.strip()
        if stripped.startswith(("DEPOLARIZE1(", "DEPOLARIZE2(")):
            targets = stripped.split(")", maxsplit=1)[1].strip()
            indent = line[: len(line) - len(line.lstrip())]
            lines.append(f"{indent}R_Z({COHERENT_RZ_HALF_TURNS}) {targets}")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _extended_text_stats(text: str) -> tuple[int, int, int, int]:
    """Count basic metadata without asking Stim to parse extended gates."""
    max_qubit = -1
    measurements = 0
    detectors = 0
    max_observable = -1

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        coord_match = re.match(r"QUBIT_COORDS\([^)]*\)\s+(\d+)$", line)
        if coord_match:
            max_qubit = max(max_qubit, int(coord_match.group(1)))

        op = line.split(maxsplit=1)[0].split("(", maxsplit=1)[0]
        operands = line.split(maxsplit=1)[1] if " " in line else ""
        if op in {"M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ"}:
            measurements += len(operands.split())
        elif op == "MPP":
            measurements += len(operands.split())
        elif op == "DETECTOR":
            detectors += 1
        elif op == "OBSERVABLE_INCLUDE":
            obs_match = re.match(r"OBSERVABLE_INCLUDE\((\d+)\)", line)
            if obs_match:
                max_observable = max(max_observable, int(obs_match.group(1)))

    return max_qubit + 1, measurements, detectors, max_observable + 1


def _metadata(
    *,
    filename: str,
    text: str,
    family: str,
    dialect: str,
    distance: int,
    rounds: int | None,
    coherent_rz_half_turns: float | None,
    correctness_status: str,
    stats: tuple[int, int, int, int],
) -> CircuitMetadata:
    qubits, measurements, detectors, observables = stats
    return CircuitMetadata(
        file=filename,
        family=family,
        dialect=dialect,
        distance=distance,
        rounds=rounds,
        physical_error_rate=PHYSICAL_ERROR_RATE,
        coherent_rz_half_turns=coherent_rz_half_turns,
        correctness_status=correctness_status,
        qubits=qubits,
        measurements=measurements,
        detectors=detectors,
        observables=observables,
        lines=len(text.splitlines()),
        bytes=len(text.encode()),
        sha256=_sha256(text),
    )


def _stim_stats(circuit: stim.Circuit) -> tuple[int, int, int, int]:
    return (
        circuit.num_qubits,
        circuit.num_measurements,
        circuit.num_detectors,
        circuit.num_observables,
    )


def _write_or_check(path: Path, text: str, *, check: bool) -> None:
    if check:
        if not path.exists():
            raise FileNotFoundError(f"Missing generated circuit: {path}")
        actual = path.read_text()
        if actual != text:
            raise RuntimeError(
                f"Generated circuit is stale: {path}\n"
                "Run benchmark/generate_circuits.py to refresh it."
            )
        return
    path.write_text(text)


def _load_vendored_near_clifford(
    filename: str,
    *,
    output_dir: Path,
    check: bool,
) -> str:
    source = DEFAULT_OUTPUT_DIR / filename
    if not source.exists():
        raise FileNotFoundError(
            f"Missing vendored near-Clifford input: {source}. "
            "These files cannot be generated by Stim."
        )

    text = source.read_text()
    expected_hash = CANONICAL_MSC_SHA256.get(filename)
    if expected_hash is not None and _sha256(text) != expected_hash:
        raise RuntimeError(
            f"Canonical Clifft circuit changed unexpectedly: {source}"
        )

    destination = output_dir / filename
    if output_dir != DEFAULT_OUTPUT_DIR:
        if check:
            if not destination.exists() or destination.read_text() != text:
                raise RuntimeError(f"Vendored circuit is stale or missing: {destination}")
        else:
            shutil.copyfile(source, destination)
    return text


def generate(*, output_dir: Path, check: bool, validate: bool) -> list[CircuitMetadata]:
    if not check:
        output_dir.mkdir(parents=True, exist_ok=True)
    elif not output_dir.is_dir():
        raise FileNotFoundError(f"Circuit directory does not exist: {output_dir}")

    records: list[CircuitMetadata] = []

    for distance, rounds in PURE_SURFACE_CONFIGS:
        circuit = _surface_code(distance, rounds)
        text = str(circuit)
        if not text.endswith("\n"):
            text += "\n"
        filename = f"pure_surface_d{distance}_r{rounds}_p1e-3.stim"
        path = output_dir / filename
        _write_or_check(path, text, check=check)
        if validate:
            parsed = stim.Circuit(text)
            parsed.detector_error_model()
        records.append(
            _metadata(
                filename=filename,
                text=text,
                family="pure_clifford_surface_code",
                dialect="stim",
                distance=distance,
                rounds=rounds,
                coherent_rz_half_turns=None,
                correctness_status="stim_generated_and_validated",
                stats=_stim_stats(circuit),
            )
        )

    for distance, rounds in COHERENT_SURFACE_CONFIGS:
        scaffold = _surface_code(distance, rounds)
        text = _coherent_noise_text(scaffold)
        filename = (
            f"coherent_surface_d{distance}_r{rounds}_p1e-3_rz0p02.stim"
        )
        _write_or_check(output_dir / filename, text, check=check)
        records.append(
            _metadata(
                filename=filename,
                text=text,
                family="near_clifford_coherent_surface_code",
                dialect="symft_clifft_extended_stim",
                distance=distance,
                rounds=rounds,
                coherent_rz_half_turns=COHERENT_RZ_HALF_TURNS,
                correctness_status="stim_scaffold_text_transform",
                stats=_stim_stats(scaffold),
            )
        )

    for filename in VENDORED_NEAR_CLIFFORD:
        text = _load_vendored_near_clifford(
            filename,
            output_dir=output_dir,
            check=check,
        )
        distance = int(re.search(r"_d(\d+)_", filename).group(1))
        records.append(
            _metadata(
                filename=filename,
                text=text,
                family=(
                    "msc_inject_cultivate"
                    if distance in {3, 5}
                    else "msc_performance_proxy"
                ),
                dialect="symft_clifft_extended_stim",
                distance=distance,
                rounds=None,
                coherent_rz_half_turns=None,
                correctness_status=(
                    "canonical_clifft_circuit"
                    if distance in {3, 5}
                    else "unverified_performance_only"
                ),
                stats=_extended_text_stats(text),
            )
        )

    records.sort(key=lambda record: record.file)
    manifest_text = json.dumps(
        {
            "schema_version": 1,
            "stim_version": stim.__version__,
            "angle_convention": "half_turns (angle * pi = radians)",
            "circuits": [asdict(record) for record in records],
        },
        indent=2,
        sort_keys=False,
    ) + "\n"
    _write_or_check(output_dir / "manifest.json", manifest_text, check=check)
    return records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Circuit output directory (default: benchmark/circuit).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if checked-in generated files differ; do not write files.",
    )
    parser.add_argument(
        "--skip-stim-validation",
        action="store_true",
        help="Skip Stim parsing/DEM validation of the pure-Clifford files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    records = generate(
        output_dir=args.output_dir.resolve(),
        check=args.check,
        validate=not args.skip_stim_validation,
    )
    action = "Checked" if args.check else "Generated"
    print(f"{action} {len(records)} circuits in {args.output_dir}")
    for record in records:
        print(
            f"  {record.file}: q={record.qubits}, "
            f"m={record.measurements}, d={record.detectors}, "
            f"sha256={record.sha256[:12]}..."
        )


if __name__ == "__main__":
    main()
