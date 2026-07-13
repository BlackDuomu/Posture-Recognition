"""Generate deterministic synthetic IMU fixtures for code-path testing.

The generated values imitate a simple swing-shaped signal, but their physical
units have not been verified against a real device.  They must not be used as
evidence of model accuracy or real-IMU performance.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np


IMU_COLUMNS = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "mag_x",
    "mag_y",
    "mag_z",
)
METADATA_COLUMNS = ("synthetic", "source")
DEFAULT_LENGTH = 100
DEFAULT_SEED = 20260713
SYNTHETIC_SOURCE = "SYNTHETIC_GENERATED_SWING"
ZERO_SOURCE = "SYNTHETIC_ZERO_PLACEHOLDER"

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = _PROJECT_ROOT / "tests" / "fixtures" / "synthetic_imu.csv"
DEFAULT_ZERO_OUTPUT = _PROJECT_ROOT / "tests" / "fixtures" / "zero_imu.csv"


def _validate_parameters(length: int, seed: int) -> None:
    if length <= 0:
        raise ValueError("length must be greater than zero")
    if seed < 0:
        raise ValueError("seed must be non-negative")


def _gaussian_pulse(phase: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((phase - center) / width) ** 2)


def generate_synthetic_imu(
    length: int = DEFAULT_LENGTH,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Return a deterministic ``(length, 9)`` swing-shaped float32 array."""
    _validate_parameters(length, seed)
    rng = np.random.default_rng(seed)
    phase = np.linspace(0.0, 1.0, num=length, endpoint=False, dtype=np.float64)

    primary = _gaussian_pulse(phase, center=0.55, width=0.055)
    follow_through = _gaussian_pulse(phase, center=0.69, width=0.085)

    acceleration = rng.normal(0.0, 0.018, size=(length, 3))
    acceleration[:, 2] += 1.0  # Stable baseline analogous to gravity.
    acceleration[:, 0] += 2.20 * primary - 0.42 * follow_through
    acceleration[:, 1] += -1.05 * primary + 0.30 * follow_through
    acceleration[:, 2] += 0.70 * primary - 0.16 * follow_through

    angular_velocity = rng.normal(0.0, 0.012, size=(length, 3))
    angular_velocity[:, 0] += 1.35 * primary - 0.35 * follow_through
    angular_velocity[:, 1] += -0.90 * primary + 0.25 * follow_through
    angular_velocity[:, 2] += 4.25 * primary - 1.10 * follow_through

    magnetic_baseline = np.array([0.32, -0.05, 0.44], dtype=np.float64)
    magnetic_field = np.broadcast_to(magnetic_baseline, (length, 3)).copy()
    magnetic_field += rng.normal(0.0, 0.0015, size=(length, 3))
    magnetic_field[:, 0] += 0.003 * np.sin(2.0 * np.pi * phase)
    magnetic_field[:, 1] += 0.002 * np.cos(2.0 * np.pi * phase)

    return np.concatenate(
        (acceleration, angular_velocity, magnetic_field), axis=1
    ).astype(np.float32)


def generate_zero_imu(length: int = DEFAULT_LENGTH) -> np.ndarray:
    """Return the explicit all-zero missing-modality placeholder fixture."""
    if length <= 0:
        raise ValueError("length must be greater than zero")
    return np.zeros((length, len(IMU_COLUMNS)), dtype=np.float32)


def write_imu_csv(path: Path | str, values: np.ndarray, source: str) -> Path:
    """Write nine IMU columns followed by non-feature synthetic metadata."""
    output_path = Path(path)
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != len(IMU_COLUMNS):
        raise ValueError(f"values must have shape (length, {len(IMU_COLUMNS)})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow((*IMU_COLUMNS, *METADATA_COLUMNS))
        for row in array:
            writer.writerow((*[f"{float(value):.8f}" for value in row], "true", source))
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic SYNTHETIC IMU fixtures for tests."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--zero-output", type=Path, default=DEFAULT_ZERO_OUTPUT)
    parser.add_argument("--length", type=int, default=DEFAULT_LENGTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        synthetic_values = generate_synthetic_imu(args.length, args.seed)
        zero_values = generate_zero_imu(args.length)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    synthetic_path = write_imu_csv(args.output, synthetic_values, SYNTHETIC_SOURCE)
    zero_path = write_imu_csv(args.zero_output, zero_values, ZERO_SOURCE)
    print(f"Wrote SYNTHETIC swing fixture: {synthetic_path}")
    print(f"Wrote SYNTHETIC zero placeholder: {zero_path}")
    print("Physical units are intentionally unspecified; these files test code paths only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
