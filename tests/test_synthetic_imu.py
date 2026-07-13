from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from data_utils.dataset_generator import extract_imu
from scripts.generate_synthetic_imu import (
    DEFAULT_LENGTH,
    DEFAULT_SEED,
    IMU_COLUMNS,
    METADATA_COLUMNS,
    SYNTHETIC_SOURCE,
    ZERO_SOURCE,
    generate_synthetic_imu,
    main,
    write_imu_csv,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
SYNTHETIC_FIXTURE = FIXTURE_DIR / "synthetic_imu.csv"
ZERO_FIXTURE = FIXTURE_DIR / "zero_imu.csv"


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _feature_values(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray(
        [[float(row[column]) for column in IMU_COLUMNS] for row in rows],
        dtype=np.float32,
    )


def test_synthetic_fixture_schema_and_float_values() -> None:
    columns, rows = _read_rows(SYNTHETIC_FIXTURE)

    assert columns == [*IMU_COLUMNS, *METADATA_COLUMNS]
    assert len(rows) == DEFAULT_LENGTH
    assert {row["synthetic"] for row in rows} == {"true"}
    assert {row["source"] for row in rows} == {SYNTHETIC_SOURCE}
    assert all(np.isfinite(float(row[column])) for row in rows for column in IMU_COLUMNS)


def test_extract_imu_reads_strict_feature_order_as_float32() -> None:
    values, energy, info = extract_imu(SYNTHETIC_FIXTURE)
    _, rows = _read_rows(SYNTHETIC_FIXTURE)
    fixture_values = _feature_values(rows)

    assert values.shape == (DEFAULT_LENGTH, len(IMU_COLUMNS))
    assert values.dtype == np.float32
    assert energy.shape == (DEFAULT_LENGTH,)
    assert info["columns"] == list(IMU_COLUMNS)
    np.testing.assert_array_equal(values[:, 1], fixture_values[:, 1])  # acc_y
    np.testing.assert_array_equal(values[:, 4], fixture_values[:, 4])  # gyro_y
    assert not any("matched incompletely" in warning for warning in info["warnings"])


def test_synthetic_signal_is_reproducible_and_matches_committed_fixture(
    tmp_path: Path,
) -> None:
    first = generate_synthetic_imu(DEFAULT_LENGTH, DEFAULT_SEED)
    second = generate_synthetic_imu(DEFAULT_LENGTH, DEFAULT_SEED)
    different_seed = generate_synthetic_imu(DEFAULT_LENGTH, DEFAULT_SEED + 1)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different_seed)

    first_path = write_imu_csv(tmp_path / "first.csv", first, SYNTHETIC_SOURCE)
    second_path = write_imu_csv(tmp_path / "second.csv", second, SYNTHETIC_SOURCE)
    assert first_path.read_bytes() == second_path.read_bytes()

    _, rows = _read_rows(SYNTHETIC_FIXTURE)
    fixture_values = _feature_values(rows)
    np.testing.assert_allclose(fixture_values, first, rtol=0.0, atol=5e-8)


def test_synthetic_signal_contains_swing_shape_without_assuming_units() -> None:
    values, _, _ = extract_imu(SYNTHETIC_FIXTURE)

    assert 0.90 < float(np.median(values[:, 2])) < 1.05
    assert float(np.max(np.abs(values[:, 3:6]))) > 3.0
    assert all(float(np.std(values[:, index])) < 0.01 for index in range(6, 9))


def test_zero_fixture_is_all_zero_float32_and_preserves_metadata() -> None:
    columns, rows = _read_rows(ZERO_FIXTURE)
    values, _, info = extract_imu(ZERO_FIXTURE)

    assert columns == [*IMU_COLUMNS, *METADATA_COLUMNS]
    assert len(rows) == DEFAULT_LENGTH
    assert {row["synthetic"] for row in rows} == {"true"}
    assert {row["source"] for row in rows} == {ZERO_SOURCE}
    assert values.shape == (DEFAULT_LENGTH, len(IMU_COLUMNS))
    assert values.dtype == np.float32
    assert np.count_nonzero(values) == 0
    assert set(info["columns"]).isdisjoint(METADATA_COLUMNS)


def test_cli_supports_custom_paths_length_and_seed(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "custom_synthetic.csv"
    zero_output = tmp_path / "nested" / "custom_zero.csv"

    assert main(
        [
            "--output",
            str(output),
            "--zero-output",
            str(zero_output),
            "--length",
            "17",
            "--seed",
            "42",
        ]
    ) == 0

    generated, _, _ = extract_imu(output)
    zero, _, _ = extract_imu(zero_output)
    assert generated.shape == (17, len(IMU_COLUMNS))
    assert zero.shape == (17, len(IMU_COLUMNS))
    assert np.count_nonzero(zero) == 0
