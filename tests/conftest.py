from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch

from app.config import Settings


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class FakeManager:
    """Small lifecycle-compatible manager used by HTTP tests."""

    def __init__(self) -> None:
        self.model_loaded = False
        self.load_calls = 0
        self.model_version = "tennis-test"
        self.device = torch.device("cpu")

    def load(self) -> object:
        self.load_calls += 1
        self.model_loaded = True
        return object()

    @property
    def info(self) -> dict[str, Any]:
        return {
            "modelVersion": self.model_version,
            "supportedInputModes": ["VIDEO_ONLY", "IMU_ONLY", "VIDEO_WITH_IMU"],
            "actionClasses": ["FOREHAND", "BACKHAND", "SERVE"],
            "subActionClasses": [
                "TOPSPIN_FOREHAND",
                "BACKHAND_DRIVE",
                "FLAT_OR_SLICE_SERVE",
            ],
            "errorClasses": ["STANDARD", "ARM_ONLY_FORCE"],
            "inputShapes": {
                "imu": [1, 100, 9],
                "poseCamA": [1, 50, 99],
                "poseCamB": [1, 50, 99],
            },
            "currentDevice": str(self.device),
            "modelLoaded": self.model_loaded,
            "cudaAvailable": False,
            "gpuName": None,
        }


class FakeInferenceEngine:
    """Captures server-side paths while returning schema-valid inference data."""

    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls: list[dict[str, Any]] = []

    def infer(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        if self.failure is not None:
            raise self.failure
        manager = kwargs["model_manager"]
        return {
            "requestId": kwargs["request_id"],
            "status": "SUCCEEDED",
            "mode": "VIDEO_WITH_IMU" if kwargs["imu_csv_path"] else "VIDEO_ONLY",
            "syntheticImu": bool(kwargs["synthetic_imu"]),
            "modelVersion": manager.model_version,
            "device": str(manager.device),
            "processingTimeMs": 1,
            "peakGpuMemoryMb": 0.0,
            "timing": {
                "modelLoadMs": 0,
                "poseExtractionMs": 0,
                "forwardMs": 1,
                "clipExportMs": 0,
            },
            "segments": [
                {
                    "index": 0,
                    "startMs": 0,
                    "endMs": 100,
                    "action": "FOREHAND",
                    "subAction": "TOPSPIN_FOREHAND",
                    "issue": "STANDARD",
                    "confidence": {"action": 0.9, "subAction": 0.8, "issue": 0.7},
                    "clipPath": None,
                }
            ],
            "warnings": [],
        }


@pytest.fixture
def repository_root() -> Path:
    return REPOSITORY_ROOT


@pytest.fixture
def real_video(repository_root: Path) -> Path:
    path = repository_root / "data" / "videos" / "video_1.mp4"
    assert path.is_file(), f"Real test video is missing: {path}"
    return path


@pytest.fixture
def synthetic_imu_csv(repository_root: Path) -> Path:
    path = repository_root / "tests" / "fixtures" / "synthetic_imu.csv"
    assert path.is_file(), f"Synthetic IMU fixture is missing: {path}"
    return path


@pytest.fixture
def fake_manager() -> FakeManager:
    return FakeManager()


@pytest.fixture
def fake_engine() -> FakeInferenceEngine:
    return FakeInferenceEngine()


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        checkpoint_path=tmp_path / "unused-checkpoint.pth",
        output_dir=tmp_path / "outputs",
        device="cpu",
        allow_cpu_fallback=False,
        model_version="tennis-test",
        max_upload_mb=10,
    )


@pytest.fixture(scope="session")
def gpu_model_manager() -> Any:
    """Load the real checkpoint once, only after an explicit GPU-test opt-in."""

    if os.getenv("RUN_GPU_TESTS") != "1":
        pytest.skip("set RUN_GPU_TESTS=1 to run real CUDA/video integration tests")
    if not torch.cuda.is_available():
        pytest.fail("RUN_GPU_TESTS=1 but torch.cuda.is_available() is False")

    checkpoint = REPOSITORY_ROOT / "checkpoints" / "tennis_multimodal_transformer.pth"
    video = REPOSITORY_ROOT / "data" / "videos" / "video_1.mp4"
    if not checkpoint.is_file():
        pytest.fail(f"Real model checkpoint is missing: {checkpoint}")
    if not video.is_file():
        pytest.fail(f"Real integration video is missing: {video}")

    from app.services.model_manager import ModelManager

    manager = ModelManager(
        checkpoint_path=checkpoint,
        device="cuda",
        allow_cpu_fallback=False,
        model_version="tennis-0.1",
    )
    manager.load()
    assert manager.load_count == 1
    return manager
