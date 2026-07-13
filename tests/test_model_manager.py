from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from app.services.model_manager import (
    CudaUnavailableError,
    ModelCheckpointError,
    ModelManager,
)


class FakeTorchModel:
    def __init__(self) -> None:
        self.loaded_state: Any = None
        self.target_device: torch.device | None = None
        self.eval_calls = 0

    def load_state_dict(self, state_dict: Any, strict: bool = True) -> None:
        assert strict is True
        self.loaded_state = state_dict

    def to(self, device: torch.device) -> "FakeTorchModel":
        self.target_device = device
        return self

    def eval(self) -> "FakeTorchModel":
        self.eval_calls += 1
        return self


def test_missing_checkpoint_has_stable_error_code(tmp_path: Path) -> None:
    manager = ModelManager(tmp_path / "missing.pth", device="cpu")

    with pytest.raises(ModelCheckpointError) as error:
        manager.load()

    assert error.value.error_code == "MODEL_CHECKPOINT_NOT_FOUND"
    assert "does not exist" in str(error.value)
    assert manager.load_count == 0


def test_cuda_unavailable_does_not_silently_fall_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.touch()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    manager = ModelManager(checkpoint, device="cuda", allow_cpu_fallback=False)

    with pytest.raises(CudaUnavailableError) as error:
        manager.load()

    assert error.value.error_code == "CUDA_UNAVAILABLE"
    assert "CPU fallback is disabled" in str(error.value)
    assert manager.load_count == 0


def test_explicit_cpu_fallback_is_honored_when_cuda_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.touch()
    fake_model = FakeTorchModel()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: {"model_state_dict": {"weight": torch.tensor([1.0])}},
    )
    monkeypatch.setattr(
        ModelManager,
        "_build_model",
        staticmethod(lambda _checkpoint: fake_model),
    )
    manager = ModelManager(checkpoint, device="cuda", allow_cpu_fallback=True)

    manager.load()

    assert manager.device == torch.device("cpu")
    assert fake_model.target_device == torch.device("cpu")


def test_checkpoint_is_loaded_and_model_initialized_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.touch()
    fake_model = FakeTorchModel()
    torch_load_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    build_calls: list[dict[str, Any]] = []
    checkpoint = {
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "optimizer_state_dict": {"ignored": True},
        "epoch": 7,
        "hierarchical": True,
    }

    def fake_torch_load(*args: Any, **kwargs: Any) -> dict[str, Any]:
        torch_load_calls.append((args, kwargs))
        return checkpoint

    def fake_build_model(value: dict[str, Any]) -> FakeTorchModel:
        build_calls.append(value)
        return fake_model

    monkeypatch.setattr(torch, "load", fake_torch_load)
    monkeypatch.setattr(ModelManager, "_build_model", staticmethod(fake_build_model))
    manager = ModelManager(checkpoint_path, device="cpu", model_version="unit-test")

    first = manager.load()
    second = manager.load()

    assert first is second is fake_model
    assert len(torch_load_calls) == 1
    assert torch_load_calls[0][1] == {"map_location": "cpu", "weights_only": True}
    assert build_calls == [checkpoint]
    assert fake_model.loaded_state is checkpoint["model_state_dict"]
    assert fake_model.target_device == torch.device("cpu")
    assert fake_model.eval_calls == 1
    assert manager.is_loaded is True
    assert manager.load_count == 1
    assert manager.model_version == "unit-test"
    assert manager.checkpoint_metadata == {"epoch": 7, "hierarchical": True}


def test_model_info_uses_stable_public_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.touch()
    fake_model = FakeTorchModel()
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: {"model_state_dict": {}, "hierarchical": True},
    )
    monkeypatch.setattr(ModelManager, "_build_model", staticmethod(lambda _: fake_model))
    manager = ModelManager(checkpoint_path, device="cpu")
    manager.load()

    info = manager.info
    assert info["modelVersion"] == "tennis-0.1"
    assert info["supportedInputModes"] == ["VIDEO_ONLY", "IMU_ONLY", "VIDEO_WITH_IMU"]
    assert info["actionClasses"] == ["FOREHAND", "BACKHAND", "SERVE"]
    assert info["subActionClasses"] == [
        "TOPSPIN_FOREHAND",
        "BACKHAND_DRIVE",
        "FLAT_OR_SLICE_SERVE",
    ]
    assert isinstance(info["errorClasses"], list)
    assert info["inputShapes"] == {
        "imu": [1, 100, 9],
        "poseCamA": [1, 50, 99],
        "poseCamB": [1, 50, 99],
    }
    assert info["currentDevice"] == "cpu"
    assert info["modelLoaded"] is True
