from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from app.schemas.inference import InferenceResponse
from inference.single_video_infer import (
    InputTensorShapeError,
    build_input_tensors,
    run_inference,
)


class CapturingHierarchicalModel:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def __call__(
        self,
        imu: torch.Tensor,
        pose_a: torch.Tensor,
        pose_b: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.calls.append((imu.detach().clone(), pose_a.detach().clone(), pose_b.detach().clone()))
        return {
            "major": torch.tensor([[8.0, 0.0, 0.0]], device=imu.device),
            "action": torch.tensor([[8.0, 0.0, 0.0]], device=imu.device),
            "quality": torch.tensor(
                [[8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=imu.device
            ),
        }


class LoadedFakeManager:
    def __init__(self, model: CapturingHierarchicalModel) -> None:
        self.model = model
        self.device = torch.device("cpu")
        self.model_version = "tennis-test"


def test_all_missing_modalities_are_exact_float32_zero_placeholders() -> None:
    tensors = build_input_tensors(device="cpu")

    assert set(tensors) == {"imu", "pose_cam_a", "pose_cam_b"}
    assert tensors["imu"].shape == (1, 100, 9)
    assert tensors["pose_cam_a"].shape == (1, 50, 99)
    assert tensors["pose_cam_b"].shape == (1, 50, 99)
    for tensor in tensors.values():
        assert tensor.dtype == torch.float32
        assert tensor.device == torch.device("cpu")
        assert torch.count_nonzero(tensor).item() == 0


def test_missing_imu_and_pose_b_follow_pose_a_batch_and_device() -> None:
    pose_a = np.ones((2, 50, 99), dtype=np.float64)

    tensors = build_input_tensors(pose_a=pose_a, device=torch.device("cpu"))

    assert tensors["imu"].shape == (2, 100, 9)
    assert tensors["pose_cam_a"].shape == (2, 50, 99)
    assert tensors["pose_cam_b"].shape == (2, 50, 99)
    assert tensors["pose_cam_a"].dtype == torch.float32
    assert torch.count_nonzero(tensors["imu"]).item() == 0
    assert torch.count_nonzero(tensors["pose_cam_b"]).item() == 0


@pytest.mark.parametrize(
    ("kwargs", "modality"),
    [
        ({"imu": np.zeros((99, 9), dtype=np.float32)}, "imu"),
        ({"imu": np.zeros((100, 8), dtype=np.float32)}, "imu"),
        ({"pose_a": np.zeros((49, 99), dtype=np.float32)}, "pose_cam_a"),
        ({"pose_b": np.zeros((50, 98), dtype=np.float32)}, "pose_cam_b"),
    ],
)
def test_wrong_input_shapes_fail_with_stable_error(
    kwargs: dict[str, np.ndarray], modality: str
) -> None:
    with pytest.raises(InputTensorShapeError) as error:
        build_input_tensors(**kwargs)

    assert error.value.error_code == "INPUT_TENSOR_SHAPE_INVALID"
    assert modality in str(error.value)


def test_modality_batch_sizes_must_match() -> None:
    with pytest.raises(InputTensorShapeError, match="same batch size"):
        build_input_tensors(
            imu=np.zeros((2, 100, 9), dtype=np.float32),
            pose_a=np.zeros((1, 50, 99), dtype=np.float32),
        )


def test_nan_input_is_rejected() -> None:
    imu = np.zeros((100, 9), dtype=np.float32)
    imu[0, 0] = np.nan

    with pytest.raises(InputTensorShapeError, match="NaN or infinite"):
        build_input_tensors(imu=imu)


def test_imu_only_runs_without_uninitialized_pose_tensors(
    synthetic_imu_csv: Path,
    tmp_path: Path,
) -> None:
    model = CapturingHierarchicalModel()
    manager = LoadedFakeManager(model)

    result = run_inference(
        model_manager=manager,
        imu_csv_path=synthetic_imu_csv,
        output_dir=tmp_path / "imu-only-output",
        request_id="imu-only-test",
    )

    assert result["status"] == "SUCCEEDED"
    assert result["mode"] == "IMU_ONLY"
    assert result["syntheticImu"] is True
    assert result["segments"]
    assert model.calls
    for imu, pose_a, pose_b in model.calls:
        assert imu.shape == (1, 100, 9)
        assert pose_a.shape == (1, 50, 99)
        assert pose_b.shape == (1, 50, 99)
        assert imu.dtype == pose_a.dtype == pose_b.dtype == torch.float32
        assert imu.device == pose_a.device == pose_b.device == torch.device("cpu")
        assert torch.count_nonzero(pose_a).item() == 0
        assert torch.count_nonzero(pose_b).item() == 0


def test_inference_response_serializes_to_stable_json_shape() -> None:
    response = InferenceResponse(
        requestId="request-123",
        mode="VIDEO_ONLY",
        syntheticImu=False,
        modelVersion="tennis-test",
        device="cuda",
        processingTimeMs=12,
        peakGpuMemoryMb=34.5,
        timing={
            "modelLoadMs": 0,
            "poseExtractionMs": 7,
            "forwardMs": 2,
            "clipExportMs": 3,
        },
        segments=[
            {
                "index": 0,
                "startMs": 100,
                "endMs": 200,
                "action": "FOREHAND",
                "subAction": "TOPSPIN_FOREHAND",
                "issue": "STANDARD",
                "confidence": {"action": 0.9, "subAction": 0.8, "issue": 0.7},
                "clipPath": "outputs/id/segment-00.mp4",
            }
        ],
        warnings=[],
    )

    payload = response.model_dump(mode="json")
    assert payload["status"] == "SUCCEEDED"
    assert payload["requestId"] == "request-123"
    assert payload["segments"][0]["confidence"] == {
        "action": 0.9,
        "subAction": 0.8,
        "issue": 0.7,
    }
    assert payload["segments"][0]["clipPath"] == "outputs/id/segment-00.mp4"
