from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.services.video_inference_service import ServiceError, VideoInferenceService
from inference.single_video_infer import _normalized_video_fps, infer_single_video


@pytest.mark.parametrize(
    ("reported_fps", "expected_fps"),
    [
        (30.0, 30.0),
        (60.0, 60.0),
        (0.0, 30.0),
        (float("nan"), 30.0),
        (90_000.0, 30.0),
    ],
)
def test_normalized_video_fps_rejects_invalid_or_container_timebase_values(
    reported_fps: float,
    expected_fps: float,
) -> None:
    assert _normalized_video_fps(reported_fps) == expected_fps


def test_model_clip_path_cannot_escape_configured_output_root(
    test_settings: Settings,
    fake_manager: Any,
) -> None:
    service = VideoInferenceService(fake_manager, test_settings)

    with pytest.raises(ServiceError) as error:
        service._public_clip_path("../outside.mp4")

    assert error.value.error_code == "INVALID_MODEL_RESPONSE"
    assert error.value.status_code == 500


@pytest.mark.gpu
@pytest.mark.integration
def test_real_gpu_new_single_video_function(
    gpu_model_manager: Any,
    real_video: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "direct-single-video"

    result = infer_single_video(
        gpu_model_manager,
        real_video,
        output_dir,
        request_id="gpu-direct-single-video",
    )

    assert gpu_model_manager.load_count == 1
    assert str(gpu_model_manager.device).startswith("cuda")
    assert result["status"] == "SUCCEEDED"
    assert result["mode"] == "VIDEO_ONLY"
    assert result["syntheticImu"] is False
    assert result["segments"]
    assert result["timing"]["poseExtractionMs"] >= 0
    assert result["timing"]["forwardMs"] >= 0
    assert result["peakGpuMemoryMb"] > 0
    assert list(output_dir.rglob("segment-*.mp4"))


@pytest.mark.gpu
@pytest.mark.integration
def test_real_gpu_fastapi_single_video(
    gpu_model_manager: Any,
    real_video: Path,
    tmp_path: Path,
) -> None:
    settings = Settings(
        checkpoint_path=gpu_model_manager.checkpoint_path,
        output_dir=tmp_path / "api-single-video",
        device="cuda",
        allow_cpu_fallback=False,
        model_version=gpu_model_manager.model_version,
        max_upload_mb=10,
    )
    app = create_app(settings=settings, manager=gpu_model_manager)

    with TestClient(app) as client, real_video.open("rb") as video_handle:
        response = client.post(
            "/api/v1/inference/video",
            files={"video": ("video_1.mp4", video_handle, "video/mp4")},
            data={"requestId": "gpu-api-single-video"},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert gpu_model_manager.load_count == 1
    assert payload["requestId"] == "gpu-api-single-video"
    assert payload["status"] == "SUCCEEDED"
    assert payload["mode"] == "VIDEO_ONLY"
    assert payload["syntheticImu"] is False
    assert payload["device"].startswith("cuda")
    assert payload["segments"]
    assert all(not path.name.startswith("_upload") for path in settings.output_dir.rglob("*"))


@pytest.mark.gpu
@pytest.mark.integration
def test_real_gpu_video_with_synthetic_imu_code_path(
    gpu_model_manager: Any,
    real_video: Path,
    synthetic_imu_csv: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic-imu-fusion"

    result = infer_single_video(
        gpu_model_manager,
        real_video,
        output_dir,
        imu_csv_path=synthetic_imu_csv,
        request_id="gpu-synthetic-fusion",
        synthetic_imu=True,
    )

    assert gpu_model_manager.load_count == 1
    assert result["status"] == "SUCCEEDED"
    assert result["mode"] == "VIDEO_WITH_IMU"
    assert result["syntheticImu"] is True
    assert "SYNTHETIC_IMU_CODE_PATH_ONLY" in result["warnings"]
    assert result["segments"]
    assert list(output_dir.rglob("segment-*.mp4"))
