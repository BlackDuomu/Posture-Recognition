from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.services.video_inference_service import VideoInferenceService


def _build_app(settings: Settings, manager: Any, engine: Any) -> Any:
    service = VideoInferenceService(manager=manager, settings=settings, engine=engine)
    return create_app(
        settings=settings,
        manager=manager,
        inference_service=service,
    )


def _post_video(
    client: TestClient,
    real_video: Path,
    *,
    filename: str = "video_1.mp4",
    imu_csv: Path | None = None,
    request_id: str | None = "api-test-request",
) -> Any:
    files: dict[str, tuple[str, bytes, str]] = {
        "video": (filename, real_video.read_bytes(), "video/mp4")
    }
    if imu_csv is not None:
        files["imuCsv"] = (
            "SYNTHETIC_imu.csv",
            imu_csv.read_bytes(),
            "text/csv",
        )
    data = {} if request_id is None else {"requestId": request_id}
    return client.post("/api/v1/inference/video", files=files, data=data)


def test_lifespan_loads_model_once_and_health_and_info_are_stable(
    test_settings: Settings,
    fake_manager: Any,
    fake_engine: Any,
) -> None:
    app = _build_app(test_settings, fake_manager, fake_engine)

    assert fake_manager.load_calls == 0
    with TestClient(app) as client:
        health_first = client.get("/health")
        health_second = client.get("/health")
        info_response = client.get("/api/v1/model/info")

    assert fake_manager.load_calls == 1
    assert health_first.status_code == health_second.status_code == 200
    assert health_first.json() == {
        "status": "UP",
        "cudaAvailable": False,
        "gpuName": None,
        "modelLoaded": True,
        "modelVersion": "tennis-test",
    }
    info = info_response.json()
    assert info_response.status_code == 200
    assert info["modelVersion"] == "tennis-test"
    assert info["supportedInputModes"] == ["VIDEO_ONLY", "IMU_ONLY", "VIDEO_WITH_IMU"]
    assert info["actionClasses"] == ["FOREHAND", "BACKHAND", "SERVE"]
    assert info["subActionClasses"] == [
        "TOPSPIN_FOREHAND",
        "BACKHAND_DRIVE",
        "FLAT_OR_SLICE_SERVE",
    ]
    assert info["issueClasses"] == ["STANDARD", "ARM_ONLY_FORCE"]
    assert info["inputShapes"] == {
        "imu": [1, 100, 9],
        "poseCamA": [1, 50, 99],
        "poseCamB": [1, 50, 99],
    }
    assert info["device"] == "cpu"


def test_single_video_request_uses_real_upload_validation_and_stable_json(
    test_settings: Settings,
    fake_manager: Any,
    fake_engine: Any,
    real_video: Path,
) -> None:
    app = _build_app(test_settings, fake_manager, fake_engine)

    with TestClient(app) as client:
        response = _post_video(client, real_video, request_id="single-video-request")

    assert response.status_code == 200
    payload = response.json()
    assert payload["requestId"] == "single-video-request"
    assert payload["status"] == "SUCCEEDED"
    assert payload["mode"] == "VIDEO_ONLY"
    assert payload["syntheticImu"] is False
    assert payload["modelVersion"] == "tennis-test"
    assert payload["segments"][0]["action"] == "FOREHAND"
    assert len(fake_engine.calls) == 1


def test_video_with_synthetic_imu_is_explicitly_marked(
    test_settings: Settings,
    fake_manager: Any,
    fake_engine: Any,
    real_video: Path,
    synthetic_imu_csv: Path,
) -> None:
    app = _build_app(test_settings, fake_manager, fake_engine)

    with TestClient(app) as client:
        response = _post_video(
            client,
            real_video,
            imu_csv=synthetic_imu_csv,
            request_id="fusion-code-path",
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "VIDEO_WITH_IMU"
    assert payload["syntheticImu"] is True
    assert fake_engine.calls[0]["synthetic_imu"] is True
    uploaded_video = Path(fake_engine.calls[0]["video_path"])
    uploaded_imu = Path(fake_engine.calls[0]["imu_csv_path"])
    assert uploaded_imu.suffix == ".csv"
    assert not uploaded_video.exists()
    assert not uploaded_imu.exists()
    assert not uploaded_video.parent.exists()


def test_missing_video_returns_stable_validation_error(
    test_settings: Settings,
    fake_manager: Any,
    fake_engine: Any,
) -> None:
    app = _build_app(test_settings, fake_manager, fake_engine)

    with TestClient(app) as client:
        response = client.post("/api/v1/inference/video")

    assert response.status_code == 422
    assert response.json() == {
        "status": "FAILED",
        "errorCode": "MISSING_VIDEO",
        "message": "必须上传 video 文件",
        "requestId": None,
    }
    assert fake_engine.calls == []


def test_invalid_video_extension_is_rejected_before_inference(
    test_settings: Settings,
    fake_manager: Any,
    fake_engine: Any,
    real_video: Path,
) -> None:
    app = _build_app(test_settings, fake_manager, fake_engine)

    with TestClient(app) as client:
        response = _post_video(client, real_video, filename="video_1.exe")

    assert response.status_code == 415
    assert response.json()["errorCode"] == "INVALID_VIDEO_EXTENSION"
    assert fake_engine.calls == []


def test_disguised_invalid_video_content_is_rejected(
    test_settings: Settings,
    fake_manager: Any,
    fake_engine: Any,
) -> None:
    app = _build_app(test_settings, fake_manager, fake_engine)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/inference/video",
            files={"video": ("fake.mp4", b"not a real mp4", "video/mp4")},
        )

    assert response.status_code == 415
    assert response.json()["errorCode"] == "INVALID_VIDEO_CONTENT"
    assert fake_engine.calls == []


def test_path_traversal_filename_is_rejected(
    test_settings: Settings,
    fake_manager: Any,
    fake_engine: Any,
    real_video: Path,
) -> None:
    app = _build_app(test_settings, fake_manager, fake_engine)

    with TestClient(app) as client:
        response = _post_video(client, real_video, filename="../video_1.mp4")

    assert response.status_code == 400
    assert response.json()["errorCode"] == "PATH_TRAVERSAL"
    assert fake_engine.calls == []


def test_engine_failure_returns_stable_error_without_internal_details(
    test_settings: Settings,
    fake_manager: Any,
    fake_engine: Any,
    real_video: Path,
) -> None:
    fake_engine.failure = RuntimeError("secret internal stack detail")
    app = _build_app(test_settings, fake_manager, fake_engine)

    with TestClient(app) as client:
        response = _post_video(client, real_video, request_id="failed-request")

    assert response.status_code == 500
    assert response.json() == {
        "status": "FAILED",
        "errorCode": "INFERENCE_FAILED",
        "message": "模型推理失败",
        "requestId": "failed-request",
    }
    assert "secret" not in response.text


def test_each_request_has_isolated_directory_and_temporary_uploads_are_removed(
    test_settings: Settings,
    fake_manager: Any,
    fake_engine: Any,
    real_video: Path,
) -> None:
    app = _build_app(test_settings, fake_manager, fake_engine)

    with TestClient(app) as client:
        first = _post_video(client, real_video, request_id="first")
        second = _post_video(client, real_video, request_id="second")

    assert first.status_code == second.status_code == 200
    assert len(fake_engine.calls) == 2
    upload_paths = [Path(call["video_path"]) for call in fake_engine.calls]
    request_dirs = [path.parent for path in upload_paths]
    assert request_dirs[0] != request_dirs[1]
    assert all(path.parent.parent == test_settings.output_dir for path in upload_paths)
    assert all(not path.exists() for path in upload_paths)
    assert all(not directory.exists() for directory in request_dirs)
