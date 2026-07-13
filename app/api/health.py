"""Health and model metadata endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from app.schemas.inference import HealthResponse, ModelInfoResponse


router = APIRouter()


def _manager_info(manager: Any) -> dict[str, Any]:
    info = getattr(manager, "info", {})
    if callable(info):
        info = info()
    return dict(info or {})


def _value(info: dict[str, Any], manager: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if name in info:
            return info[name]
        if hasattr(manager, name):
            value = getattr(manager, name)
            return value() if callable(value) else value
    return default


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    manager = request.app.state.model_manager
    info = _manager_info(manager)
    loaded = bool(_value(info, manager, "modelLoaded", "model_loaded", default=False))
    cuda_value = _value(info, manager, "cudaAvailable", "cuda_available")
    if cuda_value is None:
        current_device = str(_value(info, manager, "currentDevice", "device", default=""))
        cuda_available = current_device.startswith("cuda")
    else:
        cuda_available = bool(cuda_value)
    gpu_name = _value(info, manager, "gpuName", "gpu_name")
    if cuda_available and not gpu_name:
        try:
            import torch

            gpu_name = torch.cuda.get_device_name(getattr(manager, "device", None))
        except Exception:
            gpu_name = None
    model_version = str(
        _value(info, manager, "modelVersion", "model_version", default=request.app.state.settings.model_version)
    )
    return HealthResponse(
        status="UP" if loaded else "DEGRADED",
        cudaAvailable=cuda_available,
        gpuName=str(gpu_name) if gpu_name else None,
        modelLoaded=loaded,
        modelVersion=model_version,
    )


@router.get("/api/v1/model/info", response_model=ModelInfoResponse)
async def model_info(request: Request) -> ModelInfoResponse:
    manager = request.app.state.model_manager
    info = _manager_info(manager)
    return ModelInfoResponse(
        modelVersion=str(
            _value(info, manager, "modelVersion", "model_version", default=request.app.state.settings.model_version)
        ),
        supportedInputModes=list(
            _value(
                info,
                manager,
                "supportedInputModes",
                "supported_input_modes",
                default=["VIDEO_ONLY", "IMU_ONLY", "VIDEO_WITH_IMU"],
            )
        ),
        actionClasses=_value(
            info,
            manager,
            "actionClasses",
            "action_classes",
            default=["FOREHAND", "BACKHAND", "SERVE"],
        ),
        subActionClasses=_value(
            info,
            manager,
            "subActionClasses",
            "sub_action_classes",
            default=["TOPSPIN_FOREHAND", "BACKHAND_DRIVE", "FLAT_OR_SLICE_SERVE"],
        ),
        issueClasses=_value(
            info,
            manager,
            "issueClasses",
            "issue_classes",
            "errorClasses",
            "error_classes",
            default=[
                "STANDARD",
                "LATE_BACKSWING",
                "ARM_ONLY_FORCE",
                "CONTACT_TOO_CLOSE",
                "CONTACT_TOO_FAR",
                "WAITER_TRAY_SERVE",
                "TOSS_TOO_LOW",
            ],
        ),
        inputShapes=dict(
            _value(
                info,
                manager,
                "inputShapes",
                "input_shapes",
                default={"imu": [1, 100, 9], "poseA": [1, 50, 99], "poseB": [1, 50, 99]},
            )
        ),
        device=str(
            _value(info, manager, "currentDevice", "device", default=request.app.state.settings.device)
        ),
    )
