"""Single-video inference endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, Request, UploadFile

from app.schemas.inference import ErrorResponse, InferenceResponse
from app.services.video_inference_service import ServiceError


router = APIRouter(prefix="/api/v1/inference", tags=["inference"])
LOGGER = logging.getLogger(__name__)


@router.post(
    "/video",
    response_model=InferenceResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        507: {"model": ErrorResponse},
    },
)
async def infer_video(
    request: Request,
    video: UploadFile = File(...),
    imuCsv: UploadFile | None = File(default=None),
    cameraDirection: str | None = Form(default=None),
    requestId: str | None = Form(default=None),
) -> InferenceResponse:
    try:
        result = await request.app.state.inference_service.infer_upload(
            video=video,
            imu_csv=imuCsv,
            camera_direction=cameraDirection,
            request_id=requestId,
        )
        return InferenceResponse(**result)
    except ServiceError:
        raise
    except Exception as exc:
        LOGGER.exception("Inference service call failed")
        raise ServiceError("INFERENCE_FAILED", "模型推理失败", 500, requestId) from exc
