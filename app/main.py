"""FastAPI application factory.

The module-level ``app`` is safe to import: model weights are loaded exactly
once in the ASGI lifespan, never while importing this module and never per
request.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.health import router as health_router
from app.api.inference import router as inference_router
from app.config import Settings, get_settings
from app.schemas.inference import ErrorResponse
from app.services.video_inference_service import ServiceError, VideoInferenceService


LOGGER = logging.getLogger(__name__)


def _serialize(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _create_default_manager(settings: Settings) -> Any:
    # Lazy import keeps ``import app.main`` free from torch/model initialization.
    from app.services.model_manager import ModelManager

    candidate_values = {
        "settings": settings,
        "config": settings,
        "checkpoint_path": settings.checkpoint_path,
        "device": settings.device,
        "allow_cpu_fallback": settings.allow_cpu_fallback,
        "model_version": settings.model_version,
    }
    signature = inspect.signature(ModelManager)
    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    kwargs = {
        name: value
        for name, value in candidate_values.items()
        if accepts_kwargs or name in signature.parameters
    }
    return ModelManager(**kwargs)


def _is_model_loaded(manager: Any) -> bool:
    if hasattr(manager, "model_loaded"):
        value = manager.model_loaded
        return bool(value() if callable(value) else value)
    if hasattr(manager, "is_loaded"):
        value = manager.is_loaded
        return bool(value() if callable(value) else value)
    return getattr(manager, "model", None) is not None


async def _load_manager_once(manager: Any, lock: asyncio.Lock) -> None:
    if _is_model_loaded(manager):
        return
    async with lock:
        if _is_model_loaded(manager):
            return
        load = getattr(manager, "load", None)
        if load is None:
            raise RuntimeError("Model manager does not provide load()")
        if inspect.iscoroutinefunction(load):
            await load()
        else:
            await asyncio.to_thread(load)


def create_app(
    settings: Settings | None = None,
    manager: Any | None = None,
    inference_service: Any | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    startup_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        runtime_manager = manager or getattr(inference_service, "manager", None) or _create_default_manager(
            resolved_settings
        )
        await _load_manager_once(runtime_manager, startup_lock)
        application.state.settings = resolved_settings
        application.state.model_manager = runtime_manager
        application.state.inference_service = inference_service or VideoInferenceService(
            manager=runtime_manager,
            settings=resolved_settings,
        )
        yield

    application = FastAPI(
        title="MySport Tennis Model Service",
        version=resolved_settings.model_version,
        lifespan=lifespan,
    )
    application.include_router(health_router)
    application.include_router(inference_router)

    @application.exception_handler(ServiceError)
    async def handle_service_error(_request: Request, exc: ServiceError) -> JSONResponse:
        payload = ErrorResponse(errorCode=exc.error_code, message=exc.message, requestId=exc.request_id)
        return JSONResponse(status_code=exc.status_code, content=_serialize(payload))

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        missing_video = any(
            tuple(error.get("loc", ())) == ("body", "video") and error.get("type") in {"missing", "value_error.missing"}
            for error in exc.errors()
        )
        if missing_video:
            payload = ErrorResponse(errorCode="MISSING_VIDEO", message="必须上传 video 文件")
        else:
            payload = ErrorResponse(errorCode="INVALID_REQUEST", message="请求参数无效")
        return JSONResponse(status_code=422, content=_serialize(payload))

    @application.exception_handler(HTTPException)
    async def handle_http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        messages = {404: "请求路径不存在", 405: "请求方法不受支持"}
        payload = ErrorResponse(
            errorCode=f"HTTP_{exc.status_code}",
            message=messages.get(exc.status_code, "HTTP 请求失败"),
        )
        return JSONResponse(status_code=exc.status_code, content=_serialize(payload))

    @application.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("Unhandled model service error", exc_info=exc)
        payload = ErrorResponse(errorCode="INTERNAL_ERROR", message="模型服务内部错误")
        return JSONResponse(status_code=500, content=_serialize(payload))

    return application


app = create_app()
