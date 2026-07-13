"""Safe upload handling and serialized access to the GPU inference core."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app.config import REPOSITORY_ROOT, Settings


LOGGER = logging.getLogger(__name__)
_CHUNK_SIZE = 1024 * 1024


class ServiceError(Exception):
    """An error that is safe to serialize to an API client."""

    def __init__(self, error_code: str, message: str, status_code: int = 400, request_id: str | None = None):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.request_id = request_id


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


class VideoInferenceService:
    """Orchestrates one request without owning or reloading the model."""

    def __init__(self, manager: Any, settings: Settings, engine: Any | None = None):
        self.manager = manager
        self.settings = settings
        self.engine = engine
        self._gpu_semaphore = asyncio.Semaphore(1)

    async def infer_upload(
        self,
        video: UploadFile,
        imu_csv: UploadFile | None = None,
        camera_direction: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        response_request_id = self._normalize_request_id(request_id)
        output_root = self.settings.output_dir.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        request_dir = self._create_request_directory(output_root)
        video_path: Path | None = None
        imu_path: Path | None = None

        try:
            video_suffix = self._validate_client_filename(
                video.filename, self.settings.allowed_video_extensions, "INVALID_VIDEO_EXTENSION"
            )
            video_path = request_dir / f"_upload{video_suffix}"
            video_prefix = await self._save_limited_upload(video, video_path, self.settings.max_upload_bytes)
            self._validate_video_magic(video_suffix, video_prefix)
            await asyncio.to_thread(self._validate_opencv_video, video_path)

            synthetic_imu = False
            if imu_csv is not None:
                imu_suffix = self._validate_client_filename(imu_csv.filename, (".csv",), "INVALID_IMU_EXTENSION")
                imu_path = request_dir / f"_imu{imu_suffix}"
                imu_prefix = await self._save_limited_upload(imu_csv, imu_path, self.settings.max_upload_bytes)
                synthetic_imu = self._is_explicitly_synthetic(imu_csv.filename, imu_prefix)

            started = time.perf_counter()
            async with self._gpu_semaphore:
                try:
                    result = await asyncio.to_thread(
                        self._run_core,
                        video_path,
                        imu_path,
                        self.settings.output_dir,
                        response_request_id,
                        camera_direction,
                        synthetic_imu,
                    )
                except Exception as exc:
                    raise self._translate_inference_error(exc, response_request_id) from exc

            return self._normalize_result(
                result=result,
                request_id=response_request_id,
                has_imu=imu_path is not None,
                synthetic_imu=synthetic_imu,
                elapsed_ms=round((time.perf_counter() - started) * 1000),
            )
        except ServiceError as exc:
            if exc.request_id is None:
                exc.request_id = response_request_id
            raise
        finally:
            await self._close_upload(video)
            if imu_csv is not None:
                await self._close_upload(imu_csv)
            for temporary_path in (video_path, imu_path):
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            try:
                request_dir.rmdir()
            except OSError:
                # Exported clips remain in the per-request directory.
                pass

    def _run_core(
        self,
        video_path: Path,
        imu_path: Path | None,
        output_dir: Path,
        request_id: str,
        camera_direction: str | None,
        synthetic_imu: bool,
    ) -> Any:
        if self.engine is not None:
            method = getattr(self.engine, "run_inference", None) or getattr(self.engine, "infer", None)
            if method is None and callable(self.engine):
                method = self.engine
            if method is None:
                raise RuntimeError("The injected inference engine is not callable")
            return method(
                model_manager=self.manager,
                video_path=video_path,
                imu_csv_path=imu_path,
                output_dir=output_dir,
                request_id=request_id,
                camera_direction=camera_direction,
                synthetic_imu=synthetic_imu,
            )

        # Imported lazily so importing the web app never loads model code or weights.
        from inference import single_video_infer

        run_inference = getattr(single_video_infer, "run_inference", None)
        if run_inference is not None:
            return run_inference(
                model_manager=self.manager,
                video_path=video_path,
                imu_csv_path=imu_path,
                output_dir=output_dir,
                request_id=request_id,
                camera_direction=camera_direction,
                synthetic_imu=synthetic_imu,
            )
        return single_video_infer.infer_single_video(
            self.manager,
            video_path,
            output_dir,
            imu_csv_path=imu_path,
            request_id=request_id,
            camera_direction=camera_direction,
            synthetic_imu=synthetic_imu,
        )

    async def _save_limited_upload(self, upload: UploadFile, destination: Path, limit: int) -> bytes:
        total = 0
        prefix = bytearray()
        try:
            with destination.open("xb") as output:
                while True:
                    chunk = await upload.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > limit:
                        raise ServiceError(
                            "UPLOAD_TOO_LARGE",
                            f"上传文件超过 {self.settings.max_upload_mb} MiB 限制",
                            413,
                        )
                    if len(prefix) < 4096:
                        prefix.extend(chunk[: 4096 - len(prefix)])
                    output.write(chunk)
        except FileExistsError as exc:
            raise ServiceError("OUTPUT_CONFLICT", "请求输出目录发生冲突", 409) from exc
        if total == 0:
            raise ServiceError("EMPTY_UPLOAD", "上传文件为空", 400)
        return bytes(prefix)

    @staticmethod
    async def _close_upload(upload: UploadFile) -> None:
        try:
            await upload.close()
        except Exception:
            LOGGER.warning("Failed to close an upload stream", exc_info=True)

    @staticmethod
    def _validate_client_filename(filename: str | None, allowed: tuple[str, ...], error_code: str) -> str:
        if not filename:
            raise ServiceError("INVALID_FILENAME", "上传文件名无效", 400)
        # The name is never used as a server path; separators and traversal are
        # rejected as an additional, testable defense-in-depth boundary.
        if filename in {".", ".."} or "/" in filename or "\\" in filename or ".." in Path(filename).parts:
            raise ServiceError("PATH_TRAVERSAL", "上传文件名包含非法路径", 400)
        suffix = Path(filename).suffix.lower()
        if suffix not in allowed:
            raise ServiceError(error_code, "不支持的上传文件扩展名", 415)
        return suffix

    @staticmethod
    def _validate_video_magic(suffix: str, prefix: bytes) -> None:
        is_iso_media = len(prefix) >= 12 and prefix[4:8] == b"ftyp"
        is_avi = len(prefix) >= 12 and prefix[:4] == b"RIFF" and prefix[8:12] == b"AVI "
        is_matroska = prefix.startswith(b"\x1aE\xdf\xa3")
        valid = {
            ".mp4": is_iso_media,
            ".mov": is_iso_media,
            ".avi": is_avi,
            ".mkv": is_matroska,
        }.get(suffix, False)
        if not valid:
            raise ServiceError("INVALID_VIDEO_CONTENT", "视频内容与文件类型不匹配", 415)

    @staticmethod
    def _validate_opencv_video(path: Path) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - dependency installation error
            raise ServiceError("VIDEO_VALIDATION_UNAVAILABLE", "视频校验组件不可用", 500) from exc

        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ServiceError("INVALID_VIDEO", "视频无法打开或已损坏", 415)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ServiceError("INVALID_VIDEO", "视频不包含可读取的画面", 415)
        finally:
            capture.release()

    @staticmethod
    def _is_explicitly_synthetic(filename: str | None, prefix: bytes) -> bool:
        name_marker = "SYNTHETIC" in (filename or "").upper()
        content_marker = b"SYNTHETIC" in prefix.upper()
        return name_marker or content_marker

    @staticmethod
    def _normalize_request_id(request_id: str | None) -> str:
        if request_id is None or not request_id.strip():
            return str(uuid.uuid4())
        normalized = request_id.strip()
        if len(normalized) > 128 or any(ord(char) < 32 for char in normalized):
            raise ServiceError("INVALID_REQUEST_ID", "requestId 格式无效", 400)
        return normalized

    @staticmethod
    def _create_request_directory(output_root: Path) -> Path:
        for _ in range(5):
            request_dir = (output_root / str(uuid.uuid4())).resolve()
            if not _is_relative_to(request_dir, output_root):
                raise ServiceError("INVALID_OUTPUT_PATH", "输出目录配置无效", 500)
            try:
                request_dir.mkdir(mode=0o700, exist_ok=False)
                return request_dir
            except FileExistsError:
                continue
        raise ServiceError("OUTPUT_CONFLICT", "无法创建独立请求输出目录", 409)

    def _normalize_result(
        self,
        result: Any,
        request_id: str,
        has_imu: bool,
        synthetic_imu: bool,
        elapsed_ms: int,
    ) -> dict[str, Any]:
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        elif hasattr(result, "dict"):
            result = result.dict()
        elif hasattr(result, "to_dict"):
            result = result.to_dict()
        if not isinstance(result, dict):
            raise ServiceError("INVALID_MODEL_RESPONSE", "模型返回了无效结果", 500, request_id)

        normalized = dict(result)
        result_synthetic_imu = bool(normalized.get("syntheticImu", False))
        normalized.update(
            {
                "requestId": request_id,
                "status": "SUCCEEDED",
                "mode": "VIDEO_WITH_IMU" if has_imu else "VIDEO_ONLY",
                "syntheticImu": (result_synthetic_imu or synthetic_imu) if has_imu else False,
            }
        )
        normalized.setdefault("modelVersion", str(getattr(self.manager, "model_version", self.settings.model_version)))
        normalized.setdefault("device", str(getattr(self.manager, "device", self.settings.device)))
        normalized.setdefault("processingTimeMs", elapsed_ms)
        normalized.setdefault(
            "timing",
            {"modelLoadMs": 0, "poseExtractionMs": 0, "forwardMs": 0, "clipExportMs": 0},
        )
        normalized.setdefault("segments", [])
        normalized.setdefault("warnings", [])

        for segment in normalized["segments"]:
            clip_path = segment.get("clipPath")
            if clip_path:
                segment["clipPath"] = self._public_clip_path(clip_path)
        return normalized

    def _public_clip_path(self, clip_path: str | Path) -> str:
        candidate = Path(clip_path)
        if any(part == ".." for part in candidate.parts):
            raise ServiceError("INVALID_MODEL_RESPONSE", "模型返回了越界输出路径", 500)

        output_root = self.settings.output_dir.resolve()
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            candidate_parts = candidate.parts
            if candidate_parts and candidate_parts[0].lower() == output_root.name.lower():
                candidate = Path(*candidate_parts[1:])
            from_output = (output_root / candidate).resolve()
            from_repository = (REPOSITORY_ROOT / candidate).resolve()
            if _is_relative_to(from_repository, output_root):
                resolved = from_repository
            else:
                resolved = from_output

        if not _is_relative_to(resolved, output_root):
            raise ServiceError("INVALID_MODEL_RESPONSE", "模型返回了越界输出路径", 500)
        relative_clip = resolved.relative_to(output_root).as_posix()
        return f"{self.settings.output_dir.name}/{relative_clip}"

    @staticmethod
    def _translate_inference_error(exc: Exception, request_id: str) -> ServiceError:
        if isinstance(exc, ServiceError):
            if exc.request_id is None:
                exc.request_id = request_id
            return exc
        error_code = str(getattr(exc, "error_code", "INFERENCE_FAILED"))
        status_code = 507 if "OUT_OF_MEMORY" in error_code else 500
        if "out of memory" in str(exc).lower():
            error_code = "GPU_OUT_OF_MEMORY"
            status_code = 507
        public_messages = {
            "INVALID_VIDEO": "视频无法读取或格式无效",
            "INVALID_INPUT": "模型输入无效",
            "GPU_OUT_OF_MEMORY": "GPU 显存不足，请稍后重试或使用更短的视频",
            "CUDA_OUT_OF_MEMORY": "GPU 显存不足，请稍后重试或使用更短的视频",
        }
        LOGGER.exception("Inference request %s failed with %s", request_id, error_code)
        return ServiceError(error_code, public_messages.get(error_code, "模型推理失败"), status_code, request_id)
