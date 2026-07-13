"""Load and own the tennis model for the lifetime of the service process."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from models.model import TennisMultimodalTransformer


DEFAULT_MODEL_VERSION = "tennis-0.1"


class ModelManagerError(RuntimeError):
    """Base error carrying a stable machine-readable error code."""

    error_code = "MODEL_MANAGER_ERROR"

    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if error_code is not None:
            self.error_code = error_code


class ModelCheckpointError(ModelManagerError):
    error_code = "MODEL_CHECKPOINT_INVALID"


class CudaUnavailableError(ModelManagerError):
    error_code = "CUDA_UNAVAILABLE"


class ModelNotLoadedError(ModelManagerError):
    error_code = "MODEL_NOT_LOADED"


class ModelLoadOutOfMemoryError(ModelManagerError):
    error_code = "MODEL_LOAD_OUT_OF_MEMORY"


class ModelManager:
    """Load a checkpoint exactly once and expose the shared inference model.

    CUDA is the default. A failed CUDA request only falls back to CPU when
    ``allow_cpu_fallback`` was explicitly enabled by the caller.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        allow_cpu_fallback: bool = False,
        model_version: str = DEFAULT_MODEL_VERSION,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.requested_device = str(device).strip().lower()
        self.allow_cpu_fallback = bool(allow_cpu_fallback)
        self.model_version = model_version

        self._model: TennisMultimodalTransformer | None = None
        self._device: torch.device | None = None
        self._checkpoint_metadata: dict[str, Any] = {}
        self._inference_engine: Any | None = None
        self._load_count = 0
        self._load_ms = 0
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_loaded(self) -> bool:
        return self.is_loaded

    @property
    def cuda_available(self) -> bool:
        return bool(torch.cuda.is_available())

    @property
    def gpu_name(self) -> str | None:
        if not torch.cuda.is_available():
            return None
        device_index = self._cuda_device_index(self.device)
        return str(torch.cuda.get_device_name(device_index))

    @property
    def load_count(self) -> int:
        return self._load_count

    @property
    def load_ms(self) -> int:
        return self._load_ms

    @property
    def device(self) -> torch.device:
        if self._device is None:
            return self._resolve_device()
        return self._device

    @property
    def model(self) -> TennisMultimodalTransformer:
        if self._model is None:
            raise ModelNotLoadedError("Model has not been loaded")
        return self._model

    @property
    def checkpoint_metadata(self) -> dict[str, Any]:
        return dict(self._checkpoint_metadata)

    def _resolve_device(self) -> torch.device:
        requested = self.requested_device
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "auto":
            requested = "cuda"
        if not requested.startswith("cuda"):
            raise ModelManagerError(
                f"Unsupported model device: {self.requested_device}",
                error_code="MODEL_DEVICE_INVALID",
            )
        if torch.cuda.is_available():
            resolved = torch.device(requested)
            try:
                index = resolved.index if resolved.index is not None else torch.cuda.current_device()
                if index < 0 or index >= torch.cuda.device_count():
                    raise CudaUnavailableError(f"CUDA device index is unavailable: {requested}")
            except (AssertionError, RuntimeError) as exc:
                if not self.allow_cpu_fallback:
                    raise CudaUnavailableError(f"CUDA device is unavailable: {requested}") from exc
                return torch.device("cpu")
            return resolved
        if self.allow_cpu_fallback:
            return torch.device("cpu")
        raise CudaUnavailableError(
            "CUDA is required but torch.cuda.is_available() is False; "
            "CPU fallback is disabled"
        )

    @staticmethod
    def _build_model(checkpoint: Mapping[str, Any]) -> TennisMultimodalTransformer:
        hierarchical = bool(checkpoint.get("hierarchical", True))
        if hierarchical:
            return TennisMultimodalTransformer(
                hierarchical=True,
                num_major_classes=int(checkpoint.get("num_major_classes", 3)),
                num_action_classes=int(checkpoint.get("num_action_classes", 3)),
                num_quality_classes=int(checkpoint.get("num_quality_classes", 7)),
            )
        return TennisMultimodalTransformer(
            hierarchical=False,
            num_classes=int(checkpoint.get("num_classes", 5)),
        )

    def load(self) -> TennisMultimodalTransformer:
        """Return the shared model, loading it only on the first call."""

        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model

            path = self.checkpoint_path.resolve()
            if not path.exists() or not path.is_file():
                raise ModelCheckpointError(
                    f"Model checkpoint does not exist: {path}",
                    error_code="MODEL_CHECKPOINT_NOT_FOUND",
                )

            target_device = self._resolve_device()
            started = time.perf_counter()
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(checkpoint, Mapping):
                    raise ModelCheckpointError("Checkpoint must be a mapping")
                state_dict = checkpoint.get("model_state_dict")
                if not isinstance(state_dict, Mapping):
                    raise ModelCheckpointError(
                        "Checkpoint does not contain a model_state_dict mapping"
                    )

                model = self._build_model(checkpoint)
                model.load_state_dict(state_dict, strict=True)
                model.to(target_device)
                model.eval()
                if target_device.type == "cuda":
                    torch.cuda.synchronize(self._cuda_device_index(target_device))
            except ModelManagerError:
                raise
            except Exception as exc:
                if _is_cuda_oom(exc):
                    _empty_cuda_cache()
                    raise ModelLoadOutOfMemoryError(
                        "CUDA ran out of memory while loading the model"
                    ) from exc
                raise ModelCheckpointError(f"Unable to load model checkpoint: {exc}") from exc

            excluded = {"model_state_dict", "optimizer_state_dict"}
            self._checkpoint_metadata = {
                str(key): value for key, value in checkpoint.items() if key not in excluded
            }
            self._device = target_device
            self._model = model
            self._load_count += 1
            self._load_ms = max(0, round((time.perf_counter() - started) * 1000))
            return model

    def create_inference_engine(self) -> Any:
        """Return the process-wide inference facade backed by this manager."""

        self.load()
        if self._inference_engine is None:
            # Local import avoids a module cycle: the engine type depends on
            # ModelManager, while the manager only needs it after construction.
            from inference.single_video_infer import SingleVideoInferenceEngine

            self._inference_engine = SingleVideoInferenceEngine(self)
        return self._inference_engine

    @property
    def inference_engine(self) -> Any:
        return self.create_inference_engine()

    @property
    def info(self) -> dict[str, Any]:
        """Stable model metadata used by the API info endpoint."""

        current_device = str(self.device)
        return {
            "modelVersion": self.model_version,
            "supportedInputModes": ["VIDEO_ONLY", "IMU_ONLY", "VIDEO_WITH_IMU"],
            "actionClasses": ["FOREHAND", "BACKHAND", "SERVE"],
            "subActionClasses": [
                "TOPSPIN_FOREHAND",
                "BACKHAND_DRIVE",
                "FLAT_OR_SLICE_SERVE",
            ],
            "errorClasses": [
                "STANDARD",
                "LATE_BACKSWING",
                "ARM_ONLY_FORCE",
                "CONTACT_TOO_CLOSE",
                "CONTACT_TOO_FAR",
                "WAITER_TRAY_SERVE",
                "TOSS_TOO_LOW",
            ],
            "inputShapes": {
                "imu": [1, 100, 9],
                "poseCamA": [1, 50, 99],
                "poseCamB": [1, 50, 99],
            },
            "currentDevice": current_device,
            "modelLoaded": self.is_loaded,
            "cudaAvailable": self.cuda_available,
            "gpuName": self.gpu_name,
        }

    @staticmethod
    def _cuda_device_index(device: torch.device) -> int:
        return device.index if device.index is not None else torch.cuda.current_device()


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
