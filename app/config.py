"""Environment-backed settings for the model service.

No machine-specific absolute path or proxy setting belongs here.  Relative
defaults are resolved from the repository root so starting Uvicorn from a
different working directory is deterministic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, legacy_name: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None and legacy_name:
        value = os.getenv(legacy_name)
    return value


def _env_bool(name: str, default: bool, legacy_name: str | None = None) -> bool:
    value = _env(name, legacy_name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _env_positive_int(name: str, default: int, legacy_name: str | None = None) -> int:
    value = _env(name, legacy_name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def _resolve_path(value: str | Path | None, default: Path) -> Path:
    path = Path(value).expanduser() if value is not None else default
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings with secure, repository-relative defaults."""

    checkpoint_path: Path = field(
        default_factory=lambda: _resolve_path(
            _env("MODEL_CHECKPOINT_PATH", "CHECKPOINT_PATH"),
            REPOSITORY_ROOT / "checkpoints" / "tennis_multimodal_transformer.pth",
        )
    )
    output_dir: Path = field(
        default_factory=lambda: _resolve_path(
            _env("MODEL_OUTPUT_DIR", "OUTPUT_DIR"),
            REPOSITORY_ROOT / "outputs",
        )
    )
    device: str = field(default_factory=lambda: (_env("MODEL_DEVICE", "DEVICE") or "cuda").strip().lower())
    max_upload_mb: int = field(
        default_factory=lambda: _env_positive_int("MODEL_MAX_UPLOAD_MB", 100, "MAX_UPLOAD_MB")
    )
    allow_cpu_fallback: bool = field(
        default_factory=lambda: _env_bool("MODEL_ALLOW_CPU_FALLBACK", False, "ALLOW_CPU_FALLBACK")
    )
    model_version: str = field(default_factory=lambda: os.getenv("MODEL_VERSION", "tennis-0.1"))
    allowed_video_extensions: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv")

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", _resolve_path(self.checkpoint_path, self.checkpoint_path))
        object.__setattr__(self, "output_dir", _resolve_path(self.output_dir, self.output_dir))
        if self.device not in {"cuda", "cpu"} and not self.device.startswith("cuda:"):
            raise ValueError("MODEL_DEVICE must be 'cuda', 'cuda:<index>', or 'cpu'")
        if self.max_upload_mb <= 0:
            raise ValueError("MODEL_MAX_UPLOAD_MB must be greater than zero")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def get_settings() -> Settings:
    """Build settings at application creation time, not module import time."""

    return Settings()
