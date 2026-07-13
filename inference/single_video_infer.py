"""Reusable single-camera and IMU-only inference pipeline.

The model still receives all three modalities. Missing inputs are represented
by exact all-zero tensors, which is the missing-modality convention implemented
by :mod:`models.model`.
"""

from __future__ import annotations

import time
import uuid
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import scipy.signal
import torch

from app.services.model_manager import ModelManager
from data_utils.dataset_generator import (
    extract_imu,
    extract_pose,
    fallback_segments,
    normalize_skeleton,
    resample_sequence,
)


IMU_SHAPE = (100, 9)
POSE_SHAPE = (50, 99)
DEFAULT_CONFIDENCE_THRESHOLD = 0.50
SUPPORTED_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".m4v"})

MAJOR_CLASS_CODES = {
    0: "FOREHAND",
    1: "BACKHAND",
    2: "SERVE",
}
SUB_ACTION_CLASS_CODES = {
    0: "TOPSPIN_FOREHAND",
    1: "BACKHAND_DRIVE",
    2: "FLAT_OR_SLICE_SERVE",
}
ISSUE_CLASS_CODES = {
    0: "STANDARD",
    1: "LATE_BACKSWING",
    2: "ARM_ONLY_FORCE",
    3: "CONTACT_TOO_CLOSE",
    4: "CONTACT_TOO_FAR",
    5: "WAITER_TRAY_SERVE",
    6: "TOSS_TOO_LOW",
}


class InferenceError(RuntimeError):
    """Base inference error with a stable machine-readable code."""

    error_code = "INFERENCE_ERROR"

    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if error_code is not None:
            self.error_code = error_code


class InferenceInputError(InferenceError):
    error_code = "INFERENCE_INPUT_INVALID"


class InvalidVideoError(InferenceInputError):
    error_code = "VIDEO_INVALID"


class InputTensorShapeError(InferenceInputError):
    error_code = "INPUT_TENSOR_SHAPE_INVALID"


class ClipExportError(InferenceError):
    error_code = "CLIP_EXPORT_FAILED"


class InferenceOutOfMemoryError(InferenceError):
    error_code = "GPU_OUT_OF_MEMORY"


class InferenceExecutionError(InferenceError):
    error_code = "MODEL_INFERENCE_FAILED"


def build_input_tensors(
    imu: np.ndarray | torch.Tensor | None = None,
    pose_a: np.ndarray | torch.Tensor | None = None,
    pose_b: np.ndarray | torch.Tensor | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Build the exact float32 model contract, zero-filling missing inputs.

    Accepted non-batched inputs are ``(100, 9)`` for IMU and ``(50, 99)``
    for either pose. Batched inputs must be ``(B, 100, 9)`` and
    ``(B, 50, 99)`` respectively. This function does not resample or perform
    file I/O, so shape errors cannot be hidden by implicit preprocessing.
    """

    target_device = torch.device(device)
    prepared: dict[str, torch.Tensor | None] = {
        "imu": _prepare_tensor(imu, IMU_SHAPE, "imu", target_device),
        "pose_cam_a": _prepare_tensor(pose_a, POSE_SHAPE, "pose_cam_a", target_device),
        "pose_cam_b": _prepare_tensor(pose_b, POSE_SHAPE, "pose_cam_b", target_device),
    }
    batch_sizes = {int(t.shape[0]) for t in prepared.values() if t is not None}
    if len(batch_sizes) > 1:
        raise InputTensorShapeError("All modalities must use the same batch size")
    batch_size = next(iter(batch_sizes), 1)

    if prepared["imu"] is None:
        prepared["imu"] = torch.zeros(
            (batch_size, *IMU_SHAPE), dtype=torch.float32, device=target_device
        )
    if prepared["pose_cam_a"] is None:
        prepared["pose_cam_a"] = torch.zeros(
            (batch_size, *POSE_SHAPE), dtype=torch.float32, device=target_device
        )
    if prepared["pose_cam_b"] is None:
        prepared["pose_cam_b"] = torch.zeros(
            (batch_size, *POSE_SHAPE), dtype=torch.float32, device=target_device
        )
    return {name: tensor for name, tensor in prepared.items() if tensor is not None}


def _prepare_tensor(
    value: np.ndarray | torch.Tensor | None,
    sample_shape: tuple[int, ...],
    name: str,
    device: torch.device,
) -> torch.Tensor | None:
    if value is None:
        return None
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    except (TypeError, ValueError, RuntimeError) as exc:
        if _is_cuda_oom(exc):
            raise
        raise InputTensorShapeError(f"{name} cannot be converted to float32: {exc}") from exc
    if tuple(tensor.shape) == sample_shape:
        tensor = tensor.unsqueeze(0)
    expected_rank = len(sample_shape) + 1
    if tensor.ndim != expected_rank or tuple(tensor.shape[1:]) != sample_shape:
        raise InputTensorShapeError(
            f"{name} must have shape {sample_shape} or (B, {', '.join(map(str, sample_shape))}); "
            f"received {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise InputTensorShapeError(f"{name} contains NaN or infinite values")
    return tensor.contiguous()


def find_action_segments_topological(
    energy: np.ndarray,
    fps: float,
    noise_floor: float = 0.15,
    relative_drop: float = 0.60,
) -> list[tuple[int, int]]:
    """The current ``custom_infer.py`` topological segmentation semantics."""

    values = np.asarray(energy)
    if values.size == 0:
        return []
    max_val = np.max(values) + 1e-6
    norm_energy = values / max_val

    min_dist_frames = max(5, int(0.5 * fps))
    peaks, _ = scipy.signal.find_peaks(
        norm_energy, height=noise_floor, distance=min_dist_frames
    )
    if len(peaks) == 0:
        return []

    def find_left_valley(peak_idx: int) -> int:
        curr = peak_idx
        while curr > 0:
            if norm_energy[curr - 1] < noise_floor:
                break
            if norm_energy[curr - 1] > norm_energy[curr]:
                break
            curr -= 1
        return curr

    def find_right_valley(peak_idx: int) -> int:
        curr = peak_idx
        while curr < len(norm_energy) - 1:
            if norm_energy[curr + 1] < noise_floor:
                break
            if norm_energy[curr + 1] > norm_energy[curr]:
                break
            curr += 1
        return curr

    peak_bounds: list[list[int]] = []
    for peak in peaks:
        p = int(peak)
        peak_bounds.append([find_left_valley(p), find_right_valley(p), p])

    i = 0
    while i < len(peak_bounds) - 1:
        curr_l, _, curr_p = peak_bounds[i]
        _, next_r, next_p = peak_bounds[i + 1]
        valley_idx = curr_p + int(np.argmin(norm_energy[curr_p:next_p]))
        valley_val = norm_energy[valley_idx]
        shorter_peak_val = min(norm_energy[curr_p], norm_energy[next_p])
        drop_ratio = (shorter_peak_val - valley_val) / (shorter_peak_val + 1e-6)
        time_gap_seconds = (next_p - curr_p) / fps
        if drop_ratio < relative_drop or time_gap_seconds < 1.2:
            new_p = curr_p if norm_energy[curr_p] > norm_energy[next_p] else next_p
            peak_bounds[i] = [curr_l, next_r, new_p]
            peak_bounds.pop(i + 1)
        else:
            i += 1

    min_duration_frames = max(5, int(0.25 * fps))
    return [(left, right) for left, right, _ in peak_bounds if right - left >= min_duration_frames]


class SingleVideoInferenceEngine:
    """Thin stateful facade around the pure/reusable inference functions."""

    def __init__(self, model_manager: ModelManager) -> None:
        self.model_manager = model_manager

    def infer(
        self,
        video_path: str | Path | None = None,
        *,
        model_manager: ModelManager | None = None,
        imu_csv_path: str | Path | None = None,
        output_dir: str | Path = "outputs",
        request_id: str | None = None,
        camera_direction: str | None = None,
        synthetic_imu: bool | None = None,
        noise_floor: float = 0.15,
        relative_drop: float = 0.60,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> dict[str, Any]:
        if model_manager is not None and model_manager is not self.model_manager:
            raise InferenceInputError("Inference engine received a different model manager")
        return run_inference(
            model_manager=self.model_manager,
            video_path=video_path,
            imu_csv_path=imu_csv_path,
            output_dir=output_dir,
            request_id=request_id,
            camera_direction=camera_direction,
            synthetic_imu=synthetic_imu,
            noise_floor=noise_floor,
            relative_drop=relative_drop,
            confidence_threshold=confidence_threshold,
        )


def run_inference(
    *,
    model_manager: ModelManager,
    output_dir: str | Path,
    video_path: str | Path | None = None,
    imu_csv_path: str | Path | None = None,
    request_id: str | None = None,
    camera_direction: str | None = None,
    synthetic_imu: bool | None = None,
    noise_floor: float = 0.15,
    relative_drop: float = 0.60,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Run video-only, IMU-only, or single-video plus IMU inference."""

    del camera_direction  # Reserved API metadata; the model has no direction input.
    if video_path is None and imu_csv_path is None:
        raise InferenceInputError("At least one of video_path or imu_csv_path is required")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise InferenceInputError("confidence_threshold must be between 0 and 1")
    if not 0.0 <= noise_floor <= 1.0 or not 0.0 <= relative_drop <= 1.0:
        raise InferenceInputError("Segmentation thresholds must be between 0 and 1")

    # Accessing these properties deliberately does not auto-load the model.
    # FastAPI lifespan owns the one-time load operation.
    model = model_manager.model
    device = model_manager.device

    total_started = time.perf_counter()
    correlation_id = str(request_id).strip() if request_id is not None else str(uuid.uuid4())
    if not correlation_id:
        correlation_id = str(uuid.uuid4())
    video = Path(video_path).expanduser().resolve() if video_path is not None else None
    imu_csv = Path(imu_csv_path).expanduser().resolve() if imu_csv_path is not None else None
    artifact_id = str(uuid.uuid4())
    output_root = Path(output_dir).expanduser().resolve()
    request_output_dir = (output_root / artifact_id).resolve()
    if request_output_dir.parent != output_root:
        raise InferenceInputError("Output directory escaped the configured output root")
    request_output_dir.mkdir(parents=True, exist_ok=False)

    warnings: list[str] = ["CAM_B_MISSING_ZERO_FILLED"]
    pose_extraction_ms = 0
    forward_ms = 0
    clip_export_ms = 0
    peak_gpu_memory_mb = 0.0

    has_video = video is not None
    has_imu = imu_csv is not None
    mode = "VIDEO_WITH_IMU" if has_video and has_imu else "VIDEO_ONLY" if has_video else "IMU_ONLY"

    try:
        if device.type == "cuda":
            _cuda_synchronize(device)
            torch.cuda.reset_peak_memory_stats(_cuda_device_index(device))
        fps = 100.0 if not has_video else _validate_video(video)
        pose_a: np.ndarray | None = None
        energy_a: np.ndarray | None = None
        imu_values: np.ndarray | None = None
        imu_segments: list[tuple[int, int]] = []
        synthetic_detected = False

        if has_imu:
            _validate_imu_csv(imu_csv)
            synthetic_detected = detect_synthetic_imu(imu_csv)
            imu_values, _, imu_info = extract_imu(imu_csv)
            if len(imu_values) == 0:
                raise InferenceInputError(
                    "IMU CSV contains no data rows", error_code="IMU_DATA_EMPTY"
                )
            imu_segments = [(int(start), int(end)) for start, end in imu_info.get("segments", [])]
            if not imu_segments:
                imu_segments = fallback_segments(len(imu_values), 1)
                warnings.append("IMU_SEGMENTATION_FALLBACK")
            if imu_info.get("warnings"):
                warnings.append("IMU_PREPROCESSING_WARNING")
        else:
            warnings.append("IMU_MISSING_ZERO_FILLED")

        if has_video:
            pose_started = time.perf_counter()
            pose_a, energy_a, _ = extract_pose(
                video,
                max_width=640,
                cache_dir=request_output_dir,
                use_cache=False,
            )
            pose_extraction_ms = _elapsed_ms(pose_started)
            if len(pose_a) == 0 or energy_a is None or len(energy_a) == 0:
                raise InvalidVideoError("Video contains no decodable frames")

            percentile_98 = float(np.percentile(energy_a, 98))
            energy_a = np.clip(energy_a, 0, percentile_98)
            median_window = max(3, int(0.12 * fps))
            energy_a = (
                pd.Series(energy_a)
                .rolling(window=median_window, center=True, min_periods=1)
                .median()
                .fillna(0)
                .to_numpy()
            )
            mean_window = max(5, int(0.25 * fps))
            energy_a = (
                pd.Series(energy_a)
                .rolling(window=mean_window, center=True, min_periods=1)
                .mean()
                .fillna(0)
                .to_numpy()
            )

        final_segments: list[tuple[int, int, int, int]] = []
        if has_imu:
            assert imu_values is not None
            for imu_start, imu_end in imu_segments:
                if has_video:
                    assert pose_a is not None
                    start_ratio = imu_start / max(len(imu_values), 1)
                    end_ratio = imu_end / max(len(imu_values), 1)
                    video_start = int(start_ratio * len(pose_a))
                    video_end = int(
                        max(start_ratio * len(pose_a) + 1, end_ratio * len(pose_a))
                    )
                else:
                    video_start = 0
                    video_end = 0
                final_segments.append((imu_start, imu_end, video_start, video_end))
        else:
            assert pose_a is not None and energy_a is not None
            video_segments = find_action_segments_topological(
                energy_a,
                fps,
                noise_floor=noise_floor,
                relative_drop=relative_drop,
            )
            if not video_segments:
                video_segments = fallback_segments(len(pose_a), 1)
                warnings.append("VIDEO_SEGMENTATION_FALLBACK")
            final_segments.extend((0, 0, start, end) for start, end in video_segments)

        response_segments: list[dict[str, Any]] = []
        for index, (imu_start, imu_end, video_start, video_end) in enumerate(final_segments):
            imu_segment = None
            if has_imu:
                assert imu_values is not None
                imu_segment = resample_sequence(
                    imu_values[imu_start:imu_end], IMU_SHAPE[0], (IMU_SHAPE[1],)
                )

            pose_a_segment = None
            if has_video:
                assert pose_a is not None
                raw_pose = resample_sequence(
                    pose_a[video_start:video_end], POSE_SHAPE[0], (33, 3)
                )
                pose_a_segment = normalize_skeleton(raw_pose)

            tensors = build_input_tensors(
                imu=imu_segment,
                pose_a=pose_a_segment,
                pose_b=None,
                device=device,
            )
            _cuda_synchronize(device)
            forward_started = time.perf_counter()
            with torch.inference_mode():
                outputs = model(
                    tensors["imu"], tensors["pose_cam_a"], tensors["pose_cam_b"]
                )
            _cuda_synchronize(device)
            forward_ms += _elapsed_ms(forward_started)

            classification = _serialize_hierarchical_prediction(
                outputs, confidence_threshold
            )
            if has_video:
                start_ms = max(0, round(video_start / fps * 1000))
                end_ms = max(start_ms, round(video_end / fps * 1000))
                clip_name = f"segment-{index:02d}.mp4"
                clip_path = request_output_dir / clip_name
                clip_started = time.perf_counter()
                export_video_segment(video, video_start, video_end, clip_path)
                clip_export_ms += _elapsed_ms(clip_started)
                public_clip_path: str | None = (
                    Path(output_root.name) / artifact_id / clip_name
                ).as_posix()
            else:
                # extract_imu uses a 100 Hz segmentation basis.
                start_ms = max(0, round(imu_start / 100.0 * 1000))
                end_ms = max(start_ms, round(imu_end / 100.0 * 1000))
                public_clip_path = None

            response_segments.append(
                {
                    "index": index,
                    "startMs": start_ms,
                    "endMs": end_ms,
                    "action": classification["action"],
                    "subAction": classification["subAction"],
                    "issue": classification["issue"],
                    "confidence": classification["confidence"],
                    "clipPath": public_clip_path,
                }
            )

        synthetic_flag = bool(synthetic_imu) or synthetic_detected
        if synthetic_detected:
            warnings.append("SYNTHETIC_IMU_CODE_PATH_ONLY")
        elif synthetic_imu:
            warnings.append("SYNTHETIC_IMU_DECLARED_BY_CALLER")
        if device.type == "cuda":
            peak_gpu_memory_mb = round(
                torch.cuda.max_memory_allocated(_cuda_device_index(device)) / (1024 * 1024), 3
            )

        processing_time_ms = _elapsed_ms(total_started)
        return {
            "requestId": correlation_id,
            "status": "SUCCEEDED",
            "mode": mode,
            "syntheticImu": synthetic_flag,
            "modelVersion": model_manager.model_version,
            "device": str(device),
            "processingTimeMs": processing_time_ms,
            "timing": {
                "modelLoadMs": 0,
                "poseExtractionMs": pose_extraction_ms,
                "forwardMs": forward_ms,
                "clipExportMs": clip_export_ms,
            },
            "segments": response_segments,
            "warnings": list(dict.fromkeys(warnings)),
            "peakGpuMemoryMb": peak_gpu_memory_mb,
        }
    except InferenceError:
        _cleanup_failed_output(request_output_dir, output_root)
        raise
    except Exception as exc:
        _cleanup_failed_output(request_output_dir, output_root)
        if _is_cuda_oom(exc):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise InferenceOutOfMemoryError(
                "CUDA ran out of memory during model inference"
            ) from exc
        raise InferenceExecutionError(f"Inference failed: {exc}") from exc


def infer_single_video(
    model_manager: ModelManager,
    video_path: str | Path,
    output_dir: str | Path,
    imu_csv_path: str | Path | None = None,
    request_id: str | None = None,
    camera_direction: str | None = None,
    synthetic_imu: bool | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience entry point for the public single-video use case."""

    return run_inference(
        model_manager=model_manager,
        video_path=video_path,
        imu_csv_path=imu_csv_path,
        output_dir=output_dir,
        request_id=request_id,
        camera_direction=camera_direction,
        synthetic_imu=synthetic_imu,
        **kwargs,
    )


infer = infer_single_video


def detect_synthetic_imu(csv_path: str | Path) -> bool:
    """Detect explicit SYNTHETIC provenance markers in an IMU CSV."""

    try:
        frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise InferenceInputError(
            f"Unable to read IMU CSV: {exc}", error_code="IMU_CSV_INVALID"
        ) from exc
    columns = {str(column).strip().lower(): column for column in frame.columns}
    synthetic_column = columns.get("synthetic")
    if synthetic_column is not None:
        truthy = {"1", "true", "yes", "y", "synthetic"}
        if any(str(value).strip().lower() in truthy for value in frame[synthetic_column].dropna()):
            return True
    source_column = columns.get("source")
    if source_column is not None:
        if any("synthetic" in str(value).strip().lower() for value in frame[source_column].dropna()):
            return True
    return False


def export_video_segment(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    output_path: Path,
) -> None:
    """Export ``[start_frame, end_frame)`` using the baseline mp4v codec."""

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ClipExportError("Unable to open source video for clip export")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ClipExportError("Source video has invalid dimensions")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            writer.release()
            raise ClipExportError("Unable to create output video clip")
        written = 0
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start_frame)))
            for _ in range(max(0, int(end_frame) - int(start_frame))):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
                written += 1
        finally:
            writer.release()
        if written == 0 or not output_path.exists() or output_path.stat().st_size == 0:
            output_path.unlink(missing_ok=True)
            raise ClipExportError("The requested video segment contains no decodable frames")
    finally:
        cap.release()


def _validate_video(video_path: Path | None) -> float:
    if video_path is None:
        raise InvalidVideoError("Video path is missing")
    if not video_path.exists() or not video_path.is_file():
        raise InvalidVideoError(
            f"Video does not exist: {video_path}", error_code="VIDEO_NOT_FOUND"
        )
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise InvalidVideoError(
            f"Unsupported video extension: {video_path.suffix}",
            error_code="VIDEO_EXTENSION_UNSUPPORTED",
        )
    if video_path.stat().st_size <= 0:
        raise InvalidVideoError("Video file is empty", error_code="VIDEO_EMPTY")
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise InvalidVideoError("Video container cannot be decoded")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count <= 0 or width <= 0 or height <= 0:
            raise InvalidVideoError("Video has no valid frames or dimensions")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    return fps


def _validate_imu_csv(csv_path: Path | None) -> None:
    if csv_path is None:
        raise InferenceInputError("IMU CSV path is missing", error_code="IMU_CSV_MISSING")
    if not csv_path.exists() or not csv_path.is_file():
        raise InferenceInputError(
            f"IMU CSV does not exist: {csv_path}", error_code="IMU_CSV_NOT_FOUND"
        )
    if csv_path.suffix.lower() != ".csv":
        raise InferenceInputError(
            "IMU input must be a CSV file", error_code="IMU_CSV_INVALID_EXTENSION"
        )
    if csv_path.stat().st_size <= 0:
        raise InferenceInputError("IMU CSV is empty", error_code="IMU_CSV_EMPTY")


def _serialize_hierarchical_prediction(
    outputs: Any,
    confidence_threshold: float,
) -> dict[str, Any]:
    if not isinstance(outputs, Mapping) or not {"major", "action", "quality"}.issubset(outputs):
        raise InferenceExecutionError(
            "Model did not return hierarchical major/action/quality logits",
            error_code="MODEL_OUTPUT_INVALID",
        )
    probabilities: dict[str, tuple[float, int]] = {}
    for key in ("major", "action", "quality"):
        logits = outputs[key]
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[0] != 1:
            raise InferenceExecutionError(
                f"Model output '{key}' has an invalid shape",
                error_code="MODEL_OUTPUT_INVALID",
            )
        probability = torch.softmax(logits, dim=1)
        confidence, class_index = probability.max(dim=1)
        probabilities[key] = (float(confidence.item()), int(class_index.item()))

    major_conf, major_idx = probabilities["major"]
    action_conf, action_idx = probabilities["action"]
    issue_conf, issue_idx = probabilities["quality"]
    return {
        "action": (
            MAJOR_CLASS_CODES.get(major_idx, "UNKNOWN")
            if major_conf >= confidence_threshold
            else "UNKNOWN"
        ),
        "subAction": (
            SUB_ACTION_CLASS_CODES.get(action_idx, "UNKNOWN")
            if action_conf >= confidence_threshold
            else "UNKNOWN"
        ),
        "issue": (
            ISSUE_CLASS_CODES.get(issue_idx, "UNKNOWN")
            if issue_conf >= confidence_threshold
            else "UNKNOWN"
        ),
        "confidence": {
            "action": round(major_conf, 6),
            "subAction": round(action_conf, 6),
            "issue": round(issue_conf, 6),
        },
    }


def _cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(_cuda_device_index(device))


def _cuda_device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _cleanup_failed_output(request_output_dir: Path, output_root: Path) -> None:
    """Remove only the freshly-created UUID directory for this failed call."""

    try:
        resolved_request_dir = request_output_dir.resolve()
        resolved_output_root = output_root.resolve()
        uuid.UUID(resolved_request_dir.name)
        if resolved_request_dir.parent == resolved_output_root:
            shutil.rmtree(resolved_request_dir)
    except (OSError, ValueError):
        pass


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


__all__ = [
    "ClipExportError",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "IMU_SHAPE",
    "ISSUE_CLASS_CODES",
    "InferenceError",
    "InferenceExecutionError",
    "InferenceInputError",
    "InferenceOutOfMemoryError",
    "InputTensorShapeError",
    "InvalidVideoError",
    "MAJOR_CLASS_CODES",
    "POSE_SHAPE",
    "SUB_ACTION_CLASS_CODES",
    "SingleVideoInferenceEngine",
    "build_input_tensors",
    "detect_synthetic_imu",
    "find_action_segments_topological",
    "infer",
    "infer_single_video",
    "run_inference",
]
