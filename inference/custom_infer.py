"""CLI wrapper for the reusable single-video inference pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.model_manager import ModelManager
from inference.single_video_infer import run_inference


def _environment_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MySport single-video, IMU-only, and video-plus-IMU inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "tennis_multimodal_transformer.pth",
    )
    parser.add_argument("--cam-a", type=Path, default=None, help="Single phone video")
    parser.add_argument(
        "--cam-b",
        type=Path,
        default=None,
        help="Reserved; two-camera inference is not supported in this version",
    )
    parser.add_argument("--imu-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--device", default=os.getenv("MODEL_DEVICE", "cuda"))
    parser.add_argument(
        "--allow-cpu-fallback",
        action=argparse.BooleanOptionalAction,
        default=_environment_bool("MODEL_ALLOW_CPU_FALLBACK", False),
    )
    parser.add_argument("--model-version", default=os.getenv("MODEL_VERSION", "tennis-0.1"))
    parser.add_argument("--noise-floor", type=float, default=0.15)
    parser.add_argument("--relative-drop", type=float, default=0.60)
    parser.add_argument("--conf-threshold", type=float, default=0.50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    request_id: str | None = None
    try:
        args = build_parser().parse_args(argv)
        request_id = args.request_id
        if args.cam_b is not None:
            raise ValueError("Cam B is not supported; provide only --cam-a")
        if args.cam_a is None and args.imu_csv is None:
            raise ValueError("Provide --cam-a, --imu-csv, or both")

        manager = ModelManager(
            checkpoint_path=args.checkpoint,
            device=args.device,
            allow_cpu_fallback=args.allow_cpu_fallback,
            model_version=args.model_version,
        )
        manager.load()
        result = run_inference(
            model_manager=manager,
            video_path=args.cam_a,
            imu_csv_path=args.imu_csv,
            output_dir=args.output_dir,
            request_id=request_id,
            noise_floor=args.noise_floor,
            relative_drop=args.relative_drop,
            confidence_threshold=args.conf_threshold,
        )
        result["timing"]["modelLoadMs"] = manager.load_ms
        result["processingTimeMs"] += manager.load_ms
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        payload = {
            "requestId": request_id,
            "status": "FAILED",
            "errorCode": getattr(exc, "error_code", "INVALID_ARGUMENT"),
            "message": str(exc),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
