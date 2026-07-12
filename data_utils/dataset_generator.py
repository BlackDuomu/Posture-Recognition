from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import scipy.signal

IMU_FEATURE_KEYWORDS = [
    ("acc_x", ["加速度x", "accx", "acc_x", "accelerationx", "ax"]),
    ("acc_y", ["加速度y", "accy", "acc_y", "accelerationy", "ay"]),
    ("acc_z", ["加速度z", "accz", "acc_z", "accelerationz", "az"]),
    ("gyro_x", ["角速度x", "gyrox", "gyro_x", "gx"]),
    ("gyro_y", ["角速度y", "gyroy", "gyro_y", "gy"]),
    ("gyro_z", ["角速度z", "gyroz", "gyro_z", "gz"]),
    ("mag_x", ["磁场x", "magx", "mag_x", "mx"]),
    ("mag_y", ["磁场y", "magy", "mag_y", "my"]),
    ("mag_z", ["磁场z", "magz", "mag_z", "mz"]),
]


def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-6
    cosine_angle = np.sum(ba * bc, axis=-1) / denom
    return np.arccos(np.clip(cosine_angle, -1.0, 1.0))


def normalize_skeleton(pose):
    hip_center = (pose[:, 23] + pose[:, 24]) / 2.0  # (T, 3)
    pose_centered = pose - hip_center[:, np.newaxis, :]

    l_shoulder = pose[:, 11]
    l_hip = pose[:, 23]
    torso_len = np.linalg.norm(l_shoulder - l_hip, axis=-1, keepdims=True)[:, np.newaxis, :]  # (T, 1, 1)

    pose_normalized = pose_centered / (torso_len + 1e-6)
    return pose_normalized.reshape(pose.shape[0], -1).astype(np.float32)  # (T, 99)


def compute_auto_alignment_offset(imu_energy: np.ndarray, video_energy: np.ndarray, fps: float) -> Tuple[float, float]:
    x_imu = np.linspace(0, 1, len(imu_energy))
    x_vid = np.linspace(0, 1, len(video_energy))
    imu_resampled = np.interp(x_vid, x_imu, imu_energy)

    imu_norm = (imu_resampled - np.mean(imu_resampled)) / (np.std(imu_resampled) + 1e-6)
    vid_norm = (video_energy - np.mean(video_energy)) / (np.std(video_energy) + 1e-6)

    correlation = scipy.signal.correlate(vid_norm, imu_norm, mode='full')
    lags = scipy.signal.correlation_lags(len(vid_norm), len(imu_norm), mode='full')

    best_lag_idx = np.argmax(correlation)
    best_lag_frames = lags[best_lag_idx]

    offset_ratio = best_lag_frames / len(video_energy)

    print(f"      物理伸缩率 (Scale): 1.000000 | 物理对齐量 (Offset): {offset_ratio:.4%}")

    return 1.0, offset_ratio


def find_segments_by_imu_peaks(energy: np.ndarray, fs: float = 100.0) -> List[Tuple[int, int]]:
    min_dist_frames = int(0.15 * fs)

    peaks, _ = scipy.signal.find_peaks(
        energy,
        distance=min_dist_frames,
        height=0.20 * (np.max(energy) + 1e-6),
        prominence=0.10 * (np.max(energy) + 1e-6)
    )

    segments = []
    for p in peaks:
        peak_val = energy[p]
        limit_val = 0.15 * peak_val

        start = p
        while start > 0:
            if p - start > int(0.4 * fs):
                break
            if energy[start - 1] > energy[start] + 0.005:
                break
            if energy[start] < limit_val:
                break
            start -= 1

        # 向右（后）寻找终点
        end = p
        while end < len(energy) - 1:
            if end - p > int(0.5 * fs):
                break
            if energy[end + 1] > energy[end] + 0.005:
                break
            if energy[end] < limit_val:
                break
            end += 1

        segments.append((start, end))

    return segments


def find_segments_by_video_peaks(energy: np.ndarray, fps: float = 30.0) -> List[Tuple[int, int]]:
    min_dist_frames = int(0.8 * fps)

    peaks, _ = scipy.signal.find_peaks(
        energy,
        distance=min_dist_frames,
        height=0.12 * (np.max(energy) + 1e-6),
        prominence=0.08 * (np.max(energy) + 1e-6)
    )

    segments = []
    for p in peaks:
        peak_val = energy[p]
        limit_val = 0.15 * peak_val

        start = p
        while start > 0:
            if p - start > int(0.8 * fps):
                break
            if energy[start - 1] > energy[start] + 0.012 * peak_val:
                break
            if energy[start] < limit_val:
                break
            start -= 1

        end = p
        while end < len(energy) - 1:
            if end - p > int(1.0 * fps):
                break
            if energy[end + 1] > energy[end] + 0.012 * peak_val:
                break
            if energy[end] < limit_val:
                break
            end += 1

        segments.append((start, end))

    return segments

def find_action_segments_topological(energy: np.ndarray, fps: float, noise_floor=0.15, relative_drop=0.60) -> List[Tuple[int, int]]:
    median_window = max(3, int(0.15 * fps))
    series_f = pd.Series(energy).rolling(window=median_window, center=True, min_periods=1).median()
    mean_window = max(5, int(0.25 * fps))
    smoothed_energy = series_f.rolling(window=mean_window, center=True, min_periods=1).mean().fillna(0).to_numpy()

    max_val = np.max(smoothed_energy) + 1e-6
    norm_energy = smoothed_energy / max_val

    min_dist_frames = max(5, int(0.7 * fps))
    peaks, _ = scipy.signal.find_peaks(norm_energy, height=noise_floor, distance=min_dist_frames)

    if len(peaks) == 0:
        return []

    def find_left_valley(peak_idx):
        curr = peak_idx
        while curr > 0:
            if norm_energy[curr - 1] < noise_floor:
                break
            if norm_energy[curr - 1] > norm_energy[curr]:
                break
            curr -= 1
        return curr

    def find_right_valley(peak_idx):
        curr = peak_idx
        while curr < len(norm_energy) - 1:
            if norm_energy[curr + 1] < noise_floor:
                break
            if norm_energy[curr + 1] > norm_energy[curr]:
                break
            curr += 1
        return curr

    peak_bounds = []
    for p in peaks:
        l = find_left_valley(p)
        r = find_right_valley(p)
        peak_bounds.append([l, r, p])

    i = 0
    while i < len(peak_bounds) - 1:
        curr_l, curr_r, curr_p = peak_bounds[i]
        next_l, next_r, next_p = peak_bounds[i + 1]

        valley_idx = curr_p + np.argmin(norm_energy[curr_p:next_p])
        valley_val = norm_energy[valley_idx]
        shorter_peak_val = min(norm_energy[curr_p], norm_energy[next_p])

        # 计算跌落深度
        drop_ratio = (shorter_peak_val - valley_val) / (shorter_peak_val + 1e-6)
        time_gap_seconds = (next_p - curr_p) / fps

        if drop_ratio < 0.45 or time_gap_seconds < 0.75:
            new_p = curr_p if norm_energy[curr_p] > norm_energy[next_p] else next_p
            peak_bounds[i] = [curr_l, next_r, new_p]
            peak_bounds.pop(i + 1)
        else:
            i += 1

    min_duration_frames = max(5, int(0.3 * fps))
    final_segments = []
    for l, r, p in peak_bounds:
        if (r - l) >= min_duration_frames:
            final_segments.append((l, r))

    return final_segments


def check_and_convert_imu_path(file_path: Path | str) -> str:
    path = Path(file_path)
    if path.suffix.lower() == '.txt':
        csv_path = path.with_suffix('.csv')

        df = pd.read_csv(path, sep='\t', encoding='utf-8-sig')
        df.columns = [str(c).strip() for c in df.columns]

        target_cols = [
            '时间', '设备名称', '片上时间()',
            '加速度X(g)', '加速度Y(g)', '加速度Z(g)',
            '角速度X(°/s)', '角速度Y(°/s)', '角速度Z(°/s)',
            '角度X(°)', '角度Y(°)', '角度Z(°)',
            '磁场X(ʯt)', '磁场Y(ʯt)', '磁场Z(ʯt)',
            '温度(℃)',
            '四元数0()', '四元数1()', '四元数2()', '四元数3()'
        ]

        rename_dict = {
            '磁场X(uT)': '磁场X(ʯt)',
            '磁场Y(uT)': '磁场Y(ʯt)',
            '磁场Z(uT)': '磁场Z(ʯt)',
            '温度(°C)': '温度(℃)'
        }
        df = df.rename(columns=rename_dict)

        out_df = pd.DataFrame()
        for col in target_cols:
            if col == '时间':
                out_df['时间'] = df['时间'].apply(lambda x: str(x).split('T')[-1] if 'T' in str(x) else str(x))
            elif col == '设备名称':
                out_df['设备名称'] = df['设备名称'].apply(lambda x: str(x).split('(')[0] if '(' in str(x) else str(x))
            elif col == '片上时间()':
                out_df['片上时间()'] = df['时间'].apply(
                    lambda x: str(x).replace('T', ' ').replace('.', ':') if isinstance(x, str) else x)
            elif col in df.columns:
                out_df[col] = df[col]
            else:
                out_df[col] = "null"

        out_df.to_csv(csv_path, index=False, na_rep='null')
        print(f"   -> 格式化重构成功！新文件已自动保存至: {csv_path}")
        return str(csv_path)

    return str(file_path)


def clean_column(name: str) -> str:
    return re.sub(r"[\s\(\)（）/\\_\-°℃µμʯt]+", "", str(name).strip().lower())


def resample_sequence(data: np.ndarray, target_len: int, trailing_shape: Tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.size == 0 or arr.shape[0] == 0:
        return np.zeros((target_len, *trailing_shape), dtype=np.float32)
    arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[0] == 1:
        out = np.repeat(arr, target_len, axis=0)
    else:
        old_x = np.linspace(0.0, 1.0, arr.shape[0])
        new_x = np.linspace(0.0, 1.0, target_len)
        out = np.vstack([np.interp(new_x, old_x, arr[:, c]) for c in range(arr.shape[1])]).T
    return out.reshape((target_len, *trailing_shape)).astype(np.float32)


def find_action_segments(energy: Sequence[float], threshold: float, min_duration: int, merge_gap: int) -> List[
    Tuple[int, int]]:
    active = np.asarray(energy) > threshold
    segments: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, flag in enumerate(active):
        if flag and start is None:
            if segments and i - segments[-1][1] <= merge_gap:
                start = segments.pop()[0]
            else:
                start = i
        elif not flag and start is not None:
            if i - start >= min_duration:
                segments.append((start, i))
            start = None
    if start is not None and len(active) - start >= min_duration:
        segments.append((start, len(active)))
    return segments


def fallback_segments(length: int, count: int) -> List[Tuple[int, int]]:
    count = max(1, count)
    edges = np.linspace(0, max(length, 1), count + 1).astype(int)
    return [(int(edges[i]), int(max(edges[i] + 1, edges[i + 1]))) for i in range(count)]


def parse_recording_id(recording_id: str) -> Tuple[Optional[str], int, int, int]:
    parts = recording_id.split("_")
    if len(parts) == 3:
        return None, int(parts[0]), int(parts[1]), int(parts[2])
    if len(parts) == 4:
        return parts[0], int(parts[1]), int(parts[2]), int(parts[3])
    raise ValueError(f"Unsupported recording id format: {recording_id}")


def read_default_subject_id(data_dir: Path) -> Optional[str]:
    candidates = list(data_dir.glob("*受试者*元数据*.xlsx")) + list(data_dir.glob("*subject*metadata*.xlsx"))
    if not candidates:
        return None
    try:
        df = pd.read_excel(candidates[0])
        if df.empty:
            return None
        id_cols = [c for c in df.columns if re.search(r"subject|受试者|编号|id", str(c), re.I)]
        col = id_cols[0] if id_cols else df.columns[0]
        values = [str(v) for v in df[col].dropna().unique()]
        return values[0] if len(values) == 1 else None
    except Exception:
        return None


def extract_imu(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    cleaned = {clean_column(c): c for c in df.columns}
    cols = []
    for _, keys in IMU_FEATURE_KEYWORDS:
        found = next((cleaned[k] for k in cleaned for key in keys if key in k), None)
        cols.append(found)
    warnings = []
    if any(c is None for c in cols):
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        used = {c for c in cols if c is not None}
        fallback_iter = (c for c in numeric if c not in used)
        cols = [c if c is not None else next(fallback_iter, None) for c in cols]
        warnings.append("IMU columns matched incompletely; used numeric fallback")
    values = np.zeros((len(df), 9), dtype=np.float32)
    for i, col in enumerate(cols[:9]):
        if col is not None and col in df:
            values[:, i] = pd.to_numeric(df[col], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    gyro_mag = np.sqrt(np.sum(values[:, 3:6] ** 2, axis=1))
    energy = pd.Series(gyro_mag).rolling(window=10, center=True, min_periods=1).mean().fillna(0).to_numpy()
    threshold = float(np.mean(energy) + 0.15 * np.std(energy)) if len(energy) else 0.0
    segments = find_segments_by_imu_peaks(energy, fs=100.0)
    if not segments:
        segments = fallback_segments(len(values), 1)
        warnings.append("IMU detection failed; used whole recording fallback")
    return values, energy, {"segments": segments, "threshold": threshold, "warnings": warnings,
                            "columns": [str(c) for c in cols[:9]]}


def extract_pose(video_path: Path, max_width: int, cache_dir: Path, use_cache: bool) -> Tuple[
    np.ndarray, np.ndarray, Dict]:
    cache_path = cache_dir / f"{video_path.stem}.npz"
    if use_cache and cache_path.exists():
        z = np.load(cache_path)
        return z["pose"], z["energy"], {"cache": "hit", "frames": int(z["pose"].shape[0])}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((0, 33, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), {"cache": "miss",
                                                                                          "warning": "video open failed"}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    scale = min(1.0, float(max_width) / float(width)) if max_width else 1.0
    pose_rows, energy = [], []
    prev_wrist = None
    last_pose = np.zeros((33, 3), dtype=np.float32)
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=False, model_complexity=0, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_world_landmarks:
                pts = np.array([
                    [lm.x, lm.y, lm.z]
                    for lm in res.pose_world_landmarks.landmark
                ], dtype=np.float32)
                wrist = res.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                cur = np.array([wrist.x * width, wrist.y * height], dtype=np.float32)
                energy.append(float(np.linalg.norm(cur - prev_wrist)) if prev_wrist is not None else 0.0)
                prev_wrist = cur
                last_pose = pts
            else:
                pts = last_pose.copy()
                energy.append(0.0)
                prev_wrist = None
            pose_rows.append(pts)
    cap.release()
    pose_arr = np.asarray(pose_rows, dtype=np.float32).reshape((-1, 33, 3)) if pose_rows else np.zeros((0, 33, 3),
                                                                                                       dtype=np.float32)
    energy_arr = pd.Series(energy).rolling(window=10, center=True, min_periods=1).mean().fillna(0).to_numpy(
        dtype=np.float32)
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, pose=pose_arr, energy=energy_arr)
    return pose_arr, energy_arr, {"cache": "miss", "frames": int(pose_arr.shape[0])}


def discover_recordings(data_dir: Path) -> Dict[str, Dict[str, Path]]:
    recs: Dict[str, Dict[str, Path]] = {}

    for ext in ["*.csv", "*.txt"]:
        for imu_path in data_dir.glob(ext):
            rid = imu_path.stem
            if rid.lower().startswith("calibration"):
                continue

            if imu_path.suffix.lower() == '.txt':
                converted_csv_str = check_and_convert_imu_path(imu_path)
                imu_path = Path(converted_csv_str)
                rid = imu_path.stem

            recs.setdefault(rid, {})["csv"] = imu_path

    # 💡 扫描视频文件
    for mp4 in data_dir.glob("*.mp4"):
        if mp4.stem.lower().startswith("calibration"):
            continue
        m = re.match(r"(.+)_Cam([AB])$", mp4.stem, re.I)
        if m:
            recs.setdefault(m.group(1), {})[f"cam_{m.group(2).lower()}"] = mp4

    return {k: v for k, v in recs.items() if ("csv" in v) or ("cam_a" in v) or ("cam_b" in v)}


def build_dataset(args: argparse.Namespace) -> Tuple[List[Dict], Dict]:
    data_dir = Path(args.data_dir)
    default_subject = read_default_subject_id(data_dir)
    recordings = discover_recordings(data_dir)
    dataset, rec_summaries, warnings = [], [], []
    cache_dir = data_dir / ".cache_pose"

    for rid in sorted(recordings):
        files = recordings[rid]
        try:
            sid_from_name, major, action, quality = parse_recording_id(rid)
            subject_id = sid_from_name or default_subject or "unknown"

            has_imu = "csv" in files
            has_cam_a = "cam_a" in files
            has_cam_b = "cam_b" in files
            has_video = has_cam_a or has_cam_b

            if has_imu:
                imu, imu_energy, imu_info = extract_imu(files["csv"])
            else:
                imu = np.zeros((0, 9), dtype=np.float32)
                imu_energy = None
                imu_info = {"segments": [], "warnings": ["IMU data is missing"]}

            if has_cam_a:
                pose_a, energy_a, info_a = extract_pose(files["cam_a"], args.max_width, cache_dir, not args.no_cache)
            else:
                pose_a, energy_a, info_a = np.zeros((0, 33, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), {}

            if has_cam_b:
                pose_b, energy_b, info_b = extract_pose(files["cam_b"], args.max_width, cache_dir, not args.no_cache)
            else:
                pose_b, energy_b, info_b = np.zeros((0, 33, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), {}

            ref_pose_len = len(pose_a) if has_cam_a else len(pose_b) if has_cam_b else 0

            scale_factor, offset_ratio = 1.0, 0.0

            if has_imu and has_video:
                ref_video_path = files["cam_a"] if has_cam_a else files["cam_b"]
                cap = cv2.VideoCapture(str(ref_video_path))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()
                if fps == 0 or np.isnan(fps): fps = 30.0

                ref_energy = energy_a if has_cam_a else energy_b
                scale_factor, offset_ratio = compute_auto_alignment_offset(imu_energy, ref_energy, fps)

                segments = imu_info["segments"] or fallback_segments(len(imu), args.min_segments)
                strategy = "imu_auto_aligned_mapping"

            elif has_video:
                ref_video_path = files["cam_a"] if has_cam_a else files["cam_b"]
                cap = cv2.VideoCapture(str(ref_video_path))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()
                if fps == 0 or np.isnan(fps): fps = 30.0

                ref_energy = energy_a if has_cam_a else energy_b
                median_window = max(3, int(0.12 * fps))
                ref_energy_f = pd.Series(ref_energy).rolling(window=median_window, center=True,
                                                             min_periods=1).median().fillna(0).to_numpy()
                mean_window = max(5, int(0.25 * fps))
                ref_energy_f = pd.Series(ref_energy_f).rolling(window=mean_window, center=True,
                                                               min_periods=1).mean().fillna(0).to_numpy()

                segments = find_action_segments_topological(
                    ref_energy_f, fps,
                    noise_floor=args.noise_floor,
                    relative_drop=args.relative_drop
                )
                if not segments:
                    segments = fallback_segments(ref_pose_len, args.min_segments)
                strategy = "pure_video_topological"

            else:
                segments = imu_info["segments"] or fallback_segments(len(imu), args.min_segments)
                strategy = "pure_imu_segmentation"

            if len(segments) < args.min_segments:
                ref_len = len(imu) if has_imu else ref_pose_len
                segments = fallback_segments(ref_len, args.min_segments)

            for idx, (s, e) in enumerate(segments):
                if has_imu and has_video:
                    imu_seg = resample_sequence(imu[s:e], args.imu_len, (9,))

                    rs = scale_factor * (s / max(len(imu), 1)) + offset_ratio
                    re_ = scale_factor * (e / max(len(imu), 1)) + offset_ratio
                    rs = max(0.0, min(1.0, rs))
                    re_ = max(0.0, min(1.0, re_))

                    va0 = int(rs * len(pose_a)) if has_cam_a else 0
                    va1 = int(max(rs * len(pose_a) + 1, re_ * len(pose_a))) if has_cam_a else 0
                    vb0 = int(rs * len(pose_b)) if has_cam_b else 0
                    vb1 = int(max(rs * len(pose_b) + 1, re_ * len(pose_b))) if has_cam_b else 0

                elif has_video:
                    imu_seg = np.zeros((args.imu_len, 9), dtype=np.float32)
                    va0, va1 = (s, e) if has_cam_a else (0, 0)
                    vb0, vb1 = (s, e) if has_cam_b else (0, 0)

                else:
                    imu_seg = resample_sequence(imu[s:e], args.imu_len, (9,))
                    va0, va1 = 0, 0
                    vb0, vb1 = 0, 0

                pose_a_raw = resample_sequence(pose_a[va0:va1], args.pose_len, (33, 3))
                pose_b_raw = resample_sequence(pose_b[vb0:vb1], args.pose_len, (33, 3))

                pose_a_seg = normalize_skeleton(pose_a_raw)
                pose_b_seg = normalize_skeleton(pose_b_raw)

                dataset.append({
                    "sample_id": f"{rid}_hit{idx:03d}", "recording_id": rid, "subject_id": subject_id,
                    "major_label": major, "action_label": action, "quality_label": quality,
                    "is_correct": quality == 0, "imu": imu_seg, "pose_cam_a": pose_a_seg,
                    "pose_cam_b": pose_b_seg, "video": pose_a_seg, "label": quality,
                    "binary_label": 0 if quality == 0 else 1,
                    "source_files": {k: str(v) for k, v in files.items()},
                    "segment_info": {
                        "imu_start": int(s) if has_imu else 0,
                        "imu_end": int(e) if has_imu else 0,
                        "cam_a_start": va0, "cam_a_end": va1,
                        "cam_b_start": vb0, "cam_b_end": vb1,
                        "strategy": strategy,
                        "scale_factor": float(scale_factor),
                        "offset_ratio": float(offset_ratio)
                    },
                    "quality_info": {"quality_label": quality, "is_correct": quality == 0},
                })

            rec_summaries.append({
                "recording_id": rid,
                "num_samples": len(segments),
                "imu_segments": len(imu_info["segments"]) if has_imu else 0,
                "cam_a_frames": info_a.get("frames", 0) if has_cam_a else 0,
                "cam_b_frames": info_b.get("frames", 0) if has_cam_b else 0,
                "strategy": strategy,
                "warnings": imu_info.get("warnings", [])
            })
            warnings.extend([f"{rid}: {w}" for w in imu_info.get("warnings", [])])
        except Exception as exc:
            warnings.append(f"{rid}: failed: {exc}")
            rec_summaries.append({"recording_id": rid, "num_samples": 0, "error": str(exc)})

    return dataset, {"data_dir": str(data_dir), "num_recordings": len(recordings), "num_samples": len(dataset),
                     "recordings": rec_summaries, "warnings": warnings}


def main() -> None:
    p = argparse.ArgumentParser(description="Generate formal tennis posture dataset with modal-adaptive alignment")
    p.add_argument("--data-dir", default="/home/yezi/Desktop/Professions/BJUT/Posture-Recognition/data_utils/datas")
    p.add_argument("--output", default="/home/yezi/Desktop/Professions/BJUT/Posture-Recognition/data_utils/tennis_dataset_v1.pkl")
    p.add_argument("--summary", default="/home/yezi/Desktop/Professions/BJUT/Posture-Recognition/data_utils/tennis_dataset_summary.json")
    p.add_argument("--imu-len", type=int, default=100)
    p.add_argument("--pose-len", type=int, default=50)
    p.add_argument("--max-width", type=int, default=640)
    p.add_argument("--min-segments", type=int, default=1)
    p.add_argument("--no-cache", action="store_true")

    p.add_argument("--noise-floor", type=float, default=0.15, help="Absolute noise floor (0.0 - 1.0)")
    p.add_argument("--relative-drop", type=float, default=0.60, help="Relative drop depth (0.0 - 1.0) to trigger split")

    args = p.parse_args()
    dataset, summary = build_dataset(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(dataset, f)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(dataset)} samples to {args.output}")
    print(f"Saved summary to {args.summary}")


if __name__ == "__main__":
    main()