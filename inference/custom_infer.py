# custom_infer.py
import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import cv2
import scipy.signal

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_utils.dataset_generator import extract_imu, extract_pose, resample_sequence, fallback_segments, compute_auto_alignment_offset
from models.model import TennisMultimodalTransformer


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

def check_and_convert_imu_path(file_path):
    """
    💡 自动转换器：检测输入文件是否为维特智能原厂 .txt 格式，
    如果是，则自动将其转换并重构为标准的逗号分隔 .csv 文件。
    """
    path = Path(file_path)
    if path.suffix.lower() == '.txt':
        csv_path = path.with_suffix('.csv')
        print(f"🔄 检测到维特智能原厂 .txt 格式，正在自动将其重构并转换为标准 .csv 格式...")

        # 1. 读取 Tab 分隔的 .txt 文件
        df = pd.read_csv(path, sep='\t', encoding='utf-8-sig')
        df.columns = [str(c).strip() for c in df.columns]

        # 2. 定义标准模板的 20 列顺序
        target_cols = [
            '时间', '设备名称', '片上时间()',
            '加速度X(g)', '加速度Y(g)', '加速度Z(g)',
            '角速度X(°/s)', '角速度Y(°/s)', '角速度Z(°/s)',
            '角度X(°)', '角度Y(°)', '角度Z(°)',
            '磁场X(ʯt)', '磁场Y(ʯt)', '磁场Z(ʯt)',
            '温度(℃)',
            '四元数0()', '四元数1()', '四元数2()', '四元数3()'
        ]

        # 3. 建立原厂不规范符号映射
        rename_dict = {
            '磁场X(uT)': '磁场X(ʯt)',
            '磁场Y(uT)': '磁场Y(ʯt)',
            '磁场Z(uT)': '磁场Z(ʯt)',
            '温度(°C)': '温度(℃)'
        }
        df = df.rename(columns=rename_dict)

        # 4. 构建重组后的目标 DataFrame
        out_df = pd.DataFrame()
        for col in target_cols:
            if col == '时间':
                # 提取纯时间部分 (14:28:02.521)
                out_df['时间'] = df['时间'].apply(lambda x: str(x).split('T')[-1] if 'T' in str(x) else str(x))
            elif col == '设备名称':
                # 截断 MAC 物理地址，仅保留设备名前缀
                out_df['设备名称'] = df['设备名称'].apply(lambda x: str(x).split('(')[0] if '(' in str(x) else str(x))
            elif col == '片上时间()':
                # 格式化转换 (2026-07-02 14:28:02:521)
                out_df['片上时间()'] = df['时间'].apply(
                    lambda x: str(x).replace('T', ' ').replace('.', ':') if isinstance(x, str) else x)
            elif col in df.columns:
                out_df[col] = df[col]
            else:
                # 若源文件缺少该字段，填充 null 字符串占位
                out_df[col] = "null"

        # 5. 保存为标准逗号分隔的 .csv 格式
        out_df.to_csv(csv_path, index=False, na_rep='null')
        print(f"   -> 格式化重构成功！新文件已自动保存至: {csv_path}")
        return str(csv_path)

    return file_path

def export_video_segment(video_path, start_frame, end_frame, output_path):
    """将视频的指定帧区间导出为新的 mp4 文件"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()

def find_action_segments_topological(energy, fps, noise_floor=0.15, relative_drop=0.60):
    max_val = np.max(energy) + 1e-6
    norm_energy = energy / max_val

    min_dist_frames = max(5, int(0.5 * fps))
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

        drop_ratio = (shorter_peak_val - valley_val) / (shorter_peak_val + 1e-6)
        time_gap_seconds = (next_p - curr_p) / fps

        if drop_ratio < relative_drop or time_gap_seconds < 1.2:
            new_p = curr_p if norm_energy[curr_p] > norm_energy[next_p] else next_p
            peak_bounds[i] = [curr_l, next_r, new_p]
            peak_bounds.pop(i + 1)
        else:
            i += 1

    min_duration_frames = max(5, int(0.25 * fps))
    final_segments = []
    for l, r, p in peak_bounds:
        if (r - l) >= min_duration_frames:
            final_segments.append((l, r))

    return final_segments


def main():
    parser = argparse.ArgumentParser(description='Tennis Multimodal Unified Inference Tool')
    parser.add_argument('--checkpoint',
                        default='/home/yezi/Desktop/Professions/BJUT/Posture-Recognition/checkpoints/tennis_multimodal_transformer.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--imu-csv', default='/home/yezi/Desktop/Professions/BJUT/Posture-Recognition/data_utils/datas/tests/20260702142802.csv', help='Path to raw IMU CSV (Optional)')
    parser.add_argument('--cam-a', default=None, help='Path to CamA video (Optional)')
    parser.add_argument('--cam-b', default='/home/yezi/Desktop/Professions/BJUT/Posture-Recognition/data_utils/datas/tests/20260702142802.mp4', help='Path to CamB video (Optional)')

    # 开放拓扑自适应微调参数
    parser.add_argument('--noise-floor', type=float, default=0.15, help="Absolute noise floor (0.0 - 1.0)")
    parser.add_argument('--relative-drop', type=float, default=0.60,
                        help="Relative drop depth (0.0 - 1.0) to trigger split")
    # 开放集未知动作判定阈值
    parser.add_argument('--conf-threshold', type=float, default=0.50,
                        help='Confidence threshold to filter out unknown movements')
    parser.add_argument('--output-dir', default='inference_output', help='Directory to save predicted video segments')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载模型配置
    print("1. 正在加载多模态模型权重...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    hierarchical = checkpoint.get('hierarchical', True)

    if hierarchical:
        model = TennisMultimodalTransformer(
            hierarchical=True,
            num_major_classes=checkpoint.get('num_major_classes', 3),
            num_action_classes=checkpoint.get('num_action_classes', 3),
            num_quality_classes=checkpoint.get('num_quality_classes', 7)
        ).to(device)
    else:
        num_classes = checkpoint.get('num_classes', 5)
        model = TennisMultimodalTransformer(num_classes=num_classes).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    has_imu = args.imu_csv is not None
    has_cam_a = args.cam_a is not None
    has_cam_b = args.cam_b is not None
    has_video = has_cam_a or has_cam_b

    if not has_imu and not has_video:
        print("❌ 错误：请提供 --imu-csv、--cam-a 或 --cam-b 的其中至少一种输入数据！")
        return

    imu_vals, imu_segments = None, []
    pose_a, energy_a = None, None
    pose_b, energy_b = None, None

    if has_imu:
        csv_path_to_use = check_and_convert_imu_path(args.imu_csv)

        print("2. [IMU 提取] 正在处理 IMU 时序数据并运行波峰分切...")
        imu_vals, imu_energy, imu_info = extract_imu(Path(csv_path_to_use))
        imu_segments = imu_info["segments"]
        if not imu_segments:
            imu_segments = fallback_segments(len(imu_vals), 1)
        print(f"   -> IMU 成功分割出 {len(imu_segments)} 个动作片段。")

    if has_video:
        if has_cam_a:
            print("2. [视觉提取] 正在处理 Cam A 3D 姿态流并计算运动能量...")
            pose_a, energy_a, _ = extract_pose(Path(args.cam_a), max_width=640, cache_dir=Path('.'), use_cache=False)
            energy_a = np.clip(energy_a, 0, np.percentile(energy_a, 98))

            cap = cv2.VideoCapture(str(args.cam_a))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps == 0 or np.isnan(fps): fps = 30.0

            median_window = max(3, int(0.12 * fps))
            energy_a = pd.Series(energy_a).rolling(window=median_window, center=True, min_periods=1).median().fillna(
                0).to_numpy()
            mean_window = max(5, int(0.25 * fps))
            energy_a = pd.Series(energy_a).rolling(window=mean_window, center=True, min_periods=1).mean().fillna(
                0).to_numpy()

        if has_cam_b:
            print("2. [视觉提取] 正在处理 Cam B 3D 姿态流并计算运动能量...")
            pose_b, energy_b, _ = extract_pose(Path(args.cam_b), max_width=640, cache_dir=Path('.'), use_cache=False)
            # 异常限幅
            energy_b = np.clip(energy_b, 0, np.percentile(energy_b, 98))

            cap = cv2.VideoCapture(str(args.cam_b))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps == 0 or np.isnan(fps): fps = 30.0

            # 双重滤波
            median_window = max(3, int(0.12 * fps))
            energy_b = pd.Series(energy_b).rolling(window=median_window, center=True, min_periods=1).median().fillna(
                0).to_numpy()
            mean_window = max(5, int(0.25 * fps))
            energy_b = pd.Series(energy_b).rolling(window=mean_window, center=True, min_periods=1).mean().fillna(
                0).to_numpy()

        # 视界缺省时的单视角克隆逻辑
        if has_cam_a and not has_cam_b:
            print("   ⚠️ 提示：缺省 Cam B。")
            pose_b = None
        elif not has_cam_a and has_cam_b:
            print("   ⚠️ 提示：缺省 Cam A。")
            pose_a = None

    final_segments = []  # 格式：[(imu_start, imu_end, cam_start, cam_end), ...]

    scale_factor = 1.0
    offset_ratio = 0.0

    if has_imu and has_video:
        ref_video_path = args.cam_a if has_cam_a else args.cam_b
        cap = cv2.VideoCapture(str(ref_video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps == 0 or np.isnan(fps):
            fps = 30.0

        ref_energy = energy_a if has_cam_a else energy_b
        print("🔄 [自动对齐引擎] 正在计算 IMU 与视频之间的双锚点仿射偏置...")
        scale_factor, offset_ratio = compute_auto_alignment_offset(imu_energy, ref_energy, fps)

    if has_imu:
        # 有 IMU 时以 IMU 相对时间比例加上对齐参数映射至相机序列
        for s, e in imu_segments:
            rs = scale_factor * (s / max(len(imu_vals), 1)) + offset_ratio
            re_ = scale_factor * (e / max(len(imu_vals), 1)) + offset_ratio
            rs = max(0.0, min(1.0, rs))
            re_ = max(0.0, min(1.0, re_))

            ref_len = len(pose_a) if has_cam_a else len(pose_b) if has_cam_b else 0
            va0 = int(rs * ref_len)
            va1 = int(max(rs * ref_len + 1, re_ * ref_len))
            final_segments.append((s, e, va0, va1))
    else:
        # 纯视觉时：执行全新拓扑自适应波峰合并算法
        print("2. [纯视觉场景] 无 IMU，正在启动拓扑波峰合并引擎...")
        ref_energy = energy_a if energy_a is not None else energy_b  # 此时 pose_a 与 pose_b 必定等长且经过克隆/滤波
        video_path_for_split = args.cam_a if has_cam_a else args.cam_b

        cap = cv2.VideoCapture(str(video_path_for_split))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps == 0 or np.isnan(fps): fps = 30.0
        print(f"   [检测到视频帧率]: {fps} FPS")

        # 调用拓扑动作定位算法
        video_segments = find_action_segments_topological(
            ref_energy, fps,
            noise_floor=args.noise_floor,
            relative_drop=args.relative_drop
        )
        if not video_segments:
            video_segments = fallback_segments(len(pose_a), 1)

        print(f"   [拓扑分割配置] noise_floor={args.noise_floor}, relative_drop={args.relative_drop}")
        print(f"   -> 视觉成功分割出 {len(video_segments)} 个击球动作片段。")
        for s, e in video_segments:
            final_segments.append((0, 0, s, e))

    print("\n3. 🚀 开始时序预测（含未知动作阈值过滤）...")

    # 静态物理分类定义字典
    major_map = {0: "正手", 1: "反手", 2: "发球"}
    action_map = {0: "正手上旋球", 1: "反手抽球", 2: "平击or侧旋发球"}
    quality_map = {0: "标准动作", 1: "引拍过晚", 2: "纯手臂发力",
                   3: "击球点过近", 4: "击球点过远", 5: "发球错误-托盘式", 6: "发球错误-抛球过低"}

    for idx, (is_, ie_, vs_, ve_) in enumerate(final_segments):
        # IMU 数据载入
        if has_imu:
            imu_seg = resample_sequence(imu_vals[is_:ie_], 100, (9,))
            imu_tensor = torch.as_tensor(imu_seg, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            imu_tensor = torch.zeros((1, 100, 9), dtype=torch.float32).to(device)

        # 视频骨架载入
        if has_video:
            if pose_a is not None:
                pose_a_raw = resample_sequence(pose_a[vs_:ve_], 50, (33, 3))
                # 直接保留原始骨架 (1, 50, 33, 3)，模型内部会自动展平成 99 维
                pose_a_tensor = torch.as_tensor(pose_a_raw, dtype=torch.float32).unsqueeze(0).to(device)
            else:
                # 缺省时，将占位全零张量的形状也改回原始 3D 骨架形状 (1, 50, 33, 3)
                pose_a_tensor = torch.zeros((1, 50, 33, 3), dtype=torch.float32).to(device)

            if pose_b is not None:
                pose_b_raw = resample_sequence(pose_b[vs_:ve_], 50, (33, 3))
                # 直接保留原始骨架 (1, 50, 33, 3)
                pose_b_tensor = torch.as_tensor(pose_b_raw, dtype=torch.float32).unsqueeze(0).to(device)
            else:
                # 缺省时，将占位全零张量的形状也改回原始 3D 骨架形状 (1, 50, 33, 3)
                pose_b_tensor = torch.zeros((1, 50, 33, 3), dtype=torch.float32).to(device)

        # 模型前向推理
        with torch.no_grad():
            hand_tensor = torch.tensor([0], dtype=torch.long).to(device)  # 1表示左撇子
            backhand_tensor = torch.tensor([0], dtype=torch.long).to(device)  # 1表示单反
            ntrp_tensor = torch.tensor([[4]], dtype=torch.float32).to(device)  # 2.5分

            outputs = model(
                imu_tensor,
                pose_a_tensor,
                pose_b_tensor,
                hand_idx=hand_tensor,
                backhand_idx=backhand_tensor,
                ntrp_val=ntrp_tensor
            )

            if hierarchical:
                prob_major_2d = torch.softmax(outputs['major'], dim=1)
                prob_action_2d = torch.softmax(outputs['action'], dim=1)
                prob_quality_2d = torch.softmax(outputs['quality'], dim=1)

                conf_major, idx_major = prob_major_2d.max(dim=1)
                conf_action, idx_action = prob_action_2d.max(dim=1)
                conf_quality, idx_quality = prob_quality_2d.max(dim=1)

                prob_major = prob_major_2d[0].cpu().numpy()
                prob_action = prob_action_2d[0].cpu().numpy()
                prob_quality = prob_quality_2d[0].cpu().numpy()

                major_dist = sorted(
                    {major_map[i]: float(prob) for i, prob in enumerate(prob_major)}.items(),
                    key=lambda x: x[1], reverse=True
                )
                action_dist = sorted(
                    {action_map[i]: float(prob) for i, prob in enumerate(prob_action)}.items(),
                    key=lambda x: x[1], reverse=True
                )
                quality_dist = sorted(
                    {quality_map[i]: float(prob) for i, prob in enumerate(prob_quality)}.items(),
                    key=lambda x: x[1], reverse=True
                )

                # 提取置信度最高的结果用于终端快速打印和安全文件名生成
                pred_major_str = major_dist[0][0] if major_dist[0][1] >= args.conf_threshold else "未知大类"
                pred_action_str = action_dist[0][0] if action_dist[0][1] >= args.conf_threshold else "未知小类"
                pred_quality_str = quality_dist[0][0] if quality_dist[0][1] >= args.conf_threshold else "未知姿态"

                print(f"🎬 片段 #{idx:02d} | 预测：{pred_major_str} -> {pred_action_str} | 最可能状态：{pred_quality_str}")

                # 组装成标准的结构化 JSON 数据，用于直接喂给 LLM Agent
                segment_report = {
                    "segment_id": idx,
                    "timestamps": {
                        "video_start_frame": int(vs_),
                        "video_end_frame": int(ve_)
                    },
                    "predictions": {
                        "major_stroke_category": [{"class": k, "probability": f"{v:.2%}"} for k, v in major_dist],
                        "detailed_action_type": [{"class": k, "probability": f"{v:.2%}"} for k, v in action_dist],
                        "movement_quality_feedback": [{"class": k, "probability": f"{v:.2%}"} for k, v in quality_dist]
                    }
                }

                import json
                print("📝 [LLM 智能体数据载荷 (JSON Payload)]:")
                print(json.dumps(segment_report, ensure_ascii=False, indent=2))
                print("=" * 80)

                # ==================== 新增：自动视频切片导出逻辑 ====================
                if has_video:
                    # 确保保存目录存在
                    os.makedirs(args.output_dir, exist_ok=True)

                    # 构造您指定的文件名（使用Windows安全的全角 “｜” 和 “：”）
                    safe_filename = (
                        f"片段_{idx:02d}_"
                        f"大类_{pred_major_str}_小类_{pred_action_str}_"
                        f"状态_{pred_quality_str}_"
                        f"置信度_大类{conf_major.item():.1f}_小类{conf_action.item():.1%}_纠错{conf_quality.item():.1%}.mp4"
                    )

                    # 如果有 Cam A，导出 Cam A 的切片
                    if has_cam_a:
                        out_path_a = os.path.join(args.output_dir, f"CamA_{safe_filename}")
                        export_video_segment(args.cam_a, vs_, ve_, out_path_a)

                    # 如果有 Cam B，导出 Cam B 的切片
                    if has_cam_b:
                        out_path_b = os.path.join(args.output_dir, f"CamB_{safe_filename}")
                        export_video_segment(args.cam_b, vs_, ve_, out_path_b)
            else:
                probs = torch.softmax(outputs, dim=1)
                conf, pred_class = probs.max(dim=1)
                pred_class = int(pred_class.item())
                if conf.item() < args.conf_threshold:
                    pred_code = "未知动作/未知姿态"
                else:
                    # 使用 Checkpoint 中保存的 code 映射
                    pred_code = checkpoint['class_id_to_code'].get(str(pred_class), f"Class {pred_class}")
                print(f"🎬 片段 #{idx:02d} | 预测结果：{pred_code} | 置信度：{conf.item():.4%}")

if __name__ == '__main__':
    main()