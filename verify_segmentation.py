# verify_segmentation.py
import argparse
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal  # 需要使用 scipy 寻找波峰

from data_utils.dataset_generator import extract_imu, extract_pose, find_action_segments_topological, compute_auto_alignment_offset


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
    """将视频的指定帧片段导出为新的 mp4 文件"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
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


def main():
    parser = argparse.ArgumentParser(description="Verify Dynamic Segmentation and Alignment across Multi-channels")
    parser.add_argument('--csv', default=None, help="Path to IMU CSV file (Optional)")
    parser.add_argument('--cam-a', default=None, help="Path to CamA video file (Optional)")
    parser.add_argument('--cam-b', default='/home/yezi/Desktop/Professions/BJUT/Posture-Recognition/data_utils/datas/tests/20260702142802.mp4', help="Path to CamB video file (Optional)")
    parser.add_argument('--output-dir', default='debug_output', help="Directory to save plots and video clips")

    # 拓扑微调参数
    parser.add_argument('--noise-floor', type=float, default=0.15, help="Absolute noise floor (0.0 - 1.0)")
    parser.add_argument('--relative-drop', type=float, default=0.60,
                        help="Relative drop depth (0.0 - 1.0) to trigger split")
    args = parser.parse_args()

    has_imu = args.csv is not None
    has_cam_a = args.cam_a is not None
    has_cam_b = args.cam_b is not None

    if not has_imu and not has_cam_a and not has_cam_b:
        print("❌ 错误：请提供至少一种通道数据！")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = Path(args.csv).stem if has_imu else (Path(args.cam_a).stem if has_cam_a else Path(args.cam_b).stem)

    fps = 30.0
    if has_cam_a or has_cam_b:
        ref_video_path = args.cam_a if has_cam_a else args.cam_b
        cap = cv2.VideoCapture(str(ref_video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps == 0 or np.isnan(fps):
            fps = 30.0

    imu_vals, imu_energy = None, None
    pose_a, energy_a = None, None
    pose_b, energy_b = None, None

    segments = []
    segment_source = "imu" if has_imu else "video"

    # 1. 提取 IMU
    if has_imu:
        # ==== 新增：在此处执行自动检测并转换 ====
        csv_path_to_use = check_and_convert_imu_path(args.csv)
        print("1. 正在提取 IMU 时序特征...")
        imu_vals, imu_energy, imu_info = extract_imu(Path(csv_path_to_use))
        segments = imu_info["segments"]
        print(f"   -> [IMU] 定位到 {len(segments)} 个击球。")

    # 2. 提取视觉
    if has_cam_a:
        print("2. 正在提取 Cam A 视觉姿态特征 (BlazePose)...")
        pose_a, energy_a, _ = extract_pose(Path(args.cam_a), max_width=640, cache_dir=Path('.'), use_cache=False)
        # 异常极大值截断
        energy_a = np.clip(energy_a, 0, np.percentile(energy_a, 98))

    if has_cam_b:
        print("3. 正在提取 Cam B 视觉姿态特征 (BlazePose)...")
        pose_b, energy_b, _ = extract_pose(Path(args.cam_b), max_width=640, cache_dir=Path('.'), use_cache=False)
        # 异常极大值截断
        energy_b = np.clip(energy_b, 0, np.percentile(energy_b, 98))

    # 3. 纯视觉自适应拓扑分割
    if not has_imu:
        print("1. [纯视觉拓扑场景] 正在启动拓扑波峰合并引擎...")
        ref_energy = energy_a if has_cam_a else energy_b
        ref_pose = pose_a if has_cam_a else pose_b
        video_path_ref = args.cam_a if has_cam_a else args.cam_b

        cap = cv2.VideoCapture(str(video_path_ref))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps == 0 or np.isnan(fps): fps = 30.0
        print(f"   [检测到视频帧率]: {fps} FPS")

        median_window = max(3, int(0.12 * fps))
        ref_energy_f = pd.Series(ref_energy).rolling(window=median_window, center=True,
                                                     min_periods=1).median().fillna(0).to_numpy()
        mean_window = max(5, int(0.25 * fps))
        ref_energy_f = pd.Series(ref_energy_f).rolling(window=mean_window, center=True,
                                                       min_periods=1).mean().fillna(0).to_numpy()

        # 💡 执行新算法：拓扑自适应波峰合并
        segments = find_action_segments_topological(
            ref_energy_f, fps,
            noise_floor=args.noise_floor,
            relative_drop=args.relative_drop
        )
        print(f"   -> [拓扑分割引擎] 成功定位到 {len(segments)} 个自成一体的击球动作！")

    # 4. 💡 核心：计算自动对齐时序偏置（Auto-Alignment via Cross-Correlation）
    offset_ratio = 0.0
    if has_imu and (has_cam_a or has_cam_b):
        print("4. 🔄 [自动对齐引擎] 正在计算 IMU 与视频之间的互相关物理偏置...")
        ref_energy = energy_a if has_cam_a else energy_b
        scale_factor, offset_ratio = compute_auto_alignment_offset(imu_energy, ref_energy, fps)
        print(f"   -> 检测到时序偏置: {offset_ratio:.4%} (IMU 相比视频滞后了 {offset_ratio*100:.2f} % 相对时长)")

    # 5. 绘图与可视化（带偏置修正）
    print("5. 正在生成自动对齐图表...")
    plt.figure(figsize=(12, 6))

    if has_imu:
        # ⚠️ 画图时：将 IMU 的 X 轴时间比例向左平移 offset_ratio，实现几何对齐
        x_imu_aligned = (scale_factor * np.linspace(0, 1.0, len(imu_energy)) + offset_ratio) * 100
        norm_imu_energy = imu_energy / (np.max(imu_energy) + 1e-6)
        plt.plot(x_imu_aligned, norm_imu_energy, label='IMU Energy (Auto-Aligned)', color='blue', alpha=0.6)

    if has_cam_a:
        x_a = np.linspace(0, 100, len(energy_a))
        norm_a_energy = energy_a / (np.max(energy_a) + 1e-6)
        plt.plot(x_a, norm_a_energy, label='Cam A Pose Energy', color='red', alpha=0.6)

    if has_cam_b:
        x_b = np.linspace(0, 100, len(energy_b))
        norm_b_energy = energy_b / (np.max(energy_b) + 1e-6)
        plt.plot(x_b, norm_b_energy, label='Cam B Pose Energy', color='magenta', alpha=0.6)

    plt.axhline(args.noise_floor, color='gray', linestyle=':', label='Noise Floor (0.15)')

    # 绘制阴影覆盖区（应用自动对齐偏置）
    for idx, (s, e) in enumerate(segments):
        if segment_source == "imu":
            # IMU 动作边界通过减去对齐偏置，精确地套在视频波峰上
            rs = scale_factor * (s / max(len(imu_vals), 1)) + offset_ratio
            re_ = scale_factor * (e / max(len(imu_vals), 1)) + offset_ratio
        else:
            rs, re_ = s / max(len(ref_pose), 1), e / max(len(ref_pose), 1)

        plt.axvspan(rs * 100, re_ * 100, color='green', alpha=0.15, label='Extracted Segment' if idx == 0 else "")
        plt.text((rs + re_) / 2 * 100, 0.9, f"Hit {idx}", ha='center', color='green', fontweight='bold')

    plt.title(f"Auto-Aligned Topological Segmentation: {base_name}")
    plt.xlabel("Aligned Relative Timeline (%)")
    plt.ylabel("Normalized Kinetic Energy")
    plt.legend(loc='upper right')
    plot_path = os.path.join(args.output_dir, f"{base_name}_auto_aligned.png")
    plt.savefig(plot_path)
    print(f"   -> 自动对准图表已保存至: {plot_path}")

    # 6. 精准切片视频导出（带偏置修正）
    print(f"6. 正在利用对准参数，精准导出 {len(segments)} 个视频切片...")
    for idx, (s, e) in enumerate(segments):
        if segment_source == "imu":
            rs = scale_factor * (s / max(len(imu_vals), 1)) + offset_ratio
            re_ = scale_factor * (e / max(len(imu_vals), 1)) + offset_ratio
            # 安全越界约束
            rs = max(0.0, min(1.0, rs))
            re_ = max(0.0, min(1.0, re_))
        else:
            rs, re_ = s / max(len(ref_pose), 1), e / max(len(ref_pose), 1)

        if has_cam_a:
            va0 = int(rs * len(pose_a))
            va1 = int(max(rs * len(pose_a) + 1, re_ * len(pose_a)))
            out_a = os.path.join(args.output_dir, f"{base_name}_hit{idx:03d}_CamA.mp4")
            export_video_segment(args.cam_a, va0, va1, out_a)

        if has_cam_b:
            vb0 = int(rs * len(pose_b))
            vb1 = int(max(rs * len(pose_b) + 1, re_ * len(pose_b)))
            out_b = os.path.join(args.output_dir, f"{base_name}_hit{idx:03d}_CamB.mp4")
            export_video_segment(args.cam_b, vb0, vb1, out_b)


if __name__ == "__main__":
    main()