# data/dataset.py
import numpy as np
import os
from utils import Paths


def load_keypoints_and_labels():
    # 假设你的视频文件命名为 video_1.mp4, video_2.mp4, video_3.mp4
    video_files = ['video_1.mp4', 'video_2.mp4', 'video_3.mp4']

    # 使用统一的关键点与标签路径
    keypoints = [
        np.load(Paths.keypoints_file(os.path.splitext(video_name)[0]))
        for video_name in video_files
    ]
    labels = np.load(Paths.labels_file('labels.npy'))

    # 确保每个视频的骨架数据和标签一一对应（仅打印信息，便于调试）
    for i in range(len(keypoints)):
        print(f"Video {i+1}:")
        print(f"  Keypoints shape: {keypoints[i].shape}")
        print(f"  Label: {labels[i]}")

    return keypoints, labels

# 注意：不要在导入时执行加载，避免副作用。保留函数供训练/推理脚本调用。
