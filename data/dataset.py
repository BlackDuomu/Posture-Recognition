# data/dataset.py
import numpy as np

def load_keypoints_and_labels():
    # 假设你的视频文件命名为 video_1.mp4, video_2.mp4, video_3.mp4
    video_files = ['video_1.mp4', 'video_2.mp4', 'video_3.mp4']  # 需要与你的实际视频文件匹配
    keypoints = [np.load(f'keypoints/{video_name.split(".")[0]}_keypoints.npy') for video_name in video_files]
    labels = np.load('labels/labels.npy')  # 加载标签数据

    # 确保每个视频的骨架数据和标签一一对应
    for i in range(len(keypoints)):
        print(f"Video {i+1}:")
        print(f"  Keypoints shape: {keypoints[i].shape}")
        print(f"  Label: {labels[i]}")

    return keypoints, labels

# 调用函数以加载数据
keypoints, labels = load_keypoints_and_labels()
