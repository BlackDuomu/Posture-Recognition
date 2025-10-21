import mediapipe as mp
import cv2
import numpy as np
import os
from utils import Paths  # 使用统一路径工具类

# 初始化MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def extract_keypoints(video_paths, save_path: str = None):
    # 使用统一的关键点保存路径
    if save_path is None:
        save_path = Paths.ensure_dir(Paths.KEYPOINTS_DIR)
    else:
        save_path = Paths.ensure_dir(save_path)

    all_keypoints = []

    # 遍历视频文件列表
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)

        keypoints = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 转为RGB格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 获取骨架
            result = pose.process(rgb_frame)

            # 如果检测到姿态，保存关键点（使用 hasattr 防止静态检查告警&潜在空值）
            if hasattr(result, 'pose_landmarks') and result.pose_landmarks:
                frame_keypoints = []
                for landmark in result.pose_landmarks.landmark:
                    frame_keypoints.append([landmark.x, landmark.y, landmark.z])  # 保存x, y, z坐标
                keypoints.append(frame_keypoints)

        cap.release()

        # 获取视频文件的基本名称，去掉扩展名
        video_name = os.path.basename(video_path)
        video_name_no_extension = os.path.splitext(video_name)[0]

        # 保存该视频的骨架数据为NumPy数组（使用统一路径）
        np.save(Paths.keypoints_file(video_name_no_extension), np.array(keypoints))
        all_keypoints.append(np.array(keypoints))

        print(f"Extracted {len(keypoints)} frames from {video_name}")

    return all_keypoints


if __name__ == '__main__':
    # 示例视频路径（使用统一路径）
    video_paths = [
        os.path.join(Paths.VIDEOS_DIR, 'video_1.mp4'),
        os.path.join(Paths.VIDEOS_DIR, 'video_2.mp4'),
        os.path.join(Paths.VIDEOS_DIR, 'video_3.mp4'),
    ]
    # 确保关键点目录存在，并执行提取
    Paths.ensure_dir(Paths.KEYPOINTS_DIR)
    keypoints = extract_keypoints(video_paths, save_path=Paths.KEYPOINTS_DIR)
