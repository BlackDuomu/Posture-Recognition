import mediapipe as mp
import cv2
import numpy as np
import os

# 初始化MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def extract_keypoints(video_paths, save_path='data/keypoints'):
    # 创建保存路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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

            # 如果检测到姿态，保存关键点
            if result.pose_landmarks:
                frame_keypoints = []
                for landmark in result.pose_landmarks.landmark:
                    frame_keypoints.append([landmark.x, landmark.y, landmark.z])  # 保存x, y, z坐标
                keypoints.append(frame_keypoints)

        cap.release()

        # 获取视频文件的基本名称，去掉 .mp4 后缀
        video_name = os.path.basename(video_path)  # 获取视频文件名
        video_name_no_extension = os.path.splitext(video_name)[0]  # 去掉文件扩展名（例如 .mp4）

        # 保存该视频的骨架数据为NumPy数组
        np.save(os.path.join(save_path, f'{video_name_no_extension}_keypoints.npy'), np.array(keypoints))
        all_keypoints.append(np.array(keypoints))

        print(f"Extracted {len(keypoints)} frames from {video_name}")

    return all_keypoints


# 示例视频路径
video_paths = ['videos/video_1.mp4', 'videos/video_2.mp4', 'videos/video_3.mp4']
keypoints = extract_keypoints(video_paths)
