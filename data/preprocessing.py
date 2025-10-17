# data/preprocessing.py
import mediapipe as mp
import cv2
import numpy as np
import os

# 初始化MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def extract_keypoints(video_path, save_path='keypoints'):
    cap = cv2.VideoCapture(video_path)

    # 创建保存路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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
                frame_keypoints.append([landmark.x, landmark.y, landmark.z])
            keypoints.append(frame_keypoints)

    cap.release()

    # 将骨架数据保存为NumPy数组
    np.save(os.path.join(save_path, 'keypoints.npy'), np.array(keypoints))

    return np.array(keypoints)

def load_labels(labels_path):
    return np.load(labels_path)
