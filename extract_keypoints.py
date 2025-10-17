# scripts/extract_keypoints.py
import os, cv2, numpy as np, mediapipe as mp

def extract_video_kpts(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    kpts_list, valid_list = [], []
    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks:
            pts = [[lm.x, lm.y, lm.visibility] for lm in res.pose_landmarks.landmark]  # 33x3
            kpts_list.append(pts); valid_list.append(True)
        else:
            kpts_list.append([[0.,0.,0.]]*33); valid_list.append(False)
    cap.release()
    kpts = np.array(kpts_list, dtype=np.float32)              # [T,33,3], x/y 已是 [0,1] 归一化坐标
    valid = np.array(valid_list, dtype=bool)                  # [T]
    return kpts, valid, (H, W)

def process_folder(video_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(video_dir):
        if not fn.lower().endswith(('.mp4','.avi','.mov')): continue
        vid = os.path.splitext(fn)[0]
        kpts, valid, (H, W) = extract_video_kpts(os.path.join(video_dir, fn))
        np.savez_compressed(os.path.join(out_dir, f"{vid}.npz"),
                            kpts=kpts, valid=valid, orig_size=np.array([H, W], np.int32))
        print("saved:", vid, kpts.shape, valid.mean())

if __name__ == "__main__":
    process_folder("data/videos", "data/keypoints")
