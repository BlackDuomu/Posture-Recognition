import numpy as np
from utils import Paths


def create_labels():
    # 二分类标签：1=正确动作，0=错误/其它动作
    labels_map = {
        'video_1.mp4': 1,  # 正确示例
        'video_2.mp4': 0,  # 错误示例
        'video_3.mp4': 0   # 其它/错误示例统一归为 0
    }

    video_files = ['video_1.mp4', 'video_2.mp4', 'video_3.mp4']
    labels_array = [labels_map[video] for video in video_files]

    Paths.ensure_dir(Paths.LABELS_DIR)
    np.save(Paths.labels_file('labels.npy'), np.array(labels_array))

    print(f"Binary labels created and saved to '{Paths.labels_file('labels.npy')}'")


create_labels()
