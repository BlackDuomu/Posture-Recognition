import numpy as np

def create_labels():
    # 假设我们有3个视频：video_1.mp4, video_2.mp4, video_3.mp4
    # 这些视频的标签可能是：1（正确的跑步动作），0（错误的深蹲动作），2（跳跃动作）
    labels = {
        'video_1.mp4': 1,  # 1表示正确的跑步动作
        'video_2.mp4': 0,  # 0表示错误的深蹲动作
        'video_3.mp4': 2   # 2表示跳跃动作
    }

    # 创建一个标签数组，标签对应每个视频
    video_files = ['video_1.mp4', 'video_2.mp4', 'video_3.mp4']
    labels_array = [labels[video] for video in video_files]

    # 保存标签到npy文件
    np.save('labels/labels.npy', np.array(labels_array))

    print("Labels created and saved to 'labels.npy'")

create_labels()
