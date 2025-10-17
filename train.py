import os
import cv2
import json
import numpy as np
import mindspore
from mindspore import nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.train import Model
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.dataset import GeneratorDataset
from PoseTransformer import MediaPipePoseEstimator, PoseTransformer

# 视频标签数据存储路径
VIDEO_PATH = 'path_to_your_video_directory'
LABELS_PATH = 'path_to_your_label_jsons'


# 视频帧与标签数据提取器
class VideoPoseDataset:
    def __init__(self, video_path, labels_path, pose_estimator):
        self.video_path = video_path
        self.labels_path = labels_path
        self.pose_estimator = pose_estimator
        self.video_data = []
        self.labels = []
        self.prepare_data()

    def prepare_data(self):
        # 加载所有视频文件
        video_files = [f for f in os.listdir(self.video_path) if f.endswith('.mp4')]

        for video_file in video_files:
            # 读取标签
            label_file = os.path.join(self.labels_path, video_file.replace('.mp4', '.json'))
            with open(label_file, 'r') as f:
                video_labels = json.load(f)

            # 读取视频并提取骨架数据
            video_path = os.path.join(self.video_path, video_file)
            cap = cv2.VideoCapture(video_path)
            frame_id = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # 视频读取完毕

                # 提取骨架数据
                skeleton = self.pose_estimator.extract_skeleton(frame)
                if skeleton is None or skeleton.shape[0] != 33:
                    continue  # 跳过无效骨架数据

                # 标准化骨架数据
                skeleton /= np.array([frame.shape[1], frame.shape[0]])  # 归一化

                # 获取当前帧的标签
                current_label = next(item for item in video_labels["frames"] if item["frame_id"] == frame_id)
                keypoints = current_label["keypoints"]
                action_type = current_label["action_type"]

                # 将骨架和标签加入数据集
                self.video_data.append(skeleton)
                self.labels.append({"keypoints": keypoints, "action_type": action_type})
                frame_id += 1
            cap.release()

    def __getitem__(self, index):
        return self.video_data[index], self.labels[index]

    def __len__(self):
        return len(self.video_data)


# 训练模型
if __name__ == "__main__":
    # 初始化 MediaPipe Pose Estimator
    pose_estimator = MediaPipePoseEstimator()

    # 加载视频数据和标签
    dataset = VideoPoseDataset('path_to_your_video_directory', 'path_to_your_label_jsons', pose_estimator)

    # 准备 MindSpore 模型
    model = PoseTransformer(cnn_out_channels=64, embed_dim=128, num_heads=4, num_layers=2)

    # 损失函数和优化器
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)

    # 创建数据集
    mindspore_dataset = GeneratorDataset(dataset, ["skeleton", "label"])
    mindspore_dataset = mindspore_dataset.batch(32)

    # 训练过程
    loss_monitor = LossMonitor(per_print_times=10)
    time_monitor = TimeMonitor(data_size=mindspore_dataset.get_dataset_size())

    # 训练模型
    model_with_loss = nn.WithLossCell(model, loss_fn)
    train_net = nn.TrainOneStepCell(model_with_loss, optimizer)

    model = Model(train_net)
    model.train(epochs=10, train_dataset=mindspore_dataset, callbacks=[loss_monitor, time_monitor])

    # 保存模型
    mindspore.save_checkpoint(model, 'pose_transformer_model.ckpt')
    print("Model checkpoint saved successfully!")
