import os
import cv2
import numpy as np
import mindspore
from mindspore import nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.train import Model
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.dataset import GeneratorDataset
import pycocotools.coco as coco
import mindspore.ops as ops
import mindspore.dataset as ds

from PoseTransformer import MediaPipePoseEstimator, PoseTransformer



# COCO Dataset Loader for Pose Estimation
class CocoPoseDataset:
    def __init__(self, coco_annotation_file, image_dir, pose_estimator):
        self.coco = coco.COCO(coco_annotation_file)
        self.image_dir = image_dir
        self.pose_estimator = pose_estimator

        self.image_ids = self.coco.getImgIds()
        self.image_data = []
        self.labels = []

        self.prepare_data()

    def prepare_data(self):
        for image_id in self.image_ids:
            image_info = self.coco.loadImgs(image_id)[0]
            img_path = os.path.join(self.image_dir, image_info['file_name'])
            img = cv2.imread(img_path)
            if img is None:
                continue

            skeleton = self.pose_estimator.extract_skeleton(img)
            if skeleton is None or skeleton.shape[0] != 33:
                continue  # Skip if skeleton is not valid

            # Normalize the skeleton points between 0 and 1
            skeleton /= np.array([img.shape[1], img.shape[0]])  # Normalize by width and height
            self.image_data.append(skeleton)

            # We can choose the 'keypoints' annotation for the labels
            annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
            keypoints = []
            for annotation in annotations:
                keypoints.append(annotation['keypoints'])
            self.labels.append(keypoints)

    def __getitem__(self, index):
        return self.image_data[index], self.labels[index]

    def __len__(self):
        return len(self.image_data)


# Example usage with COCO dataset
if __name__ == "__main__":
    # Initialize MediaPipe Pose Estimator
    pose_estimator = MediaPipePoseEstimator()

    # Load COCO Dataset
    coco_annotation_file = 'path_to_coco_annotations.json'  # Path to COCO annotations
    image_dir = 'path_to_coco_images'  # Path to COCO images directory
    coco_dataset = CocoPoseDataset(coco_annotation_file, image_dir, pose_estimator)

    # Prepare MindSpore Model
    model = PoseTransformer(cnn_out_channels=64, embed_dim=128, num_heads=4, num_layers=2)

    # Loss and Optimizer
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.001)

    # Train Data
    dataset = GeneratorDataset(coco_dataset, ["skeleton", "label"])
    dataset = dataset.batch(32)

    # Model Training Loop
    loss_monitor = LossMonitor(per_print_times=10)
    time_monitor = TimeMonitor(data_size=dataset.get_dataset_size())

    # Start Training
    model_with_loss = nn.WithLossCell(model, loss_fn)
    train_net = nn.TrainOneStepCell(model_with_loss, optimizer)

    model = Model(train_net)
    model.train(epochs=10, train_dataset=dataset, callbacks=[loss_monitor, time_monitor])

    # Save model
    mindspore.save_checkpoint(model, 'pose_transformer_model.ckpt')
    print("Model saved successfully!")
