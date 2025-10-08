import cv2
import mediapipe as mp
import numpy as np
import mindspore
from mindspore import nn
from mindspore import Tensor
import mindspore.ops as ops
from mindspore import dtype as mstype


# MediaPipe Pose Estimator for Skeleton Extraction
class MediaPipePoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_skeleton(self, image):
        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Process the image and get the result
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            skeleton = []
            for landmark in results.pose_landmarks.landmark:
                skeleton.append([landmark.x, landmark.y])  # (x, y) coordinates
            return np.array(skeleton)
        else:
            return np.zeros((33, 2))  # Return 33 keypoints (x, y), 0 if no pose is detected


# CNN Module for Spatial Feature Extraction
class CNNFeatureExtractor(nn.Cell):
    def __init__(self, in_channels=3, out_channels=64):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, pad_mode="pad", padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, 3, pad_mode="pad", padding=1)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        return x


# Transformer Encoder for Temporal Feature Learning
class TransformerEncoder(nn.Cell):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Dense(33, embed_dim)  # Assuming 33 keypoints for skeleton
        self.positional_encoding = nn.Parameter(Tensor(np.zeros((1, 33, embed_dim)), dtype=mstype.float32))

        self.transformer_layers = nn.CellList([
            nn.TransformerEncoderLayer(
                embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.fc_out = nn.Dense(embed_dim, 2)  # Output layer: 2 for classification (correct/incorrect)

    def construct(self, x):
        x = self.embedding(x)
        x += self.positional_encoding

        for layer in self.transformer_layers:
            x = layer(x)

        x = self.fc_out(x[:, -1, :])  # Use the last token's output for classification
        return x


# Pose Transformer Model combining CNN and Transformer
class PoseTransformer(nn.Cell):
    def __init__(self, cnn_out_channels=64, embed_dim=128, num_heads=4, num_layers=2):
        super(PoseTransformer, self).__init__()
        self.cnn_extractor = CNNFeatureExtractor(in_channels=3, out_channels=cnn_out_channels)
        self.transformer_encoder = TransformerEncoder(embed_dim, num_heads, num_layers)

    def construct(self, x):
        # Extract spatial features using CNN
        spatial_features = self.cnn_extractor(x)
        # Reshape to match the transformer input
        spatial_features = spatial_features.view(-1, 33,
                                                 spatial_features.shape[-1])  # (batch_size, seq_len, feature_dim)

        # Apply Transformer to capture temporal features
        output = self.transformer_encoder(spatial_features)
        return output



