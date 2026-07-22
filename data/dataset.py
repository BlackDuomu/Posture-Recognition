import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import Paths


class TennisMultimodalDataset(Dataset):
    def __init__(self, dataset_path, label_key='class_id', is_training=False):
        self.is_training = is_training
        self.dataset_path = dataset_path
        self.label_key = label_key
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            data = data.get('samples', data.get('data', data.get('items', data)))
        if not isinstance(data, (list, tuple)):
            raise ValueError(f"Expected pickle to contain a list of samples, got {type(data)}")
        self.samples = list(data)

        self.sample_label_codes = [self._make_label_code(sample) for sample in self.samples]
        unique_codes = sorted(set(self.sample_label_codes), key=self._code_sort_key)
        self.code_to_class_id = {code: i for i, code in enumerate(unique_codes)}
        self.class_id_to_code = {i: code for code, i in self.code_to_class_id.items()}

        if label_key in ('class_id', 'label_code', 'composite'):
            self.value_to_class_id = self.code_to_class_id
        else:
            raw_values = sorted({sample.get(label_key, 0) for sample in self.samples})
            self.value_to_class_id = {value: i for i, value in enumerate(raw_values)}
            self.code_to_class_id = {str(value): i for value, i in self.value_to_class_id.items()}
            self.class_id_to_code = {i: str(value) for value, i in self.value_to_class_id.items()}

    @staticmethod
    def _make_label_code(sample):
        major = int(sample.get('major_label', 0))
        action = int(sample.get('action_label', 0))
        quality = int(sample.get('quality_label', sample.get('label', 0)))
        return f'{major}_{action}_{quality}'

    @staticmethod
    def _code_sort_key(code):
        return tuple(int(part) for part in str(code).split('_'))

    @property
    def num_classes(self):
        return len(self.class_id_to_code)

    def get_label_mapping(self):
        return {
            'code_to_class_id': dict(self.code_to_class_id),
            'class_id_to_code': dict(self.class_id_to_code),
            'label_key': self.label_key,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pose_a = np.asarray(sample['pose_cam_a'], dtype=np.float32).reshape(-1, 33, 3)
        pose_b = np.asarray(sample['pose_cam_b'], dtype=np.float32).reshape(-1, 33, 3)

        if self.is_training and len(pose_a) > 0:
            scale = np.random.uniform(0.95, 1.05)
            pose_a = pose_a * scale
            pose_b = pose_b * scale

            noise_a = np.random.normal(0, 0.01, size=pose_a.shape).astype(np.float32)
            noise_b = np.random.normal(0, 0.01, size=pose_b.shape).astype(np.float32)
            pose_a = pose_a + noise_a
            pose_b = pose_b + noise_b

        label_code = self.sample_label_codes[idx]
        if self.label_key in ('class_id', 'label_code', 'composite'):
            label_value = self.value_to_class_id[label_code]
        else:
            raw_value = sample.get(self.label_key, 0)
            label_value = self.value_to_class_id[raw_value]

        major_value = sample.get('major_label', 0)
        action_value = sample.get('action_label', 0)
        quality_value = sample.get('quality_label', sample.get('label', 0))

        metadata = {
            k: v for k, v in sample.items()
            if k not in {'imu', 'pose_cam_a', 'pose_cam_b', 'label', 'binary_label',
                         'major_label', 'action_label', 'quality_label'}
        }

        return {
            'imu': torch.as_tensor(np.asarray(sample['imu']), dtype=torch.float32),
            'pose_cam_a': torch.as_tensor(pose_a, dtype=torch.float32),
            'pose_cam_b': torch.as_tensor(pose_b, dtype=torch.float32),
            'label': torch.tensor(label_value, dtype=torch.long),
            'label_code': label_code,
            'major_label': torch.tensor(major_value, dtype=torch.long),
            'action_label': torch.tensor(action_value, dtype=torch.long),
            'quality_label': torch.tensor(quality_value, dtype=torch.long),
            'metadata': metadata,
        }


def load_keypoints_and_labels():
    video_files = ['video_1.mp4', 'video_2.mp4', 'video_3.mp4']

    keypoints = [
        np.load(Paths.keypoints_file(os.path.splitext(video_name)[0]))
        for video_name in video_files
    ]
    labels = np.load(Paths.labels_file('labels.npy'))

    for i in range(len(keypoints)):
        print(f"Video {i+1}:")
        print(f"  Keypoints shape: {keypoints[i].shape}")
        print(f"  Label: {labels[i]}")

    return keypoints, labels
