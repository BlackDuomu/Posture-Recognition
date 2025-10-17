# data/dataset.py
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class PoseDataset(Dataset):
    def __init__(self, keypoints_path, labels_path):
        self.keypoints = np.load(keypoints_path)  # 加载骨架数据
        self.labels = np.load(labels_path)  # 加载标签数据

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, idx):
        return torch.tensor(self.keypoints[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def create_data_loader(keypoints_path, labels_path, batch_size=32):
    dataset = PoseDataset(keypoints_path, labels_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
