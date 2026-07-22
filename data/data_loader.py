# data/data_loader.py
import math
import random
import torch
from torch.utils.data import DataLoader, Subset, random_split
from data.dataset import TennisMultimodalDataset


def tennis_collate_fn(batch):
    return {
        'imu': torch.stack([item['imu'] for item in batch], dim=0),
        'pose_cam_a': torch.stack([item['pose_cam_a'] for item in batch], dim=0),
        'pose_cam_b': torch.stack([item['pose_cam_b'] for item in batch], dim=0),
        'label': torch.stack([item['label'] for item in batch], dim=0),
        'label_code': [item['label_code'] for item in batch],
        'major_label': torch.stack([item['major_label'] for item in batch], dim=0),
        'action_label': torch.stack([item['action_label'] for item in batch], dim=0),
        'quality_label': torch.stack([item['quality_label'] for item in batch], dim=0),
        # 将 metadata 中的字符规则，转化为可计算的 Tensor
        'dominant_hand': torch.tensor(
            [1 if '左' in item['metadata'].get('dominant_hand', '') else 0 for item in batch],
            dtype=torch.long
        ),
        'backhand_type': torch.tensor(
            [1 if '单' in item['metadata'].get('backhand_type', '') else 0 for item in batch],
            dtype=torch.long
        ),
        'ntrp': torch.tensor(
            [float(item['metadata'].get('ntrp', 3.0)) for item in batch],
            dtype=torch.float32
        ),

        'metadata': [item['metadata'] for item in batch],
    }


def create_tennis_dataloaders(
        dataset_path,
        batch_size=8,
        val_ratio=0.2,
        test_ratio=0.0,
        seed=42,
        label_key='class_id',
        split_by=None,
        is_training=False
):
    dataset = TennisMultimodalDataset(dataset_path, label_key=label_key, is_training=is_training)
    n = len(dataset)
    if n == 0:
        raise ValueError(f"Empty dataset: {dataset_path}")

    if split_by is None:
        test_size = int(math.floor(n * test_ratio)) if test_ratio > 0 else 0
        val_size = int(math.floor(n * val_ratio)) if val_ratio > 0 else 0
        if val_ratio > 0 and n - test_size > 1:
            val_size = max(1, val_size)
        train_size = n - val_size - test_size
        if train_size <= 0:
            train_size = 1
            val_size = max(0, n - train_size)
            test_size = 0

        generator = torch.Generator().manual_seed(seed)
        train_set, val_set, test_set = random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )
    else:
        groups = []
        for sample in dataset.samples:
            group_val = sample.get(split_by)
            if group_val is None and 'metadata' in sample:
                group_val = sample['metadata'].get(split_by)
            groups.append(str(group_val) if group_val is not None else 'unknown')

        unique_groups = sorted(list(set(groups)))

        random.seed(seed)
        random.shuffle(unique_groups)

        num_groups = len(unique_groups)
        val_group_size = max(1, int(num_groups * val_ratio)) if val_ratio > 0 else 0
        test_group_size = int(num_groups * test_ratio) if test_ratio > 0 else 0
        train_group_size = num_groups - val_group_size - test_group_size

        train_groups = set(unique_groups[:train_group_size])
        val_groups = set(unique_groups[train_group_size:train_group_size + val_group_size])
        test_groups = set(unique_groups[train_group_size + val_group_size:])

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        val_indices = [i for i, g in enumerate(groups) if g in val_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        if not train_indices and val_indices:
            train_indices, val_indices = val_indices, train_indices

        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices) if val_indices else None
        test_set = Subset(dataset, test_indices) if test_indices else None

        print(f"=== 进行严格的 Group Split 划分 [{split_by}] ===")
        print(f"  总独立组数: {num_groups} | 划分结果: 训练组 {len(train_groups)}个, 验证组 {len(val_groups)}个")
        print(f"  训练集样本数: {len(train_indices)} (包含组: {list(train_groups)})")
        print(f"  验证集样本数: {len(val_indices)} (包含组: {list(val_groups)})")
        print("==========================================")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=tennis_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            collate_fn=tennis_collate_fn) if val_set else None
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             collate_fn=tennis_collate_fn) if test_set else None
    return train_loader, val_loader, test_loader