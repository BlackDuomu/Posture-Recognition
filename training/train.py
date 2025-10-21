# training/train.py
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.pose_transformer import PoseTransformer
from data.dataset import load_keypoints_and_labels
from utils import Paths


def flatten_keypoints_per_frame(seq: np.ndarray) -> np.ndarray:
    """
    将关键点序列从 (T, 33, 3) 展平为 (T, 99)。
    """
    if seq.ndim != 3 or seq.shape[1:] != (33, 3):
        raise ValueError(f"Expected keypoints shape (T,33,3), got {seq.shape}")
    T = seq.shape[0]
    return seq.reshape(T, 33 * 3)


class PoseSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = [flatten_keypoints_per_frame(s) for s in sequences]  # list of (T, 99)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)  # (T, 99)
        y = torch.tensor(self.labels[idx], dtype=torch.long)        # scalar 0/1
        return x, y


def pad_truncate_batch(batch, seq_len: int):
    """
    将一批变长序列填充/截断到统一长度 seq_len。
    输入: list[(T_i, 99), label]
    输出: (B, seq_len, 99), (B,)
    """
    xs, ys = zip(*batch)
    B = len(xs)
    feat_dim = xs[0].shape[1]
    out = torch.zeros((B, seq_len, feat_dim), dtype=torch.float32)
    for i, x in enumerate(xs):
        T = x.shape[0]
        if T >= seq_len:
            out[i] = x[:seq_len]
        else:
            out[i, :T] = x
    y = torch.stack(ys, dim=0)
    return out, y


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) 加载数据 (list of np.array(T,33,3), labels np.array(N,))
    keypoints_list, labels = load_keypoints_and_labels()

    # 2) 构建数据集/加载器
    seq_len = 100  # 统一序列长度，可根据数据分布调整
    dataset = PoseSequenceDataset(keypoints_list, labels)
    train_loader = DataLoader(
        dataset,
        batch_size=2,  # 样例数据较小，设置为2；可根据显存调整
        shuffle=True,
        collate_fn=lambda batch: pad_truncate_batch(batch, seq_len=seq_len),
    )

    # 3) 模型/损失/优化器
    model = PoseTransformer(input_dim=99)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()  # 二分类，用 2 类 logits + CE
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 4) 训练循环
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)          # (B, 2)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    # 5) 保存权重到统一 checkpoints 目录
    Paths.ensure_dir(Paths.CHECKPOINTS_DIR)
    ckpt_path = os.path.join(Paths.CHECKPOINTS_DIR, 'pose_transformer.pth')
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}")


if __name__ == '__main__':
    main()
