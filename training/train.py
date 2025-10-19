# training/train.py
import torch
import torch.optim as optim
import torch.nn as nn
from models.pose_transformer import PoseTransformer
from torch.utils.data import DataLoader, Dataset
from data.dataset import load_keypoints_and_labels

class PoseDataset(Dataset):
    def __init__(self, keypoints, labels):
        self.keypoints = keypoints
        self.labels = labels

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, idx):
        return torch.tensor(self.keypoints[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# 加载数据
keypoints, labels = load_keypoints_and_labels()

# 准备数据加载器
train_loader = DataLoader(PoseDataset(keypoints, labels), batch_size=32, shuffle=True)

# 模型、损失函数与优化器
model = PoseTransformer(input_dim=33)  # 假设每帧有33个关键点
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练过程
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for data, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(data)

        loss = loss_fn(outputs, labels)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# 保存模型
torch.save(model.state_dict(), 'pose_transformer.pth')
