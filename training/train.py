# training/train.py
import torch
import torch.optim as optim
import torch.nn as nn
from models.pose_transformer import PoseTransformer
from data.data_loader import create_data_loader

# 设定训练设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
keypoints_path = 'keypoints.npy'
labels_path = 'labels.npy'
train_loader = create_data_loader(keypoints_path, labels_path)

# 初始化模型、损失函数与优化器
model = PoseTransformer(input_dim=33)  # 假设每帧有33个关键点
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练过程
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(data)

        # 计算损失
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()

        # 反向传播与优化
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# 保存模型
torch.save(model.state_dict(), 'pose_transformer.pth')
