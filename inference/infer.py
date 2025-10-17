# inference/infer.py
import torch
from models.pose_transformer import PoseTransformer
import numpy as np

# 加载训练好的模型
model = PoseTransformer(input_dim=33)
model.load_state_dict(torch.load('pose_transformer.pth'))
model.eval()

# 假设你有一个测试视频并提取了骨架关键点
test_keypoints = np.load('test_keypoints.npy')
test_data = torch.tensor(test_keypoints, dtype=torch.float32).view(1, -1, 33)

# 推理
with torch.no_grad():
    output = model(test_data)

# 输出预测结果
predicted_class = output.argmax(dim=1).item()
print(f"Predicted Class: {predicted_class}")
