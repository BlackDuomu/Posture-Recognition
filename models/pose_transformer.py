# models/pose_transformer.py
import torch.nn as nn


class PoseTransformer(nn.Module):
    def __init__(self, num_heads=9, num_layers=6, input_dim=99, seq_len=100):
        super(PoseTransformer, self).__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len

        # 校验：embed_dim 必须能被 num_heads 整除
        assert input_dim % num_heads == 0, (
            f"embed_dim (input_dim={input_dim}) must be divisible by num_heads (num_heads={num_heads})"
        )

        # Transformer Encoder层
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,  # 每帧特征维度（展平 33 关节 * 3 坐标 = 99）
            nhead=num_heads,
            dim_feedforward=256,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers
        )

        # 最后的二分类层（正确/错误）
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        # 期望输入 x 形状: (batch_size, seq_len, 99)
        x = x.permute(1, 0, 2)  # -> (seq_len, batch_size, 99)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # 时序平均池化
        x = self.fc(x)
        return x
