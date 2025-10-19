# models/pose_transformer.py
import torch
import torch.nn as nn


class PoseTransformer(nn.Module):
    def __init__(self, num_heads=8, num_layers=6, input_dim=33, seq_len=100):
        super(PoseTransformer, self).__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len

        # Transformer Encoder层
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,  # 输入的特征维度
            nhead=num_heads,  # 多头注意力机制的头数
            dim_feedforward=256,  # 前馈网络的维度
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers  # Transformer的层数
        )

        # 最后的分类层
        self.fc = nn.Linear(input_dim, 2)  # 输出2类（正确/错误）

    def forward(self, x):
        # x的形状为(batch_size, seq_len, input_dim)
        x = x.permute(1, 0, 2)  # PyTorch中的Transformer要求输入的形状为(seq_len, batch_size, input_dim)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # 在seq_len维度上取平均池化
        x = self.fc(x)
        return x
