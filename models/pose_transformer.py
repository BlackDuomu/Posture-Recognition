import torch
import torch.nn as nn


class RelationalSpatialBlock(nn.Module):
    """
    负责单帧内 33个关节点 的门控与相对关系提取
    """

    def __init__(self, num_joints=33, in_features=6, d_model=32, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_joints = num_joints

        # 1. 空间门控网络 (Spatial Gating)
        # 接收每个节点的特征，输出0~1的门控权重
        self.gate_proj = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.GELU(),
            nn.Linear(in_features, 1)
        )

        # 2. 节点特征升维
        self.feature_proj = nn.Linear(in_features, d_model)

        # 3. 相对关系自注意力 (Relational Attention)
        # 用于捕捉身体部位之间的动力链关系（如：腰的转动与手臂挥动的关系）
        self.relational_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 4. 局部前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x):
        # x shape: (B, T, V, C) -> e.g. (B, 50, 33, 6)
        b, t, v, c = x.shape

        # --- A. 空间门控与残差 (Spatial Gating with Residual) ---
        # 门控权重 (B, T, 33, 1)
        gate = torch.sigmoid(self.gate_proj(x))

        # 残差设计：X_out = X + X * Gate
        # 即使 Gate 为 0，仍保留原始特征，避免梯度消失死锁
        x_gated = x + x * gate

        # --- B. 提取相对关系自注意力 (Relational Attention) ---
        # 融合 B 和 T 维度，只在空间(关节点V)维度之间做 Attention
        x_flat = x_gated.view(b * t, v, c)
        h = self.feature_proj(x_flat)  # (B*T, 33, d_model)

        # 让 33 个节点互相计算注意力，寻找当前帧最关联的节点对
        attn_out, _ = self.relational_attn(h, h, h)

        # Add & Norm (注意力层的残差)
        h = self.norm1(h + attn_out)

        # FFN 层与残差
        ffn_out = self.ffn(h)
        h = self.norm2(h + ffn_out)

        # 恢复形状 -> (B, T, V * d_model) 用于后续的时间全局Transformer
        out = h.view(b, t, v * self.feature_proj.out_features)
        return out


class PoseTransformerEncoder(nn.Module):
    def __init__(self, input_dim=99, seq_len=50, embed_dim=128, num_heads=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.num_joints = 33

        # 定义网球动作中最关键的 12 根骨骼（由哪两个 MediaPipe 关节点连接）
        # 11/12肩, 13/14肘, 15/16腕, 23/24髋, 25/26膝, 27/28踝
        self.bone_pairs = [
            (11, 12),  # 肩膀宽度
            (11, 13), (13, 15),  # 左手臂（左上臂、左前臂）
            (12, 14), (14, 16),  # 右手臂（右上臂、右前臂）
            (11, 23), (12, 24),  # 躯干两侧
            (23, 24),  # 髋关节宽度
            (23, 25), (25, 27),  # 左腿（左大腿、左小腿）
            (24, 26), (26, 28)  # 右腿（右大腿、右小腿）
        ]

        # 12 根骨头 * 3维向量 = 36维 特征
        bone_dim = len(self.bone_pairs) * 3

        # 投影层：将 36 维骨骼方向特征映射到 Transformer 的 embed_dim
        self.proj = nn.Linear(bone_dim, embed_dim)

        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (B, T, 33, 3) 或者是压平的 (B, T, 99)
        if x.dim() == 3:
            b, t, _ = x.shape
            x = x.reshape(b, t, self.num_joints, 3)

        b, t, v, c = x.shape
        assert v == self.num_joints, f"期望的关节点数为 {self.num_joints}，但输入为 {v}"
        assert c == 3, f"输入的骨骼坐标必须为 3D 空间坐标(x, y, z)，但输入为 {c}D"

        # === 核心算法：提取 100% 尺度不变的骨骼单位方向向量 ===
        bone_vectors = []
        for parent, child in self.bone_pairs:
            # 提取两个关节点的 3D 坐标 -> (B, T, 3)
            p_coord = x[:, :, parent, :]
            c_coord = x[:, :, child, :]

            # 计算骨骼指向向量
            vector = c_coord - p_coord

            # 归一化为单位向量 (消除骨骼长度差异)
            norm = torch.linalg.norm(vector, dim=-1, keepdim=True) + 1e-6
            unit_vector = vector / norm  # (B, T, 3)

            bone_vectors.append(unit_vector)

        # 拼接 12 根骨头 -> Shape: (B, T, 36)
        x_bones = torch.cat(bone_vectors, dim=-1)

        # === 送入 Transformer ===
        x_emb = self.proj(x_bones)  # (B, T, embed_dim)
        x_emb = x_emb + self.pos_embedding[:, :x_emb.size(1), :]
        out = self.encoder(x_emb)
        return self.norm(out.mean(dim=1))


class PoseTransformer(nn.Module):
    def __init__(self, num_classes=2, classify=False, **kwargs):
        super().__init__()
        self.encoder = PoseTransformerEncoder(**kwargs)
        self.classify = classify
        embed_dim = kwargs.get('embed_dim', 128)
        self.fc = nn.Linear(embed_dim, num_classes) if classify else nn.Identity()

    def forward(self, x):
        emb = self.encoder(x)
        return self.fc(emb)