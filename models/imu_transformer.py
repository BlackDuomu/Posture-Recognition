import torch
import torch.nn as nn


class IMUTransformerEncoder(nn.Module):
    def __init__(self, input_dim=9, seq_len=100, embed_dim=128, num_heads=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_norm = nn.InstanceNorm1d(input_dim, affine=True)
        self.proj = nn.Linear(input_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (B, T, C) -> e.g. (B, 100, 9)
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.input_norm(x)
        x = x.transpose(1, 2)  # (B, T, C)

        x = self.proj(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.encoder(x)
        return self.norm(x.mean(dim=1))
