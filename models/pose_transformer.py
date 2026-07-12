import torch
import torch.nn as nn


class PoseTransformerEncoder(nn.Module):
    def __init__(self, input_dim=99, seq_len=50, embed_dim=128, num_heads=4,
                 num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.proj = nn.Linear(input_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        if x.dim() == 4:
            x = x.flatten(start_dim=2)  # (B,T,33,3) -> (B,T,99)
        x = self.proj(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.encoder(x)
        return self.norm(x.mean(dim=1))


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
