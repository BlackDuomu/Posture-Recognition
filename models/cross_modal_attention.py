import torch.nn as nn


class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=1, dim_feedforward=256, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens, active_mask=None):
        if active_mask is not None:
            padding_mask = ~active_mask.bool()
            fused = self.encoder(tokens, src_key_padding_mask=padding_mask)

            mask_expanded = active_mask.unsqueeze(-1).to(fused.dtype)  # (B, M, 1)
            sum_fused = (fused * mask_expanded).sum(dim=1)
            denom = mask_expanded.sum(dim=1).clamp(min=1.0)
            return self.norm(sum_fused / denom)
        else:
            fused = self.encoder(tokens)
            return self.norm(fused.mean(dim=1))


CrossModalAttention = CrossModalFusion
