import torch
import torch.nn as nn


class ModalityDropout(nn.Module):

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, tokens):
        # tokens shape: (B, M, D)，M:（IMU, FusedCamera）
        if not self.training or self.p <= 0:
            return tokens

        b, m, d = tokens.shape

        already_missing = (tokens.abs().sum(dim=-1) < 1e-5)

        keep = (torch.rand(b, m, device=tokens.device) > self.p)

        keep = keep & (~already_missing)

        all_dropped = ~keep.any(dim=1)
        if all_dropped.any():
            for i in range(b):
                if all_dropped[i]:
                    active_indices = torch.where(~already_missing[i])[0]
                    if len(active_indices) > 0:
                        restore_idx = active_indices[torch.randint(0, len(active_indices), (1,)).item()]
                        keep[i, restore_idx] = True

        return tokens * keep.unsqueeze(-1).to(tokens.dtype)