import torch
import torch.nn as nn

from models.cross_modal_attention import CrossModalFusion
from models.dropout_mechanism import ModalityDropout
from models.imu_transformer import IMUTransformerEncoder
from models.pose_transformer import PoseTransformerEncoder


class TennisMultimodalTransformer(nn.Module):
    def __init__(self, num_classes=2, embed_dim=128, num_heads=4, num_layers=2,
                 fusion_layers=1, dropout=0.2, modality_dropout=0.15, share_pose_encoder=False,
                 hierarchical=True, num_major_classes=3, num_action_classes=3, num_quality_classes=7):
        super().__init__()

        self.imu_encoder = IMUTransformerEncoder(embed_dim=embed_dim, num_heads=num_heads,
                                                 num_layers=num_layers, dropout=dropout)
        self.pose_encoder_cam_a = PoseTransformerEncoder(embed_dim=embed_dim, num_heads=num_heads,
                                                         num_layers=num_layers, dropout=dropout)
        self.pose_encoder_cam_b = self.pose_encoder_cam_a if share_pose_encoder else PoseTransformerEncoder(
            embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout
        )

        self.camera_dropout = ModalityDropout(modality_dropout)
        self.modality_dropout = ModalityDropout(modality_dropout)
        self.pose_fusion = CrossModalFusion(embed_dim=embed_dim, num_heads=num_heads,
                                            num_layers=fusion_layers, dropout=dropout)
        self.cross_modal_fusion = CrossModalFusion(embed_dim=embed_dim, num_heads=num_heads,
                                                   num_layers=fusion_layers, dropout=dropout)

        self.hand_embed = nn.Embedding(2, embed_dim)
        self.backhand_embed = nn.Embedding(2, embed_dim)
        self.ntrp_proj = nn.Linear(1, embed_dim)
        self.meta_fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        self.hierarchical = hierarchical
        if self.hierarchical:
            self.fc_major = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_major_classes)
            )

            self.fc_action = nn.Sequential(
                nn.LayerNorm(embed_dim + num_major_classes),
                nn.Dropout(dropout),
                nn.Linear(embed_dim + num_major_classes, num_action_classes)
            )

            self.fc_quality = nn.Sequential(
                nn.LayerNorm(embed_dim + num_major_classes + num_action_classes),
                nn.Dropout(dropout),
                nn.Linear(embed_dim + num_major_classes + num_action_classes, num_quality_classes)
            )
        else:
            self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))

    def forward(self, batch_or_imu, pose_cam_a=None, pose_cam_b=None,
                hand_idx=None, backhand_idx=None, ntrp_val=None):

        if isinstance(batch_or_imu, dict):
            imu = batch_or_imu['imu']
            pose_cam_a = batch_or_imu['pose_cam_a']
            pose_cam_b = batch_or_imu['pose_cam_b']
            hand_idx = batch_or_imu.get('dominant_hand')
            backhand_idx = batch_or_imu.get('backhand_type')
            ntrp_val = batch_or_imu.get('ntrp')
        else:
            imu = batch_or_imu
            if pose_cam_a is None or pose_cam_b is None:
                raise ValueError('pose_cam_a and pose_cam_b are required when not passing a batch dict')

        device = imu.device
        b = imu.size(0)

        if hand_idx is None:
            hand_idx = torch.zeros(b, dtype=torch.long, device=device)
        if backhand_idx is None:
            backhand_idx = torch.zeros(b, dtype=torch.long, device=device)
        if ntrp_val is None:
            ntrp_val = torch.full((b, 1), 3.0, dtype=torch.float32, device=device)
        if ntrp_val.dim() == 1:
            ntrp_val = ntrp_val.unsqueeze(-1)

        hand_token = self.hand_embed(hand_idx)
        backhand_token = self.backhand_embed(backhand_idx)
        ntrp_token = self.ntrp_proj(ntrp_val)
        meta_token = self.meta_fusion(torch.cat([hand_token, backhand_token, ntrp_token], dim=-1)).unsqueeze(1)

        imu_missing = (imu.abs().reshape(imu.size(0), -1).sum(dim=1) == 0).unsqueeze(-1)
        cam_a_missing = (pose_cam_a.abs().reshape(pose_cam_a.size(0), -1).sum(dim=1) == 0).unsqueeze(-1)
        cam_b_missing = (pose_cam_b.abs().reshape(pose_cam_b.size(0), -1).sum(dim=1) == 0).unsqueeze(-1)

        imu_emb = self.imu_encoder(imu)
        imu_emb = imu_emb * (~imu_missing).to(imu_emb.dtype)

        cam_a_emb = self.pose_encoder_cam_a(pose_cam_a)
        cam_a_emb = cam_a_emb * (~cam_a_missing).to(cam_a_emb.dtype)

        cam_b_emb = self.pose_encoder_cam_b(pose_cam_b)
        cam_b_emb = cam_b_emb * (~cam_b_missing).to(cam_b_emb.dtype)

        camera_tokens = torch.stack([cam_a_emb, cam_b_emb], dim=1)
        pose_emb = self.pose_fusion(self.camera_dropout(camera_tokens))

        modality_tokens = torch.stack([meta_token.squeeze(1), imu_emb, pose_emb], dim=1)
        fused = self.cross_modal_fusion(self.modality_dropout(modality_tokens))  # (B, 128)

        if self.hierarchical:
            major_logits = self.fc_major(fused)  # (B, 3)
            major_probs = torch.softmax(major_logits, dim=-1)

            action_input = torch.cat([fused, major_probs], dim=-1)  # (B, 128 + 3)
            action_logits = self.fc_action(action_input)  # (B, 3)
            action_probs = torch.softmax(action_logits, dim=-1)

            quality_input = torch.cat([fused, major_probs, action_probs], dim=-1)  # (B, 128 + 3 + 3)
            quality_logits = self.fc_quality(quality_input)  # (B, 7)

            return {
                'major': major_logits,
                'action': action_logits,
                'quality': quality_logits
            }
        else:
            return self.classifier(fused)


MultimodalPostureModel = TennisMultimodalTransformer