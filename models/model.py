import torch
import torch.nn as nn

from models.cross_modal_attention import CrossModalFusion
from models.dropout_mechanism import ModalityDropout
from models.imu_transformer import IMUTransformerEncoder
from models.pose_transformer import PoseTransformerEncoder


class TennisMultimodalTransformer(nn.Module):
    def __init__(self, num_classes=2, embed_dim=128, num_heads=4, num_layers=2,
                 fusion_layers=1, dropout=0.1, modality_dropout=0.25, share_pose_encoder=False,
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

        self.hierarchical = hierarchical
        if self.hierarchical:
            print(
                f"   [层次化多头配置] 大类数: {num_major_classes} | 小类数: {num_action_classes} | 纠错类数: {num_quality_classes}")
            self.fc_major = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_major_classes))
            self.fc_action = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_action_classes))
            self.fc_quality = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_quality_classes))
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, num_classes),
            )

    def forward(self, batch_or_imu, pose_cam_a=None, pose_cam_b=None):
        if isinstance(batch_or_imu, dict):
            imu = batch_or_imu['imu']
            pose_cam_a = batch_or_imu['pose_cam_a']
            pose_cam_b = batch_or_imu['pose_cam_b']
        else:
            imu = batch_or_imu
            if pose_cam_a is None or pose_cam_b is None:
                raise ValueError('pose_cam_a and pose_cam_b are required when not passing a batch dict')

        # imu shape: (B, 100, 9) | pose shape: (B, 50, 33, 3)
        imu_missing = (imu.abs().sum(dim=(1, 2)) == 0).unsqueeze(-1)               # (B, 1)
        cam_a_missing = (pose_cam_a.abs().sum(dim=(1, 2)) == 0).unsqueeze(-1)  # (B, 1)
        cam_b_missing = (pose_cam_b.abs().sum(dim=(1, 2)) == 0).unsqueeze(-1)  # (B, 1)

        imu_emb = self.imu_encoder(imu)                 # (B,D)
        imu_emb = imu_emb * (~imu_missing).to(imu_emb.dtype)

        cam_a_emb = self.pose_encoder_cam_a(pose_cam_a) # (B,D)
        cam_a_emb = cam_a_emb * (~cam_a_missing).to(cam_a_emb.dtype)

        cam_b_emb = self.pose_encoder_cam_b(pose_cam_b) # (B,D)
        cam_b_emb = cam_b_emb * (~cam_b_missing).to(cam_b_emb.dtype)

        camera_tokens = torch.stack([cam_a_emb, cam_b_emb], dim=1)
        pose_emb = self.pose_fusion(self.camera_dropout(camera_tokens))

        modality_tokens = torch.stack([imu_emb, pose_emb], dim=1)
        fused = self.cross_modal_fusion(self.modality_dropout(modality_tokens))  # (B, D)

        if self.hierarchical:
            return {
                'major': self.fc_major(fused),
                'action': self.fc_action(fused),
                'quality': self.fc_quality(fused)
            }
        else:
            return self.classifier(fused)


MultimodalPostureModel = TennisMultimodalTransformer