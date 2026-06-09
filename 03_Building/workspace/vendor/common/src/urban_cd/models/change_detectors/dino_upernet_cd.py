from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..backbones.dinov3 import DINOv3FeatureExtractor
from ..decoders.upernet import UPerNetDecoder
from ..necks.temporal_fusion import TemporalConcatDiffFusion


class DINOv3UPerNetChangeDetector(nn.Module):
    """Frozen DINOv3 encoder + temporal fusion + UPerNet binary CD baseline."""

    def __init__(
        self,
        checkpoint_path: str,
        output_layers: Sequence[int],
        freeze: bool,
        drop_path_rate: float,
        unfreeze_last_n_blocks: int,
        lora: dict | None,
        fusion_out_channels: int,
        scale_factors: Sequence[float],
        ppm_bins: Sequence[int],
        fpn_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.backbone = DINOv3FeatureExtractor(
            checkpoint_path=checkpoint_path,
            output_layers=output_layers,
            freeze=freeze,
            drop_path_rate=drop_path_rate,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            lora=lora,
        )
        self.fusion = TemporalConcatDiffFusion(
            in_channels=self.backbone.hidden_size,
            out_channels=fusion_out_channels,
            scale_factors=scale_factors,
        )
        self.decoder = UPerNetDecoder(
            in_channels=[fusion_out_channels for _ in scale_factors],
            ppm_bins=ppm_bins,
            fpn_channels=fpn_channels,
            num_classes=num_classes,
        )

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        pre_features = self.backbone(t1)
        post_features = self.backbone(t2)
        fused = self.fusion(pre_features, post_features)
        logits = self.decoder(fused)
        return F.interpolate(logits, size=t1.shape[-2:], mode="bilinear", align_corners=False)


def build_dino_upernet_change_detector(config: Dict) -> DINOv3UPerNetChangeDetector:
    backbone_cfg = config["backbone"]
    neck_cfg = config["neck"]
    decoder_cfg = config["decoder"]
    return DINOv3UPerNetChangeDetector(
        checkpoint_path=backbone_cfg["checkpoint_path"],
        output_layers=backbone_cfg.get("output_layers", (6, 12, 18, 24)),
        freeze=backbone_cfg.get("freeze", True),
        drop_path_rate=backbone_cfg.get("drop_path_rate", 0.0),
        unfreeze_last_n_blocks=backbone_cfg.get("unfreeze_last_n_blocks", 0),
        lora=backbone_cfg.get("lora"),
        fusion_out_channels=neck_cfg.get("out_channels", 256),
        scale_factors=neck_cfg.get("scale_factors", (4.0, 2.0, 1.0, 0.5)),
        ppm_bins=decoder_cfg.get("ppm_bins", (1, 2, 3, 6)),
        fpn_channels=decoder_cfg.get("fpn_channels", 256),
        num_classes=decoder_cfg.get("num_classes", 1),
    )
