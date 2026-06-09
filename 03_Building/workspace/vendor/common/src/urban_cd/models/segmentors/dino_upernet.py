from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..backbones.dinov3 import DINOv3FeatureExtractor
from ..decoders.upernet import UPerNetDecoder
from ..necks.vit_pyramid import ViTPyramidAdapter


class DINOv3UPerNetSegmentor(nn.Module):
    def __init__(
        self,
        backbone: DINOv3FeatureExtractor,
        neck: ViTPyramidAdapter,
        decoder: UPerNetDecoder,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.decoder = decoder

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        pyramid = self.neck(features)
        logits = self.decoder(pyramid)
        return F.interpolate(logits, size=inputs.shape[2:], mode="bilinear", align_corners=False)


def build_dino_upernet_segmentor(model_config: dict) -> DINOv3UPerNetSegmentor:
    backbone_config = model_config["backbone"]
    neck_config = model_config["neck"]
    decoder_config = model_config["decoder"]

    backbone = DINOv3FeatureExtractor(
        checkpoint_path=backbone_config["checkpoint_path"],
        output_layers=backbone_config.get("output_layers", (6, 12, 18, 24)),
        freeze=backbone_config.get("freeze", True),
        drop_path_rate=backbone_config.get("drop_path_rate", 0.0),
        unfreeze_last_n_blocks=backbone_config.get("unfreeze_last_n_blocks", 0),
        lora=backbone_config.get("lora"),
    )
    neck = ViTPyramidAdapter(
        in_channels=backbone.hidden_size,
        out_channels=neck_config.get("out_channels", 256),
        scale_factors=neck_config.get("scale_factors", (4.0, 2.0, 1.0, 0.5)),
    )
    decoder = UPerNetDecoder(
        in_channels=[neck_config.get("out_channels", 256)] * len(backbone.output_layers),
        ppm_bins=decoder_config.get("ppm_bins", [1, 2, 3, 6]),
        fpn_channels=decoder_config.get("fpn_channels", 256),
        num_classes=decoder_config.get("num_classes", 1),
    )
    return DINOv3UPerNetSegmentor(backbone=backbone, neck=neck, decoder=decoder)
