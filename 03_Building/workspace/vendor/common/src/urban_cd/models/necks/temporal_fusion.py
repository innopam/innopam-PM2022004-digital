from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class TemporalConcatDiffFusion(nn.Module):
    """Fuse two feature streams with concat + absolute difference, then resize to a pseudo pyramid."""

    def __init__(
        self,
        in_channels: int | Sequence[int],
        out_channels: int,
        scale_factors: Sequence[float],
    ) -> None:
        super().__init__()
        if isinstance(in_channels, int):
            self.in_channels = [in_channels for _ in scale_factors]
        else:
            self.in_channels = list(in_channels)
        self.scale_factors = list(scale_factors)
        if len(self.in_channels) != len(self.scale_factors):
            raise ValueError("in_channels and scale_factors must have the same length")

        self.projections = nn.ModuleList()
        for channels in self.in_channels:
            self.projections.append(
                nn.Sequential(
                    nn.Conv2d(channels * 3, out_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=32, num_channels=out_channels),
                    nn.GELU(),
                )
            )

    def forward(self, pre_features: List[torch.Tensor], post_features: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(pre_features) != len(post_features):
            raise ValueError("pre_features and post_features must have the same length")
        if len(pre_features) != len(self.projections):
            raise ValueError("Unexpected number of feature levels")

        outputs: List[torch.Tensor] = []
        for pre_feature, post_feature, projection, scale_factor in zip(
            pre_features,
            post_features,
            self.projections,
            self.scale_factors,
        ):
            fused = torch.cat([pre_feature, post_feature, torch.abs(post_feature - pre_feature)], dim=1)
            fused = projection(fused)
            if scale_factor != 1.0:
                target_height = max(1, int(round(fused.shape[-2] * scale_factor)))
                target_width = max(1, int(round(fused.shape[-1] * scale_factor)))
                fused = F.interpolate(fused, size=(target_height, target_width), mode="bilinear", align_corners=False)
            outputs.append(fused)
        return outputs
