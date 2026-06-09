from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class ViTPyramidAdapter(nn.Module):
    """Projects same-resolution ViT features into a pseudo feature pyramid."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factors: Sequence[float] = (4.0, 2.0, 1.0, 0.5),
    ) -> None:
        super().__init__()
        self.scale_factors = tuple(scale_factors)
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                )
                for _ in self.scale_factors
            ]
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(features) != len(self.scale_factors):
            raise ValueError("The number of ViT features must match the number of pyramid scale factors")

        pyramid: List[torch.Tensor] = []
        for feature, projection, scale_factor in zip(features, self.projections, self.scale_factors):
            feature = projection(feature)
            if scale_factor != 1.0:
                target_height = max(1, int(round(feature.shape[-2] * scale_factor)))
                target_width = max(1, int(round(feature.shape[-1] * scale_factor)))
                feature = F.interpolate(
                    feature,
                    size=(target_height, target_width),
                    mode="bilinear",
                    align_corners=False,
                )
            pyramid.append(feature)
        return pyramid

