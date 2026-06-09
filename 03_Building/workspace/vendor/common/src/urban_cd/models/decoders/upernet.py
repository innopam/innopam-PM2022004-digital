from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_scales: Sequence[int]) -> None:
        super().__init__()
        self.stages = nn.ModuleList()
        for scale in pool_scales:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_scales) * out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [x]
        for stage in self.stages:
            pooled = stage(x)
            outputs.append(F.interpolate(pooled, size=x.shape[2:], mode="bilinear", align_corners=False))
        return self.bottleneck(torch.cat(outputs, dim=1))


class UPerNetDecoder(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        ppm_bins: Sequence[int],
        fpn_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.in_channels = list(in_channels)
        self.ppm = PyramidPoolingModule(in_channels[-1], fpn_channels, ppm_bins)
        self.laterals = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for channels in in_channels[:-1]:
            self.laterals.append(
                nn.Sequential(
                    nn.Conv2d(channels, fpn_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(fpn_channels),
                    nn.ReLU(inplace=True),
                )
            )
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(fpn_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(in_channels) * fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )
        self.cls_seg = nn.Conv2d(fpn_channels, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor], return_features: bool = False):
        """When return_features is True, also return the pre-cls_seg fusion tensor for auxiliary heads."""
        ppm_output = self.ppm(features[-1])
        laterals = [ppm_output]
        previous = ppm_output

        for index, feature in enumerate(features[:-1][::-1]):
            lateral = self.laterals[index](feature)
            upsampled = F.interpolate(previous, size=lateral.shape[2:], mode="bilinear", align_corners=False)
            fused = self.fpn_convs[index](lateral + upsampled)
            laterals.append(fused)
            previous = fused

        laterals = laterals[::-1]
        target_size = laterals[0].shape[2:]
        resized = [laterals[0]]
        for feature in laterals[1:]:
            resized.append(F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False))

        fusion = self.conv_fusion(torch.cat(resized, dim=1))
        logits = self.cls_seg(fusion)
        if return_features:
            return logits, fusion
        return logits

