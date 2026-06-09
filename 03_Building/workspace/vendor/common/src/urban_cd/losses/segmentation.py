from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32), persistent=False)
        self._loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self._loss.pos_weight = self.pos_weight.to(logits.device)
        return self._loss(logits, targets)


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        targets = targets.float()
        dims = (0, 2, 3)
        intersection = torch.sum(probabilities * targets, dim=dims)
        cardinality = torch.sum(probabilities + targets, dim=dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return (1.0 - dice).mean()


class CrossEntropyLoss(nn.Module):
    def __init__(self, class_weights: list[float] | None = None, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        if class_weights is None:
            self.register_buffer("class_weights", torch.empty(0), persistent=False)
        else:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32), persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 4:
            targets = targets.squeeze(1)
        targets = targets.long()
        weight = self.class_weights.to(logits.device) if self.class_weights.numel() > 0 else None
        return F.cross_entropy(logits, targets, weight=weight, ignore_index=self.ignore_index)


class MulticlassDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        include_background: bool = False,
        class_weights: list[float] | None = None,
        eps: float = 1e-6,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.eps = eps
        self.ignore_index = ignore_index
        if class_weights is None:
            self.register_buffer("class_weights", torch.empty(0), persistent=False)
        else:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32), persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 4:
            targets = targets.squeeze(1)
        targets = targets.long()
        valid = targets != self.ignore_index
        safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
        probabilities = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(safe_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1)
        probabilities = probabilities * valid
        one_hot = one_hot * valid

        start_class = 0 if self.include_background else 1
        probabilities = probabilities[:, start_class:]
        one_hot = one_hot[:, start_class:]
        dims = (0, 2, 3)
        intersection = torch.sum(probabilities * one_hot, dim=dims)
        cardinality = torch.sum(probabilities + one_hot, dim=dims)
        dice_loss = 1.0 - (2.0 * intersection + self.eps) / (cardinality + self.eps)
        if self.class_weights.numel() > 0:
            weights = self.class_weights.to(logits.device)[start_class:]
            dice_loss = dice_loss * weights / weights.clamp_min(self.eps).mean()
        return dice_loss.mean()


def build_segmentation_losses(config: dict) -> dict[str, tuple[nn.Module, float]]:
    if config.get("mode") == "multiclass":
        num_classes = int(config["num_classes"])
        class_weights = config.get("class_weights")
        ignore_index = int(config.get("ignore_index", -100))
        ce_weight = float(config.get("ce_weight", 1.0))
        dice_weight = float(config.get("dice_weight", 1.0))
        losses: dict[str, tuple[nn.Module, float]] = {}
        if ce_weight > 0:
            losses["ce"] = (CrossEntropyLoss(class_weights=class_weights, ignore_index=ignore_index), ce_weight)
        if dice_weight > 0:
            losses["multiclass_dice"] = (
                MulticlassDiceLoss(
                    num_classes=num_classes,
                    include_background=bool(config.get("dice_include_background", False)),
                    class_weights=config.get("dice_class_weights"),
                    ignore_index=ignore_index,
                ),
                dice_weight,
            )
        return losses

    pos_weight = config.get("pos_weight", 1.0)
    bce_weight = config.get("seg_bce_weight", 1.0)
    dice_weight = config.get("seg_dice_weight", 1.0)
    losses: dict[str, tuple[nn.Module, float]] = {}
    if bce_weight > 0:
        losses["bce"] = (BCEWithLogitsLoss(pos_weight=pos_weight), bce_weight)
    if dice_weight > 0:
        losses["dice"] = (DiceLoss(), dice_weight)
    return losses
