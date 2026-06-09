from __future__ import annotations

import torch


def binary_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    preds = probs > threshold
    targets = targets > threshold
    intersection = torch.logical_and(preds, targets).sum().float()
    union = torch.logical_or(preds, targets).sum().float()
    if union == 0:
        return 1.0
    return float((intersection + eps) / (union + eps))


def binary_f1(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    preds = probs > threshold
    targets = targets > threshold
    true_positive = torch.logical_and(preds, targets).sum().float()
    false_positive = torch.logical_and(preds, torch.logical_not(targets)).sum().float()
    false_negative = torch.logical_and(torch.logical_not(preds), targets).sum().float()
    denom = 2.0 * true_positive + false_positive + false_negative
    if denom == 0:
        return 1.0
    return float((2.0 * true_positive + eps) / (denom + eps))


def multiclass_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    if preds.ndim == 4:
        preds = preds.argmax(dim=1)
    if targets.ndim == 4:
        targets = targets.squeeze(1)
    preds = preds.reshape(-1).long()
    targets = targets.reshape(-1).long()
    valid = (targets >= 0) & (targets < num_classes)
    if ignore_index is not None:
        valid = valid & (targets != ignore_index)
    preds = preds[valid]
    targets = targets[valid]
    encoded = targets * num_classes + preds.clamp(0, num_classes - 1)
    matrix = torch.bincount(encoded, minlength=num_classes * num_classes)
    return matrix.reshape(num_classes, num_classes).to(torch.float64)


def metrics_from_confusion_matrix(confusion: torch.Tensor, eps: float = 1e-6) -> dict[str, float]:
    confusion = confusion.to(torch.float64)
    true_positive = torch.diag(confusion)
    pred_count = confusion.sum(dim=0)
    target_count = confusion.sum(dim=1)
    false_positive = pred_count - true_positive
    false_negative = target_count - true_positive
    precision = (true_positive + eps) / (true_positive + false_positive + eps)
    recall = (true_positive + eps) / (true_positive + false_negative + eps)
    f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)
    iou = (true_positive + eps) / (true_positive + false_positive + false_negative + eps)

    num_classes = confusion.shape[0]
    metrics: dict[str, float] = {}
    for class_index in range(num_classes):
        metrics[f"class_{class_index}_precision"] = float(precision[class_index])
        metrics[f"class_{class_index}_recall"] = float(recall[class_index])
        metrics[f"class_{class_index}_f1"] = float(f1[class_index])
        metrics[f"class_{class_index}_iou"] = float(iou[class_index])
        metrics[f"class_{class_index}_support"] = float(target_count[class_index])

    foreground = slice(1, num_classes)
    metrics["mean_iou"] = float(iou.mean())
    metrics["foreground_mean_iou"] = float(iou[foreground].mean()) if num_classes > 1 else float(iou.mean())
    metrics["mean_f1"] = float(f1.mean())
    metrics["foreground_mean_f1"] = float(f1[foreground].mean()) if num_classes > 1 else float(f1.mean())
    metrics["accuracy"] = float((true_positive.sum() + eps) / (confusion.sum() + eps))

    if num_classes > 1:
        fg_tp = confusion[1:, 1:].sum()
        fg_pred = confusion[:, 1:].sum()
        fg_target = confusion[1:, :].sum()
        fg_fp = fg_pred - fg_tp
        fg_fn = fg_target - fg_tp
        fg_precision = (fg_tp + eps) / (fg_tp + fg_fp + eps)
        fg_recall = (fg_tp + eps) / (fg_tp + fg_fn + eps)
        metrics["foreground_precision"] = float(fg_precision)
        metrics["foreground_recall"] = float(fg_recall)
        metrics["foreground_f1"] = float((2.0 * fg_precision * fg_recall + eps) / (fg_precision + fg_recall + eps))
        metrics["foreground_iou"] = float((fg_tp + eps) / (fg_tp + fg_fp + fg_fn + eps))
    return metrics
