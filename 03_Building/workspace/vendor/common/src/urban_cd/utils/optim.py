from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(model: torch.nn.Module, config: dict) -> Optimizer:
    learning_rate = config.get("lr", 1e-4)
    backbone_learning_rate = config.get("backbone_lr", learning_rate)
    weight_decay = config.get("weight_decay", 0.0)
    backbone_weight_decay = config.get("backbone_weight_decay", weight_decay)
    betas = tuple(config.get("betas", (0.9, 0.999)))

    decay_params = []
    no_decay_params = []
    backbone_decay_params = []
    backbone_no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = name.startswith("backbone.")
        if parameter.ndim == 1 or name.endswith("bias") or "norm" in name.lower():
            if is_backbone:
                backbone_no_decay_params.append(parameter)
            else:
                no_decay_params.append(parameter)
        else:
            if is_backbone:
                backbone_decay_params.append(parameter)
            else:
                decay_params.append(parameter)

    param_groups = []
    if decay_params:
        param_groups.append({"params": decay_params, "lr": learning_rate, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "lr": learning_rate, "weight_decay": 0.0})
    if backbone_decay_params:
        param_groups.append(
            {"params": backbone_decay_params, "lr": backbone_learning_rate, "weight_decay": backbone_weight_decay}
        )
    if backbone_no_decay_params:
        param_groups.append({"params": backbone_no_decay_params, "lr": backbone_learning_rate, "weight_decay": 0.0})
    return torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas)


def build_scheduler(optimizer: Optimizer, config: dict, total_steps: int) -> LambdaLR:
    warmup_epochs = config.get("warmup_epochs", 0)
    warmup_steps = config.get("warmup_steps")
    min_lr = config.get("min_lr", 1e-6)
    base_lr = optimizer.param_groups[0]["lr"]
    if warmup_steps is None:
        warmup_steps = warmup_epochs * config.get("steps_per_epoch", total_steps)
    warmup_steps = int(min(warmup_steps, total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps and warmup_steps > 0:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr / base_lr) + (1.0 - (min_lr / base_lr)) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
