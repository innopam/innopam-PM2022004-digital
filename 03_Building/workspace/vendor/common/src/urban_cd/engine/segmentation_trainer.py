from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..losses.segmentation import build_segmentation_losses
from ..metrics.segmentation import binary_f1, binary_iou
from ..utils.common import get_autocast_dtype, to_device
from ..utils.meter import AverageMeter

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class SegmentationTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        device: torch.device,
        loss_config: Dict,
        mixed_precision: str = "bf16",
        grad_accum_steps: int = 1,
        clip_grad_norm: float | None = None,
        log_interval: int = 20,
        output_dir: str | Path = "outputs",
        writer: "SummaryWriter | None" = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.loss_items = build_segmentation_losses(loss_config)
        self.grad_accum_steps = grad_accum_steps
        self.clip_grad_norm = clip_grad_norm
        self.log_interval = log_interval
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.autocast_dtype = get_autocast_dtype(mixed_precision)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.autocast_dtype == torch.float16)
        self.writer = writer
        self.global_step = 0

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0.0
        logs: Dict[str, float] = {}
        for name, (loss_fn, weight) in self.loss_items.items():
            value = loss_fn(logits, targets)
            total_loss = total_loss + weight * value
            logs[name] = value.item()
        return total_loss, logs

    def train_epoch(self, dataloader: Iterable, epoch: int) -> Dict[str, float]:
        self.model.train()
        loss_meter = AverageMeter()
        start_time = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(dataloader, start=1):
            batch = to_device(batch, self.device)
            inputs = batch["image"]
            targets = batch["mask"]
            with torch.cuda.amp.autocast(enabled=self.autocast_dtype is not None, dtype=self.autocast_dtype):
                logits = self.model(inputs)
                loss, _ = self._compute_loss(logits, targets)
                loss = loss / self.grad_accum_steps

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % self.grad_accum_steps == 0:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss_value = loss.item() * self.grad_accum_steps
            loss_meter.update(total_loss_value)
            if step % self.log_interval == 0:
                elapsed = time.time() - start_time
                learning_rate = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch} Step {step}/{len(dataloader)} "
                    f"Loss {loss_meter.avg:.4f} LR {learning_rate:.3e} Time {elapsed/step:.3f}s"
                )
                if self.writer is not None:
                    self.writer.add_scalar("train/loss", total_loss_value, self.global_step)
                    self.writer.add_scalar("train/lr", learning_rate, self.global_step)
            self.global_step += 1

        return {"loss": loss_meter.avg}

    @torch.no_grad()
    def validate(self, dataloader: Iterable) -> Dict[str, float]:
        self.model.eval()
        loss_meter = AverageMeter()
        iou_meter = AverageMeter()
        f1_meter = AverageMeter()

        for batch in dataloader:
            batch = to_device(batch, self.device)
            logits = self.model(batch["image"])
            targets = batch["mask"]
            loss, _ = self._compute_loss(logits, targets)
            loss_meter.update(loss.item())
            iou_meter.update(binary_iou(logits, targets))
            f1_meter.update(binary_f1(logits, targets))

        if self.writer is not None:
            self.writer.add_scalar("val/loss", loss_meter.avg, self.global_step)
            self.writer.add_scalar("val/iou", iou_meter.avg, self.global_step)
            self.writer.add_scalar("val/f1", f1_meter.avg, self.global_step)

        return {
            "val_loss": loss_meter.avg,
            "val_iou": iou_meter.avg,
            "val_f1": f1_meter.avg,
        }

    def save_checkpoint(self, state: Dict[str, torch.Tensor], filename: str) -> None:
        torch.save(state, self.output_dir / filename)
