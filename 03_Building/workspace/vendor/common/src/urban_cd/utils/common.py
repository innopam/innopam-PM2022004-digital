from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def get_autocast_dtype(precision: str) -> torch.dtype | None:
    precision = precision.lower()
    if precision in {"fp16", "float16"}:
        return torch.float16
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return None


def make_run_dir(base_dir: str | Path, prefix: str) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"{prefix}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

