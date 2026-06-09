from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf


def load_config(path: str | Path) -> Dict[str, Any]:
    config = OmegaConf.load(path)
    return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]

