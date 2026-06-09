from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class ChangeSample:
    file_name: str
    t1_path: Path
    t2_path: Path
    gt_path: Path


def _read_file_list(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle.readlines() if line.strip()]


class AihubBinaryChangeDataset(Dataset):
    """AIHub building change detection dataset collapsed to a binary target."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        file_list: str | Path,
        transform: Optional[
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = None,
        change_values: Sequence[int] = (1, 2, 3),
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.base_dir = self.root / split
        self.t1_dir = self.base_dir / "T1"
        self.t2_dir = self.base_dir / "T2"
        self.gt_dir = self.base_dir / "GT"
        self.transform = transform
        self.change_values = tuple(change_values)
        self.samples = self._build_index(file_list)

    def _build_index(self, file_list: str | Path) -> List[ChangeSample]:
        list_path = Path(file_list)
        if not list_path.is_absolute():
            list_path = (Path.cwd() / list_path).resolve()
        names = _read_file_list(list_path)
        samples: List[ChangeSample] = []
        for name in names:
            t1_path = self.t1_dir / name
            t2_path = self.t2_dir / name
            gt_path = self.gt_dir / name
            if not t1_path.is_file() or not t2_path.is_file() or not gt_path.is_file():
                raise FileNotFoundError(f"Missing AIHub sample for {name}")
            samples.append(ChangeSample(file_name=name, t1_path=t1_path, t2_path=t2_path, gt_path=gt_path))
        if not samples:
            raise RuntimeError(f"No change-detection samples indexed from {list_path}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        t1_np = np.array(Image.open(sample.t1_path).convert("RGB"), dtype=np.uint8)
        t2_np = np.array(Image.open(sample.t2_path).convert("RGB"), dtype=np.uint8)
        mask_np = np.array(Image.open(sample.gt_path), dtype=np.uint8)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        mask_np = np.isin(mask_np, self.change_values).astype(np.float32)

        t1 = torch.from_numpy(t1_np).permute(2, 0, 1).float() / 255.0
        t2 = torch.from_numpy(t2_np).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask_np).unsqueeze(0)

        if self.transform is not None:
            t1, t2, mask = self.transform(t1, t2, mask)

        return {
            "t1": t1,
            "t2": t2,
            "mask": mask,
            "file_name": sample.file_name,
        }


class AihubMulticlassChangeDataset(Dataset):
    """AIHub-style T1/T2/GT dataset that preserves integer change classes."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        file_list: str | Path,
        transform: Optional[
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = None,
        valid_classes: Sequence[int] = (0, 1, 2),
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.base_dir = self.root / split
        self.t1_dir = self.base_dir / "T1"
        self.t2_dir = self.base_dir / "T2"
        self.gt_dir = self.base_dir / "GT"
        self.transform = transform
        self.valid_classes = tuple(int(value) for value in valid_classes)
        self.samples = self._build_index(file_list)

    def _build_index(self, file_list: str | Path) -> List[ChangeSample]:
        list_path = Path(file_list)
        if not list_path.is_absolute():
            list_path = (Path.cwd() / list_path).resolve()
        names = _read_file_list(list_path)
        samples: List[ChangeSample] = []
        for name in names:
            t1_path = self.t1_dir / name
            t2_path = self.t2_dir / name
            gt_path = self.gt_dir / name
            if not t1_path.is_file() or not t2_path.is_file() or not gt_path.is_file():
                raise FileNotFoundError(f"Missing AIHub sample for {name}")
            samples.append(ChangeSample(file_name=name, t1_path=t1_path, t2_path=t2_path, gt_path=gt_path))
        if not samples:
            raise RuntimeError(f"No change-detection samples indexed from {list_path}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        t1_np = np.array(Image.open(sample.t1_path).convert("RGB"), dtype=np.uint8)
        t2_np = np.array(Image.open(sample.t2_path).convert("RGB"), dtype=np.uint8)
        mask_np = np.array(Image.open(sample.gt_path), dtype=np.int64)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        valid = np.isin(mask_np, self.valid_classes)
        if not valid.all():
            mask_np = np.where(valid, mask_np, 0)

        t1 = torch.from_numpy(t1_np).permute(2, 0, 1).float() / 255.0
        t2 = torch.from_numpy(t2_np).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask_np).unsqueeze(0).float()

        if self.transform is not None:
            t1, t2, mask = self.transform(t1, t2, mask)

        return {
            "t1": t1,
            "t2": t2,
            "mask": mask.squeeze(0).long(),
            "file_name": sample.file_name,
        }


def build_change_detection_dataloader(
    root: str | Path,
    split: str,
    file_list: str | Path,
    batch_size: int,
    num_workers: int,
    transform: Optional[
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ] = None,
    change_values: Sequence[int] = (1, 2, 3),
    shuffle: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> DataLoader:
    dataset = AihubBinaryChangeDataset(
        root=root,
        split=split,
        file_list=file_list,
        transform=transform,
        change_values=change_values,
    )


def build_multiclass_change_detection_dataloader(
    root: str | Path,
    split: str,
    file_list: str | Path,
    batch_size: int,
    num_workers: int,
    transform: Optional[
        Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ] = None,
    valid_classes: Sequence[int] = (0, 1, 2),
    shuffle: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> DataLoader:
    dataset = AihubMulticlassChangeDataset(
        root=root,
        split=split,
        file_list=file_list,
        transform=transform,
        valid_classes=valid_classes,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
