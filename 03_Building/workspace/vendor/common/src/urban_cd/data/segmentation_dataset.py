from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class SegmentationSample:
    image_path: str
    mask_path: str
    resolution: str


@dataclass
class ImageOnlySample:
    image_path: str
    resolution: str
    source: str
    stream: str


def _load_segmentation_pair(
    image_path: str,
    mask_path: str,
    mask_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    mask = np.array(Image.open(mask_path), dtype=np.uint8)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > (mask_threshold * 255.0)).astype(np.float32)

    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    mask_tensor = torch.from_numpy(mask).unsqueeze(0)
    return image_tensor, mask_tensor


class ResolutionSegmentationDataset(Dataset):
    """Binary segmentation dataset that merges multiple resolution folders."""

    def __init__(
        self,
        root: str,
        split: str,
        resolutions: Sequence[str],
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        mask_threshold: float = 0.5,
        valid_extensions: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.resolutions = list(resolutions)
        self.transform = transform
        self.mask_threshold = mask_threshold
        self.valid_extensions = tuple(ext.lower() for ext in valid_extensions)
        self.samples: List[SegmentationSample] = []
        self._build_index()

    def _build_index(self) -> None:
        for resolution in self.resolutions:
            image_dir = os.path.join(self.root, self.split, resolution, "images")
            mask_dir = os.path.join(self.root, self.split, resolution, "labels")
            if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
                raise FileNotFoundError(f"Missing images or labels directory for resolution {resolution}")
            for file_name in sorted(os.listdir(image_dir)):
                if not file_name.lower().endswith(self.valid_extensions):
                    continue
                image_path = os.path.join(image_dir, file_name)
                mask_path = os.path.join(mask_dir, file_name)
                if not os.path.isfile(mask_path):
                    raise FileNotFoundError(f"Label file not found for {image_path}")
                self.samples.append(
                    SegmentationSample(
                        image_path=image_path,
                        mask_path=mask_path,
                        resolution=resolution,
                    )
                )
        if not self.samples:
            raise RuntimeError(f"No segmentation samples were indexed from root={self.root} split={self.split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image_tensor, mask_tensor = _load_segmentation_pair(
            sample.image_path,
            sample.mask_path,
            self.mask_threshold,
        )

        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "resolution": sample.resolution,
            "image_path": sample.image_path,
        }


class ManifestSegmentationDataset(Dataset):
    """Binary segmentation dataset driven by explicit manifest files."""

    def __init__(
        self,
        manifest_path: str,
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        mask_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.manifest_path = manifest_path
        self.transform = transform
        self.mask_threshold = mask_threshold
        self.samples: List[SegmentationSample] = []
        self._build_index()

    def _build_index(self) -> None:
        manifest = os.path.expanduser(self.manifest_path)
        if not os.path.isfile(manifest):
            raise FileNotFoundError(f"Segmentation manifest not found: {manifest}")
        with open(manifest, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    raise ValueError(f"Invalid manifest row in {manifest}: {line}")
                image_path = parts[0]
                mask_path = parts[1]
                resolution = parts[2] if len(parts) > 2 else "unknown"
                if not os.path.isfile(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                if not os.path.isfile(mask_path):
                    raise FileNotFoundError(f"Mask file not found: {mask_path}")
                self.samples.append(
                    SegmentationSample(
                        image_path=image_path,
                        mask_path=mask_path,
                        resolution=resolution,
                    )
                )
        if not self.samples:
            raise RuntimeError(f"No segmentation samples were indexed from manifest={manifest}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image_tensor, mask_tensor = _load_segmentation_pair(
            sample.image_path,
            sample.mask_path,
            self.mask_threshold,
        )
        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "resolution": sample.resolution,
            "image_path": sample.image_path,
        }


class ImageOnlyManifestDataset(Dataset):
    """Single-image dataset driven by explicit manifests for pseudo-label generation."""

    def __init__(
        self,
        manifest_path: str,
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> None:
        super().__init__()
        self.manifest_path = manifest_path
        self.transform = transform
        self.samples: List[ImageOnlySample] = []
        self._build_index()

    def _build_index(self) -> None:
        manifest = os.path.expanduser(self.manifest_path)
        if not os.path.isfile(manifest):
            raise FileNotFoundError(f"Image-only manifest not found: {manifest}")
        with open(manifest, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 1:
                    raise ValueError(f"Invalid image-only manifest row in {manifest}: {line}")
                image_path = parts[0]
                resolution = parts[1] if len(parts) > 1 else "unknown"
                source = parts[2] if len(parts) > 2 else "unknown"
                stream = parts[3] if len(parts) > 3 else "single"
                if not os.path.isfile(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                self.samples.append(
                    ImageOnlySample(
                        image_path=image_path,
                        resolution=resolution,
                        source=source,
                        stream=stream,
                    )
                )
        if not self.samples:
            raise RuntimeError(f"No image-only samples were indexed from manifest={manifest}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = np.array(Image.open(sample.image_path).convert("RGB"), dtype=np.uint8)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if self.transform is not None:
            dummy_mask = torch.zeros((1, image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.float32)
            image_tensor, _ = self.transform(image_tensor, dummy_mask)
        return {
            "image": image_tensor,
            "resolution": sample.resolution,
            "source": sample.source,
            "stream": sample.stream,
            "image_path": sample.image_path,
        }


def build_segmentation_dataloader(
    root: str,
    split: str,
    resolutions: Sequence[str],
    batch_size: int,
    num_workers: int,
    transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    shuffle: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> DataLoader:
    dataset = ResolutionSegmentationDataset(
        root=root,
        split=split,
        resolutions=resolutions,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )


def build_manifest_segmentation_dataloader(
    manifest_path: str,
    batch_size: int,
    num_workers: int,
    transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    shuffle: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> DataLoader:
    dataset = ManifestSegmentationDataset(
        manifest_path=manifest_path,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )


def build_image_only_manifest_dataloader(
    manifest_path: str,
    batch_size: int,
    num_workers: int,
    transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    shuffle: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> DataLoader:
    dataset = ImageOnlyManifestDataset(
        manifest_path=manifest_path,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
    )
