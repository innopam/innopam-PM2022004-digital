from __future__ import annotations

import random
from typing import Optional, Sequence, Tuple

import numpy as np
from skimage.measure import label as sk_label
from skimage.measure import regionprops
import torch
import torch.nn.functional as F
from torchvision.transforms import ColorJitter


class SegmentationAugmentor:
    """Lightweight augmentation pipeline for binary segmentation."""

    def __init__(
        self,
        input_size: int,
        random_flip: bool = True,
        random_rotate: bool = True,
        color_jitter: Optional[Sequence[float]] = None,
        gaussian_noise_std: float = 0.0,
        random_scale_range: Optional[Sequence[float]] = None,
        normalize_mean: Optional[Sequence[float]] = None,
        normalize_std: Optional[Sequence[float]] = None,
    ) -> None:
        self.input_size = input_size
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.gaussian_noise_std = gaussian_noise_std
        self.random_scale_range = random_scale_range
        self.color_jitter = None
        if color_jitter is not None:
            brightness, contrast, saturation, hue = color_jitter
            self.color_jitter = ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
        self.normalize_mean = torch.tensor(normalize_mean, dtype=torch.float32) if normalize_mean is not None else None
        self.normalize_std = torch.tensor(normalize_std, dtype=torch.float32) if normalize_std is not None else None

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._random_scale(image, mask)
        if self.random_flip:
            if random.random() < 0.5:
                image = torch.flip(image, dims=[2])
                mask = torch.flip(mask, dims=[2])
            if random.random() < 0.5:
                image = torch.flip(image, dims=[1])
                mask = torch.flip(mask, dims=[1])
        if self.random_rotate:
            k = random.randint(0, 3)
            if k:
                image = torch.rot90(image, k, dims=[1, 2])
                mask = torch.rot90(mask, k, dims=[1, 2])
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        if self.gaussian_noise_std > 0:
            image = torch.clamp(image + torch.randn_like(image) * self.gaussian_noise_std, 0.0, 1.0)
        image, mask = self._crop_or_pad_to_size(image, mask, self.input_size)
        if self.normalize_mean is not None and self.normalize_std is not None:
            image = (image - self.normalize_mean[:, None, None]) / self.normalize_std[:, None, None]
        return image, mask

    def _random_scale(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.random_scale_range:
            return image, mask
        min_scale, max_scale = self.random_scale_range
        scale = random.uniform(min_scale, max_scale)
        if abs(scale - 1.0) < 1e-3:
            return image, mask
        _, height, width = image.shape
        new_height = max(8, int(round(height * scale)))
        new_width = max(8, int(round(width * scale)))
        image = F.interpolate(image.unsqueeze(0), size=(new_height, new_width), mode="bilinear", align_corners=False)[0]
        mask = F.interpolate(mask.unsqueeze(0), size=(new_height, new_width), mode="nearest")[0]
        return image, mask

    @staticmethod
    def _crop_or_pad_to_size(image: torch.Tensor, mask: torch.Tensor, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        _, height, width = image.shape
        if height < size or width < size:
            pad_height = max(size - height, 0)
            pad_width = max(size - width, 0)
            padding = (
                pad_width // 2,
                pad_width - pad_width // 2,
                pad_height // 2,
                pad_height - pad_height // 2,
            )
            image = F.pad(image, padding, value=0.0)
            mask = F.pad(mask, padding, value=0.0)
            _, height, width = image.shape
        if height == size and width == size:
            return image, mask
        top = random.randint(0, height - size)
        left = random.randint(0, width - size)
        return (
            image[:, top : top + size, left : left + size],
            mask[:, top : top + size, left : left + size],
        )


def build_segmentation_augmentor(config: dict) -> SegmentationAugmentor:
    return SegmentationAugmentor(
        input_size=config.get("input_size", 512),
        random_flip=config.get("random_flip", True),
        random_rotate=config.get("random_rotate", True),
        color_jitter=config.get("color_jitter"),
        gaussian_noise_std=config.get("gaussian_noise_std", 0.0),
        random_scale_range=config.get("random_scale_range"),
        normalize_mean=config.get("normalize_mean"),
        normalize_std=config.get("normalize_std"),
    )


class ChangeDetectionAugmentor:
    """Shared geometric transforms for T1/T2 with binary-mask aware resize/crop."""

    def __init__(
        self,
        input_size: Optional[int],
        random_flip: bool = True,
        random_rotate: bool = True,
        color_jitter: Optional[Sequence[float]] = None,
        gaussian_noise_std: float = 0.0,
        random_scale_range: Optional[Sequence[float]] = None,
        normalize_mean: Optional[Sequence[float]] = None,
        normalize_std: Optional[Sequence[float]] = None,
        independent_photometric: bool = True,
        object_aware_crop_prob: float = 0.0,
        object_crop_max_shift: int = 64,
        object_aware_crop_classes: Optional[Sequence[int]] = None,
        object_aware_crop_class_weights: Optional[Sequence[float]] = None,
    ) -> None:
        self.input_size = input_size
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.gaussian_noise_std = gaussian_noise_std
        self.random_scale_range = random_scale_range
        self.independent_photometric = independent_photometric
        self.object_aware_crop_prob = object_aware_crop_prob
        self.object_crop_max_shift = object_crop_max_shift
        self.object_aware_crop_classes = (
            [int(value) for value in object_aware_crop_classes] if object_aware_crop_classes is not None else None
        )
        self.object_aware_crop_class_weights = (
            [float(value) for value in object_aware_crop_class_weights]
            if object_aware_crop_class_weights is not None
            else None
        )
        self.color_jitter = None
        if color_jitter is not None:
            brightness, contrast, saturation, hue = color_jitter
            self.color_jitter = ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            )
        self.normalize_mean = torch.tensor(normalize_mean, dtype=torch.float32) if normalize_mean is not None else None
        self.normalize_std = torch.tensor(normalize_std, dtype=torch.float32) if normalize_std is not None else None

    def __call__(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t1, t2, mask = self._random_scale(t1, t2, mask)
        if self.random_flip:
            if random.random() < 0.5:
                t1 = torch.flip(t1, dims=[2])
                t2 = torch.flip(t2, dims=[2])
                mask = torch.flip(mask, dims=[2])
            if random.random() < 0.5:
                t1 = torch.flip(t1, dims=[1])
                t2 = torch.flip(t2, dims=[1])
                mask = torch.flip(mask, dims=[1])
        if self.random_rotate:
            k = random.randint(0, 3)
            if k:
                t1 = torch.rot90(t1, k, dims=[1, 2])
                t2 = torch.rot90(t2, k, dims=[1, 2])
                mask = torch.rot90(mask, k, dims=[1, 2])
        if self.color_jitter is not None:
            if self.independent_photometric:
                t1 = self.color_jitter(t1)
                t2 = self.color_jitter(t2)
            else:
                seed = torch.randint(0, 2**31, (1,)).item()
                torch.manual_seed(seed)
                t1 = self.color_jitter(t1)
                torch.manual_seed(seed)
                t2 = self.color_jitter(t2)
        if self.gaussian_noise_std > 0:
            t1 = torch.clamp(t1 + torch.randn_like(t1) * self.gaussian_noise_std, 0.0, 1.0)
            t2 = torch.clamp(t2 + torch.randn_like(t2) * self.gaussian_noise_std, 0.0, 1.0)
        t1, t2, mask = self._crop_or_pad_to_size(
            t1,
            t2,
            mask,
            self.input_size,
            object_aware_crop_prob=self.object_aware_crop_prob,
            object_crop_max_shift=self.object_crop_max_shift,
            object_aware_crop_classes=self.object_aware_crop_classes,
            object_aware_crop_class_weights=self.object_aware_crop_class_weights,
        )
        if self.normalize_mean is not None and self.normalize_std is not None:
            t1 = (t1 - self.normalize_mean[:, None, None]) / self.normalize_std[:, None, None]
            t2 = (t2 - self.normalize_mean[:, None, None]) / self.normalize_std[:, None, None]
        return t1, t2, mask

    def _random_scale(
        self,
        t1: torch.Tensor,
        t2: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.random_scale_range:
            return t1, t2, mask
        min_scale, max_scale = self.random_scale_range
        scale = random.uniform(min_scale, max_scale)
        if abs(scale - 1.0) < 1e-3:
            return t1, t2, mask
        _, height, width = t1.shape
        new_height = max(8, int(round(height * scale)))
        new_width = max(8, int(round(width * scale)))
        size = (new_height, new_width)
        t1 = F.interpolate(t1.unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0]
        t2 = F.interpolate(t2.unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0]
        mask = F.interpolate(mask.unsqueeze(0), size=size, mode="nearest")[0]
        return t1, t2, mask

    @staticmethod
    def _crop_or_pad_to_size(
        t1: torch.Tensor,
        t2: torch.Tensor,
        mask: torch.Tensor,
        size: Optional[int],
        object_aware_crop_prob: float = 0.0,
        object_crop_max_shift: int = 64,
        object_aware_crop_classes: Optional[Sequence[int]] = None,
        object_aware_crop_class_weights: Optional[Sequence[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if size is None:
            return t1, t2, mask
        _, height, width = t1.shape
        if height < size or width < size:
            pad_height = max(size - height, 0)
            pad_width = max(size - width, 0)
            padding = (
                pad_width // 2,
                pad_width - pad_width // 2,
                pad_height // 2,
                pad_height - pad_height // 2,
            )
            t1 = F.pad(t1, padding, value=0.0)
            t2 = F.pad(t2, padding, value=0.0)
            mask = F.pad(mask, padding, value=0.0)
            _, height, width = t1.shape
        if height == size and width == size:
            return t1, t2, mask
        if object_aware_crop_prob > 0 and random.random() < object_aware_crop_prob:
            crop_box = ChangeDetectionAugmentor._extract_object_aware_crop_box(
                mask=mask,
                crop_size=size,
                height=height,
                width=width,
                max_shift=object_crop_max_shift,
                target_classes=object_aware_crop_classes,
                class_weights=object_aware_crop_class_weights,
            )
            if crop_box is not None:
                top, left = crop_box
            else:
                top = random.randint(0, height - size)
                left = random.randint(0, width - size)
        else:
            top = random.randint(0, height - size)
            left = random.randint(0, width - size)
        return (
            t1[:, top : top + size, left : left + size],
            t2[:, top : top + size, left : left + size],
            mask[:, top : top + size, left : left + size],
        )

    @staticmethod
    def _extract_object_aware_crop_box(
        mask: torch.Tensor,
        crop_size: int,
        height: int,
        width: int,
        max_shift: int,
        target_classes: Optional[Sequence[int]] = None,
        class_weights: Optional[Sequence[float]] = None,
    ) -> Optional[Tuple[int, int]]:
        main_mask_np = mask[0].detach().cpu().numpy()
        mask_np = ChangeDetectionAugmentor._select_object_crop_mask(main_mask_np, target_classes, class_weights)
        if mask_np is None:
            return None
        labeled_array = sk_label(mask_np, connectivity=1)
        regions = regionprops(labeled_array)
        object_centers = [(int(region.centroid[0]), int(region.centroid[1])) for region in regions]
        if not object_centers:
            return None

        center_y, center_x = random.choice(object_centers)
        shift_y = 0
        shift_x = 0
        if random.random() > 0.5:
            shift_x = int(np.random.randint(-max_shift, max_shift))
            shift_y = int(np.random.randint(-max_shift, max_shift))

        top = max(0, center_y + shift_y - crop_size // 2)
        bottom = min(height, top + crop_size)
        left = max(0, center_x + shift_x - crop_size // 2)
        right = min(width, left + crop_size)

        top = max(0, bottom - crop_size)
        left = max(0, right - crop_size)
        return top, left

    @staticmethod
    def _select_object_crop_mask(
        mask_np: np.ndarray,
        target_classes: Optional[Sequence[int]],
        class_weights: Optional[Sequence[float]],
    ) -> Optional[np.ndarray]:
        if not target_classes:
            selected = mask_np > 0.5
            return selected if selected.any() else None

        present_classes = []
        present_weights = []
        for index, class_id in enumerate(target_classes):
            class_mask = np.rint(mask_np).astype(np.int64) == int(class_id)
            if not class_mask.any():
                continue
            present_classes.append(int(class_id))
            if class_weights is not None and index < len(class_weights):
                present_weights.append(max(float(class_weights[index]), 0.0))
            else:
                present_weights.append(1.0)
        if not present_classes:
            selected = mask_np > 0.5
            return selected if selected.any() else None
        total_weight = sum(present_weights)
        if total_weight <= 0:
            present_weights = [1.0] * len(present_classes)
        chosen_class = random.choices(present_classes, weights=present_weights, k=1)[0]
        return np.rint(mask_np).astype(np.int64) == chosen_class


def build_change_detection_augmentor(config: dict) -> ChangeDetectionAugmentor:
    return ChangeDetectionAugmentor(
        input_size=config.get("input_size", 512),
        random_flip=config.get("random_flip", True),
        random_rotate=config.get("random_rotate", True),
        color_jitter=config.get("color_jitter"),
        gaussian_noise_std=config.get("gaussian_noise_std", 0.0),
        random_scale_range=config.get("random_scale_range"),
        normalize_mean=config.get("normalize_mean"),
        normalize_std=config.get("normalize_std"),
        independent_photometric=config.get("independent_photometric", True),
        object_aware_crop_prob=config.get("object_aware_crop_prob", 0.0),
        object_crop_max_shift=config.get("object_crop_max_shift", 64),
        object_aware_crop_classes=config.get("object_aware_crop_classes"),
        object_aware_crop_class_weights=config.get("object_aware_crop_class_weights"),
    )
