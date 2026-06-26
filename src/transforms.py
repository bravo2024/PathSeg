"""Augmentation and preprocessing transforms for pathology tiles."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torchvision import transforms as T

_SPATIAL_AUGS = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=30, interpolation=T.InterpolationMode.NEAREST),
    T.ElasticTransform(alpha=20.0, sigma=4.0, interpolation=T.InterpolationMode.NEAREST),
])

_COLOR_AUGS = T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02)


def _to_tensor(image: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    img = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    msk = torch.from_numpy(mask).float().unsqueeze(0)
    return img, msk


def train_transform(image: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    img, msk = _to_tensor(image, mask)
    stacked = torch.cat([img, msk], dim=0)
    stacked = _SPATIAL_AUGS(stacked)
    img, msk = stacked[:3], stacked[3:4]
    img = _COLOR_AUGS(img)
    return img, msk


def val_transform(image: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return _to_tensor(image, mask)


def get_transform(split: str) -> Any:
    if split == "train":
        return train_transform
    return val_transform
