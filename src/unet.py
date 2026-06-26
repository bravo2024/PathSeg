"""U-Net segmentation model built from config specs."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import get_config


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int = 64,
        depth: int = 4,
        dropout: float = 0.1,
        out_channels: int = 1,
    ):
        super().__init__()
        self.depth = depth

        self.inc = DoubleConv(in_channels, base_filters, dropout)
        filters = [base_filters * (2 ** i) for i in range(depth + 1)]

        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(Down(filters[i], filters[i + 1], dropout))

        self.ups = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.ups.append(Up(filters[i + 1], filters[i], dropout))

        self.outc = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.inc(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = skips.pop()
        for up in self.ups:
            x = up(x, skips.pop())

        return self.outc(x)


def build_unet_from_config(cfg: dict | None = None) -> UNet:
    if cfg is None:
        cfg = get_config()
    unet_cfg = cfg.get("unet", {})
    return UNet(
        in_channels=unet_cfg.get("in_channels", 3),
        base_filters=unet_cfg.get("base_filters", 64),
        depth=unet_cfg.get("depth", 4),
        dropout=unet_cfg.get("dropout", 0.1),
    )


def dice_loss(input: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    input = input.reshape(-1)
    target = target.reshape(-1)
    intersection = (input * target).sum()
    return 1.0 - (2.0 * intersection + smooth) / (input.sum() + target.sum() + smooth)


def combined_loss(input: torch.Tensor, target: torch.Tensor, bce_weight: float = 0.5) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(input, target)
    dice = dice_loss(torch.sigmoid(input), target)
    return bce_weight * bce + (1.0 - bce_weight) * dice
