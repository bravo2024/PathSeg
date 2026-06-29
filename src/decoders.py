from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDifferentialLinearAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, kernel_size: int = 3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.gate = nn.Sequential(
            nn.Linear(dim, num_heads),
            nn.Sigmoid(),
        )
        self.local_mixer = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.proj = nn.Linear(dim, dim)
        self.channel_scale = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.flatten(2).transpose(1, 2)
        qkv = self.to_qkv(x_flat)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, N, self.num_heads, -1).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, -1).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, -1).transpose(1, 2)

        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
        attn_diff = (attn1 - attn2) * self.channel_scale[:, None, :]
        attn_diff = F.softmax(attn_diff, dim=-1)

        out = (attn_diff @ v).transpose(1, 2).reshape(B, N, C)
        gate_w = self.gate(x_flat).unsqueeze(-1)
        out = out * gate_w + x_flat * (1 - gate_w)
        out = self.proj(out).transpose(1, 2).reshape(B, C, H, W)
        out = out + self.local_mixer(x)
        return out


class GDLAEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_filters: int = 64, depth: int = 4):
        super().__init__()
        self.depth = depth
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_filters), nn.ReLU(inplace=True),
        )
        filters = [base_filters * (2 ** i) for i in range(depth + 1)]
        self.downs = nn.ModuleList()
        for i in range(depth):
            block = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(filters[i], filters[i + 1], 3, padding=1, bias=False),
                nn.BatchNorm2d(filters[i + 1]), nn.ReLU(inplace=True),
                nn.Conv2d(filters[i + 1], filters[i + 1], 3, padding=1, bias=False),
                nn.BatchNorm2d(filters[i + 1]), nn.ReLU(inplace=True),
            )
            self.downs.append(block)

    def forward(self, x):
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)
        return x, skips


class PVTGDLA(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        out_channels: int = 1,
    ):
        super().__init__()
        self.encoder = GDLAEncoder(in_channels, base_filters, depth)
        filters = [base_filters * (2 ** i) for i in range(depth + 1)]

        self.bottleneck = GatedDifferentialLinearAttention(filters[depth], num_heads)

        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_ch = filters[i + 1]
            out_ch = filters[i]
            block = nn.ModuleDict({
                "gdla": GatedDifferentialLinearAttention(in_ch, max(1, num_heads // 2)),
                "conv": nn.Sequential(
                    nn.Conv2d(in_ch + filters[i], out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                ),
            })
            self.decoders.append(block)

        self.outc = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skips = self.encoder(x)
        x = self.bottleneck(x)

        for i, decoder in enumerate(self.decoders):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = decoder["gdla"](x)
            skip = skips[-(i + 1)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = decoder["conv"](x)

        return self.outc(x)


def build_model(model_type: str = "unet", **kwargs) -> nn.Module:
    if model_type == "pvt_gdla":
        return PVTGDLA(**kwargs)
    from src.unet import UNet
    return UNet(**kwargs)
