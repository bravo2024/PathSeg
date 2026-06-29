from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenBook(nn.Module):
    def __init__(self, feat_dim: int, num_prototypes: int = 64):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feat_dim))
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, C, H, W = features.shape
        feat_flat = features.flatten(2).transpose(1, 2)
        prototypes_norm = F.normalize(self.prototypes, dim=-1)
        feat_norm = F.normalize(feat_flat, dim=-1)
        sim = feat_norm @ prototypes_norm.t()
        sim = sim / self.temperature.clamp(min=0.01)
        weights = F.softmax(sim, dim=-1)
        guide = weights @ self.prototypes
        guide = guide.transpose(1, 2).reshape(B, C, H, W)
        return guide


class GuidiNOGuidance(nn.Module):
    def __init__(self, backbone_name: str = "vit_small_patch14_dinov2", num_prototypes: int = 64):
        super().__init__()
        try:
            import timm
            self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        except ImportError:
            self.backbone = None
            return
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        feat_dim = self.backbone.embed_dim if hasattr(self.backbone, "embed_dim") else 384
        self.token_book = TokenBook(feat_dim, num_prototypes)
        self.proj = nn.Sequential(
            nn.Conv2d(feat_dim, 128, 1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone is None:
            return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        B, C, H, W = x.shape
        with torch.no_grad():
            if hasattr(self.backbone, "forward_features"):
                feats = self.backbone.forward_features(F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False))
                if feats.dim() == 3:
                    n_tokens = int(feats.shape[1] ** 0.5)
                    feats = feats[:, 1:, :].transpose(1, 2).reshape(B, -1, n_tokens, n_tokens)
            else:
                feats = self.backbone(F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False))
                if feats.dim() == 3:
                    n_tokens = int(feats.shape[1] ** 0.5)
                    feats = feats.transpose(1, 2).reshape(B, -1, n_tokens, n_tokens)
        guide = self.token_book(feats)
        guide = F.interpolate(guide, size=(H, W), mode="bilinear", align_corners=False)
        guide = self.proj(guide)
        return guide
