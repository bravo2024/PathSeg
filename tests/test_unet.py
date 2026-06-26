"""Tests for U-Net model architecture, loss functions, and training helpers."""
import numpy as np
import torch

from src.unet import DoubleConv, Down, UNet, Up, build_unet_from_config, combined_loss, dice_loss


def test_double_conv_output_shape():
    module = DoubleConv(3, 64, dropout=0.1)
    x = torch.randn(4, 3, 128, 128)
    out = module(x)
    assert out.shape == (4, 64, 128, 128)


def test_down_output_shape():
    module = Down(64, 128, dropout=0.0)
    x = torch.randn(4, 64, 64, 64)
    out = module(x)
    assert out.shape == (4, 128, 32, 32)


def test_up_output_shape():
    up = Up(128, 64, dropout=0.0)
    x1 = torch.randn(4, 128, 32, 32)
    x2 = torch.randn(4, 64, 64, 64)
    out = up(x1, x2)
    assert out.shape == (4, 64, 64, 64)


def test_up_with_odd_dimensions():
    up = Up(128, 64)
    x1 = torch.randn(4, 128, 15, 15)
    x2 = torch.randn(4, 64, 31, 31)
    out = up(x1, x2)
    assert out.shape == (4, 64, 31, 31)


def test_unet_forward_shape():
    model = UNet(in_channels=3, base_filters=16, depth=3, dropout=0.0)
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert out.shape == (2, 1, 128, 128)


def test_unet_depth_4():
    model = UNet(in_channels=3, base_filters=64, depth=4, dropout=0.1)
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out.shape == (1, 1, 128, 128)


def test_unet_depth_2():
    model = UNet(in_channels=3, base_filters=32, depth=2, dropout=0.0)
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out.shape == (1, 1, 128, 128)


def test_build_from_config():
    cfg = {
        "unet": {
            "in_channels": 3,
            "base_filters": 32,
            "depth": 3,
            "dropout": 0.05,
        }
    }
    model = build_unet_from_config(cfg)
    assert model.inc.net[0].in_channels == 3
    assert len(model.downs) == 3
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out.shape == (1, 1, 128, 128)


def test_dice_loss_perfect():
    pred = torch.sigmoid(torch.full((2, 1, 8, 8), 100.0))
    target = torch.ones(2, 1, 8, 8)
    loss = dice_loss(pred, target)
    assert loss.item() < 0.05


def test_dice_loss_no_overlap():
    pred = torch.sigmoid(torch.full((2, 1, 8, 8), -100.0))
    target = torch.ones(2, 1, 8, 8)
    loss = dice_loss(pred, target)
    assert loss.item() > 0.95


def test_combined_loss():
    logits = torch.randn(2, 1, 32, 32)
    target = (torch.sigmoid(logits) > 0.5).float()
    loss = combined_loss(logits, target, bce_weight=0.5)
    assert 0.0 < loss.item() < 1.0


def test_combined_loss_bce_weight():
    logits = torch.randn(2, 1, 32, 32)
    target = (torch.sigmoid(logits) > 0.5).float()
    loss_bce = combined_loss(logits, target, bce_weight=1.0)
    loss_dice = combined_loss(logits, target, bce_weight=0.0)
    assert loss_bce != loss_dice


def test_unet_parameter_count():
    model = UNet(in_channels=3, base_filters=64, depth=4)
    n_params = sum(p.numel() for p in model.parameters())
    # Standard U-Net with these specs ~31M params
    assert 25_000_000 < n_params < 35_000_000


def test_unet_gradients_flow():
    model = UNet(in_channels=3, base_filters=16, depth=2)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    loss = out.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum().item() > 0, f"Zero gradient for {name}"


def test_down_ups_count():
    cfg = {"unet": {"depth": 4}}
    model = build_unet_from_config(cfg)
    assert len(model.downs) == 4
    assert len(model.ups) == 4


def test_unet_output_range():
    model = UNet(in_channels=3, base_filters=16, depth=3)
    model.eval()
    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 128, 128)
    assert torch.isfinite(out).all()
