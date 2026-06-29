"""UNet training loop with early stopping, checkpointing, and logging."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.config import get_config
from src.logger import get_logger, setup_logger
from src.unet import UNet, build_unet_from_config, combined_loss
from src.decoders import build_model, PVTGDLA
from src.guidance import GuidiNOGuidance


def _segmentation_metrics_from_tensors(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict[str, float]:
    y_t = y_true.bool()
    y_p = y_pred.bool()
    tp = (y_t & y_p).sum().item()
    fp = ((~y_t) & y_p).sum().item()
    fn = (y_t & (~y_p)).sum().item()
    tn = ((~y_t) & (~y_p)).sum().item()
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    return {
        "dice": dice,
        "iou": iou,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    logger: Any,
    log_interval: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = len(loader)

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = combined_loss(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % log_interval == 0:
            logger.info(
                "Train batch [%3d/%3d] loss=%.4f",
                batch_idx + 1, n_batches, loss.item(),
            )

    return total_loss / n_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    tp = fp = fn = tn = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = combined_loss(logits, masks)
        total_loss += loss.item()

        preds = (torch.sigmoid(logits) >= threshold).bool()
        m = masks.bool()
        tp += int((m & preds).sum().item())
        fp += int((~m & preds).sum().item())
        fn += int((m & ~preds).sum().item())
        tn += int((~m & ~preds).sum().item())

    avg_loss = total_loss / len(loader)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)

    return {
        "val_loss": avg_loss,
        "dice": dice,
        "iou": iou,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def train_unet(
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_classes: int = 1,
    device: str = "cpu",
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    epochs: int = 100,
    early_stop_patience: int = 15,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 5,
    log_interval: int = 10,
    checkpoint_dir: str = "models/checkpoints",
    best_model_path: str = "models/unet_best.pt",
    seed: int = 42,
    logger_name: str = "unet",
    model_type: str = "unet",
    guidino_guidance: bool = False,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    device_obj = torch.device(device)
    logger = get_logger(logger_name)

    if model_type == "pvt_gdla":
        model = build_model("pvt_gdla", in_channels=3, base_filters=64, depth=4, num_heads=4, out_channels=1)
    else:
        model = build_unet_from_config()
    model.to(device_obj)
    guidance_module = GuidiNOGuidance() if guidino_guidance and device_obj.type == "cuda" else None

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=scheduler_factor, patience=scheduler_patience,
    )

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    best_path = Path(best_model_path)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_dice = 0.0
    best_epoch = -1
    patience_counter = 0
    history: list[dict] = []

    logger.info("Starting UNet training for up to %d epochs on %s", epochs, device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, device_obj, logger, log_interval,
        )

        val_metrics = validate(model, val_loader, device_obj)

        scheduler.step(val_metrics["dice"])

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["val_loss"], 6),
            "dice": round(val_metrics["dice"], 4),
            "iou": round(val_metrics["iou"], 4),
            "sensitivity": round(val_metrics["sensitivity"], 4),
            "specificity": round(val_metrics["specificity"], 4),
            "precision": round(val_metrics["precision"], 4),
            "lr": current_lr,
            "time_s": round(epoch_time, 2),
        }
        history.append(entry)

        logger.info(
            "Epoch %3d/%3d | train_loss=%.4f val_loss=%.4f dice=%.4f iou=%.4f sens=%.4f spec=%.4f | lr=%.6f | %.1fs",
            epoch, epochs,
            train_loss, val_metrics["val_loss"],
            val_metrics["dice"], val_metrics["iou"],
            val_metrics["sensitivity"], val_metrics["specificity"],
            current_lr, epoch_time,
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), str(best_path))
            logger.info("  -> Saved best model to %s (dice=%.4f)", best_path, best_val_dice)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch, early_stop_patience,
                )
                break

    total_time = time.time() - start_time
    logger.info("Training completed in %.1f seconds (best epoch %d)", total_time, best_epoch)

    model.load_state_dict(torch.load(str(best_path), weights_only=True))

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "total_time_s": total_time,
        "config": {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "early_stop_patience": early_stop_patience,
            "device": device,
        },
    }


def train_unet_from_config(
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict | None = None,
) -> dict[str, Any]:
    if cfg is None:
        cfg = get_config()

    setup_logger("unet", level=cfg.get("logging", {}).get("level", "INFO"))

    unet_cfg = cfg.get("unet", {})
    train_cfg = cfg.get("training", {})

    return train_unet(
        train_loader=train_loader,
        val_loader=val_loader,
        n_classes=1,
        device=train_cfg.get("device", "cpu"),
        learning_rate=unet_cfg.get("learning_rate", 0.001),
        weight_decay=unet_cfg.get("weight_decay", 0.0001),
        epochs=unet_cfg.get("epochs", 100),
        early_stop_patience=unet_cfg.get("early_stop_patience", 15),
        scheduler_factor=unet_cfg.get("scheduler_factor", 0.5),
        scheduler_patience=unet_cfg.get("scheduler_patience", 5),
        log_interval=train_cfg.get("log_interval", 10),
        checkpoint_dir=train_cfg.get("checkpoint_dir", "models/checkpoints"),
        best_model_path=train_cfg.get("best_model_path", "models/unet_best.pt"),
        seed=unet_cfg.get("random_seed", 42),
        logger_name="unet",
        model_type=cfg.get("model_type", "unet"),
        guidino_guidance=cfg.get("guidino_guidance", False),
    )


def save_unet_history(history: list[dict], path: str = "models/unet_history.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
