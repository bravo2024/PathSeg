"""train.py — Train baseline or UNet segmentation model on synthetic or real data.

Usage:
    python train.py                          # baseline (default)
    python train.py --model unet             # train UNet
    python train.py --model baseline --n 200 --seed 7
    python train.py --model unet --real data/raw/pathseg
    python train.py --model unet --epochs 50 --lr 0.0005
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_config, load_config
from src.data import H, W, make_synthetic, prepare_data
from src.evaluate import print_report, save_metrics
from src.logger import setup_logger
from src.model import fit_and_evaluate
from src.persist import save_model
from src.train_unet import save_unet_history, train_unet_from_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PathSeg model training")
    parser.add_argument("--model", choices=["baseline", "unet"], default="baseline", help="Model type")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--real", type=str, default=None, help="Path to real data folder")
    parser.add_argument("--n", type=int, default=None, help="Number of synthetic tiles")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs (UNet only)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help='Device: "cpu" or "cuda"')
    parser.add_argument("--threshold", type=float, default=None, help="Segmentation threshold")
    return parser.parse_args(argv)


def override_config(cfg: dict, args: argparse.Namespace) -> dict:
    if args.n is not None:
        cfg.setdefault("data", {})["n_synthetic"] = args.n
    if args.seed is not None:
        cfg.setdefault("data", {})["synthetic_seed"] = args.seed
    if args.epochs is not None:
        cfg.setdefault("unet", {})["epochs"] = args.epochs
    if args.lr is not None:
        if args.model == "unet":
            cfg.setdefault("unet", {})["learning_rate"] = args.lr
        else:
            cfg.setdefault("baseline", {})["learning_rate"] = args.lr
    if args.device is not None:
        cfg.setdefault("training", {})["device"] = args.device
    if args.threshold is not None and args.model == "baseline":
        cfg.setdefault("baseline", {})["threshold"] = args.threshold
    return cfg


def train_baseline(cfg: dict, args: argparse.Namespace) -> None:
    data_cfg = cfg.get("data", {})
    baseline_cfg = cfg.get("baseline", {})

    data = make_synthetic(
        n=args.n if args.n is not None else data_cfg.get("n_synthetic", 160),
        seed=args.seed if args.seed is not None else data_cfg.get("synthetic_seed", 42),
    )

    threshold = args.threshold if args.threshold is not None else 0.5
    model, metrics = fit_and_evaluate(
        data,
        threshold=threshold,
        max_pixels=baseline_cfg.get("max_pixels", 25_000),
    )

    save_model(model)
    save_metrics(metrics)
    print_report(metrics)
    print(f"\nSaved: models/model.pkl  models/metrics.json")
    print(f"Backend: numpy-pixel-logreg  |  Dice: {metrics['dice']:.4f}  IoU: {metrics['iou']:.4f}")


def train_unet(cfg: dict, args: argparse.Namespace) -> None:
    logger = setup_logger("train", level=cfg.get("logging", {}).get("level", "INFO"))
    logger.info("Preparing data for UNet training...")

    data_cfg = cfg.get("data", {})
    use_synthetic = args.real is None
    result = prepare_data(
        use_synthetic=use_synthetic,
        real_data_path=args.real,
        n_synthetic=args.n if args.n is not None else data_cfg.get("n_synthetic", 160),
        synthetic_seed=args.seed if args.seed is not None else data_cfg.get("synthetic_seed", 42),
        batch_size=data_cfg.get("batch_size", 16),
        val_batch_size=data_cfg.get("val_batch_size", 32),
        num_workers=data_cfg.get("num_workers", 0),
    )

    logger.info(
        "Data ready: %s | train=%d val=%d test=%d | pos_frac=%.3f",
        result["metadata"]["source"],
        len(result["datasets"]["train"]),
        len(result["datasets"]["val"]),
        len(result["datasets"]["test"]),
        result["metadata"]["positive_fraction"],
    )

    output = train_unet_from_config(
        train_loader=result["loaders"]["train"],
        val_loader=result["loaders"]["val"],
        cfg=cfg,
    )

    save_unet_history(output["history"])

    val_metrics = max(output["history"], key=lambda e: e["dice"])
    print_report(val_metrics)
    print(f"\nSaved: models/unet_best.pt  models/unet_history.json")
    print(f"Backend: unet-pytorch  |  Dice: {val_metrics['dice']:.4f}  IoU: {val_metrics['iou']:.4f}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = get_config()

    cfg = override_config(cfg, args)

    if args.model == "baseline":
        train_baseline(cfg, args)
    else:
        train_unet(cfg, args)


if __name__ == "__main__":
    main()
