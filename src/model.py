"""Pixel-level pathology segmentation baseline and UNet inference wrapper."""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import torch

from src.config import get_config
from src.core import LogisticRegression, Standardizer, train_test_split
from src.unet import build_unet_from_config

_MODEL_ROOT = Path(__file__).resolve().parent.parent / "models"


def _pixel_features(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images, dtype=float)
    if images.ndim == 3:
        images = images[None, ...]
    images = images / 255.0
    n_images, height, width, _ = images.shape
    red = images[..., 0]
    green = images[..., 1]
    blue = images[..., 2]
    brightness = images.mean(axis=-1)
    purple_score = blue + red - 1.35 * green
    saturation = images.max(axis=-1) - images.min(axis=-1)
    yy, xx = np.mgrid[0:height, 0:width]
    x_pos = np.tile(xx / max(width - 1, 1), (n_images, 1, 1))
    y_pos = np.tile(yy / max(height - 1, 1), (n_images, 1, 1))
    features = np.stack(
        [red, green, blue, brightness, purple_score, saturation, x_pos, y_pos],
        axis=-1,
    )
    return features.reshape(-1, features.shape[-1])


def _sample_pixels(
    images: np.ndarray,
    masks: np.ndarray,
    max_pixels: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = _pixel_features(images)
    labels = masks.reshape(-1).astype(int)
    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    half = max_pixels // 2
    pos_count = min(len(positives), half)
    neg_count = min(len(negatives), max_pixels - pos_count)
    pos_idx = rng.choice(positives, pos_count, replace=False) if pos_count else np.array([], dtype=int)
    neg_idx = rng.choice(negatives, neg_count, replace=False) if neg_count else np.array([], dtype=int)
    selected = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(selected)
    return features[selected], labels[selected]


def _segmentation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    dice = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "tn_pixels": tn,
    }


def fit_and_evaluate(data: dict, threshold: float = 0.5, max_pixels: int = 25_000) -> tuple[dict, dict]:
    cfg = get_config().get("baseline", {})
    lr = cfg.get("learning_rate", 0.28)
    epochs = cfg.get("epochs", 150)
    l2 = cfg.get("l2_reg", 2e-3)
    clf_seed = cfg.get("random_seed", 11)

    images = np.asarray(data["images"], dtype=np.uint8)
    masks = np.asarray(data["masks"], dtype=np.uint8)
    image_ids = np.arange(len(images))
    train_ids, test_ids, _, _ = train_test_split(image_ids[:, None], image_ids, test_size=0.25, seed=7)
    train_ids = train_ids.ravel().astype(int)
    test_ids = test_ids.ravel().astype(int)

    x_train, y_train = _sample_pixels(images[train_ids], masks[train_ids], max_pixels=max_pixels, seed=42)
    scaler = Standardizer().fit(x_train)
    classifier = LogisticRegression(lr=lr, epochs=epochs, l2=l2, seed=clf_seed).fit(
        scaler.transform(x_train),
        y_train,
    )

    probability_maps = predict_proba({"scaler": scaler, "clf": classifier}, images[test_ids])
    predicted_masks = probability_maps >= threshold
    metrics = _segmentation_metrics(masks[test_ids], predicted_masks)
    metrics.update(
        {
            "backend": "numpy-pixel-logreg",
            "n_train_tiles": int(len(train_ids)),
            "n_test_tiles": int(len(test_ids)),
            "sampled_train_pixels": int(len(y_train)),
            "positive_pixel_rate": float(masks.mean()),
            "threshold": float(threshold),
        }
    )
    return {
        "scaler": scaler,
        "clf": classifier,
        "features": data.get("features", []),
        "threshold": float(threshold),
    }, metrics


def predict_proba(model: dict, images: np.ndarray) -> np.ndarray:
    images = np.asarray(images, dtype=np.uint8)
    single_image = images.ndim == 3
    if single_image:
        images = images[None, ...]
    height, width = images.shape[1], images.shape[2]
    features = _pixel_features(images)
    probabilities = model["clf"].predict_proba(model["scaler"].transform(features))
    maps = probabilities.reshape(len(images), height, width)
    return maps[0] if single_image else maps


def predict_mask(model: dict, image: np.ndarray, threshold: float | None = None) -> np.ndarray:
    threshold = float(model.get("threshold", 0.5) if threshold is None else threshold)
    return (predict_proba(model, image) >= threshold).astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(255, 40, 120), alpha: float = 0.45) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    mask = np.asarray(mask).astype(bool)
    overlay = image.copy().astype(float)
    color_arr = np.asarray(color, dtype=float)
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color_arr
    return np.clip(overlay, 0, 255).astype(np.uint8)


def predict(model: dict, image: np.ndarray) -> np.ndarray:
    return predict_mask(model, image)


# ---------------------------------------------------------------------------
# UNet inference helpers
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def _load_unet_for_inference(
    model_path: str | None = None,
    device: str = "cpu",
) -> tuple[torch.nn.Module, torch.device]:
    resolved = model_path if model_path else str(_MODEL_ROOT / "unet_best.pt")
    device_obj = torch.device(device)
    model = build_unet_from_config()
    state = torch.load(resolved, map_location=device_obj, weights_only=True)
    model.load_state_dict(state)
    model.to(device_obj)
    model.eval()
    return model, device_obj


@torch.no_grad()
def predict_proba_unet(
    images: np.ndarray,
    model_path: str | None = None,
    device: str = "cpu",
) -> np.ndarray:
    images = np.asarray(images, dtype=np.uint8)
    single = images.ndim == 3
    if single:
        images = images[None, ...]

    model, device_obj = _load_unet_for_inference(model_path, device)
    n, h, w, c = images.shape
    tensor = torch.from_numpy(images).float().permute(0, 3, 1, 2) / 255.0
    tensor = tensor.to(device_obj)

    logits = model(tensor)
    probs = torch.sigmoid(logits).cpu().numpy()

    maps = probs[:, 0]
    result = maps[0] if single else maps
    return np.asarray(result, dtype=np.float32)


def predict_mask_unet(
    image: np.ndarray,
    threshold: float = 0.5,
    model_path: str | None = None,
    device: str = "cpu",
) -> np.ndarray:
    proba = predict_proba_unet(image, model_path=model_path, device=device)
    return (proba >= threshold).astype(np.uint8)


def unet_model_exists(model_path: str | Path | None = None) -> bool:
    if model_path is None:
        return (_MODEL_ROOT / "unet_best.pt").exists()
    return Path(model_path).exists()
