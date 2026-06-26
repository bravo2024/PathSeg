"""Pixel-level MLP segmenter using scikit-learn.

Architecture: 3x3 patch features (27-dim) → Dense(64) → Dense(32) → sigmoid.
Trains on randomly sampled pixels from training tiles so it stays fast on CPU.
No TensorFlow/PyTorch needed — works on any Python version and Streamlit Cloud.
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

_H = _W = 128
_PATCH_R = 1   # 3×3 neighbourhood → 3*9 = 27 features per pixel


def _patch_features(image: np.ndarray, radius: int = _PATCH_R) -> np.ndarray:
    """Extract flattened 3x3 patch features for every pixel.

    (H, W, C) float32 in [0,1] → (H*W, C*(2r+1)²) float32
    """
    H, W, C = image.shape
    r = radius
    padded = np.pad(image, ((r, r), (r, r), (0, 0)), mode="reflect")
    side = 2 * r + 1
    out = np.empty((H * W, C * side * side), dtype=np.float32)
    k = 0
    for dy in range(side):
        for dx in range(side):
            patch = padded[dy : dy + H, dx : dx + W, :]
            out[:, k : k + C] = patch.reshape(H * W, C)
            k += C
    return out


def fit_fcn(
    data: dict,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int | None = None,
    seed: int = 42,
    max_px_per_tile: int = 2000,
) -> dict:
    """Train the MLP and return a model dict with stored test predictions.

    Test predictions are stored so threshold can be changed later via
    ``evaluate_fcn()`` without retraining.
    """
    np.random.seed(seed)

    images = np.asarray(data["images"], dtype=np.float32) / 255.0
    masks  = np.asarray(data["masks"],  dtype=np.uint8)

    n = len(images)
    n_train = max(1, int(n * 0.75))
    x_tr, x_te = images[:n_train], images[n_train:]
    y_tr, y_te = masks[:n_train],  masks[n_train:]
    H, W = images.shape[1], images.shape[2]

    # Subsample pixels from each training tile to keep training fast
    rng = np.random.RandomState(seed)
    X_list, y_list = [], []
    for img, msk in zip(x_tr, y_tr):
        feats = _patch_features(img)
        flat  = msk.reshape(-1).astype(np.int32)
        n_px  = feats.shape[0]
        idx   = rng.choice(n_px, min(max_px_per_tile, n_px), replace=False)
        X_list.append(feats[idx])
        y_list.append(flat[idx])

    X_tr      = np.vstack(X_list)
    y_tr_flat = np.concatenate(y_list)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=epochs,
        learning_rate_init=lr,
        random_state=seed,
        verbose=False,
    )
    mlp.fit(X_tr_s, y_tr_flat)

    # Predict on every pixel of every test tile
    test_probas_list: list[np.ndarray] = []
    for img in x_te:
        feats = _patch_features(img)
        proba = mlp.predict_proba(scaler.transform(feats))[:, 1].reshape(H, W)
        test_probas_list.append(proba.astype(np.float32))

    test_probas = np.stack(test_probas_list) if test_probas_list else np.empty((0, H, W), np.float32)
    test_masks  = y_te.astype(np.uint8)

    return {
        "model":       mlp,
        "scaler":      scaler,
        "train_loss":  list(mlp.loss_curve_),
        "val_loss":    [],
        "epochs":      len(mlp.loss_curve_),
        "lr":          lr,
        "n_train":     n_train,
        "n_test":      n - n_train,
        "pos_rate":    float(masks.mean()),
        "test_probas": test_probas,
        "test_masks":  test_masks,
        "H":           H,
        "W":           W,
    }


def evaluate_fcn(fcn_dict: dict, threshold: float = 0.5) -> dict:
    """Re-score stored test predictions at any threshold — no retraining needed."""
    p    = fcn_dict["test_probas"]
    y    = fcn_dict["test_masks"]
    pred = (p >= threshold).astype(np.uint8)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou  = tp / (tp + fp + fn)          if tp + fp + fn      else 0.0
    sens = tp / (tp + fn)               if tp + fn            else 0.0
    spec = tn / (tn + fp)               if tn + fp            else 0.0
    prec = tp / (tp + fp)               if tp + fp            else 0.0

    H = fcn_dict.get("H", _H)
    W = fcn_dict.get("W", _W)

    return {
        "backend":              "sklearn-mlp",
        "dice":                 float(dice),
        "iou":                  float(iou),
        "sensitivity":          float(sens),
        "specificity":          float(spec),
        "precision":            float(prec),
        "tp_pixels":            tp,
        "fp_pixels":            fp,
        "fn_pixels":            fn,
        "tn_pixels":            tn,
        "threshold":            float(threshold),
        "n_train_tiles":        fcn_dict["n_train"],
        "n_test_tiles":         fcn_dict["n_test"],
        "sampled_train_pixels": fcn_dict["n_train"] * H * W,
        "positive_pixel_rate":  fcn_dict["pos_rate"],
    }


def predict_proba_fcn(fcn_dict: dict, image: np.ndarray) -> np.ndarray:
    """Return (H, W) float32 probability map for a single image."""
    img = np.asarray(image, dtype=np.float32) / 255.0
    if img.ndim == 4:
        img = img[0]
    H, W = img.shape[:2]
    feats = _patch_features(img)
    proba = fcn_dict["model"].predict_proba(fcn_dict["scaler"].transform(feats))[:, 1]
    return proba.reshape(H, W).astype(np.float32)


def predict_mask_fcn(
    fcn_dict: dict,
    image: np.ndarray,
    threshold: float | None = None,
) -> np.ndarray:
    t = threshold if threshold is not None else float(fcn_dict.get("threshold", 0.5))
    return (predict_proba_fcn(fcn_dict, image) >= t).astype(np.uint8)
