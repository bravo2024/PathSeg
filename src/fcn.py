"""Tiny Keras fully-convolutional network for binary pixel segmentation.

Architecture: Conv2D(16) → Conv2D(32) → Conv2D(1) — same spatial size in/out.
Trains on CPU in ~15 s for 120 synthetic 128×128 tiles. No GPU or checkpoint file
needed — works on Streamlit Community Cloud out of the box.
"""
from __future__ import annotations

import numpy as np

_H = _W = 128


def build_fcn(input_shape: tuple = (_H, _W, 3)):
    from tensorflow import keras  # lazy import — keeps startup fast if TF not used

    inp = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255.0)(inp)
    x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(1,  1, padding="same", activation="sigmoid")(x)
    return keras.Model(inp, x)


def fit_fcn(
    data: dict,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 8,
    seed: int = 42,
) -> dict:
    """Train the FCN and return a model dict that includes stored test predictions.

    Storing test predictions means threshold can be changed later via
    ``evaluate_fcn()`` without retraining.
    """
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(seed)
    np.random.seed(seed)

    images = np.asarray(data["images"], dtype=np.float32)               # (N, H, W, 3)
    masks  = np.asarray(data["masks"],  dtype=np.float32)[..., np.newaxis]  # (N, H, W, 1)

    n = len(images)
    n_train = max(1, int(n * 0.75))
    x_tr, x_te = images[:n_train], images[n_train:]
    y_tr, y_te = masks[:n_train],  masks[n_train:]

    model = build_fcn(input_shape=(images.shape[1], images.shape[2], images.shape[3]))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
    )

    hist = model.fit(
        x_tr, y_tr,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_te, y_te),
        verbose=0,
    )

    test_probas = model.predict(x_te, verbose=0)[..., 0]  # (N_test, H, W)
    test_masks  = y_te[..., 0].astype(np.uint8)           # (N_test, H, W)

    return {
        "model":       model,
        "train_loss":  hist.history["loss"],
        "val_loss":    hist.history.get("val_loss", []),
        "epochs":      epochs,
        "lr":          lr,
        "n_train":     n_train,
        "n_test":      n - n_train,
        "pos_rate":    float(masks.mean()),
        "test_probas": test_probas,
        "test_masks":  test_masks,
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

    return {
        "backend":              "keras-fcn",
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
        "sampled_train_pixels": fcn_dict["n_train"] * _H * _W,
        "positive_pixel_rate":  fcn_dict["pos_rate"],
    }


def predict_proba_fcn(fcn_dict: dict, image: np.ndarray) -> np.ndarray:
    """Return (H, W) float32 probability map for a single image."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3:
        img = img[np.newaxis]  # (1, H, W, 3)
    return fcn_dict["model"].predict(img, verbose=0)[0, ..., 0]


def predict_mask_fcn(
    fcn_dict: dict,
    image: np.ndarray,
    threshold: float | None = None,
) -> np.ndarray:
    t = threshold if threshold is not None else float(fcn_dict.get("threshold", 0.5))
    return (predict_proba_fcn(fcn_dict, image) >= t).astype(np.uint8)
