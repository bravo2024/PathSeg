"""End-to-end integration tests for PathSeg pipelines."""
import numpy as np
import pytest
import torch

from src.core import LogisticRegression, Standardizer, roc_auc_score, train_test_split
from src.data import H, W, PathSegDataset, create_dataloaders, create_datasets, make_synthetic
from src.evaluate import save_metrics
from src.model import (
    _segmentation_metrics,
    fit_and_evaluate,
    overlay_mask,
    predict_mask,
    predict_proba,
)
from src.persist import save_model
from src.unet import UNet, combined_loss


def test_full_baseline_pipeline():
    data = make_synthetic(n=20, seed=42)
    model, metrics = fit_and_evaluate(data, threshold=0.5, max_pixels=5_000)
    assert model["scaler"] is not None
    assert model["clf"] is not None
    assert 0.0 <= metrics["dice"] <= 1.0
    assert 0.0 <= metrics["iou"] <= 1.0
    assert "tp_pixels" in metrics
    assert "backend" in metrics


def test_predict_on_single_image():
    data = make_synthetic(n=10, seed=0)
    model, _ = fit_and_evaluate(data, max_pixels=5_000)
    img = data["images"][0]
    mask = predict_mask(model, img, threshold=0.5)
    assert mask.shape == (H, W)
    assert set(np.unique(mask)).issubset({0, 1})


def test_predict_proba_on_single_image():
    data = make_synthetic(n=10, seed=0)
    model, _ = fit_and_evaluate(data, max_pixels=5_000)
    img = data["images"][0]
    proba = predict_proba(model, img)
    assert proba.shape == (H, W)
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


def test_predict_on_batch():
    data = make_synthetic(n=10, seed=0)
    model, _ = fit_and_evaluate(data, max_pixels=5_000)
    imgs = data["images"]
    probas = predict_proba(model, imgs)
    assert probas.shape == (10, H, W)


def test_overlay_mask_shape():
    image = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    mask = np.random.randint(0, 2, (H, W), dtype=np.uint8)
    overlay = overlay_mask(image, mask)
    assert overlay.shape == (H, W, 3)
    assert overlay.dtype == np.uint8


def test_overlay_mask_different_colors():
    image = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.ones((H, W), dtype=np.uint8)
    overlay_red = overlay_mask(image, mask, color=(255, 0, 0), alpha=0.5)
    overlay_green = overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5)
    assert not np.array_equal(overlay_red, overlay_green)


def test_overlay_mask_blends_pixels():
    image = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.ones((H, W), dtype=np.uint8)
    overlay = overlay_mask(image, mask, color=(255, 0, 0), alpha=0.5)
    assert overlay[0, 0, 0] > 0


def test_segmentation_metrics_all_predicted_positive():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 1])
    m = _segmentation_metrics(y_true, y_pred)
    assert m["tp_pixels"] == 2
    assert m["fp_pixels"] == 2
    assert m["fn_pixels"] == 0
    assert m["specificity"] == 0.0


def test_segmentation_metrics_all_predicted_negative():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 0])
    m = _segmentation_metrics(y_true, y_pred)
    assert m["tp_pixels"] == 0
    assert m["fn_pixels"] == 2
    assert m["tn_pixels"] == 2
    assert m["sensitivity"] == 0.0
    assert m["dice"] == 0.0


def test_print_report(capsys):
    from src.evaluate import print_report
    metrics = {"dice": 0.85, "iou": 0.75, "n_samples": 10}
    print_report(metrics)
    captured = capsys.readouterr()
    assert "dice" in captured.out
    assert "0.8500" in captured.out
    assert "n_samples" in captured.out


def test_save_and_load_roundtrip(tmp_path):
    import json as _json
    from src.persist import load_model
    data = make_synthetic(n=10, seed=0)
    model, metrics = fit_and_evaluate(data, max_pixels=5_000)

    pkl_path = tmp_path / "model.pkl"
    json_path = tmp_path / "metrics.json"
    save_model(model, str(pkl_path))
    save_metrics(metrics, str(json_path))

    assert pkl_path.exists()
    assert json_path.exists()

    loaded_metrics = _json.loads(json_path.read_text())
    assert abs(loaded_metrics["dice"] - metrics["dice"]) < 1e-9

    loaded_model = load_model(str(pkl_path))
    img = data["images"][0]
    pred_orig = predict_mask(model, img)
    pred_loaded = predict_mask(loaded_model, img)
    assert np.array_equal(pred_orig, pred_loaded)


def test_segmentation_metrics_correct():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1])
    m = _segmentation_metrics(y_true, y_pred)
    assert m["tp_pixels"] == 2
    assert m["fp_pixels"] == 1
    assert m["fn_pixels"] == 1
    assert m["tn_pixels"] == 2
    assert abs(m["dice"] - 2 * 2 / (2 * 2 + 1 + 1)) < 1e-6
    assert abs(m["iou"] - 2 / (2 + 1 + 1)) < 1e-6


def test_segmentation_metrics_perfect():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    m = _segmentation_metrics(y_true, y_pred)
    assert m["dice"] == 1.0
    assert m["iou"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0


def test_segmentation_metrics_no_positives():
    y_true = np.zeros(10)
    y_pred = np.zeros(10)
    m = _segmentation_metrics(y_true, y_pred)
    assert m["dice"] == 0.0
    assert m["iou"] == 0.0


def test_unet_forward_and_loss():
    model = UNet(in_channels=3, base_filters=16, depth=3, dropout=0.0)
    model.eval()
    x = torch.randn(4, 3, 64, 64)
    y = (torch.rand(4, 1, 64, 64) > 0.8).float()
    with torch.no_grad():
        logits = model(x)
        loss = combined_loss(logits, y, bce_weight=0.5)
    assert logits.shape == (4, 1, 64, 64)
    assert loss.item() > 0


def test_train_test_split_proportions():
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, seed=42)
    assert len(X_tr) == 75
    assert len(X_te) == 25
    assert len(y_tr) == 75
    assert len(y_te) == 25


def test_standardizer():
    X = np.random.randn(50, 4)
    scaler = Standardizer().fit(X)
    Xt = scaler.transform(X)
    assert abs(Xt.mean(0)).max() < 1e-7
    assert abs(Xt.std(0) - 1.0).max() < 1e-6


def test_logistic_regression_converges():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (200, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    clf = LogisticRegression(lr=0.3, epochs=300, l2=1e-3, seed=0).fit(X, y)
    preds = clf.predict(X)
    acc = (preds == y).mean()
    assert acc > 0.85


def test_roc_auc():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    auc = roc_auc_score(y, s)
    assert 0.5 <= auc <= 1.0


def test_core_accuracy():
    from src.core import accuracy_score
    assert accuracy_score(np.array([1, 0, 1]), np.array([1, 0, 1])) == 1.0
    assert accuracy_score(np.array([1, 0]), np.array([0, 1])) == 0.0


def test_core_f1():
    from src.core import f1_score
    y = np.array([1, 1, 0, 0])
    p = np.array([1, 1, 1, 0])
    f1 = f1_score(y, p)
    assert 0.5 < f1 < 1.0


def test_pipeline_with_different_thresholds():
    data = make_synthetic(n=10, seed=0)
    model, metrics_high = fit_and_evaluate(data, threshold=0.8, max_pixels=5_000)
    _, metrics_low = fit_and_evaluate(data, threshold=0.2, max_pixels=5_000)
    assert metrics_high["threshold"] == 0.8
    assert metrics_low["threshold"] == 0.2


def test_dataset_with_transform():
    data = make_synthetic(n=5, seed=0)
    from src.transforms import train_transform
    ds = PathSegDataset(data, transform=train_transform)
    img, msk = ds[0]
    assert img.shape == (3, H, W)
    assert msk.shape == (1, H, W)


def test_dataloaders_batch_shapes():
    data = make_synthetic(n=16, seed=0)
    ds = PathSegDataset(data)
    loader = create_dataloaders({"train": ds}, batch_size=4)["train"]
    for images, masks in loader:
        assert images.shape == (4, 3, H, W)
        assert masks.shape == (4, 1, H, W)
        assert images.dtype == torch.float32
        assert masks.dtype == torch.float32
        break


def test_full_baseline_pipeline_metrics_range():
    for seed in [1, 7, 42]:
        data = make_synthetic(n=30, seed=seed)
        _, metrics = fit_and_evaluate(data, max_pixels=10_000)
        assert 0 <= metrics["dice"] <= 1
        assert 0 <= metrics["iou"] <= 1
        assert 0 <= metrics["sensitivity"] <= 1
        assert 0 <= metrics["specificity"] <= 1
