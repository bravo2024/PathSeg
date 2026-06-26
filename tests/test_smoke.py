"""Quick smoke test: verify the full pipeline runs end-to-end."""

from src.data import make_synthetic
from src.model import fit_and_evaluate, predict_mask, overlay_mask


def test_pipeline_runs():
    data = make_synthetic(n=10, seed=42)
    model, metrics = fit_and_evaluate(data, max_pixels=5_000)
    assert isinstance(metrics, dict) and len(metrics) >= 5
    assert model is not None
    assert metrics["dice"] > 0.0
    assert metrics["iou"] > 0.0


def test_predict_and_overlay():
    data = make_synthetic(n=5, seed=0)
    model, _ = fit_and_evaluate(data, max_pixels=5_000)
    img = data["images"][0]
    mask = predict_mask(model, img)
    assert mask.shape == img.shape[:2]
    overlay = overlay_mask(img, mask)
    assert overlay.shape == img.shape


def test_synthetic_data_structure():
    data = make_synthetic(n=5, seed=0)
    assert "images" in data
    assert "masks" in data
    assert "features" in data
    assert "target" in data
    assert len(data["images"]) == 5
