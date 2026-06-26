"""Coverage tests for config, logger, evaluate, persist, and edge cases."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from src.data import H, W, make_synthetic
from src.model import fit_and_evaluate, predict_mask


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_data_and_model():
    data = make_synthetic(n=20, seed=42)
    model, metrics = fit_and_evaluate(data, max_pixels=5_000)
    return data, model, metrics


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@pytest.fixture()
def reset_config():
    from src.config import reload_config
    reload_config()
    yield
    reload_config()


def test_load_config_returns_dict():
    from src.config import load_config
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "unet" in cfg
    assert "baseline" in cfg
    assert "data" in cfg


def test_get_config_cached():
    from src.config import get_config
    cfg1 = get_config()
    cfg2 = get_config()
    assert cfg1 is cfg2


def test_reload_config_returns_fresh(reset_config):
    from src.config import get_config, reload_config
    cfg1 = get_config()
    cfg2 = reload_config()
    assert cfg2["unet"]["epochs"] == cfg1["unet"]["epochs"]


def test_config_env_override_int(monkeypatch, reset_config):
    from src.config import reload_config
    monkeypatch.setenv("PATHSEG__UNET__EPOCHS", "200")
    cfg = reload_config()
    assert cfg["unet"]["epochs"] == 200
    assert isinstance(cfg["unet"]["epochs"], int)


def test_config_env_override_float(monkeypatch, reset_config):
    from src.config import reload_config
    monkeypatch.setenv("PATHSEG__UNET__LEARNING_RATE", "0.005")
    cfg = reload_config()
    assert abs(cfg["unet"]["learning_rate"] - 0.005) < 1e-9


def test_config_env_override_bool(monkeypatch, reset_config):
    from src.config import reload_config
    monkeypatch.setenv("PATHSEG__TRAINING__DEVICE", "cuda")
    cfg = reload_config()
    assert cfg["training"]["device"] == "cuda"


def test_deep_merge():
    from src.config import _deep_merge
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    overrides = {"a": {"y": 99, "z": 100}, "c": 4}
    result = _deep_merge(base, overrides)
    assert result["a"]["x"] == 1
    assert result["a"]["y"] == 99
    assert result["a"]["z"] == 100
    assert result["b"] == 3
    assert result["c"] == 4


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


def test_setup_logger_no_handlers():
    from src.logger import setup_logger
    logger = setup_logger("test_no_handlers", level="DEBUG", log_file=None, console=False)
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 0


def test_setup_logger_console_only():
    from src.logger import setup_logger
    logger = setup_logger("test_console", log_file=None, console=True)
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)


def test_setup_logger_with_file(tmp_path):
    from src.logger import setup_logger
    log_file = str(tmp_path / "test.log")
    logger = setup_logger("test_file_logger", log_file=log_file, console=False)
    logger.info("hello from test")
    assert Path(log_file).exists()
    assert "hello from test" in Path(log_file).read_text(encoding="utf-8")


def test_get_logger_returns_logger():
    from src.logger import get_logger
    logger = get_logger("pathseg")
    assert isinstance(logger, logging.Logger)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


def test_save_metrics_content(tmp_path):
    from src.evaluate import save_metrics
    metrics = {"dice": 0.85, "iou": 0.73, "n_tiles": 20}
    path = tmp_path / "metrics.json"
    save_metrics(metrics, str(path))
    loaded = json.loads(path.read_text())
    assert abs(loaded["dice"] - 0.85) < 1e-9
    assert abs(loaded["iou"] - 0.73) < 1e-9
    assert loaded["n_tiles"] == 20


def test_print_report_float_formatting(capsys):
    from src.evaluate import print_report
    metrics = {"dice": 0.9123, "n_samples": 42, "backend": "numpy"}
    print_report(metrics)
    out = capsys.readouterr().out
    assert "0.9123" in out
    assert "42" in out
    assert "numpy" in out
    assert "=" * 10 in out


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------


def test_load_model_roundtrip(tmp_path, small_data_and_model):
    from src.persist import load_model, save_model
    data, model, _ = small_data_and_model
    pkl_path = tmp_path / "model.pkl"
    save_model(model, str(pkl_path))
    loaded = load_model(str(pkl_path))
    img = data["images"][0]
    assert np.array_equal(predict_mask(model, img), predict_mask(loaded, img))
