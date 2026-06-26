"""Configuration loader for PathSeg.

Supports YAML config files with optional environment variable overrides
using the prefix ``PATHSEG__SECTION__KEY``.

Usage::

    from src.config import get_config
    cfg = get_config()
    cfg["unet"]["learning_rate"]   # 0.001

Environment override example::

    export PATHSEG__UNET__EPOCHS=200
    # cfg["unet"]["epochs"] is now 200
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "base.yaml"
_ENV_PATTERN = re.compile(r"^PATHSEG__(.+)__(.+)$")
_DEFAULT_CONFIG: dict[str, Any] | None = None


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base dict."""
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_env_overrides() -> dict:
    """Parse environment variables matching PATHSEG__SECTION__KEY into a nested dict."""
    overrides: dict[str, Any] = {}
    for key, raw in os.environ.items():
        match = _ENV_PATTERN.match(key)
        if not match:
            continue
        section, option = match.group(1).lower(), match.group(2).lower()
        if section not in overrides:
            overrides[section] = {}
        # Try parsing as Python literal (int, float, bool, None)
        val: str | int | float | bool | None = raw
        try:
            val = int(raw)
        except ValueError:
            try:
                val = float(raw)
            except ValueError:
                lower = raw.lower()
                if lower in ("true", "yes", "1"):
                    val = True
                elif lower in ("false", "no", "0"):
                    val = False
                elif lower in ("none", "null", ""):
                    val = None
        overrides[section][option] = val
    return overrides


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file with environment variable overrides.

    Parameters
    ----------
    path : str or Path, optional
        Path to YAML config file.  Defaults to ``config/base.yaml``.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    path = Path(path) if path else _CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as fh:
        config: dict[str, Any] = yaml.safe_load(fh) or {}

    env_overrides = _load_env_overrides()
    if env_overrides:
        config = _deep_merge(config, env_overrides)

    return config


def get_config() -> dict[str, Any]:
    """Get the cached global configuration."""
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = load_config()
    return _DEFAULT_CONFIG


def reload_config(path: str | Path | None = None) -> dict[str, Any]:
    """Force-reload config (useful in interactive/notebook contexts)."""
    global _DEFAULT_CONFIG
    _DEFAULT_CONFIG = load_config(path)
    return _DEFAULT_CONFIG
