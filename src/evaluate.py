"""Metrics persistence and console reporting."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_metrics(metrics: dict[str, Any], path: str = "models/metrics.json") -> dict[str, Any]:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(metrics, indent=2))
    return metrics


def print_report(metrics: dict[str, Any]) -> None:
    width = 44
    print("=" * width)
    print("Evaluation report")
    print("=" * width)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:<22s}: {value:.4f}")
        else:
            print(f"  {key:<22s}: {value}")
