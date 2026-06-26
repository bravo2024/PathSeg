"""Model save/load via pickle."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_model(model: Any, path: str = "models/model.pkl") -> str:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path_obj), "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(path: str = "models/model.pkl") -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
