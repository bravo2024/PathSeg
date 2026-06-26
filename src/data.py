"""Synthetic pathology tile generator, real dataset loader, and optional PyTorch Dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from src.config import get_config

H = W = 128

# ---------------------------------------------------------------------------
# Synthetic tile generator
# ---------------------------------------------------------------------------


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, int, int]) -> np.ndarray:
    coarse = rng.normal(0, 1, (shape[0] // 8 + 1, shape[1] // 8 + 1, shape[2]))
    image = np.repeat(np.repeat(coarse, 8, axis=0), 8, axis=1)[: shape[0], : shape[1]]
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image


def _ellipse_mask(
    yy: np.ndarray,
    xx: np.ndarray,
    center_y: float,
    center_x: float,
    radius_y: float,
    radius_x: float,
    angle: float,
) -> np.ndarray:
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    y, x = yy - center_y, xx - center_x
    rotated_x = cos_a * x + sin_a * y
    rotated_y = -sin_a * x + cos_a * y
    return (rotated_x / radius_x) ** 2 + (rotated_y / radius_y) ** 2 <= 1.0


def _make_tile(rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:size, 0:size]
    base = np.zeros((size, size, 3), dtype=float)
    base[..., 0] = 230
    base[..., 1] = 188
    base[..., 2] = 214
    base += 32 * (_smooth_noise(rng, (size, size, 3)) - 0.5)

    tissue_mask = _ellipse_mask(
        yy, xx,
        rng.uniform(size * 0.42, size * 0.58),
        rng.uniform(size * 0.42, size * 0.58),
        rng.uniform(size * 0.38, size * 0.54),
        rng.uniform(size * 0.36, size * 0.52),
        rng.uniform(0, np.pi),
    )
    base[~tissue_mask] = np.array([246, 239, 244]) + rng.normal(0, 3, base[~tissue_mask].shape)

    tumor_mask = np.zeros((size, size), dtype=bool)
    for _ in range(rng.integers(1, 4)):
        blob = _ellipse_mask(
            yy, xx,
            rng.uniform(size * 0.25, size * 0.75),
            rng.uniform(size * 0.25, size * 0.75),
            rng.uniform(size * 0.08, size * 0.20),
            rng.uniform(size * 0.10, size * 0.24),
            rng.uniform(0, np.pi),
        )
        tumor_mask |= blob & tissue_mask

    tissue_pixels = np.argwhere(tissue_mask)
    n_nuclei = int(size * size * rng.uniform(0.015, 0.035))
    selected = tissue_pixels[
        rng.choice(len(tissue_pixels), size=min(n_nuclei, len(tissue_pixels)), replace=False)
    ]
    for y, x in selected:
        radius = rng.integers(1, 3)
        nucleus = (yy - y) ** 2 + (xx - x) ** 2 <= radius**2
        if tumor_mask[y, x]:
            base[nucleus] = np.array([92, 38, 138]) + rng.normal(0, 8, base[nucleus].shape)
        else:
            base[nucleus] = np.array([122, 74, 160]) + rng.normal(0, 7, base[nucleus].shape)

    base[tumor_mask] += np.array([-35, -55, 32])
    base += rng.normal(0, 5, base.shape)
    image = np.clip(base, 0, 255).astype(np.uint8)
    return image, tumor_mask.astype(np.uint8)


def make_synthetic(n: int = 160, seed: int = 42, size: int = H) -> dict:
    """Generate synthetic H&E-like tiles and tumor masks.

    The generator is intentionally lightweight so the full project trains and
    deploys without external datasets. It simulates staining texture, tissue
    area, nuclei, and tumor-like regions for segmentation practice.
    """
    rng = np.random.default_rng(seed)
    images, masks = [], []
    for _ in range(n):
        image, mask = _make_tile(rng, size)
        images.append(image)
        masks.append(mask)
    return {
        "images": np.asarray(images, dtype=np.uint8),
        "masks": np.asarray(masks, dtype=np.uint8),
        "features": ["red", "green", "blue", "brightness", "purple_score", "saturation", "x", "y"],
        "target": "tumor_mask",
        "source": "Synthetic H&E-like pathology tiles",
    }


# ---------------------------------------------------------------------------
# Real data loader
# ---------------------------------------------------------------------------


def load_real(folder: str | Path) -> dict:
    """Load paired real images/masks from a folder.

    Expected structure::

        data/raw/pathseg/
          images/
            tile_001.png
          masks/
            tile_001.png
    """
    root = Path(folder)
    image_dir = root / "images"
    mask_dir = root / "masks"
    images, masks = [], []

    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        mask_path = mask_dir / image_path.with_suffix(".png").name
        if not mask_path.exists():
            continue
        image = Image.open(image_path).convert("RGB").resize((W, H))
        mask = Image.open(mask_path).convert("L").resize((W, H))
        images.append(np.asarray(image, dtype=np.uint8))
        masks.append((np.asarray(mask) > 127).astype(np.uint8))

    if not images:
        raise FileNotFoundError(f"No image/mask pairs found under {root}")

    return {
        "images": np.asarray(images, dtype=np.uint8),
        "masks": np.asarray(masks, dtype=np.uint8),
        "features": ["red", "green", "blue", "brightness", "purple_score", "saturation", "x", "y"],
        "target": "tumor_mask",
        "source": str(root),
    }


def _load_monuseg_item(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a single MoNuSeg image and its nuclear mask.

    MoNuSeg masks are multi-class (each nucleus has a unique value > 0).
    We binarize to tissue/nucleus vs background as a proxy segmentation task.
    """
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return None
    # MoNuSeg convention: image and mask share the same name, mask is in a "masks" subfolder
    mask_path = path.parent.parent / "masks" / path.name
    if not mask_path.exists():
        return None
    image = Image.open(path).convert("RGB").resize((W, H))
    mask = Image.open(mask_path).convert("L").resize((W, H))
    mask_arr = np.asarray(mask, dtype=np.uint8)
    # Binarize: any non-zero nucleus is foreground
    binary_mask = (mask_arr > 0).astype(np.uint8)
    return np.asarray(image, dtype=np.uint8), binary_mask


def load_monuseg(folder: str | Path) -> dict:
    """Load MoNuSeg dataset from a folder with images/ and masks/ subdirectories.

    Parameters
    ----------
    folder : str or Path
        Root folder containing ``images/`` and ``masks/`` subdirectories.

    Returns
    -------
    dict with keys "images", "masks", "features", "target", "source"
    """
    root = Path(folder)
    image_dir = root / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"MoNuSeg image directory not found: {image_dir}")

    images, masks = [], []
    for img_path in sorted(image_dir.iterdir()):
        result = _load_monuseg_item(img_path)
        if result is not None:
            img, msk = result
            images.append(img)
            masks.append(msk)

    if not images:
        raise FileNotFoundError(
            f"No image/mask pairs found under {root}. "
            "Expected: <root>/images/*.png and <root>/masks/*.png"
        )

    return {
        "images": np.asarray(images, dtype=np.uint8),
        "masks": np.asarray(masks, dtype=np.uint8),
        "features": ["red", "green", "blue", "brightness", "purple_score", "saturation", "x", "y"],
        "target": "tumor_mask",
        "source": str(root),
    }


# ---------------------------------------------------------------------------
# Flexible URL dataset loaders (auto-detects common real-world layouts)
# ---------------------------------------------------------------------------

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _make_result(images, masks, source):
    if not images:
        raise FileNotFoundError(f"No image/mask pairs found under {source}")
    return {
        "images": np.asarray(images, dtype=np.uint8),
        "masks": np.asarray(masks, dtype=np.uint8),
        "features": ["red", "green", "blue", "brightness", "purple_score", "saturation", "x", "y"],
        "target": "foreground_mask",
        "source": str(source),
    }


def _open_pair(img_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(img_path).convert("RGB").resize((W, H), Image.BILINEAR)
    mask = Image.open(mask_path).convert("L").resize((W, H), Image.NEAREST)
    return np.asarray(image, dtype=np.uint8), (np.asarray(mask) > 0).astype(np.uint8)


def _load_image_mask_dirs(img_dir: Path, mask_dir: Path) -> tuple[list, list]:
    """Pair images from img_dir with masks from mask_dir by stem."""
    mask_index = {}
    for p in sorted(mask_dir.glob("*")):
        if p.suffix.lower() in _IMG_EXTS:
            mask_index[p.stem] = p

    images, masks = [], []
    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in _IMG_EXTS:
            continue
        # Try exact stem match, then stem without trailing suffixes
        mp = mask_index.get(img_path.stem) or mask_index.get(img_path.stem.replace("_HE", ""))
        if mp is None:
            continue
        img, msk = _open_pair(img_path, mp)
        images.append(img)
        masks.append(msk)
    return images, masks


def detect_layout(root: Path) -> str:
    """Return the dataset layout string for a given extracted archive root."""
    # Standard images/ + masks/
    if (root / "images").is_dir() and (root / "masks").is_dir():
        return "standard"
    # NuInsSeg: organ subdirs each containing 'tissue images' + 'mask binary'
    ti_dirs = list(root.rglob("tissue images"))
    mb_dirs = list(root.rglob("mask binary"))
    if ti_dirs and mb_dirs:
        return "nuinsseg"
    # CytoDArk0: nested image/ + bwmask/ (or label/)
    img_dirs = [p for p in root.rglob("image") if p.is_dir()]
    bw_dirs = [p for p in root.rglob("bwmask") if p.is_dir()]
    if img_dirs and bw_dirs:
        return "cytodark0"
    # SegPath flat: *_HE.png paired with *_mask.png in same directory
    he_files = [p for p in root.rglob("*_HE.*") if p.suffix.lower() in _IMG_EXTS]
    mk_files = [p for p in root.rglob("*_mask.*") if p.suffix.lower() in _IMG_EXTS]
    if he_files and mk_files:
        return "segpath_flat"
    # One directory deep — try recurse
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            sub_layout = detect_layout(sub)
            if sub_layout != "unknown":
                return sub_layout
    return "unknown"


def load_url_dataset(folder: str | Path, layout: str = "auto", max_tiles: int = 500) -> dict:
    """Load a dataset downloaded from a URL, auto-detecting its structure.

    Parameters
    ----------
    folder : str or Path
        Root of the extracted archive.
    layout : str
        One of 'auto', 'standard', 'nuinsseg', 'cytodark0', 'segpath_flat'.
    max_tiles : int
        Maximum number of tiles to load (avoids OOM on very large datasets).
    """
    root = Path(folder)
    if layout == "auto":
        layout = detect_layout(root)

    images: list = []
    masks: list = []

    if layout == "standard":
        images, masks = _load_image_mask_dirs(root / "images", root / "masks")

    elif layout == "nuinsseg":
        # Each organ dir has 'tissue images/' and 'mask binary/'
        organ_roots = sorted({p.parent for p in root.rglob("tissue images") if p.is_dir()})
        for organ_root in organ_roots:
            ti_dir = organ_root / "tissue images"
            mb_dir = organ_root / "mask binary"
            if ti_dir.is_dir() and mb_dir.is_dir():
                imgs, msks = _load_image_mask_dirs(ti_dir, mb_dir)
                images.extend(imgs)
                masks.extend(msks)
            if len(images) >= max_tiles:
                break

    elif layout == "cytodark0":
        # Pick the smallest tile size (256x256) at the first magnification
        for img_dir in sorted(root.rglob("image")):
            if not img_dir.is_dir():
                continue
            bw_dir = img_dir.parent / "bwmask"
            if not bw_dir.is_dir():
                continue
            imgs, msks = _load_image_mask_dirs(img_dir, bw_dir)
            images.extend(imgs)
            masks.extend(msks)
            if len(images) >= max_tiles:
                break

    elif layout == "segpath_flat":
        # Flat dir: {stem}_HE.{ext} paired with {stem}_mask.{ext}
        mask_index: dict[str, Path] = {}
        for p in root.rglob("*_mask.*"):
            if p.suffix.lower() in _IMG_EXTS:
                stem = p.stem.replace("_mask", "")
                mask_index[stem] = p
        for img_path in sorted(root.rglob("*_HE.*")):
            if img_path.suffix.lower() not in _IMG_EXTS:
                continue
            stem = img_path.stem.replace("_HE", "")
            mp = mask_index.get(stem)
            if mp is None:
                continue
            img, msk = _open_pair(img_path, mp)
            images.append(img)
            masks.append(msk)
            if len(images) >= max_tiles:
                break

    else:
        # Fallback: scan all subdirs for any images/ + masks/ pair
        for sub in sorted(root.rglob("images")):
            if not sub.is_dir():
                continue
            mask_sub = sub.parent / "masks"
            if mask_sub.is_dir():
                imgs, msks = _load_image_mask_dirs(sub, mask_sub)
                images.extend(imgs)
                masks.extend(msks)
                if len(images) >= max_tiles:
                    break
        if not images:
            raise FileNotFoundError(
                f"Could not detect dataset layout under {root}. "
                "Expected: images/+masks/, NuInsSeg organ dirs, CytoDArk0 image/bwmask/, "
                "or flat *_HE.png+*_mask.png structure."
            )

    images = images[:max_tiles]
    masks = masks[:max_tiles]
    return _make_result(images, masks, root)


def _binarize_mask(raw, target_w: int, target_h: int) -> np.ndarray | None:
    """Convert any mask format (PIL, ndarray, list of instances) to a binary HxW uint8 array."""
    if isinstance(raw, Image.Image):
        arr = np.asarray(raw.resize((target_w, target_h), Image.NEAREST))
    elif isinstance(raw, np.ndarray):
        pil = Image.fromarray(raw if raw.ndim == 2 else raw.max(axis=-1).astype(np.uint8))
        arr = np.asarray(pil.resize((target_w, target_h), Image.NEAREST))
    elif isinstance(raw, list):
        union = np.zeros((target_h, target_w), dtype=np.uint8)
        for inst in raw:
            if isinstance(inst, Image.Image):
                m = np.asarray(inst.resize((target_w, target_h), Image.NEAREST))
                union = np.maximum(union, (m > 0).astype(np.uint8))
            elif isinstance(inst, dict):
                m = inst.get("mask") or inst.get("segmentation")
                if isinstance(m, Image.Image):
                    m = np.asarray(m.resize((target_w, target_h), Image.NEAREST))
                if isinstance(m, np.ndarray):
                    union = np.maximum(union, (m > 0).astype(np.uint8))
        return union
    else:
        return None
    if arr.ndim == 3:
        arr = arr.max(axis=-1)
    return (arr > 0).astype(np.uint8)


def load_hf_dataset(
    hf_id: str,
    image_col: str = "image",
    mask_col: str = "instances",
    split: str = "train",
    max_tiles: int = 200,
) -> dict:
    """Stream images and masks from a HuggingFace Hub dataset.

    Streams via HuggingFace's Parquet-over-HTTP infrastructure — no disk cache
    required, so this works on Streamlit Community Cloud.

    Parameters
    ----------
    hf_id : str
        HuggingFace dataset repo ID, e.g. ``"RationAI/MoNuSeg"``.
    image_col : str
        Column name containing PIL Images.
    mask_col : str
        Column name containing segmentation masks.
    split : str
        Dataset split (``"train"``, ``"test"``, ``"fold1"``, …).
    max_tiles : int
        Maximum number of tiles to stream and load.
    """
    try:
        from datasets import load_dataset as hf_load  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Install the 'datasets' package to use HuggingFace Hub: pip install datasets"
        ) from exc

    ds = hf_load(hf_id, streaming=True, split=split, trust_remote_code=False)

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for example in ds:
        if len(images) >= max_tiles:
            break
        raw_img = example.get(image_col)
        raw_msk = example.get(mask_col)
        if raw_img is None or raw_msk is None:
            continue

        if not isinstance(raw_img, Image.Image):
            try:
                raw_img = Image.fromarray(np.asarray(raw_img, dtype=np.uint8))
            except Exception:
                continue
        img = raw_img.convert("RGB").resize((W, H), Image.BILINEAR)

        binary = _binarize_mask(raw_msk, W, H)
        if binary is None:
            continue

        images.append(np.asarray(img, dtype=np.uint8))
        masks.append(binary)

    return _make_result(images, masks, f"hf:{hf_id}/{split}")


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

_SPLIT_NAMES = {"train", "val", "test"}


class PathSegDataset:
    """PyTorch-compatible Dataset for pathology segmentation.

    Wraps both synthetic and real data with configurable transforms,
    and supports train/val/test splitting by index.

    Parameters
    ----------
    data : dict
        Dictionary with keys ``"images"`` (N × H × W × 3, uint8) and
        ``"masks"`` (N × H × W, uint8 {0, 1}).
    transform : callable, optional
        Callable ``(image, mask) -> (tensor_img, tensor_mask)``.
    indices : list of int, optional
        Subset of sample indices to use.  If None, all samples are used.
    """

    def __init__(
        self,
        data: dict,
        transform: Callable | None = None,
        indices: list[int] | None = None,
    ) -> None:
        self.images = np.asarray(data["images"], dtype=np.uint8)
        self.masks = np.asarray(data["masks"], dtype=np.uint8)
        self.source: str = data.get("source", "unknown")
        self._transform = transform

        if indices is not None:
            self.images = self.images[indices]
            self.masks = self.masks[indices]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        import torch  # lazy — only needed when used as a PyTorch Dataset
        image = self.images[idx]
        mask = self.masks[idx]

        if self._transform is not None:
            return self._transform(image, mask)

        img_t = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        msk_t = torch.from_numpy(mask).float().unsqueeze(0)
        return img_t, msk_t


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------


def _get_split_indices(n_total: int, test_size: float, val_size: float, seed: int) -> dict[str, np.ndarray]:
    """Return indices for train / val / test splits."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)
    n_test = max(1, int(n_total * test_size))
    n_val = max(1, int((n_total - n_test) * val_size))
    return {
        "test": idx[:n_test],
        "val": idx[n_test : n_test + n_val],
        "train": idx[n_test + n_val :],
    }


def create_datasets(
    data: dict,
    test_size: float = 0.25,
    val_size: float = 0.15,
    seed: int = 42,
) -> dict[str, PathSegDataset]:
    """Create train/val/test datasets from a data dictionary.

    Parameters
    ----------
    data : dict
        Dictionary with ``"images"`` and ``"masks"`` keys.
    test_size : float
        Fraction of data held out for testing.
    val_size : float
        Fraction of *training* held out for validation (so actual val
        fraction = ``(1 - test_size) * val_size``).
    seed : int
        Random seed for reproducible splits.

    Returns
    -------
    dict[str, PathSegDataset]
        Keys: ``"train"``, ``"val"``, ``"test"``.
    """
    from src.transforms import get_transform

    n_total = len(data["images"])
    splits = _get_split_indices(n_total, test_size, val_size, seed)

    datasets: dict[str, PathSegDataset] = {}
    for split_name in ("train", "val", "test"):
        indices = splits[split_name].tolist()
        transform = get_transform(split_name)
        datasets[split_name] = PathSegDataset(
            data=data,
            transform=transform,
            indices=indices,
        )

    return datasets


def create_dataloaders(
    datasets: dict[str, PathSegDataset],
    batch_size: int = 16,
    val_batch_size: int = 32,
    num_workers: int = 0,
) -> dict:
    """Create DataLoaders from a dictionary of Datasets.

    Parameters
    ----------
    datasets : dict
        Must contain at least ``"train"``.  May also contain ``"val"`` and/or
        ``"test"``.
    batch_size : int
        Batch size for the training loader.
    val_batch_size : int
        Batch size for validation / test loaders.
    num_workers : int
        Number of subprocesses for data loading.  Use 0 on Windows.

    Returns
    -------
    dict[str, DataLoader]
    """
    import torch  # lazy — only needed when DataLoaders are actually used
    loaders: dict = {}
    for split_name, dataset in datasets.items():
        bs = batch_size if split_name == "train" else val_batch_size
        shuffle = split_name == "train"
        loaders[split_name] = torch.utils.data.DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
        )
    return loaders


def prepare_data(
    use_synthetic: bool = True,
    real_data_path: str | None = None,
    n_synthetic: int = 160,
    synthetic_seed: int = 42,
    test_size: float = 0.25,
    val_size: float = 0.15,
    batch_size: int = 16,
    val_batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> dict[str, Any]:
    """One-stop function: load/create data, split, wrap in DataLoaders.

    Parameters
    ----------
    use_synthetic : bool
        If True, generate synthetic tiles.  Otherwise load real data.
    real_data_path : str or None
        Path to real data folder.  Ignored if ``use_synthetic`` is True.
    n_synthetic, synthetic_seed : int
        Parameters for synthetic data generation.
    test_size, val_size : float
        Dataset split fractions.
    batch_size, val_batch_size : int
        Batch sizes.
    num_workers : int
        DataLoader workers.
    seed : int
        Random seed for splitting.

    Returns
    -------
    dict with keys "loaders" (dict of DataLoaders), "datasets" (dict of Datasets),
    "n_classes" (int), and "metadata" (dict).
    """
    if use_synthetic:
        data = make_synthetic(n=n_synthetic, seed=synthetic_seed)
    else:
        if real_data_path is None:
            cfg = get_config()
            real_data_path = cfg["data"]["real_data_path"]
        data = load_real(real_data_path)

    datasets = create_datasets(data, test_size=test_size, val_size=val_size, seed=seed)
    loaders = create_dataloaders(
        datasets, batch_size=batch_size, val_batch_size=val_batch_size, num_workers=num_workers,
    )
    n_pos = int(data["masks"].mean() * data["masks"].size)
    n_total = data["masks"].size

    return {
        "loaders": loaders,
        "datasets": datasets,
        "n_classes": 1,  # binary foreground/background
        "metadata": {
            "source": data.get("source", "unknown"),
            "n_samples": len(data["images"]),
            "tile_size": (H, W),
            "positive_fraction": float(data["masks"].mean()),
            "n_positive_pixels": n_pos,
            "n_total_pixels": n_total,
        },
    }
