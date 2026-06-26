"""PathSeg — Computational Pathology Segmentation Platform.

Interactive dashboard for training, evaluating, and reviewing
digital pathology tile segmentation models (Baseline & U-Net).
Designed for clinical research workflows and regulatory-grade documentation.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATASET_CACHE = ROOT / ".pathseg_datasets"
DATASET_CACHE.mkdir(exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import H, W, detect_layout, load_hf_dataset, load_monuseg, load_real, load_url_dataset, make_synthetic
from src.model import (
    fit_and_evaluate,
    overlay_mask,
    predict_mask,
    predict_mask_unet,
    predict_proba,
    predict_proba_unet,
    unet_model_exists,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PathSeg — Computational Pathology Segmentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Clinical-grade CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --ps-primary: #1e3a5f;
        --ps-accent: #2563eb;
        --ps-success: #059669;
        --ps-warning: #d97706;
        --ps-danger: #dc2626;
        --ps-surface: #f8fafc;
        --ps-border: #e2e8f0;
    }

    .stApp { font-family: 'Inter', sans-serif; }

    /* Hero banner */
    .ps-hero {
        padding: 1.8rem 2rem;
        border-radius: 0.75rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 40%, #1e40af 100%);
        color: white;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 24px rgba(0,0,0,0.12);
    }
    .ps-hero h1 { margin: 0 0 0.3rem 0; font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; }
    .ps-hero p { margin: 0; opacity: 0.88; font-size: 1.05rem; font-weight: 400; }
    .ps-hero .ps-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        background: rgba(255,255,255,0.15);
        margin-right: 0.4rem;
    }

    /* Clinical disclaimer */
    .ps-disclaimer {
        border-left: 4px solid var(--ps-danger);
        background: #fef2f2;
        padding: 0.85rem 1.1rem;
        border-radius: 0.35rem;
        margin: 1rem 0;
        font-size: 0.88rem;
        color: #991b1b;
    }

    /* Info box */
    .ps-info {
        border-left: 4px solid var(--ps-accent);
        background: #eff6ff;
        padding: 0.85rem 1.1rem;
        border-radius: 0.35rem;
        margin: 1rem 0;
        font-size: 0.9rem;
    }

    /* Section header */
    .ps-section {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--ps-primary);
        margin: 1.2rem 0 0.6rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid var(--ps-accent);
    }

    /* Model card box */
    .ps-card {
        border: 1px solid var(--ps-border);
        border-radius: 0.6rem;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .ps-card h3 { margin: 0 0 0.6rem 0; color: var(--ps-primary); }

    /* Metric highlight */
    .ps-metric {
        text-align: center;
        padding: 0.8rem;
        border-radius: 0.5rem;
        background: var(--ps-surface);
        border: 1px solid var(--ps-border);
    }
    .ps-metric .label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; }
    .ps-metric .value { font-size: 1.6rem; font-weight: 700; color: var(--ps-primary); }

    /* Table styling */
    .stDataFrame { border-radius: 0.5rem; overflow: hidden; }

    /* Status indicator */
    .ps-status { display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.85rem; }
    .ps-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
    .ps-dot-green { background: var(--ps-success); }
    .ps-dot-amber { background: var(--ps-warning); }
    .ps-dot-red { background: var(--ps-danger); }
    </style>
    """,
    unsafe_allow_html=True,
)

UNET_AVAILABLE = unet_model_exists()

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_tiles(n_tiles: int, seed: int, tile_size: int) -> dict:
    return make_synthetic(n=n_tiles, seed=seed, size=tile_size)


@st.cache_data(show_spinner=False)
def train_cached(n_tiles: int, seed: int, tile_size: int, threshold: float) -> tuple[dict, dict]:
    data = load_tiles(n_tiles, seed, tile_size)
    return fit_and_evaluate(data, threshold=threshold)


@st.cache_data(show_spinner=False)
def _load_unet_history(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text())


@st.cache_data(show_spinner=False)
def load_real_cached(folder: str) -> dict:
    return load_real(folder)


@st.cache_data(show_spinner=False)
def load_monuseg_cached(folder: str) -> dict:
    return load_monuseg(folder)


@st.cache_data(show_spinner=False)
def train_on_real_cached(folder: str, fmt: str, threshold: float) -> tuple[dict, dict]:
    data = load_monuseg(folder) if fmt == "monuseg" else load_real(folder)
    return fit_and_evaluate(data, threshold=threshold)


# ---------------------------------------------------------------------------
# URL dataset fetching
# ---------------------------------------------------------------------------

def _url_cache_key(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


def _url_cache_dir(url: str) -> Path:
    return DATASET_CACHE / _url_cache_key(url)


def _is_url_cached(url: str) -> bool:
    d = _url_cache_dir(url)
    return d.exists() and any(d.iterdir())


def _find_dataset_root(base: Path) -> Path:
    """Walk extracted archive to find the directory containing images/ + masks/."""
    if (base / "images").exists() and (base / "masks").exists():
        return base
    # One level deep (zip with a single root folder)
    for sub in sorted(base.iterdir()):
        if sub.is_dir():
            if (sub / "images").exists() and (sub / "masks").exists():
                return sub
            # Two levels deep
            for sub2 in sorted(sub.iterdir()):
                if sub2.is_dir() and (sub2 / "images").exists() and (sub2 / "masks").exists():
                    return sub2
    return base


def _infer_fmt_from_url(url: str) -> str:
    lower = url.lower()
    if "monuseg" in lower:
        return "monuseg"
    return "standard"


def download_url_to_cache(url: str, progress_bar=None) -> Path:
    """Download a dataset archive to the local cache directory.

    Returns the cache directory (already populated if previously downloaded).
    Supports ZIP and tar.gz archives. Raises on network or format errors.
    """
    dest = _url_cache_dir(url)
    if dest.exists() and any(dest.iterdir()):
        return dest

    dest.mkdir(parents=True, exist_ok=True)

    try:
        resp = requests.get(url, stream=True, timeout=60,
                            headers={"User-Agent": "PathSeg/1.0"})
        resp.raise_for_status()
    except requests.RequestException as exc:
        dest.rmdir()
        raise RuntimeError(f"Download failed: {exc}") from exc

    total = int(resp.headers.get("content-length", 0))
    buf = io.BytesIO()
    downloaded = 0
    chunk_size = 1 << 17  # 128 KiB

    for chunk in resp.iter_content(chunk_size=chunk_size):
        buf.write(chunk)
        downloaded += len(chunk)
        if progress_bar is not None and total:
            progress_bar.progress(min(downloaded / total, 1.0), text=f"{downloaded >> 20} / {total >> 20} MB")

    buf.seek(0)

    # Detect archive type from Content-Type or URL
    content_type = resp.headers.get("content-type", "")
    fname = urlparse(url).path.rstrip("/").split("/")[-1].lower()

    try:
        if "zip" in content_type or fname.endswith(".zip"):
            with zipfile.ZipFile(buf) as zf:
                zf.extractall(dest)
        elif "tar" in content_type or fname.endswith((".tar.gz", ".tgz", ".tar")):
            buf.seek(0)
            with tarfile.open(fileobj=buf) as tf:
                tf.extractall(dest)
        else:
            # Try ZIP first, then tar
            buf.seek(0)
            try:
                with zipfile.ZipFile(buf) as zf:
                    zf.extractall(dest)
            except zipfile.BadZipFile:
                buf.seek(0)
                try:
                    with tarfile.open(fileobj=buf) as tf:
                        tf.extractall(dest)
                except tarfile.TarError as exc:
                    # Might be a flat image directory — save raw bytes
                    raise RuntimeError(
                        "URL did not return a ZIP or tar archive. "
                        "Please provide a direct link to a .zip or .tar.gz archive "
                        "containing images/ and masks/ subdirectories."
                    ) from exc
    except Exception:
        import shutil
        shutil.rmtree(dest, ignore_errors=True)
        raise

    return dest


@st.cache_data(show_spinner=False)
def load_from_url_cached(cache_key: str, layout: str) -> dict:
    dest = DATASET_CACHE / cache_key
    return load_url_dataset(str(dest), layout=layout)


@st.cache_data(show_spinner=False)
def train_from_url_cached(cache_key: str, layout: str, threshold: float) -> tuple[dict, dict]:
    data = load_from_url_cached(cache_key, layout)
    return fit_and_evaluate(data, threshold=threshold)


@st.cache_data(show_spinner=False)
def load_hf_cached(hf_id: str, image_col: str, mask_col: str, split: str) -> dict:
    return load_hf_dataset(hf_id, image_col=image_col, mask_col=mask_col, split=split)


@st.cache_data(show_spinner=False)
def train_hf_cached(hf_id: str, image_col: str, mask_col: str, split: str, threshold: float) -> tuple[dict, dict]:
    data = load_hf_cached(hf_id, image_col, mask_col, split)
    return fit_and_evaluate(data, threshold=threshold)


def resize_uploaded_image(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file).convert("RGB").resize((W, H))
    return np.asarray(image, dtype=np.uint8)


def mask_area(mask: np.ndarray) -> float:
    return float(np.asarray(mask).mean())


def entropy_from_proba(proba: np.ndarray) -> np.ndarray:
    p = np.clip(proba, 1e-7, 1 - 1e-7)
    return -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)


def confusion_matrix_data(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_t = y_true.astype(bool)
    y_p = y_pred.astype(bool)
    return {
        "TP": int((y_t & y_p).sum()),
        "FP": int((~y_t & y_p).sum()),
        "FN": int((y_t & ~y_p).sum()),
        "TN": int((~y_t & ~y_p).sum()),
    }



# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="ps-hero">
      <h1>PathSeg</h1>
      <p>Computational Pathology Segmentation Platform — Baseline &amp; U-Net for Tissue Tile Analysis</p>
      <div style="margin-top: 0.6rem;">
        <span class="ps-badge">Research Use Only</span>
        <span class="ps-badge">HIPAA-Aligned</span>
        <span class="ps-badge">v1.0</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — Configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Configuration")

    model_choice = st.radio(
        "Model",
        ["Baseline (pixel-LR)", "U-Net"] + (["Compare both"] if UNET_AVAILABLE else []),
        help="Baseline trains on synthetic data on the fly. U-Net requires a pre-trained checkpoint.",
    )

    st.markdown("---")
    st.markdown("#### Data Source")
    data_source = st.radio(
        "Source",
        ["Synthetic (generated)", "Local folder", "Download from URL", "HuggingFace Hub"],
        help="Stream real pathology data from HuggingFace Hub — works on Streamlit Cloud without disk storage.",
    )
    real_data_folder = ""
    real_data_fmt = "standard"
    dataset_url = ""
    url_fmt = "standard"
    hf_id = ""
    hf_image_col = "image"
    hf_mask_col = "instances"
    hf_split = "train"

    if data_source == "Local folder":
        real_data_folder = st.text_input(
            "Folder path",
            placeholder="data/raw/pathseg",
            help="Folder must contain images/ and masks/ subdirectories with matching filenames.",
        )
        real_data_fmt_label = st.selectbox(
            "Format",
            ["Standard (images/ + masks/)", "MoNuSeg (multi-class → binary)"],
            key="local_fmt",
        )
        real_data_fmt = "monuseg" if "MoNuSeg" in real_data_fmt_label else "standard"

    elif data_source == "Download from URL":
        PRESETS = {
            "Custom URL…": ("", "auto"),
            "NuInsSeg — Nuclei H&E, 31 organs (1.6 GB)": (
                "https://zenodo.org/records/10518968/files/NuInsSeg.zip?download=1",
                "nuinsseg",
            ),
            "CytoDArk0 — Brain Nissl-stained nuclei (2.6 GB)": (
                "https://zenodo.org/records/13694738/files/cytoDArk0.zip?download=1",
                "cytodark0",
            ),
            "SegPath Plasma Cells — H&E (24.9 GB)": (
                "https://zenodo.org/records/7412500/files/MIST1_PlasmaCell.tar.gz?download=1",
                "segpath_flat",
            ),
        }
        preset_label = st.selectbox("Dataset", list(PRESETS.keys()), key="url_preset")
        preset_url, preset_layout = PRESETS[preset_label]

        _layout_labels = {
            "auto": "Auto-detect",
            "standard": "Standard (images/ + masks/)",
            "nuinsseg": "NuInsSeg (organ subdirs)",
            "cytodark0": "CytoDArk0 (image/ + bwmask/)",
            "segpath_flat": "SegPath flat (*_HE.png + *_mask.png)",
            "monuseg": "MoNuSeg (multi-class binarized)",
        }
        _layout_map = {v: k for k, v in _layout_labels.items()}

        if preset_label == "Custom URL…":
            dataset_url = st.text_input(
                "Dataset URL",
                placeholder="https://zenodo.org/.../dataset.zip",
                help="Direct link to a ZIP or tar.gz archive. Zenodo and GitHub releases both work.",
            )
            url_layout_label = st.selectbox(
                "Layout",
                list(_layout_labels.values()),
                key="url_layout_custom",
            )
            url_fmt = _layout_map[url_layout_label]
        else:
            dataset_url = preset_url
            url_fmt = preset_layout
            st.code(dataset_url, language="text")
            st.caption(f"Layout: `{_layout_labels[preset_layout]}`")

        if dataset_url and _is_url_cached(dataset_url):
            st.success("Cached locally — instant load.")
        elif dataset_url:
            st.info("Will download on first use.")

    elif data_source == "HuggingFace Hub":
        _HF_PRESETS = {
            "Custom HF dataset…": ("", "image", "instances", "train"),
            "MoNuSeg — 51 nuclear tiles (1000×1000, CC-BY-NC-SA)": (
                "RationAI/MoNuSeg", "image", "instances", "train",
            ),
            "PanNuke — 7.9k nuclear tiles, 19 tissue types (256×256, CC-BY-NC-SA)": (
                "RationAI/PanNuke", "image", "instances", "fold1",
            ),
            "PanOptils — 1.3k tissue tiles (1024×1024, CC0)": (
                "histolytics-hub/panoptils_refined", "image", "sem", "train",
            ),
        }
        hf_preset = st.selectbox("Dataset", list(_HF_PRESETS.keys()), key="hf_preset")
        _hf_pid, _hf_ic, _hf_mc, _hf_sp = _HF_PRESETS[hf_preset]

        if hf_preset == "Custom HF dataset…":
            hf_id = st.text_input(
                "HuggingFace dataset ID",
                placeholder="owner/dataset-name",
                help="Find datasets at huggingface.co/datasets",
            )
            hf_image_col = st.text_input("Image column", value="image", key="hf_imgcol")
            hf_mask_col = st.text_input("Mask column", value="instances", key="hf_mskcol")
            hf_split = st.text_input("Split", value="train", key="hf_split_custom")
        else:
            hf_id = _hf_pid
            hf_image_col = _hf_ic
            hf_mask_col = _hf_mc
            hf_split = _hf_sp
            st.code(hf_id, language="text")
            st.caption(f"Columns: `{hf_image_col}` / `{hf_mask_col}` · Split: `{hf_split}`")

        st.info("Streams tiles live — no disk storage needed. Works on Streamlit Cloud.")

    st.markdown("---")
    n_tiles = st.slider(
        "Synthetic tiles", 60, 260, 160, step=20,
        disabled=(data_source != "Synthetic (generated)"),
        help="Only applies when using synthetic data.",
    )
    seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42, step=1)
    threshold = st.slider(
        "Decision threshold", 0.10, 0.90, 0.50, step=0.05,
        help="Pixels with probability above this threshold are classified as foreground.",
    )

    if "U-Net" in model_choice and not UNET_AVAILABLE:
        st.warning("U-Net checkpoint not found. Run `python train.py --model unet` from terminal.")

    st.markdown("---")
    st.markdown("#### Quickstart")
    st.code("python train.py --model baseline", language="bash")
    st.code("python train.py --model unet", language="bash")
    st.code("streamlit run app.py", language="bash")

    st.markdown("---")
    st.caption(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("Python 3.11+ | PyTorch | NumPy | Streamlit")

use_unet = "U-Net" in model_choice and UNET_AVAILABLE
compare = model_choice == "Compare both"

# ---------------------------------------------------------------------------
# Data loading & model training
# ---------------------------------------------------------------------------

if data_source == "Local folder":
    if not real_data_folder:
        st.warning("Enter a folder path in the sidebar to load real data.")
        st.stop()
    try:
        with st.spinner(f"Loading real data from `{real_data_folder}`..."):
            data = (
                load_monuseg_cached(real_data_folder)
                if real_data_fmt == "monuseg"
                else load_real_cached(real_data_folder)
            )
    except FileNotFoundError as exc:
        st.error(f"Could not load data: {exc}")
        st.info(
            "Ensure the folder contains `images/` and `masks/` subdirectories "
            "with matching filenames (e.g. `tile_001.png` in both)."
        )
        st.stop()

elif data_source == "Download from URL":
    if not dataset_url:
        st.warning("Enter a dataset URL in the sidebar.")
        st.stop()
    url_key = _url_cache_key(dataset_url)
    if not _is_url_cached(dataset_url):
        st.markdown("**Downloading dataset...**")
        prog = st.progress(0, text="Starting download...")
        try:
            download_url_to_cache(dataset_url, progress_bar=prog)
            prog.empty()
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
    try:
        with st.spinner("Loading tiles from cached archive..."):
            data = load_from_url_cached(url_key, url_fmt)  # url_fmt = layout
    except FileNotFoundError as exc:
        st.error(f"Could not load data: {exc}")
        st.info(
            "The archive was downloaded but no `images/` + `masks/` directories were found. "
            "Ensure the ZIP contains matching subdirectory structure."
        )
        st.stop()

elif data_source == "HuggingFace Hub":
    if not hf_id:
        st.warning("Select a dataset or enter a HuggingFace dataset ID in the sidebar.")
        st.stop()
    try:
        with st.spinner(f"Streaming tiles from `{hf_id}` (up to 200 tiles)..."):
            data = load_hf_cached(hf_id, hf_image_col, hf_mask_col, hf_split)
    except ImportError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Failed to load from HuggingFace Hub: {exc}")
        st.info(
            "Check that the dataset ID, column names, and split are correct. "
            "The dataset must be publicly accessible."
        )
        st.stop()

else:
    with st.spinner("Generating synthetic pathology tiles..."):
        data = load_tiles(n_tiles, int(seed), H)

if compare or model_choice == "Baseline (pixel-LR)":
    with st.spinner("Training pixel-level logistic regression baseline..."):
        t0 = time.time()
        if data_source == "Local folder":
            baseline_model, baseline_metrics = train_on_real_cached(
                real_data_folder, real_data_fmt, threshold
            )
        elif data_source == "Download from URL":
            baseline_model, baseline_metrics = train_from_url_cached(
                url_key, url_fmt, threshold  # url_fmt = layout
            )
        elif data_source == "HuggingFace Hub":
            baseline_model, baseline_metrics = train_hf_cached(
                hf_id, hf_image_col, hf_mask_col, hf_split, threshold
            )
        else:
            baseline_model, baseline_metrics = train_cached(n_tiles, int(seed), H, threshold)
        train_time = time.time() - t0
else:
    baseline_model, baseline_metrics = None, None
    train_time = 0.0

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

overview_tab, data_tab, segment_tab, results_tab, models_tab = st.tabs(
    ["Overview", "Data", "Segment", "Results", "Models"]
)

# ============================================================
# TAB 1 — Overview
# ============================================================

with overview_tab:
    st.markdown('<div class="ps-section">About</div>', unsafe_allow_html=True)
    st.markdown(
        """
        PathSeg trains two segmentation models on H&E tissue tiles and lets you compare
        predictions side by side. The baseline uses per-pixel logistic regression on
        hand-crafted color features — fast, interpretable, no GPU needed. The U-Net
        learns spatial context and handles more complex tissue patterns.

        Both models work on synthetic tiles out of the box. Switch the data source in
        the sidebar to load real data from a local folder, a Zenodo URL, or stream
        directly from HuggingFace Hub.
        """
    )

    st.markdown('<div class="ps-section">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        """
        1. Pick a data source and model in the sidebar
        2. The model trains automatically (baseline ~2s on CPU, U-Net needs a pre-trained checkpoint)
        3. Go to **Segment** to look at predictions on individual tiles
        4. Go to **Results** to see aggregate metrics and the confusion matrix
        5. Upload your own image in **Segment** to run inference on any tile
        """
    )

    st.markdown('<div class="ps-section">Stats</div>', unsafe_allow_html=True)
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Models", "Baseline + U-Net")
    col_b.metric("Tile size", f"{W}\u00d7{H} px")
    col_c.metric("Metrics", "Dice, IoU, Sens, Spec")
    col_d.metric("Data sources", "4")

# ============================================================
# TAB 2 — Data
# ============================================================

with data_tab:
    st.markdown('<div class="ps-section">Dataset Overview</div>', unsafe_allow_html=True)

    n_show = min(8, len(data["images"]))
    pos_frac = float(data["masks"].mean())
    neg_frac = 1.0 - pos_frac
    is_real = data_source != "Synthetic (generated)"

    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric("Total Tiles", f"{len(data['images']):,}")
    dc2.metric("Tile Resolution", f"{W}\u00d7{H}")
    dc3.metric("Foreground Rate", f"{pos_frac:.1%}")
    dc4.metric("Background Rate", f"{neg_frac:.1%}")

    source_label = data.get("source", "Unknown")
    if data_source == "Download from URL":
        st.markdown(
            f"**{len(data['images'])}** tiles fetched from URL and loaded from local cache "
            f"(showing **{n_show}** samples). All tiles resized to {W}\u00d7{H} px."
        )
    elif data_source == "Local folder":
        st.markdown(
            f"**{len(data['images'])}** tiles loaded from `{source_label}` "
            f"(showing **{n_show}** samples). "
            f"Masks are binary (foreground = 1). All tiles resized to {W}\u00d7{H} px."
        )
    elif data_source == "HuggingFace Hub":
        st.markdown(
            f"**{len(data['images'])}** tiles streamed from `{hf_id}` (split: `{hf_split}`) "
            f"(showing **{n_show}** samples). All tiles resized to {W}\u00d7{H} px."
        )
    else:
        st.markdown(
            f"**{len(data['images'])}** synthetic tiles (showing **{n_show}** samples). "
            f"Foreground pixels represent simulated tissue/tumor regions."
        )

    sample_cols = st.columns(n_show)
    for i in range(n_show):
        sample_cols[i].image(data["images"][i], caption=f"Tile {i}", width="content")
        sample_cols[i].image(data["masks"][i] * 255, caption=f"Mask {i}", width="content")

    st.markdown('<div class="ps-section">Pixel Intensity Distribution</div>', unsafe_allow_html=True)
    r_ch = data["images"][:, :, :, 0].ravel()
    g_ch = data["images"][:, :, :, 1].ravel()
    b_ch = data["images"][:, :, :, 2].ravel()
    intensity_df = pd.DataFrame({"Red": r_ch, "Green": g_ch, "Blue": b_ch})
    st.bar_chart(intensity_df.sample(min(8000, len(intensity_df)), random_state=42), height=220)
    st.caption("RGB channel intensity distributions across all tiles.")

    st.markdown('<div class="ps-section">Dataset Provenance</div>', unsafe_allow_html=True)
    if data_source == "Download from URL":
        detected = detect_layout(DATASET_CACHE / _url_cache_key(dataset_url)) if dataset_url else "\u2014"
        prov_rows = [
            {"Field": "Source URL", "Value": dataset_url},
            {"Field": "Detected layout", "Value": detected},
            {"Field": "Layout override", "Value": url_fmt},
            {"Field": "Tiles loaded", "Value": f"{len(data['images'])}"},
            {"Field": "Tile size (internal)", "Value": f"{W}\u00d7{H} px"},
            {"Field": "Mask type", "Value": "Binary (0 = background, 1 = foreground)"},
            {"Field": "Cache directory", "Value": str(_url_cache_dir(dataset_url))},
        ]
    elif data_source == "Local folder":
        prov_rows = [
            {"Field": "Source", "Value": source_label},
            {"Field": "Format", "Value": "MoNuSeg (binarized)" if real_data_fmt == "monuseg" else "Standard (images/ + masks/)"},
            {"Field": "Tiles loaded", "Value": f"{len(data['images'])}"},
            {"Field": "Tile size (internal)", "Value": f"{W}\u00d7{H} px"},
            {"Field": "Mask type", "Value": "Binary (0 = background, 1 = foreground)"},
            {"Field": "Pixel range", "Value": "[0, 255] uint8"},
        ]
    elif data_source == "HuggingFace Hub":
        prov_rows = [
            {"Field": "HuggingFace dataset", "Value": hf_id},
            {"Field": "Split", "Value": hf_split},
            {"Field": "Image column", "Value": hf_image_col},
            {"Field": "Mask column", "Value": hf_mask_col},
            {"Field": "Tiles streamed", "Value": f"{len(data['images'])} (max 200)"},
            {"Field": "Tile size (internal)", "Value": f"{W}\u00d7{H} px"},
            {"Field": "Mask type", "Value": "Binary (binarized from source mask)"},
            {"Field": "Streaming", "Value": "Live via HuggingFace Parquet-over-HTTP"},
        ]
    else:
        prov_rows = [
            {"Field": "Source", "Value": "Synthetic generator (src.data.make_synthetic)"},
            {"Field": "Tile size", "Value": f"{W}\u00d7{H} px"},
            {"Field": "Tiles generated", "Value": f"{n_tiles}"},
            {"Field": "Random seed", "Value": str(seed)},
            {"Field": "Mask type", "Value": "Binary tumor blobs on simulated H&E background"},
            {"Field": "Pixel range", "Value": "[0, 255] uint8"},
        ]
    st.dataframe(pd.DataFrame(prov_rows), use_container_width=True, hide_index=True)

    if not is_real:
        st.markdown('<div class="ps-section">Loading Real Datasets</div>', unsafe_allow_html=True)
        st.markdown(
            """
            Switch to **Local folder**, **Download from URL**, or **HuggingFace Hub** in the sidebar to load real pathology data.

            **HuggingFace Hub** (recommended for Streamlit Cloud) streams tiles live — no disk storage required.
            Select from the presets or enter any public HF dataset ID with `image` and mask columns.

            | HF Dataset | Tiles | Resolution | Mask column | License |
            |---|---|---|---|---|
            | `RationAI/MoNuSeg` | 51 | 1000×1000 | `instances` | CC-BY-NC-SA |
            | `RationAI/PanNuke` | 7,901 | 256×256 | `instances` | CC-BY-NC-SA |
            | `histolytics-hub/panoptils_refined` | 1,349 | 1024×1024 | `sem` | CC0 |

            **Download from URL** fetches a ZIP or tar.gz archive directly from Zenodo or any public URL.
            Downloads are cached locally — subsequent runs load instantly without re-downloading.

            | Zenodo Dataset | Organ / Task | Size | Zenodo DOI |
            |---|---|---|---|
            | **NuInsSeg** | 31 organs, nuclei | 1.6 GB | [10518968](https://zenodo.org/records/10518968) |
            | **CytoDArk0** | Brain Nissl-stained nuclei | 2.6 GB | [13694738](https://zenodo.org/records/13694738) |
            | **SegPath Plasma Cells** | H&E plasma cell | 24.9 GB | [7412500](https://zenodo.org/records/7412500) |
            """
        )
    else:
        dq1, dq2, dq3 = st.columns(3)
        dq1.metric("Positive Fraction", f"{pos_frac:.2%}")
        dq2.metric("Pixel Range", "[0, 255]")
        dq3.metric("Binary Masks", "Yes (0/1)")

# ============================================================
# TAB 3 — Segment
# ============================================================

with segment_tab:
    st.markdown('<div class="ps-section">Inference & Review</div>', unsafe_allow_html=True)

    review_mode = st.radio("Data Source", ["Synthetic tile", "Upload image"], horizontal=True)

    if review_mode == "Synthetic tile":
        tile_index = st.slider("Tile index", 0, len(data["images"]) - 1, 0)
        image = data["images"][tile_index]
        true_mask = data["masks"][tile_index]
        show_truth = True
    else:
        uploaded_file = st.file_uploader(
            "Upload a pathology-like image for inference",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            help="Upload any pathology-like image. It will be resized to 128x128 for inference.",
        )
        if uploaded_file is None:
            st.info(
                "Upload an image file to run inference, or switch to **Synthetic tile** mode "
                "to review pre-generated tiles with ground-truth masks."
            )
            st.stop()
        image = resize_uploaded_image(uploaded_file)
        true_mask = None
        show_truth = False

    # --- Run inference ---
    if compare or model_choice == "Baseline (pixel-LR)":
        bl_proba = predict_proba(baseline_model, image)
        bl_mask = (bl_proba >= threshold).astype(np.uint8)
        bl_overlay = overlay_mask(image, bl_mask)
    else:
        bl_mask = bl_proba = bl_overlay = None

    if use_unet or compare:
        unet_proba = predict_proba_unet(image)
        unet_mask = (unet_proba >= threshold).astype(np.uint8)
        unet_overlay = overlay_mask(image, unet_mask, color=(30, 200, 80), alpha=0.45)
    else:
        unet_mask = unet_proba = unet_overlay = None

    # --- Visual comparison ---
    if compare:
        c_inp, c_gt, c_bl_mask, c_bl_ov, c_un_mask, c_un_ov = st.columns(6)
    else:
        c_inp, c_gt, c_mask, c_ov = st.columns(4)

    c_inp.image(image, caption="Input Tile", width="content")
    if show_truth:
        c_gt.image(true_mask * 255, caption="Ground Truth", width="content")
    else:
        c_gt.caption("No ground truth available")

    if compare:
        if bl_mask is not None:
            c_bl_mask.image(bl_mask * 255, caption="Baseline Mask", width="content")
            c_bl_ov.image(bl_overlay, caption="Baseline Overlay", width="content")
        if unet_mask is not None:
            c_un_mask.image(unet_mask * 255, caption="U-Net Mask", width="content")
            c_un_ov.image(unet_overlay, caption="U-Net Overlay", width="content")
    else:
        if bl_mask is not None:
            c_mask.image(bl_mask * 255, caption="Predicted Mask", width="content")
            c_ov.image(bl_overlay, caption="Overlay", width="content")
        elif unet_mask is not None:
            c_mask.image(unet_mask * 255, caption="Predicted Mask", width="content")
            c_ov.image(unet_overlay, caption="Overlay", width="content")

    # --- Metrics row ---
    st.markdown('<div class="ps-section">Tile-Level Metrics</div>', unsafe_allow_html=True)

    metric_items = []
    if show_truth:
        metric_items.append(("Ground Truth Area", f"{mask_area(true_mask):.1%}"))
    else:
        metric_items.append(("Tile Resolution", f"{W}\u00d7{H}"))
    if bl_mask is not None:
        metric_items.append(("Baseline Area", f"{mask_area(bl_mask):.1%}"))
        metric_items.append(("Baseline Mean Prob.", f"{float(bl_proba.mean()):.3f}"))
    if unet_mask is not None:
        metric_items.append(("U-Net Area", f"{mask_area(unet_mask):.1%}"))
        metric_items.append(("U-Net Mean Prob.", f"{float(unet_proba.mean()):.3f}"))
    mcols = st.columns(len(metric_items))
    for col, (label, value) in zip(mcols, metric_items):
        col.metric(label, value)

    # --- Probability comparison ---
    if compare and bl_proba is not None:
        with st.expander("Per-Pixel Probability Comparison (Baseline vs U-Net)", expanded=False):
            scatter_df = pd.DataFrame({
                "Baseline Probability": bl_proba.ravel(),
                "U-Net Probability": unet_proba.ravel(),
            })
            st.scatter_chart(scatter_df.sample(min(2000, len(scatter_df))), height=320)

    # --- Uncertainty map ---
    proba_for_entropy = unet_proba if unet_proba is not None else bl_proba
    if proba_for_entropy is not None:
        with st.expander("Uncertainty Map (Pixel Entropy)", expanded=False):
            st.markdown(
                "Pixel-wise entropy from predicted probabilities. "
                "Brighter = higher uncertainty."
            )
            entropy_map = entropy_from_proba(proba_for_entropy)
            max_e = entropy_map.max()
            if max_e > 0:
                entropy_display = (entropy_map / max_e * 255).astype(np.uint8)
            else:
                entropy_display = np.zeros_like(entropy_map, dtype=np.uint8)
            st.image(
                entropy_display,
                caption="Uncertainty heatmap — brighter = higher uncertainty",
                width="content",
                clamp=True,
            )

    # --- Ground truth overlay ---
    if show_truth:
        st.markdown('<div class="ps-section">Ground Truth Overlay</div>', unsafe_allow_html=True)
        truth_overlay = overlay_mask(image, true_mask, color=(30, 200, 80), alpha=0.45)
        st.image(truth_overlay, caption="Green overlay = ground-truth annotation", width="content")

    # --- Pixel-level stats for this tile ---
    if show_truth and bl_mask is not None:
        st.markdown('<div class="ps-section">Tile Pixel Statistics</div>', unsafe_allow_html=True)
        tile_metrics = confusion_matrix_data(true_mask, bl_mask)
        tp, fp, fn, tn = tile_metrics["TP"], tile_metrics["FP"], tile_metrics["FN"], tile_metrics["TN"]
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        ts1, ts2, ts3, ts4, ts5, ts6 = st.columns(6)
        ts1.metric("Sensitivity", f"{sensitivity:.3f}")
        ts2.metric("Specificity", f"{specificity:.3f}")
        ts3.metric("Dice", f"{dice:.3f}")
        ts4.metric("TP", f"{tp:,}")
        ts5.metric("FP", f"{fp:,}")
        ts6.metric("FN", f"{fn:,}")

# ============================================================
# TAB 4 — Results
# ============================================================

with results_tab:
    st.markdown('<div class="ps-section">Evaluation Results</div>', unsafe_allow_html=True)

    if baseline_metrics:
        dm = baseline_metrics

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Dice Coefficient", f"{dm['dice']:.4f}")
        r2.metric("IoU (Jaccard)", f"{dm['iou']:.4f}")
        r3.metric("Sensitivity", f"{dm['sensitivity']:.4f}")
        r4.metric("Specificity", f"{dm['specificity']:.4f}")

        r5, r6, r7, r8 = st.columns(4)
        r5.metric("Precision", f"{dm.get('precision', 0):.4f}")
        r6.metric("Threshold", f"{dm['threshold']:.2f}")
        r7.metric("Train Tiles", f"{dm['n_train_tiles']:,}")
        r8.metric("Test Tiles", f"{dm['n_test_tiles']:,}")

        # --- Confusion matrix ---
        st.markdown('<div class="ps-section">Pixel-Level Confusion Matrix</div>', unsafe_allow_html=True)
        cm = {"TP": dm["tp_pixels"], "FP": dm["fp_pixels"],
              "FN": dm["fn_pixels"], "TN": dm["tn_pixels"]}
        cm_df = pd.DataFrame([
            {"": "Predicted Positive", "Actual Positive": f"{cm['TP']:,}", "Actual Negative": f"{cm['FP']:,}"},
            {"": "Predicted Negative", "Actual Positive": f"{cm['FN']:,}", "Actual Negative": f"{cm['TN']:,}"},
        ])
        st.dataframe(cm_df, width="stretch", hide_index=True)

        total = cm["TP"] + cm["FP"] + cm["FN"] + cm["TN"]
        st.caption(
            f"Total pixels evaluated: {total:,} | "
            f"Accuracy: {(cm['TP'] + cm['TN']) / total:.4f} | "
            f"F1: {2 * dm.get('precision', 0) * dm['sensitivity'] / (dm.get('precision', 0) + dm['sensitivity'] + 1e-8):.4f}"
        )

        # --- Training summary table ---
        st.markdown('<div class="ps-section">Training Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            | Parameter | Value |
            |---|---|
            | Backend | {dm.get('backend', 'pixel-logreg')} |
            | Training tiles | {dm['n_train_tiles']:,} |
            | Test tiles | {dm['n_test_tiles']:,} |
            | Sampled pixels | {dm['sampled_train_pixels']:,} |
            | Positive pixel rate | {dm['positive_pixel_rate']:.2%} |
            | Decision threshold | {dm['threshold']:.2f} |
            | Precision | {dm.get('precision', 0):.4f} |
            | Dice coefficient | {dm['dice']:.4f} |
            | IoU (Jaccard index) | {dm['iou']:.4f} |
            | Sensitivity (recall) | {dm['sensitivity']:.4f} |
            | Specificity | {dm['specificity']:.4f} |
            """
        )
    else:
        st.info(
            "No baseline metrics available. Select **Baseline (pixel-LR)** or **Compare both** "
            "in the sidebar to train and evaluate a model."
        )

    # --- U-Net history ---
    st.markdown('<div class="ps-section">U-Net Training History</div>', unsafe_allow_html=True)

    if UNET_AVAILABLE:
        history_path = Path(ROOT) / "models" / "unet_history.json"
        history = _load_unet_history(str(history_path))
        if history:
            history_df = pd.DataFrame(history)
            best = max(history, key=lambda e: e["dice"])

            st.markdown(
                f"**Best validation epoch:** {best['epoch']} | "
                f"Dice = {best['dice']:.4f} | IoU = {best['iou']:.4f} | "
                f"val_loss = {best['val_loss']:.4f}"
            )

            chart_tabs = st.tabs(["Dice / IoU", "Loss", "Learning Rate", "Table"])
            with chart_tabs[0]:
                st.line_chart(history_df.set_index("epoch")[["dice", "iou"]], height=300)
            with chart_tabs[1]:
                st.line_chart(history_df.set_index("epoch")[["train_loss", "val_loss"]], height=300)
            with chart_tabs[2]:
                st.line_chart(history_df.set_index("epoch")[["lr"]], height=250)
            with chart_tabs[3]:
                st.dataframe(history_df, width="stretch", hide_index=True)
        else:
            st.markdown(
                "U-Net checkpoint exists but no training history found at `models/unet_history.json`. "
                "Re-run training to generate history."
            )
    else:
        st.markdown(
            "**U-Net not yet trained.** Execute `python train.py --model unet` from the terminal "
            "to train the convolutional segmentation model."
        )

# ============================================================
# TAB 5 — Models (technical)
# ============================================================

FEATURE_NAMES = ["Red", "Green", "Blue", "Brightness", "Purple score", "Saturation", "x (pos)", "y (pos)"]

with models_tab:

    # ── Logistic Regression ──────────────────────────────────
    st.markdown('<div class="ps-section">Logistic Regression</div>', unsafe_allow_html=True)

    st.latex(
        r"P(\text{foreground}\mid x) = \sigma(w^\top x + b)"
        r"= \frac{1}{1 + e^{-(w_0 r + w_1 g + w_2 b + w_3 \text{br} + w_4 \text{pu} + w_5 \text{sat} + w_6 x_c + w_7 y_c + b)}}"
    )

    feat_table = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Formula": [
            "pixel_R / 255",
            "pixel_G / 255",
            "pixel_B / 255",
            "(R + G + B) / 3",
            "B + R − 1.35·G",
            "max(R,G,B) − min(R,G,B)",
            "col / (W − 1)",
            "row / (H − 1)",
        ],
        "Intuition": [
            "H&E eosin staining",
            "Background hue",
            "H&E hematoxylin staining",
            "Overall brightness",
            "Purple/violet nuclear stain",
            "Color saturation",
            "Horizontal position",
            "Vertical position",
        ],
    })
    st.dataframe(feat_table, hide_index=True, use_container_width=True)

    st.markdown("**Gradient descent update rule** (class-balanced weights, L2 regularization):")
    st.latex(
        r"w \;\leftarrow\; w - \eta \!\left(\frac{X^\top [(p - y) \odot s]}{n} + \lambda w\right)"
        r",\qquad b \;\leftarrow\; b - \eta\,\overline{(p - y) \odot s}"
    )
    st.markdown(
        "where **s** is the per-sample class weight (`n / 2·n_pos` for positives, `n / 2·n_neg` for negatives)."
    )

    if baseline_model:
        clf = baseline_model["clf"]
        w = clf.w_
        b_val = clf.b_

        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("Bias b", f"{b_val:.4f}")
        lc2.metric("Max weight", f"{w.max():.4f}  ({FEATURE_NAMES[int(w.argmax())]})")
        lc3.metric("Min weight", f"{w.min():.4f}  ({FEATURE_NAMES[int(w.argmin())]})")

        st.markdown("**Learned weights** — positive = pushes toward foreground, negative = toward background:")
        weight_df = pd.DataFrame({"Weight": w}, index=FEATURE_NAMES).sort_values("Weight")
        st.bar_chart(weight_df, height=260)

        st.markdown(
            f"Training config:  lr η = `{clf.lr}` | epochs = `{clf.epochs}` | "
            f"L2 λ = `{clf.l2}` | pixels sampled = `25 000` (50 % pos / 50 % neg)"
        )
    else:
        st.info("Train the baseline model (select Baseline or Compare in the sidebar) to see learned weights.")

    # ── Threshold & decision boundary ────────────────────────
    st.markdown('<div class="ps-section">Threshold & Decision Boundary</div>', unsafe_allow_html=True)

    st.markdown(
        "The raw output is a probability per pixel. "
        r"A pixel is classified foreground if $p \geq \tau$, background if $p < \tau$."
    )

    if baseline_model:
        # Collect probabilities + true labels over first N tiles
        _n_sweep = min(12, len(data["images"]))
        _all_p, _all_y = [], []
        for _img, _msk in zip(data["images"][:_n_sweep], data["masks"][:_n_sweep]):
            _all_p.append(predict_proba(baseline_model, _img).ravel())
            _all_y.append(_msk.ravel())
        _all_p = np.concatenate(_all_p)
        _all_y = np.concatenate(_all_y).astype(bool)

        # Histogram: foreground vs background probabilities
        _bins = np.linspace(0, 1, 41)
        _bcenters = (_bins[:-1] + _bins[1:]) / 2
        _h_pos, _ = np.histogram(_all_p[_all_y], bins=_bins)
        _h_neg, _ = np.histogram(_all_p[~_all_y], bins=_bins)
        hist_df = pd.DataFrame(
            {"Foreground (y=1)": _h_pos, "Background (y=0)": _h_neg},
            index=np.round(_bcenters, 3),
        )
        st.markdown(f"Pixel probability distribution (first {_n_sweep} tiles). Threshold = **{threshold:.2f}**")
        st.bar_chart(hist_df, height=250)
        st.caption("Good separation → two distinct peaks. Overlap region = uncertain pixels near boundary.")

        # Threshold sweep
        _ts = np.linspace(0.05, 0.95, 46)
        _dice_s, _sens_s, _spec_s = [], [], []
        for _t in _ts:
            _pred = _all_p >= _t
            _tp = int((_all_y & _pred).sum())
            _fp = int((~_all_y & _pred).sum())
            _fn = int((_all_y & ~_pred).sum())
            _tn = int((~_all_y & ~_pred).sum())
            _dice_s.append(2 * _tp / (2 * _tp + _fp + _fn) if 2 * _tp + _fp + _fn else 0.0)
            _sens_s.append(_tp / (_tp + _fn) if _tp + _fn else 0.0)
            _spec_s.append(_tn / (_tn + _fp) if _tn + _fp else 0.0)

        sweep_df = pd.DataFrame(
            {"Dice": _dice_s, "Sensitivity": _sens_s, "Specificity": _spec_s},
            index=np.round(_ts, 3),
        )
        sweep_df.index.name = "threshold"
        _best_t = _ts[int(np.argmax(_dice_s))]
        st.markdown("Dice / Sensitivity / Specificity vs threshold:")
        st.line_chart(sweep_df, height=270)
        st.caption(
            f"Threshold that maximises Dice on these tiles: **{_best_t:.2f}** "
            f"(current sidebar value: {threshold:.2f})"
        )
    else:
        st.info("Train the baseline model to see the threshold analysis.")

    # ── U-Net architecture ────────────────────────────────────
    st.markdown('<div class="ps-section">U-Net Architecture</div>', unsafe_allow_html=True)

    unet_rows = [
        {"Stage": "Input",       "Out channels": 3,    "Spatial": f"{H}×{W}",         "Block": "RGB tile"},
        {"Stage": "Encoder 1",   "Out channels": 64,   "Spatial": f"{H}×{W}",         "Block": "DoubleConv (Conv2d 3×3 → BN → ReLU) ×2"},
        {"Stage": "Down 1",      "Out channels": 64,   "Spatial": f"{H//2}×{W//2}",   "Block": "MaxPool2d(2)"},
        {"Stage": "Encoder 2",   "Out channels": 128,  "Spatial": f"{H//2}×{W//2}",   "Block": "DoubleConv ×2"},
        {"Stage": "Down 2",      "Out channels": 128,  "Spatial": f"{H//4}×{W//4}",   "Block": "MaxPool2d(2)"},
        {"Stage": "Encoder 3",   "Out channels": 256,  "Spatial": f"{H//4}×{W//4}",   "Block": "DoubleConv ×2"},
        {"Stage": "Down 3",      "Out channels": 256,  "Spatial": f"{H//8}×{W//8}",   "Block": "MaxPool2d(2)"},
        {"Stage": "Encoder 4",   "Out channels": 512,  "Spatial": f"{H//8}×{W//8}",   "Block": "DoubleConv ×2"},
        {"Stage": "Down 4",      "Out channels": 512,  "Spatial": f"{H//16}×{W//16}", "Block": "MaxPool2d(2)"},
        {"Stage": "Bottleneck",  "Out channels": 1024, "Spatial": f"{H//16}×{W//16}", "Block": "DoubleConv ×2"},
        {"Stage": "Up 4 + skip", "Out channels": 512,  "Spatial": f"{H//8}×{W//8}",   "Block": "ConvTranspose2d(2×2) → cat(skip) → DoubleConv"},
        {"Stage": "Up 3 + skip", "Out channels": 256,  "Spatial": f"{H//4}×{W//4}",   "Block": "ConvTranspose2d(2×2) → cat(skip) → DoubleConv"},
        {"Stage": "Up 2 + skip", "Out channels": 128,  "Spatial": f"{H//2}×{W//2}",   "Block": "ConvTranspose2d(2×2) → cat(skip) → DoubleConv"},
        {"Stage": "Up 1 + skip", "Out channels": 64,   "Spatial": f"{H}×{W}",         "Block": "ConvTranspose2d(2×2) → cat(skip) → DoubleConv"},
        {"Stage": "Output",      "Out channels": 1,    "Spatial": f"{H}×{W}",         "Block": "Conv2d(1×1) → sigmoid → binary mask"},
    ]
    st.dataframe(pd.DataFrame(unet_rows), hide_index=True, use_container_width=True)
    st.caption("Skip connections (cat) concatenate encoder feature maps with decoder input, preserving spatial detail.")

    st.markdown("**Combined loss function:**")
    st.latex(r"\mathcal{L} = \tfrac{1}{2}\,\mathcal{L}_{\mathrm{BCE}} + \tfrac{1}{2}\,\mathcal{L}_{\mathrm{Dice}}")
    st.latex(
        r"\mathcal{L}_{\mathrm{BCE}} = -\frac{1}{N}\sum_{i=1}^{N}"
        r"\bigl[y_i \log p_i + (1-y_i)\log(1-p_i)\bigr]"
    )
    st.latex(
        r"\mathcal{L}_{\mathrm{Dice}} = 1 - \frac{2\sum_i y_i\,p_i + \varepsilon}"
        r"{\sum_i y_i + \sum_i p_i + \varepsilon}, \quad \varepsilon = 1"
    )
    st.markdown(
        "BCE penalises each pixel independently. "
        "Dice penalises the overlap directly — handles class imbalance better when foreground is rare."
    )

    if UNET_AVAILABLE:
        history_path = Path(ROOT) / "models" / "unet_history.json"
        history = _load_unet_history(str(history_path))
        if history:
            history_df = pd.DataFrame(history)
            best_ep = max(history, key=lambda e: e["dice"])
            st.markdown(
                f"**Best checkpoint:** epoch {best_ep['epoch']} — "
                f"Dice = `{best_ep['dice']:.4f}` | IoU = `{best_ep['iou']:.4f}` | "
                f"val_loss = `{best_ep['val_loss']:.4f}`"
            )
            ht1, ht2, ht3 = st.tabs(["Dice / IoU", "Loss", "Learning rate"])
            with ht1:
                st.line_chart(history_df.set_index("epoch")[["dice", "iou"]], height=280)
            with ht2:
                st.line_chart(history_df.set_index("epoch")[["train_loss", "val_loss"]], height=280)
            with ht3:
                st.line_chart(history_df.set_index("epoch")[["lr"]], height=230)

    # ── LR vs U-Net comparison ────────────────────────────────
    st.markdown('<div class="ps-section">LR vs U-Net</div>', unsafe_allow_html=True)

    cmp_df = pd.DataFrame([
        {"":                    "Decision unit",    "Logistic Regression": "Single pixel",                    "U-Net": "Full image (receptive field = full tile)"},
        {"":                    "Input features",   "Logistic Regression": "8 hand-crafted (color + position)", "U-Net": "Raw 3-channel RGB"},
        {"":                    "Prediction",       "Logistic Regression": "σ(w·x + b) independently per px", "U-Net": "sigmoid(conv stack) — all pixels jointly"},
        {"":                    "Spatial context",  "Logistic Regression": "None",                             "U-Net": "128×128 px receptive field via pooling"},
        {"":                    "Parameters",       "Logistic Regression": "9  (8 weights + bias)",            "U-Net": "~31 M"},
        {"":                    "Training time",    "Logistic Regression": "~2 s on CPU",                      "U-Net": "~5 min on GPU"},
        {"":                    "Loss",             "Logistic Regression": "Weighted cross-entropy",           "U-Net": "BCE + Dice (50/50)"},
        {"":                    "When it wins",     "Logistic Regression": "Strong color contrast btw classes","U-Net": "Complex morphology, texture, shape"},
    ]).set_index("")
    st.dataframe(cmp_df, use_container_width=True)
