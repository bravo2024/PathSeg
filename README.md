# PathSeg

> Computational Pathology Segmentation Platform — Tissue Tile Analysis with Pixel-Level Baseline & U-Net

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

PathSeg is a research-grade computational pathology platform for digital pathology tile segmentation. It implements two complementary approaches — a fast pixel-level logistic regression baseline and a deep convolutional U-Net — within an interactive dashboard designed for clinical research workflows.

**Intended Use:** Research and algorithm development for digital pathology. Not for clinical diagnosis.

## Quickstart

```bash
git clone <repo-url>
cd PathSeg
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py                    # train baseline (~2s on CPU)
python train.py --model unet       # train U-Net (GPU recommended)
streamlit run app.py               # launch dashboard
```

## Usage

### Train a model

```bash
# Baseline (default) — trains in ~2 seconds, no GPU needed
python train.py

# U-Net — requires PyTorch, GPU recommended for training
python train.py --model unet

# With options
python train.py --model baseline --n 200 --seed 7 --threshold 0.4
python train.py --model unet --epochs 50 --lr 0.0005 --device cuda

# Use real data (expected: data/raw/pathseg/{images,masks}/*.png)
python train.py --model unet --real data/raw/pathseg

# Use custom YAML config
python train.py --model unet --config config/custom.yaml
```

### Launch dashboard

```bash
streamlit run app.py
```

The dashboard provides six tabs:

| Tab | Description |
|---|---|
| **Overview** | Clinical context, intended use, workflow, regulatory disclaimer |
| **Data** | Dataset overview, class balance, intensity distributions, data provenance |
| **Segment** | Interactive inference, overlay visualization, uncertainty maps, clinical interpretation |
| **Results** | Evaluation metrics, confusion matrix, U-Net training history, clinical interpretation |
| **Model Card** | Architecture details, training config, strengths/limitations (Google Model Card format) |
| **Deploy** | Streamlit Cloud, Docker, REST API, CI/CD pipeline templates |

## Data

**Synthetic tiles** (default): generated on-the-fly by `src.data.make_synthetic()`. No download needed. Simulates H&E staining, tissue regions, tumor-like blobs, and nuclei.

**Real data**: place image-mask pairs in `data/raw/pathseg/`:
```
data/raw/pathseg/
  images/tile_001.png
  masks/tile_001.png
```

**MoNuSeg format**: supported via `src.data.load_monuseg()`.

## Project Structure

```
PathSeg/
  app.py                  Streamlit dashboard (6 tabs, clinical-grade UI)
  train.py                CLI entrypoint for training
  requirements.txt
  runtime.txt
  config/
    base.yaml             Hyperparameters and paths
  src/
    data.py               Data loaders, Dataset, DataLoaders
    model.py              Baseline model + U-Net inference wrapper
    unet.py               U-Net architecture + loss functions
    train_unet.py         Training loop with early stopping
    core.py               NumPy ML primitives (LogReg, Standardizer, metrics)
    evaluate.py           Metrics persistence and reporting
    persist.py            Model save/load via pickle
    transforms.py         Augmentation pipelines (spatial + color)
    logger.py             Structured logging
    config.py             YAML + environment variable config loader
  tests/
    test_smoke.py         Quick smoke tests
    test_data.py          Data pipeline tests
    test_unet.py          U-Net architecture tests
    test_integration.py   End-to-end integration tests
    test_coverage.py      Config, logger, persist coverage tests
  .streamlit/
    config.toml           Streamlit theme and server settings
  Makefile
```

## Tests

```bash
pytest -v          # run all 85 tests
pytest -q          # quick mode
```

## Deploy

### Streamlit Community Cloud
1. Push this folder to GitHub.
2. Create a new app at [share.streamlit.io](https://share.streamlit.io).
3. Set `app.py` as the entrypoint.

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python train.py
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
```

### REST API
Wrap `predict_proba` / `predict_mask` in a FastAPI endpoint for LIS/PACS integration.

## Model Performance

| Metric | Baseline | U-Net |
|---|---|---|
| Dice | ~0.98 | ~0.95+ (on real data) |
| IoU | ~0.96 | ~0.92+ |
| Training time | ~2s (CPU) | ~5min (GPU) |
| Parameters | 9 weights | ~31M |

## Regulatory Disclaimer

This is a research and portfolio demonstration. It is **not** a diagnostic medical device. Model outputs require review by a qualified pathologist or clinician before any clinical use. Not cleared by the FDA or any regulatory body.

## License

MIT
