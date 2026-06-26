# PathSeg

A Streamlit app for segmenting tissue tiles from digital pathology images. Trains two models — a simple pixel-level logistic regression baseline and a U-Net — then lets you compare predictions interactively.

Built mostly to explore how far a hand-rolled baseline can get before needing deep learning on H&E stained tiles.

![Python](https://img.shields.io/badge/python-3.11-blue) ![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red) ![PyTorch](https://img.shields.io/badge/pytorch-2.1+-orange) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Setup

```bash
git clone https://github.com/bravo2024/PathSeg.git
cd PathSeg
pip install -r requirements.txt
```

No GPU needed to run the app — the baseline trains in a couple of seconds on CPU. U-Net training is faster with a GPU but runs on CPU too.

---

## Running

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The sidebar lets you pick the model and data source.

To pre-train models before opening the app:

```bash
python train.py                          # baseline
python train.py --model unet             # U-Net (30 epochs default)
python train.py --model unet --epochs 50 --lr 0.0005
```

---

## Data

Three ways to load data from the sidebar:

**Synthetic** (default) — generated on the fly, no download. Good for testing the pipeline.

**HuggingFace Hub** — streams real H&E tiles directly, no disk storage needed. Works on Streamlit Cloud too. Presets include MoNuSeg, PanNuke, and PanOptils.

**Local folder** — point it at a folder with `images/` and `masks/` subdirectories.

**Download from URL** — paste a Zenodo or direct ZIP/tar.gz link, it downloads and caches locally.

---

## Project layout

```
app.py          main Streamlit app
train.py        CLI for training
src/
  data.py       loaders for synthetic, local, HF Hub, and URL datasets
  model.py      baseline + U-Net inference
  unet.py       U-Net architecture
  core.py       logistic regression and metrics from scratch
  transforms.py augmentation
  config.py     YAML config loader
  ...
tests/          pytest suite
config/
  base.yaml     default hyperparameters
```

---

## Tests

```bash
pytest -q
```

---

## Deploying to Streamlit Cloud

1. Fork or push to your own GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo
3. Set entry point to `app.py`

The app works on Streamlit Cloud with synthetic data or HuggingFace Hub streaming (no persistent disk needed).

---

## License

MIT
