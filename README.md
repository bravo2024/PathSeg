# PathSeg

A Streamlit app for binary segmentation of H&E tissue tiles. Trains two models — a pixel-level logistic regression baseline and a small Keras FCN — then lets you compare predictions and metrics interactively.

Built to explore how far a hand-rolled baseline gets before a convolutional net is worth the extra complexity.

![Python](https://img.shields.io/badge/python-3.11-blue) ![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red) ![TensorFlow](https://img.shields.io/badge/tensorflow--cpu-2.13+-orange) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Setup

```bash
git clone https://github.com/bravo2024/PathSeg.git
cd PathSeg
pip install -r requirements.txt
```

No GPU needed — both models train on CPU. The Keras FCN takes ~15 s for 120 synthetic tiles.

---

## Running

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The sidebar picks the model and data source. Both models train inside the app — no pre-training step needed.

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
src/
  data.py       loaders — synthetic, local folder, HF Hub, URL
  model.py      logistic regression baseline
  fcn.py        Keras FCN — build, train, evaluate, predict
  core.py       metrics (Dice, IoU, sensitivity, specificity)
  config.py     YAML config loader
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
