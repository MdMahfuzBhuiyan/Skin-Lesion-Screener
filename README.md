# Skin Lesion Screener

Lightweight **CPU-only** binary classifier (benign vs malignant) for dermoscopic images, trained on an [ISIC 2019](https://challenge.isic-archive.com/data/) subset, with a localhost web API.

**Not for clinical use** — research and education only.

## Highlights

| | Lightweight CNN | ResNet18 (frozen backbone) |
|---|---|---|
| Parameters | ~80K | ~11M (512 trainable) |
| Training | Full network on CPU | Only classifier head |
| Goal | Fast, deployable screening | Pretrained baseline comparison |

## Quick start (no ISIC download)

Verify the full pipeline on synthetic data:

```bash
cd skin-lesion-screener
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/demo_quickstart.py   # trains tiny model (~1–2 min CPU)
python scripts/run_api.py           # http://127.0.0.1:8000
```

Open http://127.0.0.1:8000 in your browser, or use the interactive docs at http://127.0.0.1:8000/docs .

## Full pipeline (ISIC 2019)

### 1. Download subset (~4,000 images)

```bash
python scripts/download_data.py
```

This fetches the official ground-truth CSV from AWS, maps 8 ISIC classes to binary labels, and downloads a stratified subset. Expect **30–60+ minutes** on a typical connection.

Options:

- `--max-images 200` — quick test with fewer downloads  
- `--skip-images` — only build CSV splits (if you already have images under `data/raw/`)

**Binary mapping**

| Malignant / suspicious | Benign |
|---|---|
| MEL, BCC, SCC, AK | NV, BKL, DF, VASC |

### 2. Train models (CPU)

```bash
python scripts/train_lightweight.py   # ~80K param CNN
python scripts/train_baseline.py      # frozen ResNet18 comparison
python scripts/compare_models.py      # side-by-side metrics
```

Metrics are saved under `results/`.

### 3. Run API

```bash
python scripts/run_api.py
```

**POST** `/predict` with multipart file field `file` (JPG/PNG).

Example:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@/path/to/lesion.jpg"
```

Response:

```json
{
  "prediction": "Benign",
  "confidence_percent": 87.42,
  "probabilities_percent": {
    "benign": 87.42,
    "malignant": 12.58
  },
  "disclaimer": "Research/demo only — not a medical device..."
}
```

Switch the API checkpoint in `config.yaml` → `models.default_for_api`.

## Project layout

```
skin-lesion-screener/
├── api/main.py              # FastAPI app
├── static/index.html        # Simple upload UI
├── config.yaml              # Data URLs, hyperparameters, API port
├── src/
│   ├── models.py            # LightweightCNN + ResNet18
│   ├── dataset.py
│   ├── train.py
│   └── inference.py
├── scripts/
│   ├── download_data.py
│   ├── train_lightweight.py
│   ├── train_baseline.py
│   ├── compare_models.py
│   ├── demo_quickstart.py
│   └── run_api.py
└── models/                  # Saved .pt checkpoints
```

## Paper / experiment notes

- **Dataset**: ISIC 2019, 3k–5k stratified subset (`config.yaml` → `subset_size`)  
- **Input**: 128×128 RGB, ImageNet normalization  
- **Metrics**: accuracy, F1, AUC, confusion matrix (see `results/*.json`)  
- **Contribution angle**: comparable screening performance with orders-of-magnitude fewer parameters and no GPU requirement  

## Configuration

Edit `config.yaml` for subset size, epochs, image size, malignant/benign class lists, and API host/port.

## License & ethics

ISIC data has its own [terms of use](https://www.isic-archive.com/). This project is a research demo — always involve qualified clinicians for real patient care.
