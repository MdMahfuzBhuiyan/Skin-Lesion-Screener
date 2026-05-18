# Skin Lesion Screener

Lightweight **CPU-only** binary classifier (benign vs malignant) for dermoscopic images, trained on a small **[DermaMNIST](https://medmnist.com/)** subset (HAM10000-derived, auto-download), with a localhost web API.


## Highlights

| | Lightweight CNN | ResNet18 (frozen backbone) |
|---|---|---|
| Parameters | ~80K | ~11M (512 trainable) |
| Training | Full network on CPU | Only classifier head |
| Goal | Fast, deployable screening | Pretrained baseline comparison |

## Dataset (DermaMNIST / MedMNIST)

| | Old (ISIC 2019) | **New (DermaMNIST)** |
|---|---|---|
| Download | Thousands of S3 image URLs | **One `pip` package, ~25 MB** |
| Default size | ~4,000 images | **1,000 images** (configurable) |
| Source | ISIC challenge | [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) via MedMNIST |

**Binary mapping (7 classes → 2)**

| Malignant | Benign |
|---|---|
| akiec, bcc, mel | bkl, df, nv, vasc |

## Quick start

```bash
cd skin-lesion-screener
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

<<<<<<< Updated upstream
python scripts/demo_quickstart.py   # trains tiny model (~1–2 min CPU)
python scripts/run_api.py           # http://127.0.0.1:8000

```
Open http://127.0.0.1:8000 in your browser, or use the interactive docs at http://127.0.0.1:8000/docs .
=======
python scripts/download_data.py        # ~1–3 min, downloads + exports subset
python scripts/train_lightweight.py
python scripts/run_api.py              # http://127.0.0.1:8000
```

**Smaller subset (e.g. 500 images):**
>>>>>>> Stashed changes

```bash
python scripts/download_data.py --subset-size 500
```

**Synthetic smoke test (no download):**

```bash
python scripts/demo_quickstart.py
python scripts/run_api.py
```

## Full pipeline

### 1. Prepare data

```bash
python scripts/download_data.py
```

Exports JPGs to `data/raw/` and CSV manifests to `data/processed/`.

### 2. Train (CPU)

```bash
python scripts/train_lightweight.py
python scripts/train_baseline.py      # optional comparison
python scripts/compare_models.py
```

### 3. API

```bash
python scripts/run_api.py
```

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@/path/to/lesion.jpg"
```

## Paper / experiment notes

- **Dataset**: DermaMNIST subset (`config.yaml` → `subset_size`, default 1000)  
- **Input**: 128×128 RGB (upscaled from 28×28 MedMNIST tiles)  
- **Metrics**: `results/*.json` — accuracy, F1, AUC, confusion matrix  
- **Citation**: Yang et al., MedMNIST v3 — see https://medmnist.com/

## Configuration

Edit `config.yaml` for `subset_size`, class IDs, epochs, and API port.

## Troubleshooting wrong predictions

1. **Re-prepare data and retrain** (required after fixes):
   ```bash
   python scripts/download_data.py
   python scripts/train_lightweight.py
   python scripts/verify_model.py   # should be ~60%+ on test tiles
   python scripts/run_api.py      # restart API after retraining
   ```

2. **Use the right image type** — model is trained on **dermoscopic** crops (HAM10000 / DermaMNIST), not general skin photos or internet pictures.

3. **Test with a known file** from the dataset:
   ```bash
   curl -X POST http://127.0.0.1:8000/predict -F "file=@data/raw/derma_test_00000.png"
   ```

4. If everything looks malignant: you may be on an **old checkpoint** — delete `models/lightweight_cnn.pt` and train again.

## License & ethics

HAM10000 / MedMNIST have their own licenses. This repo is a research demo — not a medical device.
