#!/usr/bin/env python3
"""
Create a tiny synthetic dataset, train lightweight CNN, and verify inference.
Use this to test the pipeline without downloading ISIC images.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import load_model, predict_image
from src.train import train_model


def write_split(processed: Path, name: str, benign_n: int, malignant_n: int, seed: int):
    rng = np.random.default_rng(seed)
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, n, tone in [("benign", benign_n, (120, 200)), ("malignant", malignant_n, (20, 90))]:
        for i in range(n):
            image_id = f"DEMO_{label}_{name}_{i}"
            path = raw / f"{image_id}.jpg"
            lo, hi = tone
            arr = rng.integers(lo, hi, (128, 128, 3), dtype=np.uint8)
            Image.fromarray(arr).save(path)
            rows.append({"image_id": image_id, "label": label, "filepath": str(path.relative_to(ROOT))})
    pd.DataFrame(rows).to_csv(processed / f"{name}.csv", index=False)


def main():
    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    write_split(processed, "train", 80, 80, 1)
    write_split(processed, "val", 20, 20, 2)
    write_split(processed, "test", 20, 20, 3)

    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)
    config["training"]["epochs"] = 4
    config["training"]["early_stop_patience"] = 2

    out = ROOT / config["models"]["lightweight"]
    out.parent.mkdir(exist_ok=True)
    print("Training on synthetic demo data (CPU)…")
    train_model("lightweight", config, out)

    model, image_size, _ = load_model(out, config)
    sample = Image.open(next((ROOT / "data" / "raw").glob("*.jpg")))
    print("Sample prediction:", predict_image(model, sample, image_size))
    print("\nDemo ready. Start API: python scripts/run_api.py")


if __name__ == "__main__":
    main()
