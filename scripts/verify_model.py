#!/usr/bin/env python3
"""Check predictions on held-out test images vs ground truth."""

import sys
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import load_config, load_model, predict_image
from src.paths import resolve_data_path


def main():
    config = load_config(ROOT / "config.yaml")
    ckpt = ROOT / config["models"]["lightweight"]
    if not ckpt.exists():
        print("No model found. Run train_lightweight.py first.")
        sys.exit(1)

    model, image_size, _, threshold, _ = load_model(ckpt, config)
    test_df = pd.read_csv(ROOT / config["data"]["processed_dir"] / "test.csv")

    correct = 0
    for _, row in test_df.iterrows():
        img = Image.open(resolve_data_path(row["filepath"]))
        pred = predict_image(model, img, image_size, threshold=threshold)
        ok = pred["label"] == row["label"]
        correct += int(ok)
        mark = "OK" if ok else "WRONG"
        print(f"{mark}  true={row['label']:9} pred={pred['label']:9}  mal%={pred['probabilities']['malignant']:5.1f}")

    acc = correct / len(test_df)
    print(f"\nTest accuracy: {acc:.1%} ({correct}/{len(test_df)})  threshold={threshold:.2f}")


if __name__ == "__main__":
    main()
