#!/usr/bin/env python3
"""Print side-by-side comparison of lightweight vs ResNet18 results."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_metrics(name: str) -> dict | None:
    path = RESULTS / f"{name}_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    light = load_metrics("lightweight")
    base = load_metrics("resnet18")

    if not light and not base:
        print("No results yet. Run train_lightweight.py and/or train_baseline.py first.")
        sys.exit(1)

    print("\n=== Model Comparison (test set) ===\n")
    print(f"{'Metric':<22} {'Lightweight CNN':>18} {'ResNet18 (frozen)':>18}")
    print("-" * 60)

    rows = [
        ("Parameters", "total_parameters", "{:,}"),
        ("Trainable params", "trainable_parameters", "{:,}"),
        ("Training time (s)", "training_time_sec", "{:.1f}"),
        ("Test accuracy", ("test_metrics", "accuracy"), "{:.3f}"),
        ("Test F1", ("test_metrics", "f1"), "{:.3f}"),
        ("Test AUC", ("test_metrics", "auc"), "{:.3f}"),
    ]

    for label, key, fmt in rows:
        lv, bv = "—", "—"
        if light:
            v = light[key[0]][key[1]] if isinstance(key, tuple) else light.get(key)
            if v is not None:
                lv = fmt.format(v)
        if base:
            v = base[key[0]][key[1]] if isinstance(key, tuple) else base.get(key)
            if v is not None:
                bv = fmt.format(v)
        print(f"{label:<22} {lv:>18} {bv:>18}")

    print("\nFull metrics: results/lightweight_metrics.json, results/resnet18_metrics.json")


if __name__ == "__main__":
    main()
