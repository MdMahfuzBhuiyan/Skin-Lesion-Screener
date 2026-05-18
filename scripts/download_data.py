#!/usr/bin/env python3
"""
Prepare DermaMNIST (MedMNIST) at native 128×128 — reliable auto-download.

Uses official train/val/test splits (no random re-split leakage).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.labels import class_id_to_binary


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def parse_class_id(label) -> int:
    arr = np.asarray(label).astype(int).flatten()
    return int(arr[0])


def load_dermamnist_splits(load_size: int):
    try:
        from medmnist import INFO
        import medmnist
    except ImportError as e:
        raise SystemExit(
            "Install MedMNIST first: pip install medmnist\n"
            "Then re-run: python scripts/download_data.py"
        ) from e

    info = INFO["dermamnist"]
    DataClass = getattr(medmnist, info["python_class"])

    splits = {}
    for split in ("train", "val", "test"):
        ds = DataClass(split=split, download=True, size=load_size, as_rgb=True)
        splits[split] = ds
    return splits, info


def sample_split(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Random sample keeping natural benign/malignant ratio (HAM10000-like)."""
    if len(df) <= n:
        return df.reset_index(drop=True)
    sampled = df.sample(n=n, random_state=seed).reset_index(drop=True)
    return sampled


def build_split_records(dataset, split_name: str, cfg: dict) -> pd.DataFrame:
    malignant_ids = cfg["data"]["malignant_class_ids"]
    benign_ids = cfg["data"]["benign_class_ids"]
    rows = []
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_id = parse_class_id(label)
        binary = class_id_to_binary(class_id, malignant_ids, benign_ids)
        if binary is None:
            continue
        rows.append(
            {
                "image_id": f"derma_{split_name}_{idx:05d}",
                "label": binary,
                "class_id": class_id,
                "source_split": split_name,
                "dataset_index": idx,
            }
        )
    return pd.DataFrame(rows)


def export_split(
    manifest: pd.DataFrame,
    dataset,
    raw_dir: Path,
    split_name: str,
    export_size: int,
) -> pd.DataFrame:
    paths = []
    for row in tqdm(manifest.itertuples(), total=len(manifest), desc=f"export {split_name}"):
        img, _ = dataset[row.dataset_index]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.convert("RGB")
        if export_size and (img.width != export_size or img.height != export_size):
            img = img.resize((export_size, export_size), Image.Resampling.LANCZOS)
        dest = raw_dir / f"{row.image_id}.png"
        img.save(dest)
        paths.append(str(dest.relative_to(ROOT)))
    out = manifest.copy()
    out["filepath"] = paths
    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare DermaMNIST (128px) with official splits")
    parser.add_argument("--subset-size", type=int, default=None, help="Total images (split across train/val/test)")
    args = parser.parse_args()

    cfg = load_config()
    load_size = cfg["data"].get("medmnist_load_size", 28)
    export_size = cfg["data"].get("export_size", 128)
    subset_total = args.subset_size or cfg["data"]["subset_size"]
    seed = cfg["data"]["seed"]

    train_n = int(subset_total * 0.7)
    val_n = int(subset_total * 0.15)
    test_n = subset_total - train_n - val_n

    print(f"Loading DermaMNIST {load_size}px → exporting {export_size}px…")
    splits, info = load_dermamnist_splits(load_size)
    print(f"Dataset: {info['description'][:80]}…")

    raw_dir = ROOT / cfg["data"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = ROOT / cfg["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    split_targets = {"train": train_n, "val": val_n, "test": test_n}
    exported = {}

    for split_name, target_n in split_targets.items():
        records = build_split_records(splits[split_name], split_name, cfg)
        sampled = sample_split(records, target_n, seed + hash(split_name) % 1000)
        exported[split_name] = export_split(
            sampled, splits[split_name], raw_dir, split_name, export_size
        )
        n_ben = (exported[split_name]["label"] == "benign").sum()
        n_mal = (exported[split_name]["label"] == "malignant").sum()
        print(f"  {split_name}: {len(exported[split_name])} (benign={n_ben}, malignant={n_mal})")

    cols = ["image_id", "label", "filepath", "class_id"]
    for split_name, df in exported.items():
        df[cols].to_csv(processed_dir / f"{split_name}.csv", index=False)

    full = pd.concat([exported["train"], exported["val"], exported["test"]], ignore_index=True)
    full[cols].to_csv(processed_dir / "full_manifest.csv", index=False)

    print(f"\nDone. Total {len(full)} images ({export_size}px) → {processed_dir}")
    print("Next: python scripts/train_lightweight.py")


if __name__ == "__main__":
    main()
