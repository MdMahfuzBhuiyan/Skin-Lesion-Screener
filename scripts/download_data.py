#!/usr/bin/env python3
"""Download ISIC 2019 ground truth and a stratified image subset (CPU-friendly)."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.labels import CLASS_COLUMNS, row_to_binary_label


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Already exists: {dest}")
        return
    print(f"Downloading {url} …")
    urllib.request.urlretrieve(url, dest)


def build_manifest(gt_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    malignant = cfg["data"]["malignant_classes"]
    benign = cfg["data"]["benign_classes"]

    records = []
    for _, row in gt_df.iterrows():
        image_id = row["image"]
        label = row_to_binary_label(row, malignant, benign)
        if label is None:
            continue
        records.append({"image_id": image_id, "label": label})

    manifest = pd.DataFrame(records)
    subset_size = cfg["data"]["subset_size"]
    if len(manifest) > subset_size:
        seed = cfg["data"]["seed"]
        half = subset_size // 2
        parts = []
        for label in ("benign", "malignant"):
            group = manifest[manifest["label"] == label]
            n = min(len(group), half if label == "benign" else subset_size - half)
            parts.append(group.sample(n=n, random_state=seed))
        manifest = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)
    return manifest


def download_images(manifest: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    raw_dir = ROOT / cfg["data"]["raw_dir"]
    base_url = cfg["data"]["image_base_url"].rstrip("/")
    paths = []

    for image_id in tqdm(manifest["image_id"], desc="images"):
        dest = raw_dir / f"{image_id}.jpg"
        if not dest.exists():
            url = f"{base_url}/{image_id}.jpg"
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                print(f"  skip {image_id}: {e}")
                paths.append(None)
                continue
        paths.append(str(dest.relative_to(ROOT)))

    manifest = manifest.copy()
    manifest["filepath"] = paths
    return manifest.dropna(subset=["filepath"])


def main():
    parser = argparse.ArgumentParser(description="Download ISIC 2019 subset")
    parser.add_argument("--skip-images", action="store_true", help="Only fetch CSV / build splits")
    parser.add_argument("--max-images", type=int, default=None, help="Cap downloads (debug)")
    args = parser.parse_args()

    cfg = load_config()
    gt_path = ROOT / "data" / "ISIC_2019_Training_GroundTruth.csv"
    download_file(cfg["data"]["ground_truth_url"], gt_path)

    gt_df = pd.read_csv(gt_path)
    assert "image" in gt_df.columns, "Expected 'image' column in ground truth CSV"

    manifest = build_manifest(gt_df, cfg)
    print(f"Subset: {len(manifest)} images (benign={sum(manifest['label']=='benign')}, "
          f"malignant={sum(manifest['label']=='malignant')})")

    if args.max_images:
        manifest = manifest.head(args.max_images)

    processed_dir = ROOT / cfg["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_images:
        manifest["filepath"] = manifest["image_id"].apply(
            lambda i: str((ROOT / cfg["data"]["raw_dir"] / f"{i}.jpg").relative_to(ROOT))
        )
    else:
        manifest = download_images(manifest, cfg)

    seed = cfg["data"]["seed"]
    train_ratio = cfg["data"]["train_ratio"]
    val_ratio = cfg["data"]["val_ratio"]
    test_ratio = cfg["data"]["test_ratio"]

    train_df, temp_df = train_test_split(
        manifest, test_size=(1 - train_ratio), stratify=manifest["label"], random_state=seed
    )
    rel_val = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - rel_val), stratify=temp_df["label"], random_state=seed
    )

    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)
    manifest.to_csv(processed_dir / "full_manifest.csv", index=False)

    print(f"Splits → train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"Manifests saved under {processed_dir}")


if __name__ == "__main__":
    main()
