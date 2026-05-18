"""PyTorch dataset and transforms for skin lesion images."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.labels import BINARY_LABELS

LABEL_TO_IDX = {name: i for i, name in enumerate(BINARY_LABELS)}
IDX_TO_LABEL = {i: name for name, i in LABEL_TO_IDX.items()}


def get_train_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_eval_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class SkinLesionDataset(Dataset):
    def __init__(self, manifest_csv: str | Path, transform=None):
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = LABEL_TO_IDX[row["label"]]
        return image, label


def create_dataloader(
    manifest_csv: str | Path,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    train: bool,
    num_workers: int = 0,
) -> DataLoader:
    transform = get_train_transforms(image_size) if train else get_eval_transforms(image_size)
    dataset = SkinLesionDataset(manifest_csv, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )
