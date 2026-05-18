"""Model definitions: tiny CNN (~80K params) vs frozen ResNet18 baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class LightweightCNN(nn.Module):
    """Small CNN for CPU training — ~80K parameters at 128×128 input."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_lightweight(num_classes: int = 2) -> LightweightCNN:
    return LightweightCNN(num_classes=num_classes)


def build_resnet18_baseline(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """ResNet18 pretrained — backbone frozen for fast CPU fine-tuning."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


MODEL_BUILDERS = {
    "lightweight": build_lightweight,
    "resnet18": build_resnet18_baseline,
}
