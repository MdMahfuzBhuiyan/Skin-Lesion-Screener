"""Load checkpoint and run single-image prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from src.dataset import IDX_TO_LABEL, get_eval_transforms
from src.metrics import predict_with_threshold
from src.models import MODEL_BUILDERS
from src.paths import PROJECT_ROOT


def load_config(config_path: str | Path | None = None) -> dict:
    path = config_path or PROJECT_ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str | Path, config: dict | None = None):
    config = config or load_config()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = ckpt.get("model_name", "lightweight")
    image_size = ckpt.get("image_size", config["training"]["image_size"])
    threshold = float(ckpt.get("threshold", 0.5))
    label_map = ckpt.get("label_map", IDX_TO_LABEL)

    builder = MODEL_BUILDERS[model_name]
    model = builder(num_classes=2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, image_size, model_name, threshold, label_map


def predict_image(
    model: torch.nn.Module,
    image: Image.Image,
    image_size: int,
    threshold: float = 0.5,
) -> dict:
    transform = get_eval_transforms(image_size)
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    prob_benign = float(probs[0].item())
    prob_malignant = float(probs[1].item())
    idx = int(predict_with_threshold(np.array([prob_malignant]), threshold)[0])
    label = IDX_TO_LABEL[idx]
    confidence = prob_malignant if idx == 1 else prob_benign

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "threshold_malignant": threshold,
        "probabilities": {
            "benign": round(prob_benign * 100, 2),
            "malignant": round(prob_malignant * 100, 2),
        },
    }
