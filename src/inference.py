"""Load checkpoint and run single-image prediction."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from PIL import Image

from src.dataset import IDX_TO_LABEL, get_eval_transforms
from src.models import MODEL_BUILDERS


def load_config(config_path: str | Path = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str | Path, config: dict | None = None):
    config = config or load_config()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = ckpt.get("model_name", "lightweight")
    image_size = ckpt.get("image_size", config["training"]["image_size"])

    builder = MODEL_BUILDERS[model_name]
    model = builder(num_classes=2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, image_size, model_name


def predict_image(
    model: torch.nn.Module,
    image: Image.Image,
    image_size: int,
) -> dict:
    transform = get_eval_transforms(image_size)
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    idx = int(torch.argmax(probs).item())
    label = IDX_TO_LABEL[idx]
    confidence = float(probs[idx].item())

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "probabilities": {
            "benign": round(float(probs[0].item()) * 100, 2),
            "malignant": round(float(probs[1].item()) * 100, 2),
        },
    }
