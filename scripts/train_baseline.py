#!/usr/bin/env python3
"""Train frozen ResNet18 head (heavyweight pretrained comparison) on CPU."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import load_config, train_model


def main():
    config = load_config(ROOT / "config.yaml")
    output = ROOT / config["models"]["baseline"]
    output.parent.mkdir(parents=True, exist_ok=True)
    train_model("resnet18", config, output)


if __name__ == "__main__":
    main()
