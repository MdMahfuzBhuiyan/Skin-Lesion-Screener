#!/usr/bin/env python3
"""Start the localhost API server."""

import sys
from pathlib import Path

import uvicorn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    api_cfg = cfg["api"]
    uvicorn.run(
        "api.main:app",
        host=api_cfg["host"],
        port=api_cfg["port"],
        reload=False,
        app_dir=str(ROOT),
    )


if __name__ == "__main__":
    main()
