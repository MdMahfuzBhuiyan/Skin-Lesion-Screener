#!/usr/bin/env python3
"""Print predictions for a few known test images (sanity check)."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.verify_model import main as verify_all

if __name__ == "__main__":
    print("Running full test-set check…\n")
    verify_all()
