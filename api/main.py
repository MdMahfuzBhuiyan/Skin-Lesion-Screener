"""Localhost API: upload skin lesion image → benign/malignant + confidence."""

from __future__ import annotations

import io
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import load_config, load_model, predict_image

app = FastAPI(
    title="Skin Lesion Screener",
    description="Lightweight CPU binary classifier — benign vs malignant. Not for clinical use.",
    version="1.0.0",
)

_config = None
_model = None
_image_size = None


def get_model():
    global _config, _model, _image_size
    if _model is None:
        _config = load_config(ROOT / "config.yaml")
        ckpt = ROOT / _config["models"]["default_for_api"]
        if not ckpt.exists():
            raise FileNotFoundError(
                f"No checkpoint at {ckpt}. Train first: python scripts/train_lightweight.py"
            )
        _model, _image_size, _ = load_model(ckpt, _config)
    return _model, _image_size


@app.on_event("startup")  # noqa: deprecated but simple for local demo
async def startup():
    try:
        get_model()
    except FileNotFoundError:
        pass


@app.get("/health")
def health():
    ckpt = ROOT / "config.yaml"
    has_model = False
    try:
        cfg = load_config(ckpt)
        has_model = (ROOT / cfg["models"]["default_for_api"]).exists()
    except Exception:
        pass
    return {"status": "ok", "model_loaded": has_model}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload a JPG or PNG image.")

    try:
        model, image_size = get_model()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e)) from e

    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}") from e

    result = predict_image(model, image, image_size)
    return {
        "prediction": result["label"].capitalize(),
        "confidence_percent": result["confidence"],
        "probabilities_percent": result["probabilities"],
        "disclaimer": "Research/demo only — not a medical device. See a dermatologist for diagnosis.",
    }


static_dir = ROOT / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = static_dir / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return "<p>Skin Lesion Screener API. POST an image to <a href='/docs'>/predict</a>.</p>"
