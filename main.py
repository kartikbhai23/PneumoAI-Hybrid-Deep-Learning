"""PneumoAI research demo API. Run: uv run uvicorn main:app --reload"""

from __future__ import annotations

import hashlib
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

from model import HybridModel, gradcam_overlay


# Configuration
ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "pneumonia_model.pth"
if not WEIGHTS.exists() and (ROOT / "best_model.pth").exists():
    WEIGHTS = ROOT / "best_model.pth"
LEGACY_WEIGHTS = ROOT / "pneumonia_hybrid_model.pth"
CLASSES = ["NORMAL", "PNEUMONIA"]
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 24_000_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

model: HybridModel | None = None
metrics: dict = {}
load_error: str | None = None


# Load model when FastAPI starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, metrics, load_error

    model = None
    metrics = {}
    path = WEIGHTS if WEIGHTS.exists() else LEGACY_WEIGHTS

    if path.exists():
        try:
            # Only load checkpoints from trusted sources.
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
            state = (
                checkpoint.get("model_state_dict", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )
            candidate = HybridModel().to(DEVICE)
            candidate.load_state_dict(state, strict=True)
            candidate.eval()

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                saved_classes = checkpoint.get("class_names", CLASSES)
                if list(saved_classes) != CLASSES:
                    raise ValueError(
                        "Checkpoint class order does not match expected NORMAL, PNEUMONIA"
                    )
                metrics = checkpoint.get("metrics", {}) or {}

            model = candidate
            load_error = None
        except Exception as exc:
            model = None
            load_error = f"{type(exc).__name__}: {exc}"
    else:
        load_error = (
            "Model weights not found. Place pneumonia_hybrid_model.pth, "
            "pneumonia_model.pth or best_model.pth beside main.py."
        )

    yield


app = FastAPI(title="PneumoAI Research API", version="1.0.0", lifespan=lifespan)


@app.get("/")
def home():
    return FileResponse(ROOT / "index.html")


@app.get("/health")
def health():
    return {
        "model_ready": model is not None,
        "device": str(DEVICE),
        "message": load_error,
    }


@app.get("/metrics")
def get_metrics():
    if metrics:
        return {
            "available": True,
            "metrics": metrics,
            "note": "Metrics loaded from the saved model checkpoint.",
        }

    # For older weights-only checkpoints, use metrics only when hash-matched.
    recorded = ROOT / "evaluation_metrics.json"
    path = WEIGHTS if WEIGHTS.exists() else LEGACY_WEIGHTS
    if recorded.exists() and path.exists():
        try:
            report = json.loads(recorded.read_text(encoding="utf-8"))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if report["checkpoint_sha256"] == digest:
                return {
                    "available": True,
                    "metrics": report["metrics"],
                    "note": report["note"],
                }
        except (OSError, ValueError, KeyError):
            pass

    return {
        "available": False,
        "metrics": {},
        "note": "No evaluation results are saved for this checkpoint.",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=load_error or "Trained model unavailable",
        )

    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(
            status_code=415, detail="Upload a JPG or PNG chest X-ray image"
        )

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be smaller than 10 MB")

    try:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
        image.load()
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise HTTPException(
            status_code=400, detail="Invalid or unsupported image"
        ) from exc

    tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()

    predicted = int(np.argmax(probabilities))
    try:
        heatmap = gradcam_overlay(model, image, tensor, predicted)
        heatmap_error = None
    except Exception:
        heatmap = None
        heatmap_error = "Heatmap unavailable for this image."

    return {
        "prediction": CLASSES[predicted],
        "confidence": round(probabilities[predicted], 6),
        "scores": {
            label: round(float(probabilities[i]), 6)
            for i, label in enumerate(CLASSES)
        },
        "gradcam_image": heatmap,
        "gradcam_error": heatmap_error,
        "disclaimer": "Experimental research output, not a medical diagnosis.",
    }
