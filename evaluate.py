"""Evaluate the saved checkpoint: .venv/Scripts/python.exe evaluate.py"""

import hashlib
import json
import sys
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from main import CLASSES, LEGACY_WEIGHTS, PREPROCESS, ROOT, WEIGHTS
from model import HybridModel


def evaluate():
    torch.set_num_threads(4)
    path = WEIGHTS if WEIGHTS.exists() else LEGACY_WEIGHTS
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if "--if-needed" in sys.argv:
        try:
            report = json.loads((ROOT / "evaluation_metrics.json").read_text(encoding="utf-8"))
            if report["checkpoint_sha256"] == digest and report.get("metrics"):
                print("Verified evaluation results already available.", flush=True)
                return
        except (OSError, ValueError, KeyError):
            pass
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    if "model_state_dict" in checkpoint:
        if list(checkpoint.get("class_names", CLASSES)) != CLASSES:
            raise ValueError("Checkpoint class order differs from the application")
    model = HybridModel()
    model.load_state_dict(state, strict=True)
    model.eval()
    dataset = ImageFolder(ROOT / "dataset/chest_xray/test", transform=PREPROCESS)
    if dataset.classes != CLASSES:
        raise ValueError("Test dataset class order differs from the application")
    matrix = torch.zeros((2, 2), dtype=torch.int64)
    with torch.inference_mode():
        completed = 0
        for images, labels in DataLoader(dataset, batch_size=16, shuffle=False):
            predicted = model(images).argmax(dim=1)
            matrix += torch.bincount(labels * 2 + predicted, minlength=4).reshape(2, 2)
            completed += len(labels)
            print(f"Evaluated {completed}/{len(dataset)}", flush=True)
    counts = matrix.to(torch.float64)
    true_positive = counts.diag()
    precision = true_positive / counts.sum(0).clamp(min=1)
    recall = true_positive / counts.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-15)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Checkpoint changed during evaluation; run again")
    report = {
        "checkpoint_sha256": digest,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "class_names": CLASSES,
        "test_samples": len(dataset),
        "metrics": {
            "accuracy": (true_positive.sum() / counts.sum()).item(),
            "precision": precision.mean().item(),
            "recall": recall.mean().item(),
            "f1_score": f1.mean().item(),
            "confusion_matrix": matrix.tolist(),
        },
        "note": f"Evaluated on {len(dataset)} images in dataset/chest_xray/test. "
        "Precision, recall and F1 are macro averages. Results verified against the checkpoint SHA-256.",
    }
    output = ROOT / "evaluation_metrics.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    evaluate()
