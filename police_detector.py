"""
police_detector.py

Drop-in replacement using the trained MobileNetV2 classifier.
Loads police_classifier.pt once at import time, scores crops via
is_police_vehicle() — same interface as the original heuristic version.
"""

import cv2
import numpy as np
import os

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ── Load model at import time ─────────────────────────────────────────────────

_MODEL_PATH = os.environ.get("POLICE_MODEL", "police_classifier.pt")
if torch.cuda.is_available():
    _DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    _DEVICE = torch.device("mps")
else:
    _DEVICE = torch.device("cpu")
_MODEL = None

_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
    )

    if os.path.exists(_MODEL_PATH):
        model.load_state_dict(torch.load(_MODEL_PATH, map_location=_DEVICE))
        model.to(_DEVICE)
        model.eval()
        _MODEL = model
        print(f"[police_detector] Loaded {_MODEL_PATH} on {_DEVICE}")
    else:
        print(f"[police_detector] WARNING: {_MODEL_PATH} not found — all detections will return False")

    return _MODEL


def is_police_vehicle(crop: np.ndarray) -> dict:
    """
    Score a BGR vehicle crop. Returns dict compatible with detect.py:
      is_police, confidence, battenburg, text_found, text_detected
    """
    model = _load_model()

    if model is None or crop is None or crop.size == 0:
        return {
            "is_police": False,
            "confidence": 0.0,
            "battenburg": False,
            "text_found": "",
            "text_detected": False,
        }

    try:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor = _TRANSFORM(img).unsqueeze(0).to(_DEVICE)

        with torch.no_grad():
            logit = model(tensor).squeeze()
            prob = torch.sigmoid(logit).item()
    except Exception:
        prob = 0.0

    return {
        "is_police":     prob >= 0.5,
        "confidence":    round(prob, 3),
        "battenburg":    prob >= 0.5,
        "text_found":    "",
        "text_detected": False,
    }
