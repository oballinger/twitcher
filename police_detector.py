"""
police_detector.py

Heuristic police vehicle detector using:
  1. Battenburg pattern detection (blue/yellow HSV checkerboard)
  2. "POLICE" text detection via OCR (pytesseract)

Also saves flagged + unflagged crops to disk for dataset generation.
"""

import cv2
import numpy as np
import pytesseract
import os
import time
import hashlib

# ── Dataset output dirs ───────────────────────────────────────────────────────
DATASET_DIR  = "dataset"
POLICE_DIR   = os.path.join(DATASET_DIR, "police")
VEHICLE_DIR  = os.path.join(DATASET_DIR, "vehicle")   # non-police vehicles
os.makedirs(POLICE_DIR,  exist_ok=True)
os.makedirs(VEHICLE_DIR, exist_ok=True)

# ── Battenburg HSV ranges ─────────────────────────────────────────────────────
# UK police Battenburg: retro-reflective yellow + blue checks
# Loose ranges to maximise recall on compressed, low-res JamCam footage
YELLOW_LOWER = np.array([18,  80,  80], dtype=np.uint8)
YELLOW_UPPER = np.array([35, 255, 255], dtype=np.uint8)

BLUE_LOWER   = np.array([95,  60,  60], dtype=np.uint8)
BLUE_UPPER   = np.array([135, 255, 255], dtype=np.uint8)

YELLOW_THRESH  = 0.04
BLUE_THRESH    = 0.03
CHECKER_BLOCKS = 4

# ── OCR config ────────────────────────────────────────────────────────────────
# Whitelist only uppercase letters — faster and more accurate for vehicle text
TESS_CONFIG = r"--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
POLICE_KEYWORDS = {"POLICE", "POLCE", "POL1CE"}  # common OCR misreads included


def detect_battenburg(crop: np.ndarray) -> tuple[bool, float]:
    """
    Check a vehicle crop for Battenburg markings.
    Returns (detected: bool, confidence: float 0-1).
    """
    hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total = crop.shape[0] * crop.shape[1]

    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    blue_mask   = cv2.inRange(hsv, BLUE_LOWER,   BLUE_UPPER)

    yellow_frac = cv2.countNonZero(yellow_mask) / total
    blue_frac   = cv2.countNonZero(blue_mask)   / total

    if yellow_frac < YELLOW_THRESH or blue_frac < BLUE_THRESH:
        return False, 0.0

    # Check for checkerboard alternation: scan horizontal strips,
    # count transitions between yellow-dominant and blue-dominant columns.
    # Scale required transitions down for small crops (JamCam resolution).
    h, w = crop.shape[:2]

    # Upscale tiny crops before analysing pattern
    min_dim = 80
    if h < min_dim or w < min_dim:
        scale  = max(min_dim / h, min_dim / w)
        crop_r = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        hsv_r  = cv2.cvtColor(crop_r, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv_r, YELLOW_LOWER, YELLOW_UPPER)
        blue_mask   = cv2.inRange(hsv_r, BLUE_LOWER,   BLUE_UPPER)
        h, w = crop_r.shape[:2]
    
    strip_h = max(1, h // 6)
    transitions = 0

    for row in range(0, h - strip_h, strip_h):
        y_strip = yellow_mask[row:row + strip_h, :]
        b_strip = blue_mask[row:row + strip_h,   :]

        y_cols = (y_strip.sum(axis=0) > strip_h * 60).astype(np.uint8)
        b_cols = (b_strip.sum(axis=0) > strip_h * 60).astype(np.uint8)

        col_w = max(1, w // 12)
        prev  = None
        for col in range(0, w, col_w):
            y_block = y_cols[col:col + col_w].sum()
            b_block = b_cols[col:col + col_w].sum()
            dominant = "y" if y_block > b_block else ("b" if b_block > y_block else None)
            if dominant and dominant != prev:
                transitions += 1
                prev = dominant

    # Scale required blocks to crop size — small crops need fewer transitions
    required = max(2, min(CHECKER_BLOCKS, w // 20))

    detected   = transitions >= required
    confidence = min(1.0, (yellow_frac + blue_frac) * 5 + transitions * 0.05)

    return detected, round(confidence, 3)


def detect_police_text(crop: np.ndarray) -> tuple[bool, str]:
    """
    Run OCR on a vehicle crop and look for POLICE text.
    Returns (detected: bool, found_text: str).
    """
    # Upscale small crops — OCR works poorly below ~100px height
    h, w = crop.shape[:2]
    scale = max(1, 100 // h)
    if scale > 1:
        crop = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # Pre-process: greyscale + threshold to improve OCR on vehicle livery
    grey  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Try both light-on-dark and dark-on-light
    results = set()
    for thresh_img in [
        cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY     + cv2.THRESH_OTSU)[1],
        cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
    ]:
        text = pytesseract.image_to_string(thresh_img, config=TESS_CONFIG).upper()
        for word in text.split():
            word = word.strip(".,;:-|()[]")
            if word in POLICE_KEYWORDS:
                results.add(word)

    detected = len(results) > 0
    return detected, ", ".join(results)


def is_police_vehicle(crop: np.ndarray) -> dict:
    """
    Run both heuristics on a vehicle crop.
    Returns a result dict.
    """
    batt_detected, batt_conf = detect_battenburg(crop)
    text_detected, text_found = detect_police_text(crop)

    is_police = batt_detected or text_detected
    confidence = max(batt_conf, 0.9 if text_detected else 0.0)

    return {
        "is_police":       is_police,
        "confidence":      confidence,
        "battenburg":      batt_detected,
        "batt_confidence": batt_conf,
        "text_detected":   text_detected,
        "text_found":      text_found,
    }


def save_crop(crop: np.ndarray, result: dict, camera_id: str = "unknown") -> str:
    """
    Save a vehicle crop to the appropriate dataset folder.
    Filename encodes camera + timestamp + content hash (avoids duplicates).
    Returns the saved filepath.
    """
    h = hashlib.md5(crop.tobytes()).hexdigest()[:8]
    ts = int(time.time())
    fname = f"{camera_id}_{ts}_{h}.jpg"

    folder = POLICE_DIR if result["is_police"] else VEHICLE_DIR
    path   = os.path.join(folder, fname)

    # Add a small label overlay for human review
    annotated = crop.copy()
    label = "POLICE" if result["is_police"] else "vehicle"
    color = (0, 0, 255) if result["is_police"] else (0, 200, 80)
    cv2.putText(annotated, label, (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    cv2.imwrite(path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path
