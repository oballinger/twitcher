"""
collect.py

Standalone data collection pipeline.
- Fetches all available TfL JamCam feeds
- Runs YOLO on every frame across all cameras concurrently
- Saves clean (unannotated) vehicle crops to dataset/<label>/
- Only saves crops with detection confidence > 60%
- Applies police heuristics and saves to dataset/police/ if flagged

Usage:
    python collect.py                        # run forever
    python collect.py --passes 3             # 3 full passes over all cameras
    python collect.py --workers 8            # concurrent camera threads (default 4)
    python collect.py --fps 5               # frames per second to sample (default 5)
    python collect.py --dataset ./dataset   # output dir (default ./dataset)
"""

import os
import cv2
import time
import hashlib
import argparse
import threading
import requests
import numpy as np
import torch
from datetime import datetime
from ultralytics import YOLO
from police_detector import is_police_vehicle

if torch.cuda.is_available():
    _device = "cuda"
elif torch.backends.mps.is_available():
    _device = "mps"
else:
    _device = "cpu"

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--passes",  type=int,   default=0,           help="Number of passes (0 = infinite)")
parser.add_argument("--workers", type=int,   default=4,           help="Concurrent camera threads")
parser.add_argument("--fps",     type=float, default=5,           help="Frames per second to sample")
parser.add_argument("--dataset", type=str,   default="./dataset", help="Output dataset directory")
parser.add_argument("--tfl-key", type=str,   default=os.environ.get("TFL_KEY", ""), help="TfL API key")
parser.add_argument("--types",   type=str,   nargs="+",
                    default=["police"],
                    choices=["car", "truck", "bus", "motorcycle", "police"],
                    help="Vehicle types to collect (default: police)")
args = parser.parse_args()

CONF_THRESHOLD = 0.60
DATASET_DIR    = args.dataset
TFL_KEY        = args.tfl_key
COLLECT_TYPES  = set(args.types)

# ── Setup dataset folders ─────────────────────────────────────────────────────
LABELS = sorted(COLLECT_TYPES)
for label in LABELS:
    os.makedirs(os.path.join(DATASET_DIR, label), exist_ok=True)

# ── Shared stats (thread-safe via lock) ───────────────────────────────────────
stats_lock  = threading.Lock()
stats = {label: 0 for label in LABELS}
stats["skipped"] = 0

# Per-class subsampling: max crops saved per class per pass through a camera.
# Keeps the dataset balanced — cars are ~8x more common than buses.
# Police has no cap (every hit is precious).
CLASS_CAP = {
    "car":        5,    # heavily overseen — strict cap
    "truck":      15,
    "bus":        20,
    "motorcycle": 50,   # rare — save everything above threshold
    "police":     None, # no cap
}

# Per-camera per-class counters, reset each camera pass
# Structure: { camera_id: { label: count } }
camera_counts      = {}
camera_counts_lock = threading.Lock()

# Single shared YOLO model — GPU inference serialised via lock.
# Workers parallelise network I/O and video decode; GPU runs one batch at a time.
_model      = None
_model_lock = threading.Lock()

def get_model():
    global _model
    if _model is None:
        _model = YOLO("yolo11l.pt")
        _model.to(_device)
    return _model


# ── Helpers ───────────────────────────────────────────────────────────────────
def crop_hash(crop: np.ndarray) -> str:
    return hashlib.md5(crop.tobytes()).hexdigest()[:10]


def save_crop(crop: np.ndarray, label: str, camera_id: str, conf: float = 1.0):
    """Save a clean (unannotated) crop to dataset/<label>/."""
    ts    = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    h     = crop_hash(crop)
    conf_pct = int(round(conf * 100))
    fname = f"{camera_id}_{ts}_{conf_pct}pct_{h}.jpg"
    path  = os.path.join(DATASET_DIR, label, fname)
    cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

    with stats_lock:
        stats[label] = stats.get(label, 0) + 1


def process_frame(frame: np.ndarray, camera_id: str):
    """Run YOLO + police heuristics on a frame, save qualifying crops."""
    with _model_lock:
        model   = get_model()
        results = model(frame, verbose=False, device=_device)[0]
        names   = model.names

    for box in results.boxes:
        cls   = int(box.cls[0])
        label = names[cls]
        conf  = float(box.conf[0])

        if label not in ["car", "truck", "bus", "motorcycle"]:
            continue

        if conf < CONF_THRESHOLD:
            with stats_lock:
                stats["skipped"] += 1
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Police check overrides the vehicle label (skip if not collecting police)
        if "police" in COLLECT_TYPES:
            police = is_police_vehicle(crop)
            save_label = "police" if police.get("is_police") else label
        else:
            save_label = label

        # Skip labels we're not collecting
        if save_label not in COLLECT_TYPES:
            with stats_lock:
                stats["skipped"] += 1
            continue

        # Subsample: check per-camera per-class cap
        cap_val = CLASS_CAP.get(save_label)
        if cap_val is not None:
            with camera_counts_lock:
                count = camera_counts.get(camera_id, {}).get(save_label, 0)
                if count >= cap_val:
                    with stats_lock:
                        stats["skipped"] += 1
                    continue
                camera_counts.setdefault(camera_id, {})[save_label] = count + 1

        save_crop(crop, save_label, camera_id, conf)


def collect_camera(cam: dict, target_fps: float, passes: int):
    """
    Open a camera's video feed, sample at target_fps, run detection.
    Loops `passes` times (0 = infinite).
    """
    video_url = cam["videoUrl"]
    camera_id = cam["id"].replace("/", "_").replace(".", "_")
    pass_n    = 0

    print(f"  [+] Starting {cam['name']} ({camera_id})")

    while passes == 0 or pass_n < passes:
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            print(f"  [!] Cannot open {camera_id} — skipping")
            break

        src_fps  = cap.get(cv2.CAP_PROP_FPS) or 10
        skip     = max(1, int(src_fps / target_fps))
        frame_i  = 0
        interval = 1.0 / target_fps
        last     = 0

        # Reset per-class counter for this camera pass
        with camera_counts_lock:
            camera_counts[camera_id] = {label: 0 for label in COLLECT_TYPES}

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # end of clip → next pass

            if frame_i % skip == 0:
                process_frame(frame, camera_id)

                now = time.time()
                if now - last < interval:
                    time.sleep(interval - (now - last))
                last = time.time()

            frame_i += 1

        cap.release()
        pass_n += 1

    print(f"  [-] Done {camera_id}")


def fetch_cameras(tfl_key: str) -> list:
    url  = f"https://api.tfl.gov.uk/Place/Type/JamCam?app_key={tfl_key}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    raw = resp.json()

    cameras = []
    for cam in raw:
        props = {p["key"]: p["value"] for p in cam.get("additionalProperties", [])}
        if props.get("available") != "true":
            continue
        if not props.get("videoUrl"):
            continue
        cameras.append({
            "id":       cam["id"],
            "name":     cam["commonName"],
            "videoUrl": props["videoUrl"],
        })

    return cameras


def print_stats():
    with stats_lock:
        total = sum(v for k, v in stats.items() if k != "skipped")
        parts = "  ".join(f"{k}={stats[k]}" for k in LABELS)
        print(f"\r  Saved: {parts}  |  total={total}  skipped={stats['skipped']}",
              end="", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not TFL_KEY:
        print("[err] TFL_KEY not set. Export it or pass --tfl-key <key>")
        exit(1)

    print(f"[twitcher] Loading YOLO model… (device: {_device})")
    get_model()
    print(f"[twitcher] Model on {next(get_model().model.parameters()).device}")

    print("[twitcher] Fetching camera list…")
    cameras = fetch_cameras(TFL_KEY)
    print(f"[twitcher] {len(cameras)} cameras available\n")

    print(f"[twitcher] Starting collection")
    print(f"           workers : {args.workers}")
    print(f"           fps     : {args.fps}")
    print(f"           passes  : {args.passes or '∞'}")
    print(f"           conf    : >{CONF_THRESHOLD:.0%}")
    print(f"           types   : {', '.join(LABELS)}")
    print(f"           device  : {_device}")
    print(f"           dataset : {DATASET_DIR}\n")

    semaphore = threading.Semaphore(args.workers)
    threads   = []

    def run_camera(cam):
        with semaphore:
            collect_camera(cam, args.fps, args.passes)

    for cam in cameras:
        t = threading.Thread(target=run_camera, args=(cam,), daemon=True)
        t.start()
        threads.append(t)

    # Print stats every 5 seconds
    try:
        while any(t.is_alive() for t in threads):
            print_stats()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n[twitcher] Interrupted")

    print_stats()
    print("\n[twitcher] Collection complete")