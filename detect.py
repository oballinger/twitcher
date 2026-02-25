from ultralytics import YOLO
import cv2
import numpy as np
import requests
import traceback
import time
from flask import Flask, request, Response, jsonify

app = Flask(__name__)

model = YOLO("yolo11n.pt")

VEHICLES = ["car", "truck", "bus", "motorcycle"]

LABEL_COLORS = {
    "car":        (0, 255, 136),
    "truck":      (0, 170, 255),
    "bus":        (136, 68, 255),
    "motorcycle": (255, 100, 0),
}

@app.route("/")
def home():
    return "Detection server running"


def draw_boxes(frame, vehicles):
    """Draw bounding boxes + labels onto a frame in-place."""
    for v in vehicles:
        x1, y1, x2, y2 = v["box"]
        label = v["label"]
        conf  = v["confidence"]
        color = LABEL_COLORS.get(label, (255, 255, 255))

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # Corner ticks
        tick = max(6, int(min(x2 - x1, y2 - y1) * 0.15))
        for cx, cy, sx, sy in [
            (x1, y1,  1,  1),
            (x2, y1, -1,  1),
            (x1, y2,  1, -1),
            (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (cx, cy), (cx + sx * tick, cy), color, 1)
            cv2.line(frame, (cx, cy), (cx, cy + sy * tick), color, 1)

        # Label pill background
        text  = f"{label.upper()} {conf:.0%}"
        scale = 0.3
        thick = 1
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        pill_y1 = max(0, y1 - th - 5)
        cv2.rectangle(frame, (x1, pill_y1), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            frame, text,
            (x1 + 3, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA
        )

    # Vehicle count overlay (top-right)
    count = len(vehicles)
    count_text = f"VEHICLES: {count}"
    (cw, ch), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    fw = frame.shape[1]
    rx2 = fw - 6
    rx1 = rx2 - cw - 10
    cv2.rectangle(frame, (rx1, 6), (rx2, ch + 14), (0, 0, 0), -1)
    cv2.putText(
        frame, count_text,
        (rx1 + 5, ch + 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 136), 1, cv2.LINE_AA
    )

    return frame


def generate_stream(video_url, target_fps=10):
    """
    Open a video URL, run YOLO on every frame, yield annotated JPEG frames
    as a multipart MJPEG stream.
    """
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        # Yield a single error frame
        err_frame = np.zeros((240, 426, 3), dtype=np.uint8)
        cv2.putText(err_frame, "Cannot open stream", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        _, buf = cv2.imencode(".jpg", err_frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        return

    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 25
    # Only run YOLO every N source frames to keep up
    skip     = max(1, int(src_fps / target_fps))
    interval = 1.0 / target_fps
    frame_i  = 0
    last_vehicles = []
    last_sent = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            # Loop: rewind to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        # Run YOLO only on every `skip`-th frame
        if frame_i % skip == 0:
            results = model(frame, verbose=False)[0]
            last_vehicles = []
            for box in results.boxes:
                cls   = int(box.cls[0])
                label = model.names[cls]
                conf  = float(box.conf[0])
                if label not in VEHICLES:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                last_vehicles.append({
                    "label": label, "confidence": conf, "box": [x1, y1, x2, y2]
                })

        # Always draw last known boxes (smooth appearance between detections)
        annotated = draw_boxes(frame.copy(), last_vehicles)

        # Throttle output to target_fps
        now = time.time()
        elapsed = now - last_sent
        if elapsed < interval:
            time.sleep(interval - elapsed)

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")

        last_sent = time.time()
        frame_i  += 1

    cap.release()


@app.route("/stream")
def stream():
    """
    GET /stream?videoUrl=<url>&fps=10
    Returns an MJPEG stream of annotated frames.
    """
    video_url  = request.args.get("videoUrl")
    target_fps = int(request.args.get("fps", 10))

    if not video_url:
        return "missing videoUrl", 400

    return Response(
        generate_stream(video_url, target_fps),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.route("/detect", methods=["POST"])
def detect():
    """Single-frame JSON detection (kept for compatibility)."""
    try:
        data = request.get_json()
        url  = data.get("videoUrl") or data.get("imageUrl")
        if not url:
            return jsonify({"error": "missing videoUrl"}), 400

        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            return jsonify({"error": f"Cannot open: {url}"}), 500

        for _ in range(5):
            cap.grab()
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return jsonify({"error": "No frame read"}), 500

        results  = model(frame, verbose=False)[0]
        vehicles = []
        for box in results.boxes:
            cls   = int(box.cls[0])
            label = model.names[cls]
            conf  = float(box.conf[0])
            if label not in VEHICLES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vehicles.append({"label": label, "confidence": round(conf, 3), "box": [x1, y1, x2, y2]})

        h, w = frame.shape[:2]
        return jsonify({"count": len(vehicles), "vehicles": vehicles, "frameWidth": w, "frameHeight": h})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # threaded=True so multiple popup streams can run concurrently
    app.run(host="0.0.0.0", port=5000, threaded=True)