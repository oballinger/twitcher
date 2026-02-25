from ultralytics import YOLO
import cv2
import numpy as np
import traceback
import time
from flask import Flask, request, Response, jsonify
from police_detector import is_police_vehicle

app = Flask(__name__)

model = YOLO("yolo11l.pt")

VEHICLES = ["car", "truck", "bus", "motorcycle"]

LABEL_COLORS = {
    "car":        (136, 255, 0),
    "truck":      (255, 170, 0),
    "bus":        (255, 68, 136),
    "motorcycle": (0, 100, 255),
}
POLICE_COLOR = (0, 0, 255)


@app.route("/")
def home():
    return "Detection server running"


def draw_boxes(frame, vehicles):
    for v in vehicles:
        x1, y1, x2, y2 = v["box"]
        label     = v["label"]
        conf      = v["confidence"]
        is_police = v.get("is_police", False)
        color     = POLICE_COLOR if is_police else LABEL_COLORS.get(label, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        tick = max(6, int(min(x2 - x1, y2 - y1) * 0.15))
        for cx, cy, sx, sy in [
            (x1, y1,  1,  1), (x2, y1, -1,  1),
            (x1, y2,  1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(frame, (cx, cy), (cx + sx * tick, cy), color, 1)
            cv2.line(frame, (cx, cy), (cx, cy + sy * tick), color, 1)

        display_label = "POLICE" if is_police else label.upper()
        text  = f"{display_label} {conf:.0%}"
        scale = 0.3
        thick = 1
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        pill_y1 = max(0, y1 - th - 5)
        cv2.rectangle(frame, (x1, pill_y1), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)

    police_count  = sum(1 for v in vehicles if v.get("is_police"))
    vehicle_count = len(vehicles)
    count_text    = f"VEHICLES: {vehicle_count}"
    if police_count:
        count_text += f"  POLICE: {police_count}"

    (cw, ch), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    fw  = frame.shape[1]
    rx2 = fw - 6
    rx1 = rx2 - cw - 10
    cv2.rectangle(frame, (rx1, 6), (rx2, ch + 14), (0, 0, 0), -1)
    text_color = POLICE_COLOR if police_count else (0, 255, 136)
    cv2.putText(frame, count_text, (rx1 + 5, ch + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)

    return frame


def run_detection(frame):
    results  = model(frame, verbose=False)[0]
    vehicles = []

    for box in results.boxes:
        cls   = int(box.cls[0])
        label = model.names[cls]
        conf  = float(box.conf[0])
        if label not in VEHICLES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)

        crop   = frame[y1:y2, x1:x2]
        police = is_police_vehicle(crop) if crop.size > 0 else {}

        vehicles.append({
            "label":       label,
            "confidence":  round(conf, 3),
            "box":         [x1, y1, x2, y2],
            "is_police":   police.get("is_police", False),
            "battenburg":  police.get("battenburg", False),
            "text_found":  police.get("text_found", ""),
            "police_conf": police.get("confidence", 0.0),
        })

    return vehicles


def generate_stream(video_url, target_fps=10):
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        err_frame = np.zeros((240, 426, 3), dtype=np.uint8)
        cv2.putText(err_frame, "Cannot open stream", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        _, buf = cv2.imencode(".jpg", err_frame)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        return

    src_fps       = cap.get(cv2.CAP_PROP_FPS) or 25
    skip          = max(1, int(src_fps / target_fps))
    interval      = 1.0 / target_fps
    frame_i       = 0
    last_vehicles = []
    last_sent     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        if frame_i % skip == 0:
            last_vehicles = run_detection(frame)

        annotated = draw_boxes(frame.copy(), last_vehicles)

        now     = time.time()
        elapsed = now - last_sent
        if elapsed < interval:
            time.sleep(interval - elapsed)

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

        last_sent = time.time()
        frame_i  += 1

    cap.release()


@app.route("/stream")
def stream():
    video_url  = request.args.get("videoUrl")
    target_fps = int(request.args.get("fps", 10))
    if not video_url:
        return "missing videoUrl", 400
    return Response(
        generate_stream(video_url, target_fps),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"}
    )


@app.route("/detect", methods=["POST"])
def detect():
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

        vehicles = run_detection(frame)
        h, w     = frame.shape[:2]
        return jsonify({
            "count":        len(vehicles),
            "police_count": sum(1 for v in vehicles if v["is_police"]),
            "vehicles":     vehicles,
            "frameWidth":   w,
            "frameHeight":  h,
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
