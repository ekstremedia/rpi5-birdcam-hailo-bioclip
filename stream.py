#!/usr/bin/env python3
"""HTTP MJPEG streaming server with Hailo AI detection and custom labels."""

import os
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import Hailo

PORT = 8888
MODEL_PATH = "/usr/share/hailo-models/yolov8s_h8.hef"
CONFIDENCE_THRESHOLD = 0.5

# COCO class labels - customize any of these!
LABELS = [
    "idiot", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Colors per class (will cycle)
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (128, 0, 255), (0, 128, 255), (255, 0, 128), (0, 255, 128),
]

# Shared state
latest_frame = None
frame_lock = threading.Lock()
frame_event = threading.Event()


def extract_detections(output, w, h):
    """Parse YOLOv8 NMS output: list of 80 arrays, each (N, 5) -> detections."""
    results = []
    for class_id, dets in enumerate(output):
        if dets.shape[0] == 0:
            continue
        for det in dets:
            score = det[4]
            if score < CONFIDENCE_THRESHOLD:
                continue
            y0 = int(det[0] * h)
            x0 = int(det[1] * w)
            y1 = int(det[2] * h)
            x1 = int(det[3] * w)
            label = LABELS[class_id] if class_id < len(LABELS) else f"class_{class_id}"
            color = COLORS[class_id % len(COLORS)]
            results.append((label, score, (x0, y0, x1, y1), color))
    return results


def draw_detections(frame, detections):
    """Draw bounding boxes and labels on the frame."""
    for label, score, (x0, y0, x1, y1), color in detections:
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
        text = f"{label} {score:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x0, y0 - th - 8), (x0 + tw + 4, y0), color, -1)
        cv2.putText(frame, text, (x0 + 2, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def camera_loop():
    """Main camera + inference loop."""
    global latest_frame

    hailo = Hailo(MODEL_PATH)
    model_h, model_w, _ = hailo.get_input_shape()

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (1280, 960), "format": "RGB888"},
        lores={"size": (model_w, model_h), "format": "RGB888"},
        buffer_count=4,
    )
    picam2.configure(config)
    picam2.start()
    print("Camera started", flush=True)

    try:
        while True:
            (main_frame, lores_frame), metadata = picam2.capture_arrays(
                ["main", "lores"])

            # picamera2 RGB888 is already BGR in memory
            main_bgr = main_frame

            # Run inference on the low-res frame
            lores_input = np.ascontiguousarray(lores_frame)
            results = hailo.run(lores_input)

            # Parse detections and draw
            h, w = main_bgr.shape[:2]
            detections = extract_detections(results, w, h)
            draw_detections(main_bgr, detections)

            _, jpeg = cv2.imencode('.jpg', main_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

            with frame_lock:
                latest_frame = jpeg.tobytes()
            frame_event.set()
    finally:
        picam2.stop()
        hailo.close()


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""<!DOCTYPE html>
<html><head><title>Bird Cam</title></head>
<body style="margin:0;background:#000;display:flex;align-items:center;justify-content:center;height:100vh">
<img src="/stream" style="max-width:100%;max-height:100vh">
</body></html>""")
            return

        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            try:
                while True:
                    frame_event.wait(timeout=1.0)
                    with frame_lock:
                        frame = latest_frame
                    if frame is None:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(frame)}\r\n".encode())
                    self.wfile.write(b"\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    time.sleep(0.033)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

    print("Starting camera with Hailo AI...", flush=True)
    frame_event.wait(timeout=30)
    if latest_frame:
        print(f"First frame received ({len(latest_frame)} bytes)", flush=True)
    else:
        print("WARNING: No frames received yet", flush=True)

    server = ThreadedHTTPServer(("0.0.0.0", PORT), StreamHandler)
    print(f"Stream ready at http://0.0.0.0:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
