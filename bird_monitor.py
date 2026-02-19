#!/usr/bin/env python3
"""Bird Feeder Monitor - Pi 5 + Hailo-8 AI HAT+ bird detection and counting system.

Captures video from picamera2, runs YOLOv8s on Hailo-8 for object detection,
filters for birds (COCO class 14), tracks them across frames, logs visits
to SQLite, saves crop images, and serves a live web dashboard.

Usage:
    python3 bird_monitor.py

Then open http://<PI_IP>:8888 in a browser.
"""

import cv2
import json
import numpy as np
import os
import sqlite3
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime, date
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

from picamera2 import Picamera2
from picamera2.devices import Hailo

from species_classifier import SpeciesClassifier

# ============================================================
# Configuration
# ============================================================

PORT = 8888
MODEL_PATH = "/usr/share/hailo-models/yolov8s_h8.hef"
BIRD_CLASS_ID = 14  # COCO class index for "bird"
CONFIDENCE_THRESHOLD = 0.4
CROP_DIR = "/home/pi/ai/bird_crops"
DB_PATH = "/home/pi/ai/birds.db"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 960
MODEL_INPUT_SIZE = 640

# Tracker settings
MAX_DISAPPEARED = 15   # frames before a bird is considered "gone" (~1s at 15fps)
MAX_DISTANCE = 150     # max pixel distance to match same bird across frames
CROP_SAVE_INTERVAL = 3.0  # seconds between crop saves for the same tracked bird

# Species classifier (Phase 2 — BioCLIP zero-shot)
SPECIES_LIST_PATH = "/home/pi/ai/models/norwegian_species.txt"
SPECIES_RELOAD_INTERVAL = 60  # seconds between hot-reload checks

# COCO class names (80 classes, index 14 = bird)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
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


# ============================================================
# SQLite Database
# ============================================================

class BirdDB:
    """Thread-safe SQLite database for logging bird visits and detections."""

    def __init__(self, db_path):
        self.db_path = db_path
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self):
        """Get a thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_schema(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                species TEXT,
                max_confidence REAL,
                crop_path TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_visits_start ON visits(start_time);

            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                class_id INTEGER NOT NULL,
                confidence REAL NOT NULL,
                bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
                crop_path TEXT,
                track_id INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_det_ts ON detections(timestamp);
        """)
        conn.commit()
        conn.close()

    def log_visit_start(self, track_id, timestamp, confidence, crop_path=None, species=None):
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO visits (track_id, start_time, max_confidence, crop_path, species) "
            "VALUES (?, ?, ?, ?, ?)",
            (track_id, timestamp, confidence, crop_path, species),
        )
        conn.commit()

    def log_visit_end(self, track_id, timestamp):
        conn = self._get_conn()
        conn.execute(
            "UPDATE visits SET end_time = ? WHERE track_id = ? AND end_time IS NULL",
            (timestamp, track_id),
        )
        conn.commit()

    def get_today_count(self):
        conn = self._get_conn()
        today = date.today().isoformat()
        row = conn.execute(
            "SELECT COUNT(*) FROM visits WHERE start_time >= ?", (today,)
        ).fetchone()
        return row[0] if row else 0

    def get_recent_visits(self, limit=20):
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM visits ORDER BY start_time DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# ============================================================
# Centroid Tracker
# ============================================================

class CentroidTracker:
    """Simple centroid-based object tracker for counting bird visits.

    Matches detections across frames by minimizing centroid distance.
    Tracks arrivals (new birds) and departures (birds gone for N frames).
    """

    def __init__(self, max_disappeared=15, max_distance=150):
        self.next_id = 0
        self.objects = OrderedDict()       # track_id -> (cx, cy)
        self.bboxes = OrderedDict()        # track_id -> (x1, y1, x2, y2)
        self.confidences = OrderedDict()   # track_id -> best confidence
        self.disappeared = OrderedDict()   # track_id -> frames since last seen
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        # Per-frame events
        self.arrivals = []    # track_ids that appeared this frame
        self.departures = []  # track_ids that disappeared this frame
        self.total_arrivals = 0

    def _register(self, centroid, bbox, confidence):
        tid = self.next_id
        self.objects[tid] = centroid
        self.bboxes[tid] = bbox
        self.confidences[tid] = confidence
        self.disappeared[tid] = 0
        self.next_id += 1
        self.total_arrivals += 1
        self.arrivals.append(tid)
        return tid

    def _deregister(self, tid):
        self.departures.append(tid)
        del self.objects[tid]
        del self.bboxes[tid]
        del self.confidences[tid]
        del self.disappeared[tid]

    def update(self, detections):
        """Update with new frame's detections. Returns current tracked objects.

        Each detection is a dict with x_min, y_min, x_max, y_max, confidence.
        """
        self.arrivals = []
        self.departures = []

        # No detections: increment disappeared for all
        if len(detections) == 0:
            for tid in list(self.disappeared.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    self._deregister(tid)
            return self.objects

        # Compute centroids of input detections
        input_centroids = []
        input_bboxes = []
        input_confs = []
        for d in detections:
            cx = (d["x_min"] + d["x_max"]) / 2.0
            cy = (d["y_min"] + d["y_max"]) / 2.0
            input_centroids.append((cx, cy))
            input_bboxes.append((d["x_min"], d["y_min"], d["x_max"], d["y_max"]))
            input_confs.append(d["confidence"])

        # First detections ever: register all
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self._register(input_centroids[i], input_bboxes[i], input_confs[i])
            return self.objects

        # Compute distance matrix between existing objects and new detections
        obj_ids = list(self.objects.keys())
        obj_centroids = list(self.objects.values())

        D = np.zeros((len(obj_centroids), len(input_centroids)))
        for i, oc in enumerate(obj_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = np.sqrt((oc[0] - ic[0]) ** 2 + (oc[1] - ic[1]) ** 2)

        # Greedy matching: sort by distance, match closest first
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue

            tid = obj_ids[row]
            self.objects[tid] = input_centroids[col]
            self.bboxes[tid] = input_bboxes[col]
            self.confidences[tid] = max(self.confidences[tid], input_confs[col])
            self.disappeared[tid] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Unmatched existing objects: increment disappeared
        for row in range(len(obj_centroids)):
            if row not in used_rows:
                tid = obj_ids[row]
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    self._deregister(tid)

        # Unmatched new detections: register as new
        for col in range(len(input_centroids)):
            if col not in used_cols:
                self._register(input_centroids[col], input_bboxes[col], input_confs[col])

        return self.objects


# ============================================================
# Bird Monitor (core processing)
# ============================================================

class BirdMonitor:
    """Main bird detection and monitoring pipeline.

    Captures frames from picamera2, runs YOLOv8 on Hailo-8, filters for birds,
    tracks them, logs to DB, saves crops, and provides frames for the web UI.
    """

    def __init__(self):
        self.running = False
        self.latest_frame_jpeg = None
        self.frame_lock = threading.Lock()
        self.frame_event = threading.Event()

        self.current_bird_count = 0
        self.today_visits = 0
        self.fps = 0.0

        self.tracker = CentroidTracker(
            max_disappeared=MAX_DISAPPEARED, max_distance=MAX_DISTANCE
        )
        self.db = BirdDB(DB_PATH)
        self.today_visits = self.db.get_today_count()

        # Crop save rate limiting
        self.last_crop_time = {}  # track_id -> timestamp

        # Species classifier (loaded lazily — graceful degradation if missing)
        self.classifier = None
        self.species_labels = {}  # track_id -> (species, confidence)
        self.last_reload_check = 0
        try:
            self.classifier = SpeciesClassifier(SPECIES_LIST_PATH)
        except (FileNotFoundError, ImportError, ValueError) as e:
            print(f"  Artsklassifisering ikke tilgjengelig: {e}", flush=True)
            print("  Kjører uten artsklassifisering (fugler vises som 'Fugl')", flush=True)

        os.makedirs(CROP_DIR, exist_ok=True)

    def start(self):
        """Initialize hardware and start the processing loop."""
        self.running = True

        # Initialize Hailo
        print("Starter Hailo-modell...", flush=True)
        try:
            self.hailo = Hailo(MODEL_PATH)
        except Exception as e:
            print(f"FEIL: Kunne ikke starte Hailo: {e}", file=sys.stderr)
            print("Kjører stream.py eller en annen Hailo-prosess fortsatt?", file=sys.stderr)
            print("Stopp den med: pkill -f stream.py", file=sys.stderr)
            sys.exit(1)

        input_shape = self.hailo.get_input_shape()
        inputs_info, outputs_info = self.hailo.describe()
        print(f"  Model input:  {input_shape}")
        for name, shape, fmt in outputs_info:
            print(f"  Model output: {name} shape={shape} format={fmt}")

        # Initialize camera
        print("Starter kamera...", flush=True)
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1)  # let auto-exposure settle
        print(f"  Kamera klart: {FRAME_WIDTH}x{FRAME_HEIGHT}", flush=True)

        # Start processing in background thread
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        print(f"Behandlingsløkke startet. Dashbord på http://0.0.0.0:{PORT}", flush=True)

    def stop(self):
        """Clean shutdown - waits for processing thread to finish first."""
        self.running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=5)
        if hasattr(self, "picam2"):
            try:
                self.picam2.stop()
            except Exception:
                pass
        if hasattr(self, "hailo"):
            try:
                self.hailo.close()
            except Exception:
                pass
        if self.classifier:
            self.classifier.close()
        print("Fuglevakt stoppet.", flush=True)

    # ---- Main processing loop ----

    def _process_loop(self):
        frame_count = 0
        fps_start = time.time()
        first_frame = True

        while self.running:
            try:
                # 1. Capture frame (RGB, shape: H x W x 3)
                frame_rgb = self.picam2.capture_array("main")

                # 2. Prepare model input: resize to 640x640
                model_input = cv2.resize(frame_rgb, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))

                # 3. Run Hailo inference
                raw_output = self.hailo.run(model_input)

                # Debug: print output info on first frame
                if first_frame:
                    self._debug_output(raw_output)
                    first_frame = False

                # 4. Parse detections
                all_detections = self._parse_detections(raw_output)
                bird_detections = [
                    d for d in all_detections if d["class_id"] == BIRD_CLASS_ID
                ]

                # 5. Update tracker
                self.tracker.update(bird_detections)
                self.current_bird_count = len(self.tracker.objects)

                # 6. Handle arrivals and departures
                now_str = datetime.now().isoformat(timespec="seconds")

                for tid in self.tracker.arrivals:
                    conf = self.tracker.confidences.get(tid, 0)
                    bbox = self.tracker.bboxes.get(tid)
                    crop_path = self._save_crop(frame_rgb, bbox, tid)

                    # Species classification on arrival
                    species = None
                    species_conf = 0.0
                    if self.classifier and bbox:
                        species, species_conf = self._classify_bird(frame_rgb, bbox)
                        if species:
                            self.species_labels[tid] = (species, species_conf)

                    self.db.log_visit_start(tid, now_str, conf, crop_path, species=species)
                    self.today_visits = self.db.get_today_count()

                    species_str = f" art={species} ({species_conf:.0%})" if species else ""
                    print(
                        f"  [{now_str}] Fugl ankommet! spor={tid} konf={conf:.2f}{species_str}",
                        flush=True,
                    )

                for tid in self.tracker.departures:
                    self.db.log_visit_end(tid, now_str)
                    self.species_labels.pop(tid, None)
                    print(f"  [{now_str}] Fugl forlot. spor={tid}", flush=True)

                # Periodic hot-reload check for species model
                now_time = time.time()
                if self.classifier and now_time - self.last_reload_check > SPECIES_RELOAD_INTERVAL:
                    self.last_reload_check = now_time
                    self.classifier.check_reload()

                # 7. Draw overlays and encode JPEG
                # picamera2 "RGB888" actually delivers BGR on Pi 5, so no conversion needed
                display = frame_rgb
                self._draw_overlays(display, bird_detections, all_detections)
                _, jpeg_buf = cv2.imencode(
                    ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )

                with self.frame_lock:
                    self.latest_frame_jpeg = jpeg_buf.tobytes()
                self.frame_event.set()

                # 8. FPS tracking
                frame_count += 1
                elapsed = time.time() - fps_start
                if elapsed >= 2.0:
                    self.fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = time.time()

            except Exception as e:
                if not self.running:
                    break  # expected during shutdown
                import traceback

                print(f"Feil i behandlingsløkke: {e}", flush=True)
                traceback.print_exc()
                time.sleep(0.5)

    # ---- Detection parsing ----

    def _debug_output(self, raw_output):
        """Print raw output details on first inference (for debugging tensor format)."""
        if isinstance(raw_output, list):
            # List of per-class detection arrays: [(N,5), (M,5), ...]
            print(f"  Output: list of {len(raw_output)} class arrays (NMS by class)")
            for i, arr in enumerate(raw_output):
                if isinstance(arr, np.ndarray) and arr.size > 0:
                    name = COCO_CLASSES[i] if i < len(COCO_CLASSES) else str(i)
                    print(f"    class {i} ({name}): {arr.shape[0]} detections")
                    for j in range(min(2, arr.shape[0])):
                        vals = arr[j]
                        print(f"      det[{j}]: y_min={vals[0]:.3f} x_min={vals[1]:.3f} "
                              f"y_max={vals[2]:.3f} x_max={vals[3]:.3f} conf={vals[4]:.3f}")
        elif isinstance(raw_output, dict):
            for name, arr in raw_output.items():
                print(f"  Output '{name}': shape={arr.shape} dtype={arr.dtype}")
        elif isinstance(raw_output, np.ndarray):
            print(f"  Output: shape={raw_output.shape} dtype={raw_output.dtype}")
        else:
            print(f"  Output type: {type(raw_output)}")

    def _parse_detections(self, raw_output):
        """Parse YOLOv8 NMS output from Hailo into a list of detection dicts.

        The Hailo wrapper returns a list of 80 numpy arrays (one per COCO class).
        Each array has shape (N, 5) where N = number of detections for that class.
        The 5 values per detection are: [y_min, x_min, y_max, x_max, confidence]
        All coordinates are normalized to [0, 1].
        """
        detections = []

        if isinstance(raw_output, list):
            # List of per-class arrays: output[cls_id] = array of shape (N, 5)
            for cls_id, cls_dets in enumerate(raw_output):
                if not isinstance(cls_dets, np.ndarray) or cls_dets.size == 0:
                    continue
                for row in cls_dets:
                    if len(row) < 5:
                        continue
                    confidence = float(row[4])
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue
                    y_min, x_min, y_max, x_max = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    detections.append({
                        "class_id": cls_id,
                        "class_name": COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id),
                        "confidence": confidence,
                        "x_min": x_min * FRAME_WIDTH,
                        "y_min": y_min * FRAME_HEIGHT,
                        "x_max": x_max * FRAME_WIDTH,
                        "y_max": y_max * FRAME_HEIGHT,
                    })

        elif isinstance(raw_output, np.ndarray) and raw_output.ndim == 3:
            # Fallback: single tensor of shape (80, 5, 100)
            num_classes, _, max_dets = raw_output.shape
            for cls in range(num_classes):
                for det in range(max_dets):
                    confidence = float(raw_output[cls, 4, det])
                    if confidence < CONFIDENCE_THRESHOLD:
                        continue
                    y_min = float(raw_output[cls, 0, det])
                    x_min = float(raw_output[cls, 1, det])
                    y_max = float(raw_output[cls, 2, det])
                    x_max = float(raw_output[cls, 3, det])
                    detections.append({
                        "class_id": cls,
                        "class_name": COCO_CLASSES[cls] if cls < len(COCO_CLASSES) else str(cls),
                        "confidence": confidence,
                        "x_min": x_min * FRAME_WIDTH,
                        "y_min": y_min * FRAME_HEIGHT,
                        "x_max": x_max * FRAME_WIDTH,
                        "y_max": y_max * FRAME_HEIGHT,
                    })

        return detections

    # ---- Drawing ----

    def _get_species_label(self, x1, y1, x2, y2, det_conf):
        """Get the best label for a bird detection, using species if available.

        Matches the detection bbox to tracked birds by finding the closest
        tracked bbox, then looks up its species classification.
        """
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        best_tid = None
        best_dist = float("inf")
        for tid, (tcx, tcy) in self.tracker.objects.items():
            dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_tid = tid

        if best_tid is not None and best_tid in self.species_labels:
            species, sp_conf = self.species_labels[best_tid]
            # Shorten long species names for the overlay
            short = species.split("(")[0].strip()
            if len(short) > 25:
                short = short[:23] + ".."
            return f"{short} {sp_conf:.0%}"

        return f"Fugl {det_conf:.0%}"

    def _draw_overlays(self, frame_bgr, bird_dets, all_dets):
        """Draw bounding boxes and status overlay on the frame (BGR, in-place)."""
        # Bird detections: bright green, thick
        for d in bird_dets:
            x1, y1, x2, y2 = int(d["x_min"]), int(d["y_min"]), int(d["x_max"]), int(d["y_max"])
            conf = d["confidence"]
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Use species label if available for any tracked bird at this location
            label = self._get_species_label(x1, y1, x2, y2, conf)

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame_bgr, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 255, 0), -1)
            cv2.putText(frame_bgr, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Other detections: dim gray, thin (for debug context)
        for d in all_dets:
            if d["class_id"] == BIRD_CLASS_ID:
                continue
            x1, y1, x2, y2 = int(d["x_min"]), int(d["y_min"]), int(d["x_max"]), int(d["y_max"])
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (100, 100, 100), 1)
            cv2.putText(frame_bgr, d["class_name"], (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # Status overlay (top-left)
        lines = [
            f"Fugler na: {self.current_bird_count}",
            f"I dag: {self.today_visits} besok",
            f"FPS: {self.fps:.1f}",
        ]
        y = 30
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(frame_bgr, (8, y - th - 4), (8 + tw + 10, y + 6), (0, 0, 0), -1)
            cv2.putText(frame_bgr, line, (13, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            y += th + 18

    # ---- Crop saving ----

    def _save_crop(self, frame_rgb, bbox, track_id):
        """Save a cropped bird image to disk. Returns the file path or None."""
        if bbox is None:
            return None

        now = time.time()
        if track_id in self.last_crop_time:
            if now - self.last_crop_time[track_id] < CROP_SAVE_INTERVAL:
                return None
        self.last_crop_time[track_id] = now

        x1, y1, x2, y2 = (int(v) for v in bbox)
        h, w = frame_rgb.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 20 or y2 - y1 < 20:
            return None

        crop = frame_rgb[y1:y2, x1:x2]
        crop_bgr = crop  # already BGR from picamera2

        date_dir = os.path.join(CROP_DIR, datetime.now().strftime("%Y-%m-%d"))
        os.makedirs(date_dir, exist_ok=True)
        filename = f"{datetime.now().strftime('%H-%M-%S')}_track{track_id}.jpg"
        filepath = os.path.join(date_dir, filename)
        cv2.imwrite(filepath, crop_bgr)
        return filepath

    # ---- Species classification ----

    def _classify_bird(self, frame_rgb, bbox):
        """Classify a bird's species from its bounding box region.

        Returns (species_name, confidence) or (None, 0.0) on failure.
        """
        try:
            x1, y1, x2, y2 = (int(v) for v in bbox)
            h, w = frame_rgb.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 20 or y2 - y1 < 20:
                return None, 0.0

            crop = frame_rgb[y1:y2, x1:x2]
            species, confidence = self.classifier.classify(crop)

            # Filter out "not_a_bird" predictions
            if species.lower() == "not_a_bird":
                return None, 0.0

            return species, confidence
        except Exception as e:
            print(f"  Feil ved artsklassifisering: {e}", flush=True)
            return None, 0.0

    # ---- Public API for HTTP server ----

    def get_stats(self):
        return {
            "current_birds": self.current_bird_count,
            "today_visits": self.today_visits,
            "fps": round(self.fps, 1),
            "tracked_ids": list(self.tracker.objects.keys()),
            "total_arrivals_session": self.tracker.total_arrivals,
        }

    def get_recent_birds(self, limit=20):
        return self.db.get_recent_visits(limit)


# ============================================================
# HTTP Server + Dashboard
# ============================================================

# Global reference set from main()
monitor: BirdMonitor = None  # type: ignore


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="no">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fuglematerstasjon</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#1a1a2e; color:#e0e0e0; font-family:-apple-system,BlinkMacSystemFont,sans-serif; }
  .header { background:#16213e; padding:12px 20px; display:flex; align-items:center; gap:12px;
             border-bottom:2px solid #4ecca3; }
  .header h1 { font-size:1.2em; color:#4ecca3; font-weight:600; }
  .header .fps { color:#666; font-size:0.85em; margin-left:auto; }
  .container { display:grid; grid-template-columns:1fr 280px; gap:12px; padding:12px;
               max-width:1400px; margin:0 auto; height:calc(100vh - 52px); }
  .video-box { background:#000; border-radius:8px; overflow:hidden; min-height:0; }
  .video-box img { width:100%; height:100%; object-fit:contain; display:block; }
  .sidebar { display:flex; flex-direction:column; gap:12px; min-height:0; }
  .card { background:#16213e; border-radius:8px; padding:18px; text-align:center; }
  .card .val { font-size:2.8em; font-weight:700; color:#4ecca3; line-height:1; }
  .card .lbl { font-size:0.85em; color:#888; margin-top:4px; }
  .recent { background:#16213e; border-radius:8px; padding:14px; flex:1;
             overflow-y:auto; min-height:0; }
  .recent h3 { color:#4ecca3; font-size:0.95em; margin-bottom:8px; }
  .entry { padding:6px 0; border-bottom:1px solid #2a2a4a; display:flex;
           justify-content:space-between; align-items:center; font-size:0.85em; }
  .entry:last-child { border-bottom:none; }
  .entry .tm { color:#888; }
  .entry .cf { color:#4ecca3; }
  .no-birds { color:#555; font-style:italic; padding:20px 0; text-align:center; }
  @media(max-width:800px) {
    .container { grid-template-columns:1fr; height:auto; }
    .video-box { height:60vw; }
  }
</style>
</head>
<body>
  <div class="header">
    <h1>Fuglematerstasjon</h1>
    <span class="fps" id="fps"></span>
  </div>
  <div class="container">
    <div class="video-box">
      <img src="/stream" alt="Direktesending">
    </div>
    <div class="sidebar">
      <div class="card">
        <div class="val" id="now">0</div>
        <div class="lbl">Fugler akkurat n&aring;</div>
      </div>
      <div class="card">
        <div class="val" id="today">0</div>
        <div class="lbl">Bes&oslash;k i dag</div>
      </div>
      <div class="recent" id="recent">
        <h3>Siste bes&oslash;kende</h3>
        <div id="list"><div class="no-birds">Ingen fugler oppdaget enn&aring;</div></div>
      </div>
    </div>
  </div>
<script>
async function poll() {
  try {
    const r = await fetch("/api/stats");
    const d = await r.json();
    document.getElementById("now").textContent = d.current_birds;
    document.getElementById("today").textContent = d.today_visits;
    document.getElementById("fps").textContent = d.fps + " FPS";
  } catch(e) {}
}
async function birds() {
  try {
    const r = await fetch("/api/birds");
    const list = await r.json();
    const el = document.getElementById("list");
    if (!list.length) { el.innerHTML = '<div class="no-birds">Ingen fugler oppdaget enn&aring;</div>'; return; }
    el.innerHTML = list.map(b => {
      const t = new Date(b.start_time).toLocaleTimeString();
      const c = b.max_confidence ? (b.max_confidence*100).toFixed(0)+"%" : "";
      const s = b.species || "";
      return '<div class="entry"><span class="tm">'+t+'</span><span>'+s+'</span><span class="cf">'+c+'</span></div>';
    }).join("");
  } catch(e) {}
}
setInterval(poll, 1000);
setInterval(birds, 5000);
poll(); birds();
</script>
</body>
</html>"""


class BirdHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the bird monitor web interface."""

    def do_GET(self):
        if self.path == "/":
            self._serve_dashboard()
        elif self.path == "/stream":
            self._serve_mjpeg()
        elif self.path == "/api/stats":
            self._serve_json(monitor.get_stats())
        elif self.path == "/api/birds":
            self._serve_json(monitor.get_recent_birds())
        elif self.path.startswith("/crops/"):
            self._serve_crop()
        else:
            self.send_error(404)

    def _serve_dashboard(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(DASHBOARD_HTML.encode())

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.end_headers()
        try:
            while monitor.running:
                monitor.frame_event.wait(timeout=2.0)
                with monitor.frame_lock:
                    jpeg = monitor.latest_frame_jpeg
                if jpeg is None:
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n".encode())
                self.wfile.write(b"\r\n")
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(0.05)  # ~20fps max for streaming
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_json(self, data):
        body = json.dumps(data, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_crop(self):
        # Serve images from bird_crops/
        rel_path = self.path[len("/crops/"):]
        full_path = os.path.join(CROP_DIR, rel_path)
        full_path = os.path.realpath(full_path)
        # Security: ensure path is under CROP_DIR
        if not full_path.startswith(os.path.realpath(CROP_DIR)):
            self.send_error(403)
            return
        if not os.path.isfile(full_path):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.end_headers()
        with open(full_path, "rb") as f:
            self.wfile.write(f.read())

    def log_message(self, format, *args):
        pass  # suppress request logs


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ============================================================
# Main
# ============================================================

def main():
    global monitor

    monitor = BirdMonitor()
    monitor.start()

    server = ThreadedHTTPServer(("0.0.0.0", PORT), BirdHandler)
    print(f"Dashbord: http://0.0.0.0:{PORT}", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAvslutter...", flush=True)
    finally:
        server.server_close()
        monitor.stop()


if __name__ == "__main__":
    main()
