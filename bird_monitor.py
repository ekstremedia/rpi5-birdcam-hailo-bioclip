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
import urllib.request
from collections import OrderedDict
from datetime import datetime, date
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn

from PIL import Image, ImageDraw, ImageFont

try:
    from picamera2 import Picamera2
    from picamera2.devices import Hailo
except ImportError:
    raise ImportError(
        "picamera2 is required. Install with: sudo apt install python3-picamera2"
    )

# ============================================================
# Configuration (loaded from .env file, see .env.example)
# ============================================================

def _load_env(path):
    """Load key=value pairs from a .env file into os.environ."""
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

_load_env(Path(__file__).parent / ".env")

def _env(key, default):
    return os.environ.get(key, default)

def _env_int(key, default):
    return int(os.environ.get(key, default))

def _env_float(key, default):
    return float(os.environ.get(key, default))

PORT = _env_int("PORT", 8888)
MODEL_PATH = _env("MODEL_PATH", "/usr/share/hailo-models/yolov8s_h8.hef")
BIRD_CLASS_ID = 14  # COCO class index for "bird"
CONFIDENCE_THRESHOLD = _env_float("CONFIDENCE_THRESHOLD", 0.4)
CROP_DIR = _env("CROP_DIR", "/home/pi/ai/bird_crops")
DB_PATH = _env("DB_PATH", "/home/pi/ai/birds.db")
MODEL_INPUT_SIZE = 640

# --- Camera Configuration ---
CAMERA_SOURCE = _env("CAMERA_SOURCE", "elgato")  # "elgato" or "picamera2"
ELGATO_DEVICE = _env("ELGATO_DEVICE", "/dev/v4l/by-id/usb-Elgato_Cam_Link_4K_0005723438000-video-index0")
ELGATO_WIDTH = _env_int("ELGATO_WIDTH", 1920)
ELGATO_HEIGHT = _env_int("ELGATO_HEIGHT", 1080)
PICAMERA2_WIDTH = _env_int("PICAMERA2_WIDTH", 1280)
PICAMERA2_HEIGHT = _env_int("PICAMERA2_HEIGHT", 960)

# Set frame dimensions based on camera source
if CAMERA_SOURCE == "elgato":
    FRAME_WIDTH = ELGATO_WIDTH
    FRAME_HEIGHT = ELGATO_HEIGHT
else:
    FRAME_WIDTH = PICAMERA2_WIDTH
    FRAME_HEIGHT = PICAMERA2_HEIGHT

# Tracker settings
MAX_DISAPPEARED = _env_int("MAX_DISAPPEARED", 15)
MAX_DISTANCE = _env_int("MAX_DISTANCE", 150)
CROP_SAVE_INTERVAL = _env_float("CROP_SAVE_INTERVAL", 3.0)

# Species classification API (BioCLIP on NUC via Docker)
SPECIES_API_URL = _env("SPECIES_API_URL", "http://192.168.1.64:5555")
SPECIES_API_TIMEOUT = _env_int("SPECIES_API_TIMEOUT", 5)
SPECIES_HEALTH_INTERVAL = _env_int("SPECIES_HEALTH_INTERVAL", 15)
SPECIES_RECLASSIFY_THRESHOLD = _env_float("SPECIES_RECLASSIFY_THRESHOLD", 0.70)
SPECIES_RECLASSIFY_INTERVAL = _env_float("SPECIES_RECLASSIFY_INTERVAL", 1.5)  # seconds between retries
SPECIES_RECLASSIFY_MAX = _env_int("SPECIES_RECLASSIFY_MAX", 3)  # max retries per bird
SPECIES_LIST_PATH = _env("SPECIES_LIST_PATH", "/home/pi/ai/models/norwegian_species.txt")

NETATMO_URL = _env("NETATMO_URL", "https://ekstremedia.no/api/netatmo/stations/2c4735da-abbe-425e-a2ba-1006e786554c")
NETATMO_INTERVAL = _env_int("NETATMO_INTERVAL", 600)  # seconds between fetches

# Fonts for PIL text rendering (supports æøå, unlike OpenCV putText)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_LABEL = ImageFont.truetype(FONT_PATH, 20)    # bird labels on bboxes
FONT_HUD = ImageFont.truetype(FONT_PATH, 22)      # status overlay (top-left)
FONT_SMALL = ImageFont.truetype(FONT_PATH, 14)     # non-bird class names


def pil_text_size(text, font):
    """Get (width, height) of rendered text using PIL."""
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


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
        # Migration: add labeling columns to visits table
        for col, coltype in [("user_species", "TEXT"), ("labeled_at", "TEXT")]:
            try:
                conn.execute(f"ALTER TABLE visits ADD COLUMN {col} {coltype}")
            except sqlite3.OperationalError:
                pass  # column already exists
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

    def update_visit_species(self, track_id, species):
        """Update the auto-classified species for a visit (called from background thread)."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE visits SET species = ? WHERE track_id = ? AND end_time IS NULL",
            (species, track_id),
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

    def get_unlabeled_visits(self, limit=1):
        """Get visits that have crop images but no user label yet."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, track_id, start_time, species, max_confidence, crop_path "
            "FROM visits WHERE crop_path IS NOT NULL AND user_species IS NULL "
            "ORDER BY start_time DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def set_label(self, visit_id, species):
        """Set the human-assigned species label for a visit."""
        conn = self._get_conn()
        now = datetime.now().isoformat(timespec="seconds")
        conn.execute(
            "UPDATE visits SET user_species = ?, labeled_at = ? WHERE id = ?",
            (species, now, visit_id),
        )
        conn.commit()

    def skip_visit(self, visit_id):
        """Mark a visit as skipped (not useful for training)."""
        self.set_label(visit_id, "skip")

    def undo_label(self, visit_id):
        """Remove the label from a visit so it appears unlabeled again."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE visits SET user_species = NULL, labeled_at = NULL WHERE id = ?",
            (visit_id,),
        )
        conn.commit()

    def get_label_stats(self):
        """Get labeling progress counts."""
        conn = self._get_conn()
        total = conn.execute(
            "SELECT COUNT(*) FROM visits WHERE crop_path IS NOT NULL"
        ).fetchone()[0]
        labeled = conn.execute(
            "SELECT COUNT(*) FROM visits WHERE crop_path IS NOT NULL "
            "AND user_species IS NOT NULL AND user_species != 'skip'"
        ).fetchone()[0]
        skipped = conn.execute(
            "SELECT COUNT(*) FROM visits WHERE user_species = 'skip'"
        ).fetchone()[0]
        return {
            "total": total,
            "labeled": labeled,
            "skipped": skipped,
            "unlabeled": total - labeled - skipped,
        }


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

        # Remote species classification API (BioCLIP on NUC)
        self.species_api_ok = False  # True when NUC API is reachable
        self.species_labels = {}  # track_id -> (species, confidence)
        self.classify_attempts = {}  # track_id -> number of classify attempts
        self.classify_last_time = {}  # track_id -> timestamp of last attempt
        self.last_health_check = 0

        # Netatmo outdoor temperature
        self.outdoor_temp = None  # float or None if never fetched
        self._last_temp_fetch = 0

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
        print(f"Starter kamera ({CAMERA_SOURCE})...", flush=True)
        if CAMERA_SOURCE == "elgato":
            self.cap = cv2.VideoCapture(ELGATO_DEVICE, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, ELGATO_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ELGATO_HEIGHT)
            if not self.cap.isOpened():
                print(f"FEIL: Kunne ikke åpne Elgato på {ELGATO_DEVICE}", file=sys.stderr)
                sys.exit(1)
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"  Elgato Cam Link: {actual_w}x{actual_h} @ {actual_fps:.0f}fps", flush=True)
        else:
            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration(
                main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(1)  # let auto-exposure settle
            print(f"  PiCamera2: {FRAME_WIDTH}x{FRAME_HEIGHT}", flush=True)

        # Start processing in background thread
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        print(f"Behandlingsløkke startet. Dashbord på http://0.0.0.0:{PORT}", flush=True)

        # Check species API on NUC
        self._check_species_api()
        self._health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_thread.start()

        # Netatmo outdoor temperature
        self._temp_thread = threading.Thread(target=self._temp_fetch_loop, daemon=True)
        self._temp_thread.start()

    def _check_species_api(self):
        """Ping the remote species API health endpoint."""
        try:
            req = urllib.request.Request(f"{SPECIES_API_URL}/health", method="GET")
            with urllib.request.urlopen(req, timeout=SPECIES_API_TIMEOUT) as resp:
                data = json.loads(resp.read())
                was_ok = self.species_api_ok
                self.species_api_ok = data.get("status") == "ok"
                if self.species_api_ok and not was_ok:
                    count = data.get("species_count", "?")
                    print(f"  Klassifiseringsmotor: På ({count} arter)", flush=True)
                elif not self.species_api_ok and was_ok:
                    print("  Klassifiseringsmotor: Frakoblet", flush=True)
        except Exception:
            if self.species_api_ok:
                print("  Klassifiseringsmotor: Frakoblet", flush=True)
            self.species_api_ok = False

    def _health_check_loop(self):
        """Periodically check if the remote species API is reachable."""
        while self.running:
            time.sleep(SPECIES_HEALTH_INTERVAL)
            self._check_species_api()

    def _fetch_outdoor_temp(self):
        """Fetch outdoor temperature from Netatmo API."""
        try:
            req = urllib.request.Request(NETATMO_URL, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                for module in data.get("modules", []):
                    if module.get("type") == "Outdoor Module":
                        temp = module.get("measurements", {}).get("Temperature")
                        if temp is not None:
                            self.outdoor_temp = float(temp)
                            break
        except Exception as e:
            print(f"  Netatmo henting feilet: {e}", flush=True)

    def _temp_fetch_loop(self):
        """Periodically fetch outdoor temperature."""
        self._fetch_outdoor_temp()
        while self.running:
            time.sleep(NETATMO_INTERVAL)
            self._fetch_outdoor_temp()

    def stop(self):
        """Clean shutdown - waits for processing thread to finish first."""
        self.running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=5)
        if hasattr(self, "cap"):
            try:
                self.cap.release()
            except Exception:
                pass
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
        print("Fuglevakt stoppet.", flush=True)

    # ---- Main processing loop ----

    def _process_loop(self):
        frame_count = 0
        fps_start = time.time()
        first_frame = True

        while self.running:
            try:
                # 1. Capture frame (BGR, shape: H x W x 3)
                if CAMERA_SOURCE == "elgato":
                    ret, frame_rgb = self.cap.read()
                    if not ret:
                        time.sleep(0.01)
                        continue
                else:
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

                    # Log visit immediately (species filled in later by background thread)
                    self.db.log_visit_start(tid, now_str, conf, crop_path, species=None)
                    self.today_visits = self.db.get_today_count()
                    print(
                        f"  [{now_str}] Fugl ankommet! spor={tid} konf={conf:.2f}",
                        flush=True,
                    )

                    # Submit species classification to NUC API (non-blocking)
                    if self.species_api_ok and bbox:
                        crop_copy = frame_rgb[
                            max(0, int(bbox[1])):min(frame_rgb.shape[0], int(bbox[3])),
                            max(0, int(bbox[0])):min(frame_rgb.shape[1], int(bbox[2]))
                        ].copy()
                        if crop_copy.shape[0] >= 20 and crop_copy.shape[1] >= 20:
                            self.classify_attempts[tid] = 1
                            self.classify_last_time[tid] = time.time()
                            threading.Thread(
                                target=self._classify_bird_async,
                                args=(tid, crop_copy),
                                daemon=True,
                            ).start()

                for tid in self.tracker.departures:
                    self.db.log_visit_end(tid, now_str)
                    self.species_labels.pop(tid, None)
                    self.classify_attempts.pop(tid, None)
                    self.classify_last_time.pop(tid, None)
                    print(f"  [{now_str}] Fugl forlot. spor={tid}", flush=True)

                # Reclassify low-confidence birds that are still tracked
                if self.species_api_ok:
                    now_time = time.time()
                    for tid in list(self.tracker.objects.keys()):
                        if tid not in self.species_labels:
                            continue
                        sp, conf = self.species_labels[tid]
                        attempts = self.classify_attempts.get(tid, 1)
                        last_t = self.classify_last_time.get(tid, 0)
                        if (conf < SPECIES_RECLASSIFY_THRESHOLD
                                and attempts < SPECIES_RECLASSIFY_MAX
                                and now_time - last_t >= SPECIES_RECLASSIFY_INTERVAL):
                            bbox = self.tracker.bboxes.get(tid)
                            if bbox:
                                crop_copy = frame_rgb[
                                    max(0, int(bbox[1])):min(frame_rgb.shape[0], int(bbox[3])),
                                    max(0, int(bbox[0])):min(frame_rgb.shape[1], int(bbox[2]))
                                ].copy()
                                if crop_copy.shape[0] >= 20 and crop_copy.shape[1] >= 20:
                                    self.classify_last_time[tid] = now_time
                                    self.classify_attempts[tid] = attempts + 1
                                    threading.Thread(
                                        target=self._classify_bird_async,
                                        args=(tid, crop_copy),
                                        daemon=True,
                                    ).start()

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
        """Draw bounding boxes and status overlay on the frame (BGR, in-place).

        Uses OpenCV for rectangles/lines (fast) and PIL for text (Unicode support).
        """
        # Collect text draws: (text, x, y, font, fill, bg)
        text_draws = []

        # Bird detections: bright green, thick
        for d in bird_dets:
            x1, y1, x2, y2 = int(d["x_min"]), int(d["y_min"]), int(d["x_max"]), int(d["y_max"])
            conf = d["confidence"]
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)

            label = self._get_species_label(x1, y1, x2, y2, conf)
            tw, th = pil_text_size(label, FONT_LABEL)
            cv2.rectangle(frame_bgr, (x1, y1 - th - 10), (x1 + tw + 8, y1), (0, 255, 0), -1)
            text_draws.append((label, x1 + 4, y1 - th - 7, FONT_LABEL, (0, 0, 0)))

        # Other detections: dim gray, thin
        for d in all_dets:
            if d["class_id"] == BIRD_CLASS_ID:
                continue
            x1, y1, x2, y2 = int(d["x_min"]), int(d["y_min"]), int(d["x_max"]), int(d["y_max"])
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (100, 100, 100), 1)
            text_draws.append((d["class_name"], x1, y1 - 4, FONT_SMALL, (100, 100, 100)))

        # Status bar (two lines across the top, 70% transparent black)
        DAGER = ["man", "tir", "ons", "tor", "fre", "lør", "søn"]
        MAANEDER = ["jan", "feb", "mar", "apr", "mai", "jun",
                     "jul", "aug", "sep", "okt", "nov", "des"]
        now = datetime.now()
        dag = DAGER[now.weekday()]
        mnd = MAANEDER[now.month - 1]
        klokke = f"{dag} {now.day}. {mnd} {now.strftime('%H:%M:%S')}"

        api_icon = "●" if self.species_api_ok else "○"
        temp_str = f"{self.outdoor_temp:.1f}°C" if self.outdoor_temp is not None else ""
        line1 = (f"{klokke}  |  "
                 f"Fugler: {self.current_bird_count}  |  "
                 f"I dag: {self.today_visits} besøk  |  "
                 + (f"{temp_str}  |  " if temp_str else "")
                 + f"FPS: {self.fps:.0f}  |  "
                 f"API {api_icon}")

        # Species summary from currently tracked birds
        species_counts = {}
        for tid_key in self.tracker.objects:
            if tid_key in self.species_labels:
                sp_name = self.species_labels[tid_key][0].split("(")[0].strip()
                species_counts[sp_name] = species_counts.get(sp_name, 0) + 1
        if species_counts:
            parts = [f"{cnt} {name}" for name, cnt in sorted(species_counts.items(), key=lambda x: -x[1])]
            line2 = ", ".join(parts)
        else:
            line2 = ""

        _, th1 = pil_text_size(line1, FONT_HUD)
        bar_h = th1 + 12
        if line2:
            _, th2 = pil_text_size(line2, FONT_LABEL)
            bar_h += th2 + 6

        # Darken the bar region in-place (70% opacity black = multiply by 0.3)
        frame_bgr[0:bar_h, :] = (frame_bgr[0:bar_h, :] * 0.3).astype(np.uint8)

        text_draws.append((line1, 8, 4, FONT_HUD, (78, 204, 163)))
        if line2:
            text_draws.append((line2, 8, th1 + 10, FONT_LABEL, (220, 220, 220)))

        # Render all text in one PIL pass
        img_pil = Image.fromarray(frame_bgr)
        draw = ImageDraw.Draw(img_pil)
        for text, tx, ty, font, fill in text_draws:
            draw.text((tx, ty), text, font=font, fill=fill)
        frame_bgr[:] = np.array(img_pil)

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

    def _classify_bird_async(self, track_id, crop):
        """Classify a bird by sending the crop to the remote NUC API.

        Called from a daemon thread so the video loop never blocks.
        Only updates the label if the new confidence is higher than the existing one.
        """
        try:
            _, jpeg_buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            jpeg_bytes = jpeg_buf.tobytes()

            req = urllib.request.Request(
                f"{SPECIES_API_URL}/classify",
                data=jpeg_bytes,
                method="POST",
                headers={"Content-Type": "application/octet-stream"},
            )
            with urllib.request.urlopen(req, timeout=SPECIES_API_TIMEOUT) as resp:
                result = json.loads(resp.read())

            species = result.get("species")
            confidence = result.get("confidence", 0)
            inf_time = result.get("inference_time", 0)

            if not species:
                return

            # Only update if better than existing classification
            old = self.species_labels.get(track_id)
            attempt = self.classify_attempts.get(track_id, 1)
            if old and old[1] >= confidence:
                print(
                    f"  [art] spor={track_id} forsøk {attempt}: {species} ({confidence:.0%}) "
                    f"← beholdt {old[0]} ({old[1]:.0%})",
                    flush=True,
                )
                return

            self.species_labels[track_id] = (species, confidence)
            self.db.update_visit_species(track_id, species)
            retry_tag = f" forsøk {attempt}" if attempt > 1 else ""
            print(
                f"  [art] spor={track_id}{retry_tag} → {species} ({confidence:.0%}) [{inf_time:.2f}s]",
                flush=True,
            )
        except Exception as e:
            print(f"  Feil ved artsklassifisering: {e}", flush=True)

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


LABEL_HTML = """<!DOCTYPE html>
<html lang="no">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Merking av fuglebilder</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#1a1a2e; color:#e0e0e0; font-family:-apple-system,BlinkMacSystemFont,sans-serif; }
  .header { background:#16213e; padding:12px 20px; display:flex; align-items:center; gap:12px;
             border-bottom:2px solid #4ecca3; flex-wrap:wrap; }
  .header h1 { font-size:1.2em; color:#4ecca3; font-weight:600; }
  .header .progress { color:#888; font-size:0.9em; margin-left:auto; }
  .header a { color:#4ecca3; text-decoration:none; font-size:0.85em; }
  .container { max-width:700px; margin:0 auto; padding:16px; }
  .crop-box { background:#000; border-radius:8px; overflow:hidden; text-align:center;
              margin-bottom:12px; min-height:200px; display:flex; align-items:center;
              justify-content:center; }
  .crop-box img { max-width:100%; max-height:50vh; object-fit:contain; display:block; margin:auto; }
  .crop-box .empty { color:#555; font-style:italic; padding:40px; }
  .suggestion { background:#16213e; border-radius:8px; padding:12px 16px; margin-bottom:12px;
                font-size:1em; text-align:center; }
  .suggestion .species { color:#4ecca3; font-weight:600; }
  .suggestion .conf { color:#888; }
  .species-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(130px, 1fr));
                  gap:8px; margin-bottom:12px; }
  .species-btn { background:#16213e; border:2px solid #2a2a4a; border-radius:8px; padding:10px 6px;
                 color:#e0e0e0; font-size:0.9em; cursor:pointer; text-align:center;
                 transition:all 0.15s; }
  .species-btn:hover { background:#1f2b47; border-color:#4ecca3; }
  .species-btn:active { transform:scale(0.96); }
  .species-btn.suggested { border-color:#4ecca3; background:#1a3a2e; }
  .species-btn .key { display:inline-block; background:#2a2a4a; color:#888; border-radius:3px;
                      font-size:0.7em; padding:1px 5px; margin-right:4px; vertical-align:middle; }
  .actions { display:flex; gap:10px; margin-bottom:16px; }
  .actions button { flex:1; padding:12px; border:2px solid #2a2a4a; border-radius:8px;
                    font-size:0.95em; cursor:pointer; transition:all 0.15s; }
  .skip-btn { background:#2a1a1a; color:#e07070; border-color:#5a2a2a !important; }
  .skip-btn:hover { background:#3a2020; border-color:#e07070 !important; }
  .undo-btn { background:#1a1a2e; color:#aaa; }
  .undo-btn:hover { background:#252540; border-color:#888 !important; }
  .undo-btn:disabled { opacity:0.3; cursor:not-allowed; }
  .done { text-align:center; padding:60px 20px; }
  .done h2 { color:#4ecca3; margin-bottom:10px; }
  .toast { position:fixed; bottom:20px; left:50%; transform:translateX(-50%); background:#4ecca3;
           color:#1a1a2e; padding:10px 24px; border-radius:20px; font-weight:600; opacity:0;
           transition:opacity 0.3s; pointer-events:none; z-index:100; }
  .toast.show { opacity:1; }
  .shortcuts { color:#555; font-size:0.75em; text-align:center; margin-top:8px; }
</style>
</head>
<body>
  <div class="header">
    <h1>Merking av fuglebilder</h1>
    <span class="progress" id="progress"></span>
    <a href="/">&larr; Dashbord</a>
  </div>
  <div class="container">
    <div class="crop-box" id="cropBox">
      <div class="empty">Laster...</div>
    </div>
    <div class="suggestion" id="suggestion" style="display:none"></div>
    <div class="species-grid" id="speciesGrid"></div>
    <div class="actions" id="actions" style="display:none">
      <button class="skip-btn" onclick="skipImage()">Hopp over (S)</button>
      <button class="undo-btn" id="undoBtn" onclick="undoLast()" disabled>Angre siste (Z)</button>
    </div>
    <div class="shortcuts">Tastatursnarveier: 1-9 = velg art, S = hopp over, Z = angre</div>
  </div>
  <div class="toast" id="toast"></div>

<script>
let currentVisit = null;
let speciesList = [];
let lastLabeledId = null;
let suggestedSpecies = '';

async function loadSpecies() {
  try {
    const r = await fetch('/api/label/species');
    speciesList = await r.json();
    renderSpeciesButtons();
  } catch(e) { console.error('Feil ved lasting av artsliste:', e); }
}

function renderSpeciesButtons() {
  const grid = document.getElementById('speciesGrid');
  grid.innerHTML = speciesList.map((sp, i) => {
    const isSuggested = suggestedSpecies && sp.norwegian === suggestedSpecies;
    const keyLabel = i < 9 ? '<span class="key">' + (i+1) + '</span>' : '';
    return '<button class="species-btn' + (isSuggested ? ' suggested' : '') +
           '" onclick="labelImage(\'' + sp.norwegian.replace(/'/g, "\\\\'") + '\')">' +
           keyLabel + sp.norwegian + '</button>';
  }).join('');
}

async function loadNext() {
  try {
    const r = await fetch('/api/label/queue');
    const data = await r.json();
    if (!data || !data.id) {
      showDone();
      return;
    }
    currentVisit = data;
    const cropBox = document.getElementById('cropBox');
    const cropUrl = data.crop_path.replace('/home/pi/ai/bird_crops/', '/crops/');
    cropBox.innerHTML = '<img src="' + cropUrl + '" alt="Fugl">';

    const suggestion = document.getElementById('suggestion');
    if (data.species) {
      suggestedSpecies = data.species;
      const confPct = data.max_confidence ? (data.max_confidence * 100).toFixed(0) + '%' : '';
      suggestion.innerHTML = 'BioCLIP: <span class="species">' + data.species +
                             '</span> <span class="conf">(' + confPct + ')</span>';
      suggestion.style.display = '';
    } else {
      suggestedSpecies = '';
      suggestion.style.display = 'none';
    }
    renderSpeciesButtons();
    document.getElementById('actions').style.display = '';
    updateProgress();
  } catch(e) { console.error('Feil ved lasting av bilde:', e); }
}

async function labelImage(species) {
  if (!currentVisit) return;
  const visitId = currentVisit.id;
  try {
    await fetch('/api/label/save', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({visit_id: visitId, species: species})
    });
    lastLabeledId = visitId;
    document.getElementById('undoBtn').disabled = false;
    showToast('Merket: ' + species);
    loadNext();
  } catch(e) { console.error('Feil ved lagring:', e); }
}

async function skipImage() {
  if (!currentVisit) return;
  const visitId = currentVisit.id;
  try {
    await fetch('/api/label/skip', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({visit_id: visitId})
    });
    lastLabeledId = visitId;
    document.getElementById('undoBtn').disabled = false;
    showToast('Hoppet over');
    loadNext();
  } catch(e) { console.error('Feil ved hopping:', e); }
}

async function undoLast() {
  if (!lastLabeledId) return;
  try {
    await fetch('/api/label/undo', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({visit_id: lastLabeledId})
    });
    showToast('Angret');
    lastLabeledId = null;
    document.getElementById('undoBtn').disabled = true;
    loadNext();
  } catch(e) { console.error('Feil ved angring:', e); }
}

async function updateProgress() {
  try {
    const r = await fetch('/api/label/stats');
    const s = await r.json();
    document.getElementById('progress').textContent =
      s.labeled + ' av ' + s.total + ' merket' + (s.skipped ? ' (' + s.skipped + ' hoppet over)' : '');
  } catch(e) {}
}

function showDone() {
  document.querySelector('.container').innerHTML =
    '<div class="done"><h2>Ferdig!</h2><p>Alle bilder er merket.</p>' +
    '<p style="margin-top:16px"><a href="/" style="color:#4ecca3">Tilbake til dashbord</a></p></div>';
  updateProgress();
}

function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 1200);
}

document.addEventListener('keydown', function(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 's' || e.key === 'S') { skipImage(); return; }
  if (e.key === 'z' || e.key === 'Z') { undoLast(); return; }
  const num = parseInt(e.key);
  if (num >= 1 && num <= 9 && num <= speciesList.length) {
    labelImage(speciesList[num - 1].norwegian);
  }
});

loadSpecies();
loadNext();
</script>
</body>
</html>"""


class BirdHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the bird monitor web interface."""

    def do_GET(self):
        if self.path == "/":
            self._serve_dashboard()
        elif self.path == "/label":
            self._serve_label_page()
        elif self.path == "/stream":
            self._serve_mjpeg()
        elif self.path == "/api/stats":
            self._serve_json(monitor.get_stats())
        elif self.path == "/api/birds":
            self._serve_json(monitor.get_recent_birds())
        elif self.path == "/api/label/queue":
            self._serve_label_queue()
        elif self.path == "/api/label/species":
            self._serve_label_species()
        elif self.path == "/api/label/stats":
            self._serve_json(monitor.db.get_label_stats())
        elif self.path.startswith("/crops/"):
            self._serve_crop()
        else:
            self.send_error(404)

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len else b""
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.send_error(400, "Ugyldig JSON")
            return

        if self.path == "/api/label/save":
            visit_id = data.get("visit_id")
            species = data.get("species")
            if not visit_id or not species:
                self.send_error(400, "Mangler visit_id eller species")
                return
            monitor.db.set_label(visit_id, species)
            self._serve_json({"ok": True})
        elif self.path == "/api/label/skip":
            visit_id = data.get("visit_id")
            if not visit_id:
                self.send_error(400, "Mangler visit_id")
                return
            monitor.db.skip_visit(visit_id)
            self._serve_json({"ok": True})
        elif self.path == "/api/label/undo":
            visit_id = data.get("visit_id")
            if not visit_id:
                self.send_error(400, "Mangler visit_id")
                return
            monitor.db.undo_label(visit_id)
            self._serve_json({"ok": True})
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

    def _serve_label_page(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(LABEL_HTML.encode())

    def _serve_label_queue(self):
        visits = monitor.db.get_unlabeled_visits(limit=1)
        if not visits:
            self._serve_json({})
            return
        v = visits[0]
        self._serve_json({
            "id": v["id"],
            "crop_path": v["crop_path"],
            "species": v.get("species"),  # BioCLIP auto-species (Norwegian name)
            "max_confidence": v["max_confidence"],
            "start_time": v["start_time"],
        })

    def _serve_label_species(self):
        """Return the species list with Norwegian names for the labeling UI."""
        species = []
        try:
            with open(SPECIES_LIST_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "|" in line:
                        en, no = line.split("|", 1)
                        species.append({"english": en.strip(), "norwegian": no.strip()})
        except FileNotFoundError:
            pass
        self._serve_json(species)

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
