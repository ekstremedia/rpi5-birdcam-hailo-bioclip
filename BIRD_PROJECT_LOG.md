# Bird Feeder Camera Project Log

## Goal
Use Raspberry Pi 5 + AI HAT+ to recognize and count birds at a bird feeder.

## Hardware
- Raspberry Pi 5 Model B Rev 1.0
- Hailo-8 AI HAT+ (detected, firmware 4.23.0)
- Camera: **IMX477 (Raspberry Pi HQ Camera)** - 12.3MP, up to 4056x3040
- OS: Debian 13 (trixie), kernel 6.12.62+rpt-rpi-2712

## Setup Log

### 2026-02-18 - Initial Setup
- Followed guide: https://www.raspberrypi.com/documentation/computers/ai.html
- AI HAT+ physically installed and detected by system
- Hailo packages installed:
  - hailo-all 5.1.1
  - hailort 4.23.0
  - hailo-tappas-core 5.1.0
  - hailo-models 1.0.0-2
  - rpicam-apps-hailo-postprocess 1.11.1-1
- rpicam-apps installed and working
- **Camera: not yet connected** - `rpicam-hello` reports "No cameras available!"
- Hailo apps installing in background (per guide)

### 2026-02-18 - Headless Streaming Setup
- Pi is headless (no monitor/GUI) — decided on TCP + MJPEG streaming via `rpicam-vid`
- View on laptop using VLC (Open Network Stream → `tcp/mjpeg://PI_IP:8888`)
- All software already installed and ready:
  - rpicam-vid v1.11.1 with libav support
  - rpicam-apps-hailo-postprocess for AI overlay
  - 9 Hailo post-process configs available (YOLOv6, YOLOv8, pose, segmentation, etc.)
- Plain stream: `rpicam-vid -t 0 --width 1280 --height 960 --codec mjpeg --listen -o tcp://0.0.0.0:8888`
- AI stream: add `--post-process-file /usr/share/rpi-camera-assets/hailo_yolov8_inference.json`
- Waiting on camera to test

### 2026-02-18 - Camera Connected!
- Connected **IMX477 (Raspberry Pi HQ Camera)** via CSI
- Kernel detected it automatically: `imx477 10-001a: Device found is imx477`
- Registered as `/dev/video0` through `/dev/video7`
- Capabilities: 12.3MP (4056x3040), up to ~148fps at lower res
- Test photo captured successfully (`test_photo.jpg`, 2028x1520, 311K)
- Camera is working perfectly - ready for AI demos

### 2026-02-18 - Live AI Stream Working!
- Got live MJPEG stream with Hailo YOLOv8 AI detection working over HTTP
- Model: YOLOv8s on Hailo-8 (COCO 80 classes, includes "bird")
- Stream runs at 1280x960 @ 30fps with AI bounding box overlay
- Viewable in any browser on the local network

#### Quick reference commands

**Take a test photo:**
```bash
rpicam-still -o test_photo.jpg --width 2028 --height 1520 -t 2000
```

**Start the AI stream server:**
```bash
python3 /home/pi/ai/stream.py
```
Then open `http://192.168.1.176:8888` in a browser.

**Run in background:**
```bash
nohup python3 /home/pi/ai/stream.py > /tmp/stream-server.log 2>&1 &
```

**Stop the stream:**
```bash
pkill -f stream.py
```

**Check camera is detected:**
```bash
rpicam-hello --list-cameras
```

#### How the stream server works (`stream.py`)
- Launches `rpicam-vid` with `--codec mjpeg --flush` and Hailo YOLOv8 post-processing
- Reads raw MJPEG frames from stdout (FFD8..FFD9 markers)
- Serves them over HTTP as `multipart/x-mixed-replace` (standard browser MJPEG)
- Threaded server supports multiple viewers

#### Available Hailo post-process configs
| Config file | Description |
|---|---|
| `hailo_yolov8_inference.json` | General object detection (80 COCO classes) |
| `hailo_yolov6_inference.json` | YOLOv6 object detection |
| `hailo_yolov8_pose.json` | Pose estimation |
| `hailo_yolov5_segmentation.json` | Instance segmentation |
| `hailo_yolov5_personface.json` | Person + face detection |

Configs are in `/usr/share/rpi-camera-assets/`. Swap via `--post-process-file` flag.

### 2026-02-19 - Bird Monitor Application Built!

Built `bird_monitor.py` — a complete bird detection, tracking, and monitoring system.

#### Architecture
Two-stage pipeline (Stage 1 complete, Stage 2 planned):
```
Camera (picamera2, 1280x960 RGB)
  → Resize to 640x640
  → YOLOv8s on Hailo-8 (Stage 1: Detection)
  → Filter COCO class 14 ("bird")
  → Centroid tracker (arrival/departure counting)
  → SQLite logging + crop saving
  → Web dashboard with MJPEG stream
```

#### Key discoveries during implementation

**Hailo output format**: The `picamera2.devices.Hailo` wrapper returns a **list of 80 numpy arrays** (one per COCO class), NOT a single (80, 5, 100) tensor as the model shape suggests. Each array has shape `(N, 5)` where N = number of detections for that class. The 5 values are `[y_min, x_min, y_max, x_max, confidence]`, all normalized [0, 1].

**Performance**: Running at **30 FPS** with full YOLOv8s inference on Hailo-8. The pipeline is not CPU-bound — Hailo handles inference, OpenCV handles drawing/encoding.

**Camera resource management**: The Hailo device and camera can only be used by one process at a time. Must stop `stream.py` before running `bird_monitor.py` (and vice versa). The `picamera2` library needs a clean shutdown or the camera stays locked.

#### What bird_monitor.py includes
1. **Camera capture** via `picamera2` (1280x960 @ 30fps)
2. **Hailo inference** via `picamera2.devices.Hailo` class
3. **Detection parsing** for all 80 COCO classes, filtering for birds (class 14)
4. **Centroid-based tracker** — tracks birds across frames, counts arrivals/departures
5. **SQLite database** (`birds.db`) — logs each bird visit with timestamps, confidence, crop path
6. **Auto crop saving** — saves bird images to `bird_crops/YYYY-MM-DD/` with rate limiting
7. **Web dashboard** at port 8888:
   - `/` — Live dashboard with bird count, visit counter, recent visitors
   - `/stream` — Raw MJPEG stream with bounding box overlays
   - `/api/stats` — JSON stats (current birds, today visits, FPS)
   - `/api/birds` — JSON list of recent bird visits
   - `/crops/<path>` — Serves saved crop images
8. **Overlay drawing** — Green boxes for birds, gray for other objects, status HUD

#### Quick reference

**Start the bird monitor:**
```bash
python3 /home/pi/ai/bird_monitor.py
```
Then open `http://192.168.1.176:8888` in a browser.

**Run in background:**
```bash
nohup python3 /home/pi/ai/bird_monitor.py > /tmp/bird-monitor.log 2>&1 &
```

**Stop:**
```bash
pkill -f bird_monitor.py
```

**Important:** Only one of `stream.py` or `bird_monitor.py` can run at a time (they share the camera and Hailo device).

#### Files
| File | Purpose |
|---|---|
| `bird_monitor.py` | Main application (~500 lines) |
| `birds.db` | SQLite database (auto-created) |
| `bird_crops/` | Saved bird crop images (auto-created) |
| `stream.py` | Original simple stream (kept as fallback) |

### 2026-02-19 - Phase 2: Species Classification Added!

Initially tried EfficientNet-B7 from HuggingFace (n2b8/backyard-birds), but it was trained
on North American species only — completely wrong for Norwegian birds.

**Switched to BioCLIP** ([imageomics/bioclip](https://huggingface.co/imageomics/bioclip)) —
a zero-shot vision-language model trained on 10M+ biological images (CVPR 2024 Best Student Paper).

#### How it works
- On bird arrival, the detection crop is classified by BioCLIP zero-shot
- Species list is loaded from `models/norwegian_species.txt` (42 common Norwegian birds)
- Model matches the image against all species names — no retraining needed
- Norwegian names shown on bounding boxes ("Kjottmeis 87%" instead of "Fugl 92%")
- Species logged to `birds.db` and shown on dashboard
- ~4s per classification on Pi 5 CPU (only on arrival, not every frame)
- To add new species: just edit `models/norwegian_species.txt` (hot-reloads every 60s)

#### Why BioCLIP over traditional classifiers
- **Zero-shot**: works with ANY species — just add the name to the text file
- **Global coverage**: trained on 450K+ taxa from iNaturalist, not just North American birds
- **No retraining needed**: adding a new species = adding a line to a text file
- **Tested on crops**: Pilfink 99-100%, Graspurv 76%, Ravn 58%

#### Model details
- **Model**: BioCLIP 2 (ViT-B/16), ~600MB cached from HuggingFace
- **Startup**: ~15 min first run (downloads model), ~2 min from cache
- **Inference**: ~4s per classification on Pi 5 ARM64 CPU
- **RAM**: ~2.6GB when loaded (Pi 5 has 8GB, plenty of room)

#### Files
| File | Purpose |
|---|---|
| `species_classifier.py` | BioCLIP species classifier module |
| `train_species.py` | Retraining script for desktop/GPU (kept for future use) |
| `models/norwegian_species.txt` | 42 Norwegian bird species (English + Norwegian names) |

#### All user-facing text in Norwegian
- Dashboard: "Fuglematerstasjon", "Fugler akkurat na", "Besok i dag", etc.
- Overlay: "Fugler na:", species names in Norwegian
- Console: All startup, arrival, departure messages in Norwegian

## Next Steps
- [ ] Point camera at bird feeder and test species identification with real birds
- [x] Add species classification (Phase 2) — BioCLIP zero-shot
- [x] Add retraining pipeline for learning new species (Phase 3)
- [x] Norwegian translation of all user-facing text
- [ ] Set up auto-start on boot (systemd service)
- [ ] Add daily/weekly bird statistics to dashboard
