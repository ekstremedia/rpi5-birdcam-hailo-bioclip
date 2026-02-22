# Bird Feeder Camera

An AI-powered bird feeder monitor that detects, tracks, and identifies bird species in real time — then streams it live to the web. This is a hobby/learning project built with a mix of off-the-shelf hardware and open-source AI models. If you're curious about building something similar, hopefully this gives you a good starting point.

Three machines work together: a **Raspberry Pi 5** (a small, affordable single-board computer) runs real-time detection at the feeder, an **Intel NUC** (a small, quiet desktop PC) handles species classification, and a **VPS** (a rented cloud server) serves the live stream to the public.

## Architecture

Here's how the three machines talk to each other. The Pi captures video and detects birds, sends crops to the NUC for species identification, and streams everything to the VPS so anyone can watch online:

```text
Pi 5                          NUC                           VPS
────────────────────          ────────────────────          ────────────────────
Canon G25 → Elgato            species-api :5555
  → YOLOv8s on Hailo-8         BioCLIP classification
  → Centroid tracker
  → Web dashboard :8888
     │                                                           mediamtx
     ├─ /stream (MJPEG) ──────→ stream-relay ──── RTMP ────→   WebRTC / HLS
     ├─ /api/stats      ──────→ stats-pusher ── HTTPS ─────→ Laravel → Reverb
     ├─ /api/birds      ──────→                               → /fuglekamera
     └─ /classify (POST) ←────  BioCLIP API
```

**Live at**: [ekstremedia.no/fuglekamera](https://ekstremedia.no/fuglekamera)

### Why three machines?

The **Pi** sits at the bird feeder doing real-time detection — it needs to be small, quiet, and close to the camera. Species classification (figuring out *which* bird it is) is much more demanding, taking ~4 seconds on the Pi but only ~0.2 seconds on the **NUC**, which has a proper Intel i7 processor. The **VPS** handles the public-facing stream because a home internet connection can't serve video to more than a handful of viewers at once.

---

## Raspberry Pi 5

The Pi runs `bird_monitor.py` — the main application that captures video, detects birds, tracks them, and serves a live dashboard.

### Hardware

- Raspberry Pi 5 (8GB)
- [Hailo-8 AI HAT+](https://www.raspberrypi.com/products/ai-hat) — a plug-in accelerator board for running AI models, rated at 26 TOPS (tera operations per second — a measure of AI processing speed)
- Camera: Canon LEGRIA HF G25 via [Elgato Cam Link 4K](https://www.elgato.com/cam-link-4k) (captures the camera's HDMI output as a USB webcam)
- Backup camera: Raspberry Pi HQ Camera (IMX477, CSI)
- OS: Debian 13 (trixie)

### Detection pipeline

1. Camera captures 1920x1080 @ 25fps via Elgato Cam Link
2. Each frame is resized to 640x640 and fed to **YOLOv8s** (a fast object detection model — "You Only Look Once") running on the Hailo-8
3. Detections are filtered for COCO class 14 (bird) — COCO is a standard dataset of everyday objects used to train detection models, and "bird" happens to be class number 14 — with a confidence threshold above 0.4
4. A centroid-based tracker (an algorithm that follows objects between frames by their center point) assigns IDs, counts arrivals and departures
5. On bird arrival, the crop is sent to the NUC for species classification
6. Results are logged to SQLite (a simple file-based database), crops saved to disk

### Setup

```bash
git clone https://github.com/ekstremedia/rpi5-birdcam-hailo-bioclip.git
cd rpi5-birdcam-hailo-bioclip

cp .env.example .env
# Edit .env — set CAMERA_SOURCE, SPECIES_API_URL, etc.

# Required system packages (should already be on Pi OS)
sudo apt install python3-picamera2 python3-opencv python3-pil

python3 bird_monitor.py
```

Then open `http://PI_IP:8888` in a browser.

### Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_SOURCE` | `elgato` | `elgato` or `picamera2` |
| `SPECIES_API_URL` | `http://NUC_IP:5555` | NUC classification server |
| `PORT` | `8888` | Web dashboard port |
| `CONFIDENCE_THRESHOLD` | `0.4` | Minimum detection confidence |
| `CROP_DIR` | `/home/pi/ai/bird_crops` | Where bird crop images are saved |
| `DB_PATH` | `/home/pi/ai/birds.db` | SQLite database path |

### Web dashboard

| Endpoint | Description |
|----------|-------------|
| `/` | Live dashboard with stream, bird count, recent visitors |
| `/stream` | Raw MJPEG stream (a series of JPEG images sent as video) with bounding box overlays |
| `/label` | Web UI for manually labeling bird crops (training data) |
| `/api/stats` | JSON: current birds, today's visits, FPS |
| `/api/birds` | JSON: recent bird visits with species |

### HUD overlay

The video stream shows a status bar with:
- Norwegian date/time (e.g. "tor 20. feb 19:08:54")
- Current bird count and today's visit total
- Processing FPS
- Classification server status (På/Frakoblet)
- Species summary when birds are present (e.g. "2 blåmeis, 1 spettmeis")

### Files (Pi)

| File | Description |
|------|-------------|
| `bird_monitor.py` | Main application — detection, tracking, classification, web dashboard |
| `stream.py` | Simple MJPEG stream with Hailo AI overlay (fallback/testing) |
| `species_classifier.py` | Local BioCLIP classifier module (unused, kept for reference) |
| `models/norwegian_species.txt` | 42 bird species (English + Norwegian names) |
| `.env.example` | Configuration template |

---

## NUC (Classification + Relay)

The NUC runs three Docker (a way to run apps in isolated containers) services that support the Pi and connect it to the public web.
All files are in the [`nuc/`](nuc/) directory.

| Service | Image | Purpose |
|---------|-------|---------|
| `species-api` | Custom (BioCLIP + Flask) | Species classification API on port 5555 |
| `stream-relay` | linuxserver/ffmpeg | Relays Pi MJPEG → H.264 → RTMP to VPS |
| `stats-pusher` | python:3.11-slim | Polls Pi stats every 5s, pushes to Laravel |

### Setup

```bash
cd nuc
cp .env.example .env
# Edit .env — set HF_TOKEN, PI_STREAM, VPS_RTMP, LARAVEL_URL, LARAVEL_TOKEN

docker compose up -d

# Verify
docker compose ps
curl http://localhost:5555/health
```

### Configuration (nuc/.env)

```bash
# BioCLIP model download (build-time only)
HF_TOKEN=hf_your_token

# Stream relay
PI_STREAM=http://PI_IP:8888/stream
VPS_RTMP=rtmp://VPS_IP:1935/birdcam/live

# Stats pusher
PI_URL=http://PI_IP:8888
LARAVEL_URL=https://ekstremedia.no
LARAVEL_TOKEN=your_shared_secret_token
POLL_INTERVAL=5
```

### Species classification API

The `species-api` service runs [BioCLIP](https://huggingface.co/imageomics/bioclip) (an AI model trained on millions of biology images) as a Flask (lightweight Python web framework) HTTP API.
When `bird_monitor.py` detects a bird, it POSTs the crop to `/classify` and gets back the species name in Norwegian.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/classify` | POST | Send JPEG bytes → `{species, species_en, confidence, inference_time}` |
| `/health` | GET | Status check → `{status, model, species_count}` |

Classification takes ~0.2s on the NUC (i7) vs ~4s on the Pi 5.

Low-confidence results (<70%) are automatically reclassified up to 3 times using fresh crops, keeping the best result.

### Stream relay

The `stream-relay` service reads the Pi's MJPEG stream, encodes it to H.264 with libx264 (ultrafast preset, zerolatency tune, 1.5 Mbps), and pushes it via RTMP (a video streaming protocol) to the VPS.

### Stats pusher

The `stats-pusher` service polls the Pi's `/api/stats` and `/api/birds` endpoints every 5 seconds and POSTs the combined data to the Laravel (a PHP web framework) API. Laravel caches the stats and broadcasts them via [Reverb](https://reverb.laravel.com/) (Laravel's real-time WebSocket server) to all connected browsers.

---

## VPS (Public Stream)

The VPS runs [mediamtx](https://github.com/bluenviron/mediamtx) to receive the RTMP stream and serve it to browsers, plus a Laravel/Vue application for the public page.

Setup instructions are in [`vpssetup.md`](vpssetup.md).

### Components

| Component | Purpose |
|-----------|---------|
| [mediamtx](https://github.com/bluenviron/mediamtx) v1.11.3 | Receives RTMP on :1935, serves WebRTC/WHEP (a low-latency browser video protocol) on :8889, HLS (another streaming format — higher latency but wider support) on :8888 |
| Apache2 reverse proxy | Proxies `/birdcam/live/whep` and `/birdcam/live/` through HTTPS |
| Laravel + Inertia + Vue 3 | `/fuglekamera` page with WebRTC player + live stats sidebar |
| [Laravel Reverb](https://reverb.laravel.com/) | Broadcasts bird stats to connected browsers in real time |

### Public page features

- WebRTC video player using WHEP protocol (lowest latency)
- Live stats sidebar: current bird count, today's visits, species list with confidence
- Real-time updates via Reverb websockets — no polling
- LIVE/OFFLINE badge based on stream connectivity
- All text in Norwegian

---

## Species List

The file `models/norwegian_species.txt` (Pi) and `nuc/norwegian_species.txt` (NUC) define which species the system recognizes. Format: `English Name | Norsk navn`, one per line.

```text
Great Tit | Kjøttmeis
Blue Tit | Blåmeis
Eurasian Magpie | Skjære
...
```

42 common Norwegian feeder birds are included. To add a species, just add a line — the classification server hot-reloads the file automatically. No retraining needed.

Species names follow the [IOC World Bird List](https://www.worldbirdnames.org/) (English) and [Norsk Ornitologisk Forening](https://www.birdlife.no/) (Norwegian).

---

## How Classification Works

Unlike a traditional classifier with fixed output classes, BioCLIP uses zero-shot classification — it can identify things it wasn't explicitly trained on by matching images against text descriptions, rather than being limited to a fixed list baked into the model.

1. At startup, species names are loaded and turned into text prompts ("a photo of a Great Tit", ...)
2. All prompts are encoded into text embeddings (cached in memory)
3. When a bird crop arrives, it's encoded into an image embedding
4. Cosine similarity is computed between the image and all species text embeddings
5. Softmax gives a probability for each species — the top match is returned

This means adding a new species is just adding one line to a text file. No model retraining.

---

## Credits and Acknowledgments

### Models

- **[YOLOv8s](https://github.com/ultralytics/ultralytics)** by Ultralytics — real-time object detection model running on the Hailo-8 accelerator. Trained on the [COCO dataset](https://cocodataset.org/) (80 classes including "bird"). License: AGPL-3.0.

- **[BioCLIP](https://huggingface.co/imageomics/bioclip)** by the [Imageomics Institute](https://imageomics.org/) — vision foundation model for biological image classification, built on OpenAI's CLIP (ViT-B/16) and fine-tuned on the [TreeOfLife-10M](https://huggingface.co/datasets/imageomics/TreeOfLife-10M) dataset (10M+ biological images, 450K+ taxa). CVPR 2024 Best Student Paper. License: MIT.
  - Paper: [BioCLIP: A Vision Foundation Model for the Tree of Life](https://arxiv.org/abs/2311.18803)
  - Authors: Samuel Stevens, Jiaman Wu, Matthew J. Thompson, Elizabeth G. Campolongo, Chan Hee Song, David Edward Carlyn, Li Dong, Wasila M. Dahdul, Charles Stewart, Tanya Berger-Wolf, Wei-Lun Chao, Yu Su — The Ohio State University
  - GitHub: [Imageomics/BioCLIP](https://github.com/Imageomics/BioCLIP)

### Libraries and Tools

- **[OpenCLIP](https://github.com/mlfoundations/open_clip)** — open source implementation of CLIP used to run BioCLIP. License: MIT.
- **[Hailo SDK](https://hailo.ai/)** — neural network compiler and runtime for the Hailo-8 AI accelerator.
- **[picamera2](https://github.com/raspberrypi/picamera2)** — Python interface for Raspberry Pi cameras.
- **[OpenCV](https://opencv.org/)** — computer vision library for image processing and video capture.
- **[Flask](https://flask.palletsprojects.com/)** — lightweight Python web framework for the classification API.
- **[mediamtx](https://github.com/bluenviron/mediamtx)** by bluenviron — zero-dependency media server for RTMP, WebRTC, HLS, and more. License: MIT.
- **[FFmpeg](https://ffmpeg.org/)** — video encoding and streaming toolkit.
- **[hls.js](https://github.com/video-dev/hls.js)** — HLS client for browsers without native support.
- **[Laravel Reverb](https://reverb.laravel.com/)** — first-party WebSocket server for Laravel.

### Datasets

- **[COCO](https://cocodataset.org/)** (Common Objects in Context) — object detection dataset used to train YOLOv8.
- **[TreeOfLife-10M](https://huggingface.co/datasets/imageomics/TreeOfLife-10M)** — 10M biological images from iNaturalist, BIOSCAN-1M, and Encyclopedia of Life, used to train BioCLIP.

### Platforms

- **[Hugging Face](https://huggingface.co/)** — model hosting and distribution for BioCLIP.
- **[Raspberry Pi](https://www.raspberrypi.com/)** — Pi 5 hardware and software ecosystem.

---

## Project Log

Detailed build log with discoveries, architecture decisions, and implementation notes:
[BIRD_PROJECT_LOG.md](BIRD_PROJECT_LOG.md)
