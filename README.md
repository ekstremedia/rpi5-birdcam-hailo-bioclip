# Bird Feeder Camera

AI-powered bird feeder monitor that detects, tracks, and identifies bird species in real time.

Two machines work together: a **Raspberry Pi 5** handles detection and tracking at the feeder,
while a **classification server** runs the species identification model over the network.

## Architecture

```text
┌─────────────────────────────┐         ┌──────────────────────────────┐
│  Raspberry Pi 5             │         │  Classification Server       │
│                             │         │                              │
│  Camera (IMX477 / Elgato)   │         │  BioCLIP (ViT-B/16)         │
│  → YOLOv8s on Hailo-8      │  HTTP   │  via open_clip + Flask       │
│  → Bird detected? Crop it   ├────────►│  → Species identification   │
│  → Centroid tracker         │  ~0.2s  │  → Norwegian name returned  │
│  → Web dashboard :8888      │◄────────┤                              │
│  → SQLite logging           │         │  Docker · port 5555          │
└─────────────────────────────┘         └──────────────────────────────┘
```

**Detection**: YOLOv8s runs on the Hailo-8 AI accelerator at 30 FPS, filtering for COCO class 14 (bird).

**Tracking**: Centroid-based tracker counts arrivals and departures, logs each visit to SQLite.

**Classification**: When a bird arrives, its crop is sent to the classification server. BioCLIP
(zero-shot, 42 Norwegian species) returns the species name in ~0.2s.

**Dashboard**: Live MJPEG stream with bounding boxes, species labels, visit counter, and FPS overlay.

## Raspberry Pi 5

### Hardware

- Raspberry Pi 5 (8GB)
- Hailo-8 AI HAT+ (26 TOPS neural accelerator)
- Camera: Raspberry Pi HQ Camera (IMX477) or Elgato Cam Link 4K
- OS: Debian 13 (trixie)

### Files

| File | Description |
|------|-------------|
| `bird_monitor.py` | Main application — detection, tracking, web dashboard |
| `stream.py` | Simple MJPEG stream with Hailo AI overlay (fallback/testing) |
| `species_classifier.py` | Local BioCLIP classifier module (unused now, kept for reference) |
| `models/norwegian_species.txt` | 42 bird species (English + Norwegian names) |
| `.env.example` | Configuration template |

### Setup

```bash
# Clone the repo
git clone https://github.com/ekstremedia/pi5-ai.git
cd pi5-ai

# Copy and edit config
cp .env.example .env
# Edit .env — set CAMERA_SOURCE, SPECIES_API_URL, paths, etc.

# Required system packages (should already be installed on Pi OS)
sudo apt install python3-picamera2 python3-opencv python3-pil
```

### Configuration

All settings are in `.env` (see `.env.example` for defaults):

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_SOURCE` | `elgato` | `elgato` or `picamera2` |
| `SPECIES_API_URL` | `http://192.168.1.64:5555` | Classification server address |
| `PORT` | `8888` | Web dashboard port |
| `CONFIDENCE_THRESHOLD` | `0.4` | Minimum detection confidence |
| `CROP_DIR` | `/home/pi/ai/bird_crops` | Where bird crop images are saved |
| `DB_PATH` | `/home/pi/ai/birds.db` | SQLite database path |

### Running

```bash
# Start the bird monitor
python3 bird_monitor.py

# Run in background
nohup python3 bird_monitor.py > /tmp/bird-monitor.log 2>&1 &

# Stop
pkill -f bird_monitor.py
```

Then open `http://PI_IP:8888` in a browser.

### Web dashboard

| Endpoint | Description |
|----------|-------------|
| `/` | Live dashboard with stream, bird count, recent visitors |
| `/stream` | Raw MJPEG stream with bounding box overlays |
| `/label` | Web UI for manually labeling bird crops (training data) |
| `/api/stats` | JSON: current birds, today's visits, FPS |
| `/api/birds` | JSON: recent bird visits |

### HUD overlay

The video stream shows:
- Green bounding boxes around detected birds with species labels
- **Fugler nå**: Current bird count
- **I dag**: Today's visit count
- **FPS**: Processing frame rate
- **Klassifiseringsmotor: På/Frakoblet**: Whether the classification server is reachable

## NUC Services

A separate machine (NUC, any x86 box with a few GB of RAM) runs three Docker services.
All files are in the [`nuc/`](nuc/) directory — see [`nuc/README.md`](nuc/README.md) for full details.

| Service | Purpose |
|---------|---------|
| `species-api` | BioCLIP species classification API (port 5555) |
| `stream-relay` | ffmpeg MJPEG→H.264→RTMP relay to VPS |
| `stats-pusher` | Polls Pi stats, pushes to Laravel via Reverb |

### Quick setup

```bash
cd nuc
cp .env.example .env
# Edit .env with your values

docker compose up -d

# Verify
curl http://localhost:5555/health
# → {"status":"ok","model":"bioclip","species_count":42}
```

### API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/classify` | POST | Send JPEG bytes, get `{species, species_en, confidence, inference_time}` |
| `/health` | GET | Status + species count |

### How classification works

BioCLIP is a CLIP model fine-tuned on 10M biological images. It matches bird crops against
text prompts ("a photo of a Great Tit", "a photo of a Blue Tit", ...) using cosine similarity.
No training needed — just list species names in `norwegian_species.txt`.

## Species list

The file `models/norwegian_species.txt` defines which species the system recognizes.
Format: `English Name | Norsk navn`, one per line.

```text
Great Tit | Kjøttmeis
Blue Tit | Blåmeis
Eurasian Magpie | Skjære
...
```

42 common Norwegian feeder birds are included. To add a species, just add a line — the
classification server hot-reloads the file automatically.

## Project log

Detailed build log with discoveries, architecture decisions, and implementation notes:
[BIRD_PROJECT_LOG.md](BIRD_PROJECT_LOG.md)
