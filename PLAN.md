# Bird Feeder Live Stream — Project Plan

## Overview

A three-machine system to detect and classify birds at a feeder, then stream the annotated video live to a public website via WebRTC.

**Machines:**

| Machine | Role | Key Hardware |
|---------|------|-------------|
| **Pi 5** (192.168.1.176) | Camera capture + real-time AI detection | Hailo-8 AI HAT+ (26 TOPS), Canon LEGRIA HF G25 via Elgato Cam Link 4K, Raspberry Pi HQ Camera (backup) |
| **NUC** (Intel i7-6770HQ, 32GB RAM, Debian 13) | Species classification + H.264 encoding + stream relay | AVX2 CPU, Iris Pro 580 GPU, 500GB NVMe |
| **VPS** (public web server) | Public stream delivery + web page | mediamtx, nginx, existing web server |

**Data flow:**

```
Canon G25 → Elgato Cam Link 4K → Pi 5
                                    │
                    ┌───────────────┤
                    │               │
              Hailo YOLOv8    On bird arrival:
              (30fps detection)   POST crop to NUC
                    │               │
                    │               ▼
              Annotated        NUC: BioCLIP API
              MJPEG stream     (species in ~0.5-1s)
              on LAN               │
                    │          species label
                    │          returned to Pi
                    ▼          (updates overlay)
                   NUC
                    │
              ffmpeg: read MJPEG
              encode H.264
              push RTMP
                    │
                    ▼
                   VPS
              mediamtx receives RTMP
                    │
              ┌─────┴─────┐
              │            │
           WebRTC        HLS
           (low latency) (fallback)
              │            │
              └─────┬──────┘
                    ▼
              Public web page
              https://yoursite.com/birdcam
```

---

## Phase 1: Camera Switch (Pi 5)

### Goal
Replace picamera2 (CSI camera) with OpenCV VideoCapture reading from the Elgato Cam Link 4K (HDMI capture from Canon G25).

### Prerequisites
- Canon LEGRIA HF G25 connected via mini HDMI → Elgato Cam Link 4K → USB 3.0 port on Pi 5
- G25 set to output mode with all OSD overlays disabled (no battery icon, no recording indicator)
- G25 powered via AC adapter (continuous operation)

### Tasks

#### 1.1 Verify Elgato Cam Link on Pi 5
```bash
# Check device appears
v4l2-ctl --list-devices

# Verify USB 3.0 connection (should show 5000M)
lsusb -t

# Check supported formats
v4l2-ctl -d /dev/videoX --list-formats-ext

# Test capture
ffmpeg -f v4l2 -i /dev/videoX -frames 1 test_elgato.jpg
```

#### 1.2 Modify bird_monitor.py — Camera Abstraction

Add a config section at the top of `bird_monitor.py`:

```python
# --- Camera Configuration ---
CAMERA_SOURCE = "elgato"  # "elgato" or "picamera2"
ELGATO_DEVICE = "/dev/video0"  # adjust based on v4l2-ctl output
ELGATO_WIDTH = 1280
ELGATO_HEIGHT = 720
```

Replace the picamera2 camera initialization with a camera abstraction:

```python
if CAMERA_SOURCE == "elgato":
    import cv2
    cap = cv2.VideoCapture(ELGATO_DEVICE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ELGATO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ELGATO_HEIGHT)
    # Frame capture: ret, frame = cap.read()  → BGR numpy array
else:
    # Existing picamera2 code (keep as fallback)
    from picamera2 import Picamera2
    # ...existing setup...
```

The Hailo inference stays the same regardless of camera source — it just needs a numpy array resized to 640x640.

#### 1.3 Decouple Hailo from picamera2

Currently, Hailo inference may be tightly coupled to picamera2's pipeline. Refactor to use the Hailo device independently:

```python
from picamera2.devices import Hailo

hailo = Hailo("yolov8s_h8l.hf")

def run_detection(frame):
    """Run YOLOv8 on a BGR numpy array, return detections."""
    input_frame = cv2.resize(frame, (640, 640))
    results = hailo.run(input_frame)
    # results is list of 80 numpy arrays, one per COCO class
    # Each array shape (N, 5): [y_min, x_min, y_max, x_max, confidence]
    return results
```

#### 1.4 Test End-to-End
- Verify G25 → Elgato → Pi 5 → Hailo detection pipeline works
- Confirm bounding boxes draw correctly on the HDMI-captured frames
- Check FPS is stable (target: 15-30fps)
- Verify bird detection works as before (COCO class 14)

### Files Modified
- `bird_monitor.py` — camera abstraction, config options

---

## Phase 2: BioCLIP Classification API (NUC)

### Goal
Move species classification off the Pi and onto the NUC as an HTTP API, reducing classification time from ~4s to ~0.5-1s.

### Prerequisites
- Python 3.11+ on NUC
- Network connectivity between Pi and NUC (same LAN or VPN)

### Tasks

#### 2.1 Install Dependencies on NUC
```bash
# PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --break-system-packages

# BioCLIP
pip install open_clip_torch --break-system-packages

# API server
pip install flask pillow --break-system-packages

# Optional: Intel acceleration
pip install openvino openvino-tokenizers --break-system-packages
```

#### 2.2 Create Species Classification API

Create `/home/terje/birdcam/species_api.py` on the NUC:

```python
"""
BioCLIP Species Classification API
Receives bird crop images via POST, returns species + confidence.
Runs on NUC for fast inference (~0.5-1s vs ~4s on Pi 5).
"""
from flask import Flask, request, jsonify
import open_clip
from PIL import Image
import torch
import io
import time
import os

app = Flask(__name__)

# --- Configuration ---
SPECIES_FILE = os.path.join(os.path.dirname(__file__), 'norwegian_species.txt')
HOST = '0.0.0.0'
PORT = 5555

# --- Model Loading ---
print("Laster BioCLIP-modell...")
start = time.time()
model, _, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
model.eval()
print(f"Modell lastet på {time.time() - start:.1f}s")

# --- Species List ---
def load_species():
    with open(SPECIES_FILE, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

species_list = load_species()
species_mtime = os.path.getmtime(SPECIES_FILE)

# Precompute text embeddings
def compute_text_features(species):
    texts = tokenizer(species)
    with torch.no_grad():
        features = model.encode_text(texts)
        features /= features.norm(dim=-1, keepdim=True)
    return features

text_features = compute_text_features(species_list)
print(f"{len(species_list)} arter lastet")

@app.route('/classify', methods=['POST'])
def classify():
    """Classify a bird crop image. POST raw JPEG/PNG bytes."""
    global text_features, species_list, species_mtime

    # Hot-reload species file if changed
    try:
        mtime = os.path.getmtime(SPECIES_FILE)
        if mtime != species_mtime:
            species_list = load_species()
            text_features = compute_text_features(species_list)
            species_mtime = mtime
            print(f"Artsliste oppdatert: {len(species_list)} arter")
    except Exception as e:
        print(f"Feil ved oppdatering av artsliste: {e}")

    start = time.time()

    img = Image.open(io.BytesIO(request.data)).convert('RGB')
    img_tensor = preprocess_val(img).unsqueeze(0)

    with torch.no_grad():
        img_features = model.encode_image(img_tensor)
        img_features /= img_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * img_features @ text_features.T).softmax(dim=-1)

    top_idx = probs[0].argmax().item()
    confidence = probs[0][top_idx].item()
    elapsed = time.time() - start

    result = {
        'species': species_list[top_idx],
        'confidence': round(confidence, 3),
        'inference_time': round(elapsed, 3)
    }
    print(f"Klassifisert: {result['species']} ({confidence:.1%}) på {elapsed:.2f}s")
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model': 'BioCLIP',
        'species_count': len(species_list)
    })

if __name__ == '__main__':
    app.run(host=HOST, port=PORT)
```

#### 2.3 Copy Species File to NUC
Copy `models/norwegian_species.txt` from the Pi to `/home/terje/birdcam/norwegian_species.txt` on the NUC. Keep both in sync.

#### 2.4 Create systemd Service for Species API

Create `/etc/systemd/system/species-api.service`:

```ini
[Unit]
Description=BioCLIP Species Classification API
After=network.target

[Service]
User=terje
WorkingDirectory=/home/terje/birdcam
ExecStart=/usr/bin/python3 /home/terje/birdcam/species_api.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now species-api
```

#### 2.5 Modify bird_monitor.py — Remote Classification (Pi 5)

Replace the local BioCLIP classification in `bird_monitor.py` with an HTTP call to the NUC:

```python
import requests

# --- Classification Configuration ---
CLASSIFICATION_MODE = "remote"  # "remote" or "local"
CLASSIFICATION_URL = "http://NUC_IP:5555/classify"

def classify_bird_remote(crop_bytes):
    """Send crop to NUC for species classification."""
    try:
        r = requests.post(CLASSIFICATION_URL, data=crop_bytes, timeout=10,
                         headers={'Content-Type': 'application/octet-stream'})
        if r.status_code == 200:
            result = r.json()
            return result['species'], result['confidence']
    except requests.exceptions.RequestException as e:
        print(f"Klassifiseringsfeil: {e}")
    return None, 0.0
```

In the background classification thread, replace the local BioCLIP call:

```python
def classification_worker(crop_bytes, visit_id):
    if CLASSIFICATION_MODE == "remote":
        species, confidence = classify_bird_remote(crop_bytes)
    else:
        species, confidence = classify_bird_local(crop_bytes)  # existing code

    if species:
        update_species_label(visit_id, species, confidence)
```

#### 2.6 Test Classification API
```bash
# From the Pi, test with a saved crop:
curl -X POST --data-binary @bird_crops/2026-02-19/some_crop.jpg \
  http://NUC_IP:5555/classify

# Expected response:
# {"species": "Kjøttmeis (Great Tit)", "confidence": 0.92, "inference_time": 0.7}

# Health check:
curl http://NUC_IP:5555/health
```

### Files Created (NUC)
- `/home/terje/birdcam/species_api.py`
- `/home/terje/birdcam/norwegian_species.txt`
- `/etc/systemd/system/species-api.service`

### Files Modified (Pi 5)
- `bird_monitor.py` — remote classification support

---

## Phase 3: Stream Relay (NUC → VPS)

### Goal
The NUC reads the Pi's MJPEG stream, encodes it to H.264, and pushes it via RTMP to the VPS where mediamtx serves it as WebRTC.

### Tasks

#### 3.1 Install ffmpeg on NUC
```bash
sudo apt install ffmpeg
```

#### 3.2 Create Stream Relay Script on NUC

Create `/home/terje/birdcam/stream_relay.sh`:

```bash
#!/bin/bash
# Reads MJPEG from Pi's bird_monitor.py, encodes H.264, pushes RTMP to VPS
# Restarts automatically on failure

PI_STREAM="http://192.168.1.176:8888/stream"
VPS_RTMP="rtmp://YOUR_VPS_IP:1935/birdcam"

# RTMP auth (must match mediamtx config)
# If using auth: VPS_RTMP="rtmp://pi:yourSecretHere@YOUR_VPS_IP:1935/birdcam"

while true; do
    echo "$(date): Starter strømming til VPS..."

    ffmpeg -hide_banner -loglevel warning \
        -f mjpeg -i "$PI_STREAM" \
        -c:v libx264 \
        -preset ultrafast \
        -tune zerolatency \
        -b:v 1500k \
        -maxrate 1500k \
        -bufsize 3000k \
        -g 30 \
        -keyint_min 30 \
        -sc_threshold 0 \
        -f flv \
        "$VPS_RTMP"

    echo "$(date): Strøm avbrutt, prøver igjen om 5 sekunder..."
    sleep 5
done
```

```bash
chmod +x /home/terje/birdcam/stream_relay.sh
```

#### 3.3 Create systemd Service for Stream Relay

Create `/etc/systemd/system/stream-relay.service`:

```ini
[Unit]
Description=Bird Feeder Stream Relay (MJPEG → H.264 → RTMP)
After=network.target species-api.service

[Service]
User=terje
ExecStart=/home/terje/birdcam/stream_relay.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now stream-relay
```

#### 3.4 Tuning Notes

- **Bitrate:** 1500kbps is good for a mostly-static bird feeder scene. Reduce to 800-1000kbps if upload bandwidth is limited. Increase to 2500kbps for more detail.
- **Resolution:** The Pi serves 1280x960 (or 1280x720 with Elgato). No rescaling needed.
- **FPS:** ffmpeg will match the source framerate. If CPU load is too high on the NUC (unlikely), add `-r 15` to cap at 15fps.
- **Bandwidth:** At 1500kbps ≈ 650 MB/hour upload from home. Verify your upload speed can sustain this.

### Files Created (NUC)
- `/home/terje/birdcam/stream_relay.sh`
- `/etc/systemd/system/stream-relay.service`

---

## Phase 4: mediamtx on VPS

### Goal
Install and configure mediamtx on the VPS to receive the RTMP stream from the NUC and serve it to viewers via WebRTC (low latency) and HLS (fallback).

### Tasks

#### 4.1 Install mediamtx

```bash
cd /opt
sudo wget https://github.com/bluenviron/mediamtx/releases/download/v1.12.2/mediamtx_v1.12.2_linux_amd64.tar.gz
sudo tar xzf mediamtx_v1.12.2_linux_amd64.tar.gz
sudo rm mediamtx_v1.12.2_linux_amd64.tar.gz
```

Check for latest version at: https://github.com/bluenviron/mediamtx/releases

#### 4.2 Configure mediamtx

Edit `/opt/mediamtx.yml`:

```yaml
###############################################
# General settings

# Log level (debug, info, warn, error)
logLevel: info

###############################################
# RTMP server (receives stream from NUC)
rtmpAddress: :1935

###############################################
# WebRTC server (serves to viewers)
webrtcAddress: :8889

###############################################
# HLS server (fallback for older browsers)
hlsAddress: :8888

###############################################
# Stream paths

paths:
  birdcam:
    source: publisher

    # Authentication for stream publisher (NUC)
    publishUser: pi
    publishPass: CHANGE_THIS_TO_A_STRONG_PASSWORD

    # Allow anyone to view (no read auth)
    # readUser:
    # readPass:
```

#### 4.3 Create systemd Service

Create `/etc/systemd/system/mediamtx.service`:

```ini
[Unit]
Description=mediamtx Media Server
After=network.target

[Service]
ExecStart=/opt/mediamtx /opt/mediamtx.yml
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now mediamtx
```

#### 4.4 Firewall Rules

Open the required ports (adjust for your firewall tool):

```bash
# RTMP ingest (only allow from your home IP for security)
sudo ufw allow from YOUR_HOME_IP to any port 1935

# WebRTC (public)
sudo ufw allow 8889/tcp

# HLS (public, optional fallback)
sudo ufw allow 8888/tcp

# WebRTC uses UDP for media (allow a range)
sudo ufw allow 8189/udp
```

Better yet: use a WireGuard tunnel between NUC and VPS for the RTMP ingest so port 1935 isn't exposed at all.

#### 4.5 nginx Reverse Proxy (HTTPS)

Add to your nginx config to serve WebRTC behind HTTPS on your domain:

```nginx
# WebRTC WHEP endpoint (for the JS player)
location /birdcam/whep {
    proxy_pass http://127.0.0.1:8889/birdcam/whep;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}

# WebRTC WHIP endpoint (if needed)
location /birdcam/whip {
    proxy_pass http://127.0.0.1:8889/birdcam/whip;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
}

# ICE candidates (WebRTC signaling)
location ~ ^/birdcam/whep/.*$ {
    proxy_pass http://127.0.0.1:8889;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}

# HLS fallback
location /birdcam/ {
    proxy_pass http://127.0.0.1:8888/birdcam/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
}

# mediamtx built-in player (optional, for quick testing)
location /birdcam/player {
    proxy_pass http://127.0.0.1:8889/birdcam;
    proxy_http_version 1.1;
}
```

#### 4.6 Test mediamtx

```bash
# Check service is running
sudo systemctl status mediamtx

# Check logs
sudo journalctl -u mediamtx -f

# Test the built-in player (after stream is active)
# Visit: http://YOUR_VPS_IP:8889/birdcam
```

### Files Created (VPS)
- `/opt/mediamtx` (binary)
- `/opt/mediamtx.yml` (config)
- `/etc/systemd/system/mediamtx.service`
- nginx config additions

---

## Phase 5: Public Web Page (VPS)

### Goal
A public-facing web page that shows the live bird feeder stream via WebRTC with HLS fallback.

### Tasks

#### 5.1 Create the Live Page

Create a page on your web server (location depends on your setup). Core HTML:

```html
<!DOCTYPE html>
<html lang="no">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fuglekamera — Live</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #1a1a2e;
            color: #e0e0e0;
            font-family: system-ui, -apple-system, sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        header {
            padding: 1.5rem;
            text-align: center;
        }
        h1 { font-size: 1.8rem; color: #fff; }
        .subtitle { color: #8888aa; margin-top: 0.3rem; }
        .video-container {
            width: 100%;
            max-width: 960px;
            aspect-ratio: 4/3;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            position: relative;
        }
        video {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .status {
            position: absolute;
            top: 12px;
            right: 12px;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .status.live { background: #e53935; color: #fff; }
        .status.offline { background: #555; color: #ccc; }
        .info {
            max-width: 960px;
            padding: 1.5rem;
            text-align: center;
            color: #8888aa;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>

<header>
    <h1>🐦 Fuglekamera — Vesterålen</h1>
    <p class="subtitle">AI-drevet fuglematerstasjon med artsgjenkjenning</p>
</header>

<div class="video-container">
    <video id="birdcam" autoplay muted playsinline></video>
    <div class="status offline" id="status">OFFLINE</div>
</div>

<div class="info">
    <p>Strømmen bruker WebRTC for lav forsinkelse. Fugler oppdages med YOLOv8 på Hailo-8, arter klassifiseres med BioCLIP.</p>
</div>

<script>
    const video = document.getElementById('birdcam');
    const status = document.getElementById('status');
    const whepUrl = '/birdcam/whep';  // via nginx proxy

    async function startWebRTC() {
        const pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        });

        pc.ontrack = (event) => {
            video.srcObject = event.streams[0];
            status.textContent = 'LIVE';
            status.className = 'status live';
        };

        pc.oniceconnectionstatechange = () => {
            if (pc.iceConnectionState === 'disconnected' || pc.iceConnectionState === 'failed') {
                status.textContent = 'OFFLINE';
                status.className = 'status offline';
                // Retry after delay
                setTimeout(startWebRTC, 5000);
            }
        };

        pc.addTransceiver('video', { direction: 'recvonly' });

        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        try {
            const response = await fetch(whepUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/sdp' },
                body: pc.localDescription.sdp
            });

            if (response.ok) {
                const answer = await response.text();
                await pc.setRemoteDescription({ type: 'answer', sdp: answer });

                // Handle ICE candidates from server (trickle ICE via WHEP)
                const location = response.headers.get('Location');
                if (location) {
                    // WHEP spec: poll for server ICE candidates
                    const patchResponse = await fetch(location, {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/trickle-ice-sdpfrag' },
                        body: ''
                    });
                }
            } else {
                // Stream not available, retry
                setTimeout(startWebRTC, 5000);
            }
        } catch (e) {
            console.error('WebRTC error:', e);
            setTimeout(startWebRTC, 5000);
        }
    }

    startWebRTC();
</script>

</body>
</html>
```

> **Note:** The WHEP client implementation above is simplified. mediamtx also provides a ready-made player page at `http://VPS:8889/birdcam` which handles all the WebRTC negotiation. You can iframe that, or use it as reference for a more robust WHEP client. There are also lightweight WHEP client libraries available on npm/GitHub.

#### 5.2 HLS Fallback (Optional)

For browsers that don't support WebRTC or when latency doesn't matter, add an HLS fallback using hls.js:

```html
<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
<script>
    // If WebRTC fails, fall back to HLS
    function startHLS() {
        if (Hls.isSupported()) {
            const hls = new Hls();
            hls.loadSource('/birdcam/index.m3u8');
            hls.attachMedia(video);
            status.textContent = 'LIVE (HLS)';
            status.className = 'status live';
        }
    }
</script>
```

### Files Created (VPS)
- Live page HTML (location depends on your web server setup)

---

## Phase 6: Autostart & Reliability

### Goal
All services start on boot and recover from failures.

### Pi 5 — bird_monitor.py

Create `/etc/systemd/system/bird-monitor.service`:

```ini
[Unit]
Description=Bird Feeder AI Monitor
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/ai
ExecStart=/usr/bin/python3 /home/pi/ai/bird_monitor.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

### NUC Services
Already defined in Phase 2 and Phase 3:
- `species-api.service` — BioCLIP classification API
- `stream-relay.service` — ffmpeg MJPEG → H.264 → RTMP relay

### VPS Service
Already defined in Phase 4:
- `mediamtx.service` — media relay server

### Startup Order
1. Pi boots → `bird-monitor.service` starts → camera + Hailo detection running, MJPEG stream available
2. NUC boots → `species-api.service` starts → BioCLIP ready for classification requests
3. NUC boots → `stream-relay.service` starts → connects to Pi MJPEG, encodes, pushes RTMP to VPS
4. VPS (always running) → `mediamtx.service` → receives RTMP, serves WebRTC to viewers

---

## File Structure Summary

### Pi 5 (`/home/pi/ai/`)
```
ai/
├── bird_monitor.py          # Main app (modified: camera abstraction + remote classification)
├── stream.py                # Original simple stream (kept as fallback)
├── species_classifier.py    # Local BioCLIP (kept as fallback, no longer primary)
├── models/
│   └── norwegian_species.txt
├── birds.db                 # SQLite database
└── bird_crops/              # Saved crop images
```

### NUC (`/home/terje/birdcam/`)
```
birdcam/
├── species_api.py           # BioCLIP HTTP API
├── stream_relay.sh          # ffmpeg relay script
└── norwegian_species.txt    # Species list (copy from Pi)
```

### VPS (`/opt/`)
```
/opt/
├── mediamtx                 # Binary
└── mediamtx.yml             # Config

/var/www/yoursite/            # Or wherever your web root is
└── birdcam/
    └── index.html           # Public live page
```

---

## Implementation Order

| Step | Machine | What | Depends On |
|------|---------|------|-----------|
| 1 | Pi 5 | Camera switch: Elgato Cam Link support in bird_monitor.py | Canon G25 + Elgato connected |
| 2 | NUC | Install Python + PyTorch + BioCLIP + Flask | Debian 13 installed |
| 3 | NUC | Create and test `species_api.py` | Step 2 |
| 4 | Pi 5 | Switch bird_monitor.py to remote classification | Step 3 (NUC API running) |
| 5 | VPS | Install and configure mediamtx | None |
| 6 | NUC | Create `stream_relay.sh`, test ffmpeg → VPS | Step 5 (mediamtx running) + Step 1 (Pi stream) |
| 7 | VPS | nginx reverse proxy + HTTPS | Step 5 |
| 8 | VPS | Create public web page | Step 7 |
| 9 | All | systemd services + autostart | Steps 1-8 working |

Steps 1-4 (camera + classification) and Steps 5-8 (streaming) can be done in parallel.

---

## Security Considerations

- **RTMP ingest:** Restrict port 1935 on VPS to your home IP, or better: use WireGuard tunnel between NUC and VPS
- **mediamtx auth:** Set publishUser/publishPass to prevent unauthorized stream injection
- **BioCLIP API:** Only accessible on LAN (or WireGuard between Pi and NUC if on different networks)
- **HTTPS:** Serve the public web page and WebRTC endpoint behind HTTPS via nginx

---

## Monitoring & Debugging

```bash
# Pi — check bird monitor
sudo journalctl -u bird-monitor -f
curl http://localhost:8888/api/stats

# NUC — check classification API
sudo journalctl -u species-api -f
curl http://localhost:5555/health

# NUC — check stream relay
sudo journalctl -u stream-relay -f

# VPS — check mediamtx
sudo journalctl -u mediamtx -f
# mediamtx also has an API: http://localhost:9997/v3/paths/list
```

---

## Future Enhancements

- **Stats API on VPS:** Forward bird stats from Pi → VPS so the web page can show species counts, daily visitors, etc.
- **Dashboard proxy:** Proxy the Pi's dashboard through the VPS so the labeling UI and stats are publicly accessible
- **Recording:** Configure mediamtx to record segments, or use ffmpeg on the NUC to save timestamped clips
- **Multi-camera:** Add the Pi HQ Camera as a second angle, switch between feeds
- **Notifications:** Push notifications when rare species are detected
- **OpenVINO:** Optimize BioCLIP inference on NUC using Intel's OpenVINO for faster classification

