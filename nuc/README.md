# Birdcam NUC Services

Docker Compose services running on the NUC to support the bird feeder camera system.
Part of the [pi5-ai](https://github.com/ekstremedia/pi5-ai) repo — clone and run from `nuc/`.

## Services

| Service | Image | Purpose |
|---------|-------|---------|
| `species-api` | Custom (BioCLIP) | Species classification API on port 5555 |
| `stream-relay` | linuxserver/ffmpeg | Relays Pi MJPEG → H.264 → RTMP to VPS |
| `stats-pusher` | python:3.11-slim | Polls Pi stats, POSTs to Laravel every 5s |

## Quick start

```bash
git clone https://github.com/ekstremedia/pi5-ai.git
cd pi5-ai/nuc
cp .env.example .env
# Edit .env with your values

docker compose up -d          # Start all services
docker compose ps             # Check status
docker compose logs -f        # Follow all logs
```

## Configuration (.env)

```bash
# BioCLIP model download (build-time only)
HF_TOKEN=hf_your_token

# Stream relay
PI_STREAM=http://192.168.1.176:8888/stream
VPS_RTMP=rtmp://185.14.97.143:1935/birdcam/live

# Stats pusher
PI_URL=http://192.168.1.176:8888
LARAVEL_URL=https://ekstremedia.no
LARAVEL_TOKEN=your_shared_secret_token
POLL_INTERVAL=5
```

## Architecture

```text
Pi 5 (192.168.1.176)          NUC (this machine)           VPS (ekstremedia.no)
────────────────────          ────────────────────          ────────────────────
bird_monitor.py               species-api :5555
  ├─ /stream (MJPEG) ──────→ stream-relay ──── RTMP ────→ mediamtx → WebRTC/HLS
  ├─ /api/stats      ──────→ stats-pusher ── HTTPS ─────→ Laravel → Reverb WS
  ├─ /api/birds      ──────→                               → /fuglekamera page
  └─ /classify (POST) ←────  BioCLIP API
```

## Per-service details

### stream-relay

Reads the Pi's MJPEG stream, encodes H.264 with libx264 (ultrafast, zerolatency, 1.5 Mbps), and pushes RTMP to mediamtx on the VPS. Auto-restarts on failure.

```bash
docker compose logs stream-relay --tail 20
docker compose restart stream-relay
```

### stats-pusher

Polls `/api/stats` and `/api/birds` on the Pi every 5 seconds, combines the data, and POSTs to the Laravel endpoint with Bearer token auth. Laravel caches the stats and broadcasts via Reverb websockets. Exponential backoff on errors (max 60s).

```bash
docker compose logs stats-pusher --tail 20
docker compose restart stats-pusher
```

### species-api

Flask API serving BioCLIP zero-shot species classification. Called by `bird_monitor.py` on the Pi when a bird arrives.

```bash
curl http://localhost:5555/health
curl -X POST --data-binary @bird.jpg http://localhost:5555/classify
```

## Prerequisites

- Pi running `bird_monitor.py` at the configured IP
- VPS running mediamtx (RTMP on :1935)
- Laravel endpoint deployed with matching Bearer token
