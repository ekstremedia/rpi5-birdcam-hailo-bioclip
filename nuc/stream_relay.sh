#!/bin/bash
# stream_relay.sh — Relay MJPEG from Pi → H.264 → RTMP to mediamtx on VPS
#
# Reads the bird_monitor MJPEG stream, re-encodes to H.264,
# and pushes via RTMP to the VPS for WebRTC/HLS distribution.

set -euo pipefail

PI_STREAM="${PI_STREAM:-http://192.168.1.176:8888/stream}"
VPS_RTMP="${VPS_RTMP:-rtmp://ekstremedia.no:1935/birdcam/live}"
RETRY_DELAY=5

echo "Stream relay starting"
echo "  Source: $PI_STREAM"
echo "  Target: $VPS_RTMP"

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ffmpeg..."

    ffmpeg -hide_banner -loglevel warning \
        -f mjpeg -reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 10 \
        -i "$PI_STREAM" \
        -c:v libx264 -preset ultrafast -tune zerolatency \
        -b:v 1500k -maxrate 1500k -bufsize 3000k \
        -g 50 -keyint_min 25 \
        -pix_fmt yuv420p \
        -f flv \
        "$VPS_RTMP" || true

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ffmpeg exited, restarting in ${RETRY_DELAY}s..."
    sleep "$RETRY_DELAY"
done
