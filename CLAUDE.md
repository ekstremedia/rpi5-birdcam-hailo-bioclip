# Bird Feeder Camera Project

## What this is
Raspberry Pi 5 + AI HAT+ (Hailo-8) project to recognize and count birds at a bird feeder.

## Current status
- AI HAT+ installed and detected (Hailo-8, fw 4.23.0)
- Camera: Canon LEGRIA HF G25 via Elgato Cam Link 4K (1920x1080 @ 25fps)
- `bird_monitor.py` — full detection + tracking + remote species classification + web dashboard
- Running YOLOv8s on Hailo at ~22 FPS, filtering for birds (COCO class 14)
- Species classification via BioCLIP on NUC (remote API, ~0.2s per bird)
- Low-confidence birds (<70%) auto-reclassified up to 3 times, keeping best result
- Web dashboard at http://PI_IP:8888 with live stream, species labels, counts, and API
- Video overlay: Norwegian date/time, bird count, FPS, species summary on transparent bar
- Config via `.env` file (camera source, API URL, thresholds, etc.)

## Project log
All progress is tracked in BIRD_PROJECT_LOG.md - update it as we go.

## Preferences
- Log everything we do to BIRD_PROJECT_LOG.md with dates
- This is a learning/hobby project - explain things along the way
