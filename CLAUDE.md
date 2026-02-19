# Bird Feeder Camera Project

## What this is
Raspberry Pi 5 + AI HAT+ (Hailo-8) project to recognize and count birds at a bird feeder.

## Current status
- AI HAT+ installed and detected (Hailo-8, fw 4.23.0)
- IMX477 HQ Camera connected and working (1280x960 @ 30fps)
- `bird_monitor.py` — full detection + tracking + species classification + web dashboard
- Running YOLOv8s on Hailo at 30 FPS, filtering for birds (COCO class 14)
- Species classification via EfficientNet-B7 ONNX (556 species, ~4s per classification on arrival)
- Web dashboard at http://PI_IP:8888 with live stream, species labels, counts, and API
- Retraining pipeline (`train_species.py`) ready for fine-tuning with labeled crops
- Next: point at bird feeder, test with real birds

## Project log
All progress is tracked in BIRD_PROJECT_LOG.md - update it as we go.

## Preferences
- Log everything we do to BIRD_PROJECT_LOG.md with dates
- This is a learning/hobby project - explain things along the way
