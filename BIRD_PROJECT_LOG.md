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
- Model: YOLOv8m on Hailo-8 (COCO 80 classes, includes "bird")
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

## Next Steps
- [ ] Point camera at bird feeder
- [ ] Adapt/build bird-specific detection + counting pipeline
- [ ] Set up auto-start on boot
