#!/usr/bin/env python3
"""HTTP MJPEG streaming server using rpicam-vid with native Hailo AI detection."""

import os
import subprocess
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

PORT = 8888

# Shared state
latest_frame = None
frame_lock = threading.Lock()
frame_event = threading.Event()


def camera_reader():
    """Read MJPEG frames from rpicam-vid stdout."""
    global latest_frame
    cmd = [
        "rpicam-vid", "-t", "0",
        "--width", "1280", "--height", "960",
        "--framerate", "30",
        "--codec", "mjpeg",
        "--flush",
        "--post-process-file",
        "/usr/share/rpi-camera-assets/hailo_yolov8_inference.json",
        "-o", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=0)
    fd = proc.stdout.fileno()
    buf = bytearray()
    while True:
        data = os.read(fd, 65536)
        if not data:
            break
        buf.extend(data)
        while True:
            start = buf.find(b"\xff\xd8")
            if start == -1:
                buf = bytearray()
                break
            end = buf.find(b"\xff\xd9", start + 2)
            if end == -1:
                if start > 0:
                    del buf[:start]
                break
            frame = bytes(buf[start:end + 2])
            del buf[:end + 2]
            with frame_lock:
                latest_frame = frame
            frame_event.set()


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"""<!DOCTYPE html>
<html><head><title>Bird Cam</title></head>
<body style="margin:0;background:#000;display:flex;align-items:center;justify-content:center;height:100vh">
<img src="/stream" style="max-width:100%;max-height:100vh">
</body></html>""")
            return

        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            try:
                while True:
                    frame_event.wait(timeout=1.0)
                    with frame_lock:
                        frame = latest_frame
                    if frame is None:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(frame)}\r\n".encode())
                    self.wfile.write(b"\r\n")
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    time.sleep(0.033)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    t = threading.Thread(target=camera_reader, daemon=True)
    t.start()

    print("Starting camera with Hailo AI...", flush=True)
    frame_event.wait(timeout=30)
    if latest_frame:
        print(f"First frame received ({len(latest_frame)} bytes)", flush=True)
    else:
        print("WARNING: No frames received yet", flush=True)

    server = ThreadedHTTPServer(("0.0.0.0", PORT), StreamHandler)
    print(f"Stream ready at http://0.0.0.0:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
