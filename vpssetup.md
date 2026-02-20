# VPS Setup — mediamtx + Apache proxy for birdcam stream

The VPS (185.14.97.143) runs Debian 11 with Apache2 and the Laravel project at `/home/terje/sites/nesthus-prod`. The birdcam stream is served at `https://ekstremedia.no/fuglekamera` — no subdomains needed.

## Architecture

```
NUC (home LAN)                         VPS (185.14.97.143)
─────────────                          ────────────────────
ffmpeg                                 mediamtx
  MJPEG from Pi → H.264 ──RTMP:1935──→  receives RTMP on :1935
                                         serves WebRTC on :8889
                                         serves HLS on :8888
                                              │
                                         Apache2 reverse proxy
                                         /birdcam/live/whep → :8889
                                         /birdcam/live/     → :8888
                                              │
                                         Laravel (Inertia + Vue 3)
                                         /fuglekamera → Fuglekamera.vue
                                              │
                                         Browser (WebRTC player)
```

## 1. Install mediamtx

```bash
cd /tmp
wget https://github.com/bluenviron/mediamtx/releases/download/v1.11.3/mediamtx_v1.11.3_linux_amd64.tar.gz
tar xzf mediamtx_v1.11.3_linux_amd64.tar.gz
sudo mv mediamtx /usr/local/bin/
```

## 2. Configure mediamtx

Write `/etc/mediamtx.yml`:

```yaml
logLevel: info

# RTMP — receives stream from NUC
rtmp: yes
rtmpAddress: :1935

# WebRTC — serves to browser clients (low latency)
webrtc: yes
webrtcAddress: :8889

# HLS — fallback for browsers without WebRTC
hls: yes
hlsAddress: :8888

# API
api: yes
apiAddress: :9997

# Path configuration
paths:
  birdcam/live:
    source: publisher
```

**Security**: Restrict RTMP port to the NUC's IP so only it can publish:
```bash
sudo ufw allow from <NUC_PUBLIC_IP> to any port 1935 proto tcp
```

## 3. systemd service

Create `/etc/systemd/system/mediamtx.service`:

```ini
[Unit]
Description=MediaMTX media server
After=network.target

[Service]
ExecStart=/usr/local/bin/mediamtx /etc/mediamtx.yml
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now mediamtx
```

## 4. Apache reverse proxy

Add these proxy rules to the **existing** SSL VirtualHost in `/etc/apache2/sites-enabled/000-default-le-ssl.conf`.

The key is that `ProxyPass` directives must be **directly in the VirtualHost** (not inside `<Location>` blocks) so they take priority over Laravel's `.htaccess` rewrite rules.

```apache
# Inside <VirtualHost *:443> block, before </VirtualHost>

# mediamtx stream proxy for /fuglekamera
ProxyPreserveHost On

ProxyPass /birdcam/live/whep http://127.0.0.1:8889/birdcam/live/whep
ProxyPassReverse /birdcam/live/whep http://127.0.0.1:8889/birdcam/live/whep

ProxyPassMatch ^/birdcam/live/whep/(.+)$ http://127.0.0.1:8889/birdcam/live/whep/$1
ProxyPassReverse /birdcam/live/whep/ http://127.0.0.1:8889/birdcam/live/whep/

ProxyPass /birdcam/live/ http://127.0.0.1:8888/birdcam/live/
ProxyPassReverse /birdcam/live/ http://127.0.0.1:8888/birdcam/live/
<Location "/birdcam/live/">
    Header set Access-Control-Allow-Origin "*"
    Header set Access-Control-Allow-Methods "GET, OPTIONS"
</Location>
```

Enable required modules and reload:

```bash
sudo a2enmod proxy_http headers
sudo apache2ctl configtest
sudo systemctl reload apache2
```

### Why direct ProxyPass, not inside Location blocks?

Laravel's `.htaccess` (`AllowOverride All`) rewrites all requests to `index.php`. If `ProxyPass` is inside a `<Location>` block, the `.htaccess` rewrite runs first and Laravel returns 404. Direct `ProxyPass` in the VirtualHost runs before the filesystem handler.

## 5. Laravel route + Vue page

Route in `routes/web.php`:
```php
Route::get('/fuglekamera', fn () => Inertia::render('Public/Fuglekamera'))->name('fuglekamera');
```

Vue page at `resources/js/pages/Public/Fuglekamera.vue`:
- WebRTC player using WHEP protocol at `/birdcam/live/whep`
- Auto-reconnect on disconnect (3s delay)
- Loading spinner, error state with retry button
- "Direkte" (live) badge when connected
- All text in Norwegian

Build after changes:
```bash
cd /home/terje/sites/nesthus-prod
npm run build
```

## 6. Verify

```bash
# Check mediamtx is running and receiving stream
curl -s http://localhost:9997/v3/paths/list | python3 -m json.tool

# Check ports
ss -tlnp | grep -E '1935|8888|8889'

# Test WHEP through Apache (should return 400 = reaching mediamtx)
curl -s -X POST https://ekstremedia.no/birdcam/live/whep \
    -H "Content-Type: application/sdp" -d "test" -o /dev/null -w "%{http_code}"

# Open in browser
# https://ekstremedia.no/fuglekamera
```

## 7. NUC side — push RTMP stream

From the NUC, ffmpeg reads the Pi's MJPEG stream and pushes H.264 RTMP to the VPS:

```bash
ffmpeg -i http://PI_IP:8888/stream \
    -c:v libx264 -preset ultrafast -tune zerolatency \
    -b:v 1500k -maxrate 1500k -bufsize 3000k \
    -g 50 -keyint_min 25 \
    -f flv rtmp://185.14.97.143:1935/birdcam/live
```

## Troubleshooting

### WHEP returns 404
Apache's `.htaccess` is intercepting the request. Make sure `ProxyPass` directives are directly in the VirtualHost, not inside `<Location>` blocks.

### WHEP returns 301
mediamtx is redirecting and the `/stream/` prefix approach causes path mismatches. Use direct `/birdcam/live/` paths (no prefix).

### Stream not showing in browser
Check that mediamtx has an active path: `curl http://localhost:9997/v3/paths/list` — look for `"ready": true`.

### mediamtx ports
| Port | Protocol | Purpose |
|------|----------|---------|
| 1935 | RTMP | Receives stream from NUC (expose to NUC's public IP) |
| 8888 | HTTP | HLS playback (proxied through Apache) |
| 8889 | HTTP | WebRTC/WHEP playback (proxied through Apache) |
| 9997 | HTTP | API (localhost only) |
| 8554 | RTSP | RTSP playback (not used, stays open) |
| 8189 | UDP | WebRTC ICE (needs to be reachable from clients) |
