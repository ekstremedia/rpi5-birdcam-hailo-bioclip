# VPS Setup — mediamtx + nginx for birdcam stream

## 1. Install mediamtx

```bash
# Download latest release (check https://github.com/bluenviron/mediamtx/releases)
cd /tmp
wget https://github.com/bluenviron/mediamtx/releases/download/v1.11.3/mediamtx_v1.11.3_linux_amd64.tar.gz
tar xzf mediamtx_v1.11.3_linux_amd64.tar.gz
sudo mv mediamtx /usr/local/bin/
sudo mv mediamtx.yml /etc/mediamtx.yml
```

## 2. Configure mediamtx

Edit `/etc/mediamtx.yml`:

```yaml
# /etc/mediamtx.yml

# Logging
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
    # Only accept this specific stream
    source: publisher
```

## 3. Firewall rules

```bash
# RTMP from NUC only (replace with NUC's public IP)
sudo ufw allow from NUC_PUBLIC_IP to any port 1935

# WebRTC and HLS will be proxied through nginx, no need to expose directly
# API port stays internal only
```

## 4. systemd service

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
sudo systemctl status mediamtx
```

## 5. nginx reverse proxy (HTTPS)

Add to your nginx config (assuming you already have SSL for ekstremedia.no):

```nginx
# /etc/nginx/sites-available/birdcam.ekstremedia.no

server {
    listen 443 ssl http2;
    server_name birdcam.ekstremedia.no;

    ssl_certificate /etc/letsencrypt/live/birdcam.ekstremedia.no/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/birdcam.ekstremedia.no/privkey.pem;

    # WebRTC WHEP endpoint
    location /birdcam/live/whep {
        proxy_pass http://127.0.0.1:8889;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # WebRTC ICE candidates (needed for WHEP)
    location ~ ^/birdcam/live/whep/(.+) {
        proxy_pass http://127.0.0.1:8889;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # HLS fallback
    location /birdcam/live/ {
        proxy_pass http://127.0.0.1:8888;
        proxy_http_version 1.1;
        proxy_set_header Host $host;

        # CORS for hls.js
        add_header Access-Control-Allow-Origin *;
        add_header Access-Control-Allow-Methods "GET, OPTIONS";
    }
}
```

```bash
# Get SSL cert
sudo certbot certonly --nginx -d birdcam.ekstremedia.no

# Enable site
sudo ln -s /etc/nginx/sites-available/birdcam.ekstremedia.no /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

## 6. DNS

Add an A record for `birdcam.ekstremedia.no` pointing to the VPS IP.

## 7. Verify

```bash
# Check mediamtx is running
curl http://localhost:9997/v3/paths/list

# After NUC starts streaming, check the path is active
curl http://localhost:9997/v3/paths/list | jq '.items[] | .name'

# Test HLS
curl -I https://birdcam.ekstremedia.no/birdcam/live/index.m3u8

# Test WHEP
curl -X POST https://birdcam.ekstremedia.no/birdcam/live/whep \
    -H "Content-Type: application/sdp" -d "test" -v
# (will get an error but should connect to the endpoint)
```
