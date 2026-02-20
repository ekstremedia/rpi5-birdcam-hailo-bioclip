# Ideas

## Offload Species Classification to a PC

**Problem**: BioCLIP takes ~4s per classification on the Pi 5 CPU. The Pi handles YOLOv8 detection at 30 FPS fine (Hailo does the work), but species ID is the bottleneck.

**Idea**: Run BioCLIP on a more powerful PC (with GPU) on the local network. Classification would drop to ~100-300ms round trip instead of 4s.

**How it would work**:
1. `species_server.py` on the PC — small FastAPI/Flask server that loads BioCLIP once, exposes `POST /classify` accepting a JPEG crop, returns `{species, confidence}`
2. Pi's `_classify_bird_async` sends the crop over HTTP instead of running BioCLIP locally
3. Optional: keep local BioCLIP as fallback if the server is unreachable
4. BioCLIP's ~2.6GB RAM footprint could be removed from the Pi entirely

**Benefits**:
- Pi stays lean: just detection + tracking + streaming
- Much faster species ID (~200ms vs 4s)
- Could run a bigger/better model on the PC (e.g. ViT-L instead of ViT-B)
- Could also run multiple models or ensemble for better accuracy

**What's needed**:
- A PC on the same LAN with a GPU (even a laptop with a decent GPU would work)
- Python + pybioclip installed on that PC
- Config in bird_monitor.py for the server URL (e.g. `SPECIES_SERVER = "http://192.168.1.x:5000"`)
