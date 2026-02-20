# Bird Species Classification API

HTTP API that identifies bird species from photos using zero-shot classification.
Runs as a Docker container on the NUC, serves the Pi 5 bird monitor over the network.

## How it works

1. The Pi 5 detects a bird (YOLOv8s on Hailo-8) and crops the image
2. The crop is POST-ed as JPEG bytes to `http://NUC_IP:5555/classify`
3. The API runs the crop through BioCLIP and returns the species name in Norwegian

Classification takes ~0.2s on the NUC (i7) vs ~4s on the Pi 5.

## The model: BioCLIP

**BioCLIP** is a vision foundation model for biological image classification, built on
OpenAI's CLIP (ViT-B/16) and fine-tuned on 10 million biological images.

| | |
|---|---|
| **Model** | [imageomics/bioclip](https://huggingface.co/imageomics/bioclip) on HuggingFace |
| **Architecture** | CLIP ViT-B/16 (Vision Transformer, 224x224 input) |
| **Training data** | TreeOfLife-10M — 10M images, 450K+ taxa from iNaturalist, BIOSCAN-1M, Encyclopedia of Life |
| **Paper** | [BioCLIP: A Vision Foundation Model for the Tree of Life](https://arxiv.org/abs/2311.18803) (CVPR 2024) |
| **Award** | CVPR 2024 Best Student Paper |
| **License** | MIT |
| **Image size** | ~1.74 GB Docker image (model weights baked in) |

### Authors

Samuel Stevens, Jiaman Wu, Matthew J. Thompson, Elizabeth G. Campolongo, Chan Hee Song,
David Edward Carlyn, Li Dong, Wasila M. Dahdul, Charles Stewart, Tanya Berger-Wolf,
Wei-Lun Chao, Yu Su — The Ohio State University / [Imageomics Institute](https://imageomics.org)

### Why BioCLIP

- **Zero-shot**: No training needed — just list species names in a text file
- **Biology-specific**: 72% zero-shot accuracy on bird species (vs 50% for generic CLIP)
- **Global coverage**: Trained on 450K+ taxa worldwide, not just North American birds
- **Easy to extend**: Adding a new species = adding one line to `norwegian_species.txt`

### Links

- HuggingFace model: https://huggingface.co/imageomics/bioclip
- Demo: https://huggingface.co/spaces/imageomics/bioclip-demo
- GitHub: https://github.com/Imageomics/BioCLIP
- Paper: https://arxiv.org/abs/2311.18803
- Dataset: https://huggingface.co/datasets/imageomics/TreeOfLife-10M

## How classification works (zero-shot)

Unlike a traditional classifier with a fixed set of output classes, BioCLIP uses
CLIP-style contrastive learning to match images against text descriptions.

At startup, the API:
1. Reads species names from `norwegian_species.txt` (e.g. "Great Tit", "Blue Tit", ...)
2. Creates text prompts: `"a photo of a Great Tit"`, `"a photo of a Blue Tit"`, ...
3. Encodes all prompts into text embeddings (once, cached in memory)

On each classification request:
1. The bird crop is preprocessed (resize to 224x224, normalize)
2. The image is encoded into an image embedding
3. Cosine similarity is computed between the image and all species text embeddings
4. Softmax over similarities gives a probability for each species
5. The top species (Norwegian name) and confidence are returned

This means you can add or remove species just by editing the text file — no retraining.

## Files

```
/home/terje/birdcam/
├── species_api.py           # Flask API server
├── norwegian_species.txt    # Species list (mounted into container)
├── Dockerfile               # Python 3.11 + PyTorch CPU + BioCLIP
├── docker-compose.yml       # Container config
├── .env                     # HuggingFace token (gitignored)
└── .gitignore
```

## Setup from scratch

### Prerequisites

- Docker and Docker Compose installed
- A HuggingFace account and access token (free): https://huggingface.co/settings/tokens

### 1. Create the `.env` file

```bash
cd /home/terje/birdcam
echo "HF_TOKEN=hf_your_token_here" > .env
```

The token is only used during `docker compose build` to download the model from
HuggingFace. It is not stored in the final image.

### 2. Build the Docker image

```bash
docker compose build
```

This takes a few minutes. It:
- Installs PyTorch (CPU-only) + open_clip + Flask
- Downloads the BioCLIP model weights (~600 MB) from HuggingFace
- Bakes everything into the image so the container starts fast

### 3. Start the container

```bash
docker compose up -d
```

### 4. Verify it's running

```bash
# Health check
curl http://localhost:5555/health
# → {"model":"bioclip","species_count":42,"status":"ok"}

# Classify an image
curl -X POST --data-binary @some_bird.jpg http://localhost:5555/classify
# → {"species":"Kjøttmeis","species_en":"Great Tit","confidence":0.87,"inference_time":0.19}
```

### 5. View logs

```bash
docker compose logs -f
```

## API reference

### POST /classify

Send a JPEG image as the request body. Returns the most likely species.

```bash
curl -X POST --data-binary @bird_crop.jpg http://localhost:5555/classify
```

**Response:**
```json
{
  "species": "Kjøttmeis",
  "species_en": "Great Tit",
  "confidence": 0.87,
  "inference_time": 0.19
}
```

| Field | Description |
|-------|-------------|
| `species` | Norwegian name |
| `species_en` | English name (used internally by the model) |
| `confidence` | 0.0–1.0, softmax probability |
| `inference_time` | Seconds spent on classification |

### GET /health

```bash
curl http://localhost:5555/health
```

**Response:**
```json
{
  "status": "ok",
  "model": "bioclip",
  "species_count": 42
}
```

Returns HTTP 503 if the model hasn't finished loading yet.

## Species list

The file `norwegian_species.txt` contains 42 common Norwegian feeder birds, hand-curated
for this project. The English names follow the [IOC World Bird List](https://www.worldbirdnames.org/)
standard — these are what BioCLIP recognizes. The Norwegian names are the standard
norske artsnavn from [Norsk Ornitologisk Forening](https://www.birdlife.no/).

The list is organized by family (meiser, spurver/finker, troster, spetter, kråkefugler,
etc.) and focuses on species that actually show up at Norwegian bird feeders. It's not
meant to be exhaustive — add more species as you see them.

Format is one species per line: `English Name | Norsk navn`

```
Great Tit | Kjøttmeis
Blue Tit | Blåmeis
Eurasian Magpie | Skjære
...
```

Lines starting with `#` are comments. Blank lines are ignored.

**Hot-reload**: The API checks the file's modification time on every `/classify` request.
If the file has changed, species and text embeddings are reloaded automatically — no
container restart needed.

**To add a species**: Edit `norwegian_species.txt` on the host (it's volume-mounted into
the container). The next classification request will pick up the change.

## Docker details

### Why Docker

The NUC runs Debian 13 with Python 3.13, which has compatibility issues with PyTorch
and open_clip. Docker with Python 3.11-slim avoids this entirely and keeps the host clean.

### Why CPU-only PyTorch

The NUC has no GPU. CPU-only PyTorch is ~190 MB instead of ~2 GB, and classification
still takes only ~0.2s per image on an i7.

### Image size

The Docker image is ~1.7 GB, mostly PyTorch (~190 MB) and the BioCLIP model weights
(~600 MB). The model is downloaded during build so the container starts in ~3 seconds.

### Rebuilding

```bash
cd /home/terje/birdcam
docker compose build --no-cache   # Full rebuild
docker compose up -d              # Restart with new image
```

## Troubleshooting

**Container won't start / health returns 503:**
Check logs with `docker compose logs`. Most likely the model file is corrupted —
rebuild with `docker compose build --no-cache`.

**Classification returns low confidence for everything:**
This is normal for ambiguous crops (partial bird, blurry, too far away). The Pi's
detection threshold filters out most of these. Confidence above ~0.3 is usually correct.

**Want to test with the HuggingFace demo:**
Upload a bird photo at https://huggingface.co/spaces/imageomics/bioclip-demo to compare
results with the API.
