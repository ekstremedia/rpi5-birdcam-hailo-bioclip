"""BioCLIP species classification API.

Accepts JPEG bird crops via POST /classify and returns the most likely
species with confidence score. Species names are returned in Norwegian.
"""

import io
import os
import time
import logging

import torch
import open_clip
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

SPECIES_FILE = os.environ.get("SPECIES_FILE", "/data/norwegian_species.txt")

# Globals filled by load_model() and load_species()
model = None
tokenizer = None
preprocess = None
device = "cpu"

species_english = []   # English names (used for prompts)
species_norwegian = [] # Norwegian names (returned in response)
text_features = None   # Precomputed text embeddings
species_file_mtime = 0


def load_model():
    """Load BioCLIP model once at startup."""
    global model, tokenizer, preprocess
    log.info("Loading BioCLIP model...")
    t0 = time.time()
    model, preprocess_train, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    model.eval()
    log.info("Model loaded in %.1fs", time.time() - t0)


def load_species():
    """Parse species file and precompute text embeddings."""
    global species_english, species_norwegian, text_features, species_file_mtime

    try:
        mtime = os.path.getmtime(SPECIES_FILE)
    except OSError:
        log.error("Species file not found: %s", SPECIES_FILE)
        return

    if mtime == species_file_mtime and text_features is not None:
        return  # No change

    log.info("Loading species from %s", SPECIES_FILE)
    eng, nor = [], []
    with open(SPECIES_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) == 2:
                eng.append(parts[0].strip())
                nor.append(parts[1].strip())

    if not eng:
        log.error("No species parsed from %s", SPECIES_FILE)
        return

    species_english = eng
    species_norwegian = nor

    # Precompute text embeddings
    prompts = [f"a photo of a {name}" for name in species_english]
    tokens = tokenizer(prompts)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    species_file_mtime = mtime
    log.info("Loaded %d species, text embeddings ready", len(species_english))


@app.route("/health", methods=["GET"])
def health():
    ok = model is not None and text_features is not None
    return jsonify({
        "status": "ok" if ok else "not ready",
        "model": "bioclip",
        "species_count": len(species_english),
    }), 200 if ok else 503


@app.route("/classify", methods=["POST"])
def classify():
    # Hot-reload species if file changed
    load_species()

    data = request.get_data()
    if not data:
        return jsonify({"error": "No image data"}), 400

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image"}), 400

    t0 = time.time()
    img_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        image_features = model.encode_image(img_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp()
        similarity = (logit_scale * image_features @ text_features.T).squeeze(0)
        probs = similarity.softmax(dim=-1)

    best_idx = probs.argmax().item()
    confidence = probs[best_idx].item()
    inference_time = time.time() - t0

    return jsonify({
        "species": species_norwegian[best_idx],
        "species_en": species_english[best_idx],
        "confidence": round(confidence, 4),
        "inference_time": round(inference_time, 3),
    })


if __name__ == "__main__":
    load_model()
    load_species()
    app.run(host="0.0.0.0", port=5555)
