#!/usr/bin/env python3
"""Species classifier for bird crops using BioCLIP zero-shot classification.

Uses BioCLIP (a CLIP model trained on 10M+ biological images) to identify
bird species from crop images. Works with any species — no retraining needed.

The species list is loaded from a text file (one species per line, format:
"English Name | Norsk navn"). English names are used for the model,
Norwegian names are returned for display.

Usage standalone (for testing):
    python3 species_classifier.py path/to/bird_crop.jpg
"""

import os
import sys
import time

import cv2

try:
    from bioclip import CustomLabelsClassifier
except ImportError:
    CustomLabelsClassifier = None

# Default paths
DEFAULT_SPECIES_PATH = "/home/pi/ai/models/norwegian_species.txt"


def _load_species_file(path):
    """Load species file. Each line: 'English Name | Norsk navn'.

    Returns (english_names, norwegian_map) where norwegian_map maps
    english -> norwegian for display.
    """
    english_names = []
    norwegian_map = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                en, no = line.split("|", 1)
                en, no = en.strip(), no.strip()
            else:
                en = line
                no = line
            english_names.append(en)
            norwegian_map[en] = no

    return english_names, norwegian_map


class SpeciesClassifier:
    """BioCLIP-based zero-shot bird species classifier.

    Uses a vision-language model to match bird crops against a list of
    species names. Can identify any species without explicit training.
    """

    def __init__(self, species_path=DEFAULT_SPECIES_PATH):
        if CustomLabelsClassifier is None:
            raise ImportError("pybioclip er ikke installert. Kjor: pip3 install pybioclip")

        if not os.path.isfile(species_path):
            raise FileNotFoundError(f"Artsliste ikke funnet: {species_path}")

        self.species_path = species_path
        self.species_mtime = os.path.getmtime(species_path)

        # Load species list
        self.english_names, self.norwegian_map = _load_species_file(species_path)

        if len(self.english_names) < 2:
            raise ValueError(f"Trenger minst 2 arter i {species_path}")

        # Initialize BioCLIP classifier
        print(f"  Laster BioCLIP-modell ({len(self.english_names)} arter)...", flush=True)
        self.classifier = CustomLabelsClassifier(self.english_names)
        print(f"  BioCLIP klar!", flush=True)

    def classify(self, crop_bgr):
        """Classify a bird crop image.

        Args:
            crop_bgr: BGR numpy array (as returned by cv2 / picamera2)

        Returns:
            (species_name, confidence) tuple. Species name is in Norwegian
            if a translation exists, otherwise English.
        """
        # Save crop to a temp file (pybioclip expects a file path)
        tmp_path = "/tmp/bird_classify_tmp.jpg"
        cv2.imwrite(tmp_path, crop_bgr)

        # Run BioCLIP zero-shot classification
        predictions = self.classifier.predict(tmp_path)

        if not predictions:
            return None, 0.0

        top = predictions[0]
        english_name = top["classification"]
        confidence = float(top["score"])

        # Map to Norwegian name for display
        norwegian_name = self.norwegian_map.get(english_name, english_name)

        return norwegian_name, confidence

    def check_reload(self):
        """Check if species list has been updated and reload if so."""
        try:
            current_mtime = os.path.getmtime(self.species_path)
        except OSError:
            return False

        if current_mtime > self.species_mtime:
            print("  Artsliste endret — laster på nytt...", flush=True)
            try:
                self.english_names, self.norwegian_map = _load_species_file(self.species_path)
                self.classifier = CustomLabelsClassifier(self.english_names)
                self.species_mtime = current_mtime
                print(f"  Lastet på nytt: {len(self.english_names)} arter", flush=True)
                return True
            except Exception as e:
                print(f"  Kunne ikke laste artsliste på nytt: {e}", flush=True)
                return False
        return False

    def close(self):
        """Release resources."""
        self.classifier = None


# ---- Standalone test ----

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Bruk: python3 species_classifier.py <bildesti> [artslistesti]")
        sys.exit(1)

    image_path = sys.argv[1]
    species_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SPECIES_PATH

    img = cv2.imread(image_path)
    if img is None:
        print(f"Kunne ikke lese bilde: {image_path}")
        sys.exit(1)

    print(f"Bilde: {image_path} ({img.shape[1]}x{img.shape[0]})")
    print(f"Artsliste: {species_path}")

    classifier = SpeciesClassifier(species_path)

    t0 = time.time()
    species, confidence = classifier.classify(img)
    elapsed = time.time() - t0

    print(f"\nResultat: {species} ({confidence:.1%})")
    print(f"Inferenstid: {elapsed:.2f}s")

    classifier.close()
