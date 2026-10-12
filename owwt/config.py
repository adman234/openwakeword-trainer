"""Paths, ports and training presets shared by the web app and the trainer."""
import os
from pathlib import Path

VERSION = "1.0.0"

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
# Read-only files baked into the image (Kokoro weights and voices, openWakeWord feature models)
ASSET_DIR = Path(os.environ.get("ASSET_DIR", "/opt/assets"))

PROJECTS_DIR = DATA_DIR / "projects"
DOWNLOADS_DIR = DATA_DIR / "downloads"
CUSTOM_BACKGROUND_DIR = DATA_DIR / "background"
CERT_DIR = DATA_DIR / "certs"
JOB_DIR = DATA_DIR / "job"
# Wake word clips to review, e.g. Home Assistant debug recordings trimmed by wake-clip-keeper
IMPORT_DIR = Path(os.environ.get("IMPORT_DIR", "/import"))

KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_DIR = ASSET_DIR / "kokoro"
MELSPEC_MODEL = ASSET_DIR / "oww" / "melspectrogram.onnx"
EMBEDDING_MODEL = ASSET_DIR / "oww" / "embedding_model.onnx"
TFLITE_PYTHON = Path(os.environ.get("TFLITE_PYTHON", "/opt/tfenv/bin/python"))

HTTP_PORT = int(os.environ.get("HTTP_PORT", "8080"))
HTTPS_PORT = int(os.environ.get("HTTPS_PORT", "8443"))

SR = 16000
FEATURE_DIM = 96
FRAME_SECONDS = 0.08  # one openWakeWord embedding every 80 ms

RECORDING_KINDS = ("positive", "negative", "background")

# openwakeword: models for the Home Assistant openWakeWord add-on. microwakeword: on-device models for ESPHome.
MODEL_TYPES = ("openwakeword", "microwakeword")

PRESETS = {
    "quick": {
        "label": "Quick test",
        "description": "Checks that everything works and gives a rough model.",
        "n_positive": 2000,
        "n_negative": 2000,
        "steps": 20000,
        "layer_size": 32,
        "micro_steps": 5000,
    },
    "standard": {
        "label": "Standard",
        "description": "A good everyday model. Recommended.",
        "n_positive": 10000,
        "n_negative": 10000,
        "steps": 50000,
        "layer_size": 32,
        "micro_steps": 20000,
    },
    "max": {
        "label": "Maximum",
        "description": "More voices and longer training for the fewest false triggers.",
        "n_positive": 25000,
        "n_negative": 25000,
        "steps": 100000,
        "layer_size": 64,
        "micro_steps": 40000,
    },
}

ADVANCED_DEFAULTS = {
    "max_negative_weight": 1500,
    "target_fp_per_hour": 0.2,
    "real_weight": 0.25,
    "voices": "all",
    "n_blocks": 1,
    "micro_target_faph": 0.5,
}

ADVANCED_LIMITS = {
    "n_positive": (200, 100000),
    "n_negative": (200, 100000),
    "steps": (1000, 200000),
    "layer_size": (16, 256),
    "max_negative_weight": (1, 10000),
    "target_fp_per_hour": (0.01, 10),
    "real_weight": (0, 0.9),
    "n_blocks": (1, 4),
    "micro_steps": (1000, 200000),
    "micro_target_faph": (0.01, 10),
}


def training_settings(project: dict) -> dict:
    """Merge a project's preset with any advanced overrides it sets."""
    settings = dict(PRESETS.get(project.get("preset"), PRESETS["standard"]))
    settings.update(ADVANCED_DEFAULTS)
    for key, value in (project.get("advanced") or {}).items():
        if value is None or value == "":
            continue
        if key in ADVANCED_LIMITS:
            lo, hi = ADVANCED_LIMITS[key]
            value = type(hi)(min(max(float(value), lo), hi))
        settings[key] = value
    return settings
