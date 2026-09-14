"""Build-time download of everything needed offline: Kokoro weights and voices, openWakeWord feature models.

Run with --smoke-test to also phonemize and synthesize a short clip, which catches missing
runtime downloads (spaCy models, espeak) while the image is still being built.
"""
import os
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

ASSET_DIR = Path(os.environ.get("ASSET_DIR", "/opt/assets"))
KOKORO_REPO = "hexgrad/Kokoro-82M"
OWW_RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"


def main():
    kokoro_dir = ASSET_DIR / "kokoro"
    files = ["config.json", "kokoro-v1_0.pth"]
    files += sorted(f for f in list_repo_files(KOKORO_REPO) if f.startswith("voices/") and f.endswith(".pt"))
    for name in files:
        if not (kokoro_dir / name).exists():
            print("kokoro:", name, flush=True)
            hf_hub_download(KOKORO_REPO, name, local_dir=kokoro_dir)

    oww_dir = ASSET_DIR / "oww"
    oww_dir.mkdir(parents=True, exist_ok=True)
    for name in ["melspectrogram.onnx", "embedding_model.onnx"]:
        dest = oww_dir / name
        if not dest.exists():
            print("openwakeword:", name, flush=True)
            urllib.request.urlretrieve(f"{OWW_RELEASE}/{name}", dest)

    if "--smoke-test" in sys.argv:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from owwt.tts import Kokoro

        tts = Kokoro("cpu")
        for british in (False, True):
            ps = tts.phonemize("Hey Jarvis!", british=british)
            audio = tts.synth(ps, "af_heart" if not british else "bm_george")
            print(f"smoke test ok: phonemes={ps!r} samples={len(audio)}", flush=True)


if __name__ == "__main__":
    main()
