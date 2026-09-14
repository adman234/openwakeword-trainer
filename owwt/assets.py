"""Shared training data, downloaded once into /data/downloads. Downloads resume if interrupted.

Run directly (python -m owwt.assets) to download everything as a background job.
"""
import os
import shutil
import time
from pathlib import Path

import numpy as np
import requests

from .audio import decode, to_int16
from .config import CUSTOM_BACKGROUND_DIR, DOWNLOADS_DIR

HF = "https://huggingface.co"
RIR_API = f"{HF}/api/datasets/davidscripka/MIT_environmental_impulse_responses/tree/main/16khz"
RIR_URL = f"{HF}/datasets/davidscripka/MIT_environmental_impulse_responses/resolve/main/16khz/"
FEATURES_URL = f"{HF}/datasets/davidscripka/openwakeword_features/resolve/main/"
AUDIOSET_URL = f"{HF}/datasets/agkphysics/AudioSet/resolve/main/"
AUDIOSET_SHARDS = ["data/bal_train/00.parquet", "data/bal_train/01.parquet"]

RIR_DIR = DOWNLOADS_DIR / "rirs"
BACKGROUND_BANK = DOWNLOADS_DIR / "background" / "audioset_bank.npy"
ACAV_FEATURES = DOWNLOADS_DIR / "features" / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
VALIDATION_FEATURES = DOWNLOADS_DIR / "features" / "validation_set_features.npy"

ASSETS = [
    {"id": "rirs", "label": "Room acoustics", "size_bytes": 8_400_000,
     "description": "270 real room impulse responses (MIT), so clips sound like they were said across a room."},
    {"id": "background", "label": "Background sounds", "size_bytes": 1_390_000_000,
     "description": "About 2.5 hours of everyday audio from AudioSet, mixed under the training clips as noise."},
    {"id": "validation", "label": "False activation test set", "size_bytes": 184_836_608,
     "description": "11 hours of speech, music and noise used to measure false activations per hour."},
    {"id": "acav", "label": "Negative speech (2,000 hours)", "size_bytes": 17_280_000_128,
     "description": "Pre-computed features from 2,000 hours of ACAV100M audio. This is what teaches the model what is NOT the wake word."},
    {"id": "microwakeword", "label": "ESPHome negative data", "size_bytes": 5_709_958_106, "optional": True,
     "description": "Speech, noise and dinner party spectrograms from microWakeWord. Only needed for ESPHome models, and downloaded automatically the first time you train one."},
]

MICRO_URL = f"{HF}/datasets/kahrendt/microwakeword/resolve/main/"
MICRO_DIR = DOWNLOADS_DIR / "microwakeword"
MICRO_SETS = {"dinner_party": 444_310_142, "dinner_party_eval": 82_329_019,
              "no_speech": 2_000_317_854, "speech": 3_183_001_091}


def micro_ready() -> bool:
    return all((MICRO_DIR / name / ".complete").exists() for name in MICRO_SETS)


def ensure_microwakeword(report):
    """Download and unpack microWakeWord's negative spectrogram sets. report(fraction, message)."""
    import zipfile

    total = sum(MICRO_SETS.values())
    done = 0
    for name, size in MICRO_SETS.items():
        target = MICRO_DIR / name
        if (target / ".complete").exists():
            done += size
            continue
        zip_path = MICRO_DIR / f"{name}.zip"
        if not zip_path.exists():
            download(MICRO_URL + zip_path.name, zip_path,
                     progress=lambda d, t, base=done: report((base + d) / total,
                                                            f"ESPHome negative data: {(base + d) / 1e9:.1f} of {total / 1e9:.1f} GB"))
        report((done + size) / total, f"Unpacking ESPHome negative data: {name}")
        shutil.rmtree(target, ignore_errors=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(MICRO_DIR, [m for m in z.namelist() if not m.startswith("__MACOSX") and not m.endswith(".DS_Store")])
        (target / ".complete").touch()
        zip_path.unlink()
        done += size
    report(1.0, "ESPHome negative data ready")


def _ready(asset_id: str) -> bool:
    if asset_id == "microwakeword":
        return micro_ready()
    if asset_id == "rirs":
        return (RIR_DIR / ".complete").exists()
    if asset_id == "background":
        return BACKGROUND_BANK.exists()
    if asset_id == "validation":
        return VALIDATION_FEATURES.exists()
    if asset_id == "acav":
        return ACAV_FEATURES.exists()
    return False


def _partial_bytes(asset_id: str) -> int:
    part = {"acav": ACAV_FEATURES, "validation": VALIDATION_FEATURES}.get(asset_id)
    if part is not None:
        p = part.with_name(part.name + ".part")
        return p.stat().st_size if p.exists() else 0
    return 0


def status() -> list[dict]:
    return [{**a, "ready": _ready(a["id"]), "partial_bytes": _partial_bytes(a["id"])} for a in ASSETS]


def all_ready() -> bool:
    """True when everything openWakeWord training needs is present (optional ESPHome data excluded)."""
    return all(_ready(a["id"]) for a in ASSETS if not a.get("optional"))


def missing() -> list[str]:
    return [a["label"] for a in ASSETS if not a.get("optional") and not _ready(a["id"])]


def _headers():
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def download(url: str, dest: Path, progress=None, attempts: int = 20):
    """Resumable download with retries, for multi-gigabyte files over flaky connections."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(attempts):
        pos = part.stat().st_size if part.exists() else 0
        headers = dict(_headers())
        if pos:
            headers["Range"] = f"bytes={pos}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:  # already complete
                    break
                r.raise_for_status()
                if pos and r.status_code != 206:
                    pos = 0  # server ignored the range, start over
                total = pos + int(r.headers.get("Content-Length", 0))
                last = 0.0
                with open(part, "ab" if pos else "wb") as f:
                    for chunk in r.iter_content(chunk_size=4 << 20):
                        f.write(chunk)
                        pos += len(chunk)
                        if progress and time.time() - last > 1:
                            progress(pos, total)
                            last = time.time()
                if total and pos < total:
                    raise IOError(f"connection closed at {pos} of {total} bytes")
            break
        except (requests.RequestException, IOError) as exc:
            if attempt == attempts - 1:
                raise
            wait = min(60, 5 * (attempt + 1))
            print(f"Download interrupted ({exc}), retrying in {wait}s", flush=True)
            time.sleep(wait)
    os.replace(part, dest)


def ensure_rirs(progress=None):
    if _ready("rirs"):
        return
    RIR_DIR.mkdir(parents=True, exist_ok=True)
    files = [f["path"].split("/")[-1] for f in requests.get(RIR_API, headers=_headers(), timeout=60).json()
             if f["type"] == "file" and f["path"].endswith(".wav")]
    for i, name in enumerate(files):
        if not (RIR_DIR / name).exists():
            download(RIR_URL + name, RIR_DIR / name)
        if progress:
            progress(i + 1, len(files))
    (RIR_DIR / ".complete").touch()


def ensure_background(progress=None):
    if _ready("background"):
        return
    import pyarrow.parquet as pq

    tmp_dir = DOWNLOADS_DIR / "background" / "tmp"
    chunks = []
    for i, shard in enumerate(AUDIOSET_SHARDS):
        local = tmp_dir / Path(shard).name
        if not local.exists():
            download(AUDIOSET_URL + shard, local,
                     progress=(lambda d, t, i=i: progress(i + d / max(t, 1) * 0.8, len(AUDIOSET_SHARDS))) if progress else None)
        parquet = pq.ParquetFile(local)
        for batch in parquet.iter_batches(batch_size=16, columns=["audio"]):
            for item in batch.column(0).to_pylist():
                data = item.get("bytes") if isinstance(item, dict) else item
                if not data:
                    continue
                try:
                    chunks.append(to_int16(decode(data)))
                except Exception:
                    continue
        parquet.close()
        if progress:
            progress(i + 1, len(AUDIOSET_SHARDS))
    bank = np.concatenate(chunks)
    part = BACKGROUND_BANK.with_name("audioset_bank.partial.npy")
    np.save(part, bank)
    os.replace(part, BACKGROUND_BANK)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def ensure_features(progress=None):
    if not _ready("validation"):
        download(FEATURES_URL + VALIDATION_FEATURES.name, VALIDATION_FEATURES,
                 progress=(lambda d, t: progress("validation", d, t)) if progress else None)
    if not _ready("acav"):
        download(FEATURES_URL + ACAV_FEATURES.name, ACAV_FEATURES,
                 progress=(lambda d, t: progress("acav", d, t)) if progress else None)


def custom_background_files(extra_dirs=()) -> list[Path]:
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".opus"}
    dirs = [CUSTOM_BACKGROUND_DIR, *extra_dirs]
    return sorted(p for d in dirs if Path(d).exists() for p in Path(d).rglob("*") if p.suffix.lower() in exts)


def rir_paths() -> list[str]:
    return sorted(str(p) for p in RIR_DIR.glob("*.wav"))


def ensure_augmentation(report):
    """Just the room acoustics and background audio, which microWakeWord-only runs need."""
    ensure_rirs(lambda i, n: report(0.1 * i / n, f"Room acoustics: {i} of {n} files"))
    ensure_background(lambda i, n: report(0.1 + 0.9 * i / n, f"Background sounds: part {min(int(i) + 1, n)} of {n}"))
    report(1.0, "Room acoustics and background sounds ready")


def ensure_all(report):
    """report(fraction, message) is called as downloads progress."""
    gb = 1024 ** 3
    report(0.0, "Downloading room acoustics")
    ensure_rirs(lambda i, n: report(0.02 * i / n, f"Room acoustics: {i} of {n} files"))
    ensure_background(lambda i, n: report(0.02 + 0.08 * i / n, f"Background sounds: part {min(int(i) + 1, n)} of {n}"))

    def features_progress(which, done, total):
        if which == "validation":
            report(0.10 + 0.02 * done / max(total, 1), f"False activation test set: {done / gb:.2f} of {total / gb:.2f} GB")
        else:
            report(0.12 + 0.88 * done / max(total, 1), f"Negative speech features: {done / gb:.1f} of {total / gb:.1f} GB")

    ensure_features(features_progress)
    report(1.0, "All training data downloaded")


if __name__ == "__main__":
    from .jobs import JobReporter

    reporter = JobReporter(stages=["Download training data"])
    reporter.run(lambda: (reporter.stage(0), ensure_all(reporter.progress)))
