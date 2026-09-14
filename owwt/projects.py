"""Wake word projects on disk: settings, voice recordings and training runs.

/data/projects/<id>/
  project.json
  recordings/{positive,negative,background}/*.wav
  work/        generated clips and cached features (safe to delete, rebuilt on the next training run)
  runs/<run>/  <model>.onnx, <model>.tflite, report.json
"""
import json
import os
import random
import re
import shutil
import time
import uuid

import soundfile as sf

from .audio import write_wav
from .config import OUTPUT_DIR, PRESETS, PROJECTS_DIR, RECORDING_KINDS, SR

ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
FILE_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,120}\.wav$")
RUN_RE = re.compile(r"^\d{8}-\d{6}$")
MODEL_FILE_RE = re.compile(r"^[a-z0-9_]{1,64}\.(onnx|tflite|json)$")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48] or "wake_word"


def project_dir(pid: str):
    if not ID_RE.match(pid or ""):
        raise KeyError(pid)
    d = PROJECTS_DIR / pid
    if not (d / "project.json").exists():
        raise KeyError(pid)
    return d


def _write_json(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def load(pid: str) -> dict:
    return json.loads((project_dir(pid) / "project.json").read_text())


def save(project: dict):
    project["updated_at"] = time.time()
    _write_json(project_dir(project["id"]) / "project.json", project)


def create(phrase: str, preset: str = "standard") -> dict:
    phrase = " ".join((phrase or "").split())
    if not phrase:
        raise ValueError("Enter a wake word phrase")
    if len(phrase) > 60:
        raise ValueError("Keep the wake word under 60 characters")
    if preset not in PRESETS:
        preset = "standard"
    base = slugify(phrase)
    pid, i = base, 2
    while (PROJECTS_DIR / pid).exists():
        pid, i = f"{base}_{i}", i + 1
    d = PROJECTS_DIR / pid
    for kind in RECORDING_KINDS:
        (d / "recordings" / kind).mkdir(parents=True, exist_ok=True)
    now = time.time()
    project = {
        "id": pid, "name": phrase.title(), "phrase": phrase, "model_name": pid, "variants": [],
        "pronunciation": "", "custom_negatives": [], "preset": preset, "advanced": {},
        "publish_to_output": True, "created_at": now, "updated_at": now,
    }
    _write_json(d / "project.json", project)
    return project


EDITABLE = {"name": str, "phrase": str, "model_name": str, "variants": list, "pronunciation": str,
            "custom_negatives": list, "preset": str, "advanced": dict, "publish_to_output": bool}


def update(pid: str, patch: dict) -> dict:
    project = load(pid)
    for key, value in patch.items():
        if key not in EDITABLE:
            continue
        if not isinstance(value, EDITABLE[key]):
            raise ValueError(f"Invalid value for {key}")
        if key == "phrase":
            value = " ".join(value.split())
            if not value:
                raise ValueError("The wake word phrase can't be empty")
        elif key == "preset" and value not in PRESETS:
            raise ValueError("Unknown preset")
        elif key in ("variants", "custom_negatives"):
            value = [" ".join(str(v).split()) for v in value if str(v).strip()][:200]
        elif key == "model_name":
            value = slugify(value)
        elif key == "pronunciation":
            value = value.strip()[:400]
        project[key] = value
    save(project)
    return project


def delete(pid: str):
    shutil.rmtree(project_dir(pid))


# Recordings -----------------------------------------------------------------------------------
def recordings(pid: str) -> dict:
    root = project_dir(pid) / "recordings"
    out = {}
    for kind in RECORDING_KINDS:
        items = []
        folder = root / kind
        folder.mkdir(parents=True, exist_ok=True)
        for p in sorted(folder.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                info = sf.info(str(p))
            except Exception:
                continue
            items.append({"name": p.name, "duration": round(info.frames / info.samplerate, 2),
                          "created_at": p.stat().st_mtime})
        out[kind] = items
    return out


def recording_files(pid: str, kind: str) -> list:
    return sorted((project_dir(pid) / "recordings" / kind).glob("*.wav"))


def add_recording(pid: str, kind: str, audio) -> dict:
    if kind not in RECORDING_KINDS:
        raise ValueError("Unknown recording type")
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.wav"
    folder = project_dir(pid) / "recordings" / kind
    folder.mkdir(parents=True, exist_ok=True)
    write_wav(folder / name, audio)
    return {"name": name, "duration": round(len(audio) / SR, 2), "created_at": time.time()}


def recording_path(pid: str, kind: str, name: str):
    if kind not in RECORDING_KINDS or not FILE_RE.match(name or ""):
        raise KeyError(name)
    p = project_dir(pid) / "recordings" / kind / name
    if not p.exists():
        raise KeyError(name)
    return p


# Generated samples ------------------------------------------------------------------------------
SAMPLE_DIRS = {"positive": "tts/pos_train", "negative": "tts/neg_train"}


def generated_samples(pid: str, kind: str, k: int = 12) -> list[str]:
    folder = project_dir(pid) / "work" / SAMPLE_DIRS[kind]
    if not folder.exists():
        return []
    names = [p.name for p in folder.glob("*.wav")]
    return random.sample(names, min(k, len(names)))


def generated_sample_path(pid: str, kind: str, name: str):
    if kind not in SAMPLE_DIRS or not FILE_RE.match(name or ""):
        raise KeyError(name)
    p = project_dir(pid) / "work" / SAMPLE_DIRS[kind] / name
    if not p.exists():
        raise KeyError(name)
    return p


# Runs -----------------------------------------------------------------------------------------
def runs(pid: str) -> list[dict]:
    root = project_dir(pid) / "runs"
    out = []
    if root.exists():
        for rd in sorted(root.iterdir(), reverse=True):
            report = rd / "report.json"
            if RUN_RE.match(rd.name) and report.exists():
                data = json.loads(report.read_text())
                data["files"] = sorted(f.name for f in rd.iterdir() if f.suffix in (".onnx", ".tflite"))
                out.append(data)
    return out


def run_dir(pid: str, run_id: str):
    if not RUN_RE.match(run_id or ""):
        raise KeyError(run_id)
    d = project_dir(pid) / "runs" / run_id
    if not d.exists():
        raise KeyError(run_id)
    return d


def run_file(pid: str, run_id: str, filename: str):
    if not MODEL_FILE_RE.match(filename or ""):
        raise KeyError(filename)
    p = run_dir(pid, run_id) / filename
    if not p.exists():
        raise KeyError(filename)
    return p


def publish(pid: str, run_id: str) -> list[str]:
    d = run_dir(pid, run_id)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for f in sorted(d.iterdir()):
        if f.suffix in (".onnx", ".tflite"):
            shutil.copy2(f, OUTPUT_DIR / f.name)
            copied.append(str(OUTPUT_DIR / f.name))
    return copied


def summary(project: dict) -> dict:
    recs = recordings(project["id"])
    latest = next(iter(runs(project["id"])), None)
    return {
        **project,
        "recording_counts": {k: len(v) for k, v in recs.items()},
        "latest_run": None if latest is None else {
            "run_id": latest["run_id"], "created_at": latest["created_at"], "files": latest["files"],
            "recommended_threshold": latest.get("recommended_threshold"),
            "headline": latest.get("headline"),
        },
    }


def list_projects() -> list[dict]:
    if not PROJECTS_DIR.exists():
        return []
    items = []
    for d in PROJECTS_DIR.iterdir():
        if (d / "project.json").exists() and ID_RE.match(d.name):
            try:
                items.append(summary(load(d.name)))
            except Exception:
                continue
    return sorted(items, key=lambda p: p.get("updated_at", 0), reverse=True)
