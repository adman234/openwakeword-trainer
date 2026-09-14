"""FastAPI app: REST API, the static web UI and the live test WebSocket."""
import json
import os
import random
import shutil
import threading
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
from fastapi import Body, FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import adversarial, assets, projects
from .audio import decode, level_report, trim_silence, wav_bytes
from .config import (ADVANCED_DEFAULTS, CUSTOM_BACKGROUND_DIR, DATA_DIR, HTTP_PORT, HTTPS_PORT, OUTPUT_DIR, PRESETS,
                     RECORDING_KINDS, SR, TFLITE_PYTHON, VERSION)
from .device import gpu_summary
from .jobs import JobManager

STATIC = Path(__file__).with_name("static")
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

app = FastAPI(title="openWakeWord Trainer", docs_url="/api/docs", redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
jobs = JobManager()

_tts = None
_tts_lock = threading.Lock()
_gpu_cache = {"at": 0.0, "value": None}


def tts():
    global _tts
    with _tts_lock:
        if _tts is None:
            from .tts import Kokoro

            _tts = Kokoro(device=os.environ.get("PREVIEW_DEVICE", "cpu"))
    return _tts


@app.exception_handler(KeyError)
async def not_found(request: Request, exc: KeyError):
    return JSONResponse({"detail": "Not found"}, status_code=404)


@app.exception_handler(ValueError)
async def bad_request(request: Request, exc: ValueError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.exception_handler(RuntimeError)
async def conflict(request: Request, exc: RuntimeError):
    return JSONResponse({"detail": str(exc)}, status_code=409)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


# System and downloads -------------------------------------------------------------------------
@app.get("/api/system")
def system():
    if time.time() - _gpu_cache["at"] > 15:
        _gpu_cache.update(at=time.time(), value=gpu_summary())
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(DATA_DIR)
    return {
        "version": VERSION, "gpu": _gpu_cache["value"], "assets": assets.status(), "assets_ready": assets.all_ready(),
        "data_dir": str(DATA_DIR), "output_dir": str(OUTPUT_DIR), "custom_background_dir": str(CUSTOM_BACKGROUND_DIR),
        "disk_free_bytes": disk.free, "http_port": HTTP_PORT, "https_port": HTTPS_PORT, "presets": PRESETS,
        "advanced_defaults": ADVANCED_DEFAULTS, "tflite_available": TFLITE_PYTHON.exists(),
    }


@app.post("/api/assets/download")
def download_assets():
    return jobs.start("download", "owwt.assets", [], "Downloading training data")


@app.get("/api/job")
def job():
    return jobs.current()


@app.get("/api/job/log")
def job_log(offset: int = 0):
    return jobs.log_tail(offset)


@app.post("/api/job/cancel")
def cancel_job():
    jobs.cancel()
    return {"ok": True}


# Kokoro ---------------------------------------------------------------------------------------
@app.get("/api/voices")
def voices():
    from .tts import list_voices

    return list_voices()


class PhonemizeBody(BaseModel):
    text: str


@app.post("/api/tts/phonemize")
async def phonemize(body: PhonemizeBody):
    def work():
        engine = tts()
        with engine.lock:
            return {"american": engine.phonemize(body.text), "british": engine.phonemize(body.text, british=True)}

    return await run_in_threadpool(work)


class PreviewBody(BaseModel):
    text: str = ""
    phonemes: str = ""
    voice: str = "af_heart"
    speed: float = 1.0
    random_voice: bool = False


@app.post("/api/tts/preview")
async def preview(body: PreviewBody):
    if not body.text.strip() and not body.phonemes.strip():
        raise ValueError("Enter a phrase to preview")

    def work():
        engine = tts()
        with engine.lock:
            voice = engine.random_blend(random.Random()) if body.random_voice else body.voice
            main_voice = max(voice, key=voice.get) if isinstance(voice, dict) else voice
            phonemes = body.phonemes.strip() or engine.phonemize(body.text, british=main_voice.startswith("b"))
            unknown = engine.unknown_symbols(phonemes)
            audio = engine.synth(phonemes, voice, max(0.5, min(2.0, body.speed)))
        return audio, phonemes, voice, unknown

    audio, phonemes, voice, unknown = await run_in_threadpool(work)
    headers = {"X-Phonemes": quote(phonemes), "X-Voice": quote(json.dumps(voice)),
               "X-Unknown": quote("".join(unknown)), "Cache-Control": "no-store"}
    return Response(wav_bytes(audio), media_type="audio/wav", headers=headers)


# Projects -------------------------------------------------------------------------------------
class CreateBody(BaseModel):
    phrase: str
    preset: str = "standard"


@app.get("/api/projects")
def list_projects():
    return projects.list_projects()


@app.post("/api/projects")
def create_project(body: CreateBody):
    return projects.create(body.phrase, body.preset)


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    project = projects.summary(projects.load(pid))
    project["recordings"] = projects.recordings(pid)
    project["runs"] = [{k: r.get(k) for k in ("run_id", "created_at", "files", "recommended_threshold", "headline", "duration_seconds")}
                       for r in projects.runs(pid)]
    return project


@app.patch("/api/projects/{pid}")
def update_project(pid: str, patch: dict = Body(...)):
    return projects.update(pid, patch)


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    status = jobs.current()
    if status.get("running") and status.get("project_id") == pid:
        raise RuntimeError("Cancel the training job for this wake word before deleting it")
    projects.delete(pid)
    return {"ok": True}


@app.delete("/api/projects/{pid}/cache")
def clear_cache(pid: str):
    status = jobs.current()
    if status.get("running") and status.get("project_id") == pid:
        raise RuntimeError("Can't clear the cache while this wake word is training")
    shutil.rmtree(projects.project_dir(pid) / "work", ignore_errors=True)
    return {"ok": True}


@app.get("/api/projects/{pid}/suggestions")
async def suggestions(pid: str):
    project = projects.load(pid)
    return await run_in_threadpool(adversarial.suggestions, project["phrase"], project.get("custom_negatives", []))


@app.get("/api/projects/{pid}/recordings")
def list_recordings(pid: str):
    return projects.recordings(pid)


@app.post("/api/projects/{pid}/recordings/{kind}")
async def upload_recordings(pid: str, kind: str, files: list[UploadFile] = File(...), trim: bool = True):
    projects.project_dir(pid)
    if kind not in RECORDING_KINDS:
        raise KeyError(kind)
    results = []
    for upload in files:
        data = await upload.read()
        if len(data) > MAX_UPLOAD_BYTES:
            results.append({"filename": upload.filename, "error": "File is larger than 200 MB"})
            continue

        def work(data=data):
            x = decode(data)
            if kind != "background" and trim:
                x = trim_silence(x)
            if len(x) < 0.3 * SR:
                raise ValueError("Too short (under 0.3 seconds). Nothing was heard, check the microphone.")
            report = level_report(x)
            if kind == "background":
                report["warnings"] = []  # room noise is supposed to be quiet
            if kind == "positive" and len(x) > 4 * SR:
                report["warnings"].append("Longer than 4 seconds. Wake word clips work best as just the phrase.")
            item = projects.add_recording(pid, kind, x)
            item.update(report)
            return item

        try:
            results.append({"filename": upload.filename, **(await run_in_threadpool(work))})
        except Exception as exc:
            results.append({"filename": upload.filename, "error": str(exc)})
    return {"items": results}


@app.get("/api/projects/{pid}/recordings/{kind}/{name}")
def get_recording(pid: str, kind: str, name: str):
    return FileResponse(projects.recording_path(pid, kind, name), media_type="audio/wav")


@app.delete("/api/projects/{pid}/recordings/{kind}/{name}")
def delete_recording(pid: str, kind: str, name: str):
    projects.recording_path(pid, kind, name).unlink()
    return {"ok": True}


@app.get("/api/projects/{pid}/samples/{kind}")
def list_samples(pid: str, kind: str):
    if kind not in projects.SAMPLE_DIRS:
        raise KeyError(kind)
    return projects.generated_samples(pid, kind)


@app.get("/api/projects/{pid}/samples/{kind}/{name}")
def get_sample(pid: str, kind: str, name: str):
    return FileResponse(projects.generated_sample_path(pid, kind, name), media_type="audio/wav")


@app.post("/api/projects/{pid}/train")
def train(pid: str):
    project = projects.load(pid)
    return jobs.start("train", "owwt.train", [pid], f"Training \"{project['phrase']}\"", project_id=pid)


@app.get("/api/projects/{pid}/runs")
def list_runs(pid: str):
    return projects.runs(pid)


@app.get("/api/projects/{pid}/runs/{run_id}/files/{filename}")
def get_run_file(pid: str, run_id: str, filename: str):
    return FileResponse(projects.run_file(pid, run_id, filename), filename=filename,
                        media_type="application/octet-stream")


@app.post("/api/projects/{pid}/runs/{run_id}/publish")
def publish_run(pid: str, run_id: str):
    return {"copied": projects.publish(pid, run_id)}


# Live test ------------------------------------------------------------------------------------
@app.websocket("/api/projects/{pid}/live")
async def live(ws: WebSocket, pid: str, run: str = ""):
    from .live import LiveDetector

    await ws.accept()
    try:
        run_id = run or next(iter(projects.runs(pid)))["run_id"]
        model = next(projects.run_dir(pid, run_id).glob("*.onnx"))
        detector = await run_in_threadpool(LiveDetector, model)
    except (KeyError, StopIteration):
        await ws.send_json({"type": "error", "message": "No trained model yet. Train one first."})
        await ws.close()
        return

    await ws.send_json({"type": "ready", "run_id": run_id, "n_frames": detector.n_frames})
    threshold, last_detection, detection_audio = 0.5, 0.0, None
    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes"):
                pcm = np.frombuffer(message["bytes"], dtype="<i2")
                scores = await run_in_threadpool(detector.feed, pcm)
                for score in scores:
                    detected = score >= threshold and time.time() - last_detection > 1.5
                    if detected:
                        last_detection = time.time()
                        detection_audio = detector.last_audio(2.5)
                    await ws.send_json({"type": "score", "score": round(score, 4), "detected": detected})
            elif message.get("text"):
                data = json.loads(message["text"])
                if data.get("action") == "threshold":
                    threshold = min(0.99, max(0.01, float(data.get("value", 0.5))))
                elif data.get("action") == "save" and data.get("kind") in ("positive", "negative"):
                    if data.get("source") == "detection" and detection_audio is not None:
                        audio = detection_audio
                    else:
                        audio = detector.last_audio(float(data.get("seconds", 4)))
                    clip = trim_silence(audio.astype(np.float32) / 32768.0)
                    if len(clip) < 0.3 * SR:
                        await ws.send_json({"type": "error", "message": "Nothing to save yet"})
                        continue
                    item = await run_in_threadpool(projects.add_recording, pid, data["kind"], clip)
                    await ws.send_json({"type": "saved", "kind": data["kind"], "item": item})
    except WebSocketDisconnect:
        pass
