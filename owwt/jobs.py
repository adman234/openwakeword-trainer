"""Background jobs (training or downloads), one at a time, run as a subprocess the web UI watches and can cancel.

The web process uses JobManager to launch and observe. The job process uses JobReporter to publish its
stage and progress to /data/job/status.json and its log to stdout (which the manager points at log.txt).
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

from .config import JOB_DIR

STATUS_FILE = JOB_DIR / "status.json"
LOG_FILE = JOB_DIR / "log.txt"
CANCEL_FILE = JOB_DIR / "cancel"
APP_ROOT = Path(__file__).resolve().parent.parent


def _read_status() -> dict:
    try:
        return json.loads(STATUS_FILE.read_text())
    except Exception:
        return {}


def _write_status(status: dict, required: bool = True):
    """Atomically replace status.json. Retries briefly because the web process may be reading it at that
    moment (Windows refuses to replace an open file). Progress updates pass required=False, so a missed
    write never takes down a training run."""
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    status["updated_at"] = time.time()
    tmp = STATUS_FILE.with_name(f"status.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(status))
    for attempt in range(20):
        try:
            os.replace(tmp, STATUS_FILE)
            return
        except PermissionError:
            if attempt == 19:
                if required:
                    raise
                return
            time.sleep(0.05)


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


class JobReporter:
    """Used inside the job subprocess."""

    def __init__(self, stages: list[str]):
        self.status = _read_status()
        self.status.update(stages=stages, stage=-1, stage_progress=0.0, message="Starting", state="running",
                           pid=os.getpid())
        self._last_write = 0.0
        _write_status(self.status)

    def log(self, message: str):
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    def stage(self, index: int, message: str = ""):
        self.status.update(stage=index, stage_progress=0.0, message=message or self.status["stages"][index])
        self.log(f"=== {self.status['stages'][index]} ===")
        _write_status(self.status)

    def progress(self, fraction: float, message: str | None = None):
        self.status["stage_progress"] = max(0.0, min(1.0, float(fraction)))
        if message:
            self.status["message"] = message
        now = time.time()
        if now - self._last_write > 0.5 or fraction >= 1.0:
            self._last_write = now
            _write_status(self.status, required=False)

    def set(self, **values):
        self.status.update(values)
        _write_status(self.status)

    def should_stop(self) -> bool:
        return CANCEL_FILE.exists()

    def check_stop(self):
        if self.should_stop():
            raise KeyboardInterrupt("cancelled")

    def run(self, fn):
        code = 0
        try:
            fn()
            self.status.update(state="done", message="Finished", stage_progress=1.0)
            self.log("Job finished")
        except KeyboardInterrupt:
            self.status.update(state="cancelled", message="Cancelled")
            self.log("Job cancelled")
            code = 130
        except Exception as exc:
            self.log(traceback.format_exc())
            self.status.update(state="failed", message="Failed", error=str(exc) or exc.__class__.__name__)
            code = 1
        finally:
            self.status["finished_at"] = time.time()
            _write_status(self.status)
        sys.exit(code)


class JobManager:
    """Used by the web process."""

    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()
        status = _read_status()
        if status.get("state") == "running" and not _pid_alive(status.get("pid")):
            status.update(state="interrupted", message="Interrupted (the container restarted)")
            _write_status(status)

    def running(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        status = _read_status()
        return status.get("state") == "running" and _pid_alive(status.get("pid"))

    def current(self) -> dict:
        with self.lock:
            status = _read_status()
            if self.proc is not None and self.proc.poll() is not None and status.get("state") == "running":
                status.update(state="failed", message="Failed", error=f"Job process exited with code {self.proc.returncode}",
                              finished_at=time.time())
                _write_status(status)
            status["running"] = status.get("state") == "running"
            return status

    def start(self, kind: str, module: str, args: list[str], title: str, project_id: str | None = None) -> dict:
        with self.lock:
            if self.running():
                raise RuntimeError("Another job is already running. Wait for it to finish or cancel it first.")
            JOB_DIR.mkdir(parents=True, exist_ok=True)
            CANCEL_FILE.unlink(missing_ok=True)
            status = {"id": uuid.uuid4().hex[:12], "kind": kind, "title": title, "project_id": project_id,
                      "state": "running", "stages": [], "stage": -1, "stage_progress": 0.0, "message": "Starting",
                      "started_at": time.time()}
            _write_status(status)
            log = open(LOG_FILE, "w", encoding="utf-8")
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            popen_kwargs = {"start_new_session": True} if os.name == "posix" else {}
            self.proc = subprocess.Popen([sys.executable, "-m", module, *args], cwd=str(APP_ROOT), stdout=log,
                                         stderr=subprocess.STDOUT, env=env, **popen_kwargs)
            log.close()
            status["pid"] = self.proc.pid
            _write_status(status)
            return status

    def cancel(self):
        CANCEL_FILE.touch()
        proc = self.proc
        pid = proc.pid if proc else _read_status().get("pid")

        def force_kill():
            time.sleep(20)
            alive = proc.poll() is None if proc else _pid_alive(pid)
            if not alive:
                return
            try:
                if os.name == "posix":
                    os.killpg(int(pid), signal.SIGKILL)
                elif proc:
                    proc.kill()
            except Exception:
                pass
            status = _read_status()
            if status.get("state") == "running":
                status.update(state="cancelled", message="Cancelled", finished_at=time.time())
                _write_status(status)

        threading.Thread(target=force_kill, daemon=True).start()

    def shutdown(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(self.proc.pid, signal.SIGTERM)
                else:
                    self.proc.terminate()
            except Exception:
                pass

    @staticmethod
    def log_tail(offset: int = 0, limit: int = 256_000) -> dict:
        if not LOG_FILE.exists():
            return {"text": "", "offset": 0}
        size = LOG_FILE.stat().st_size
        if offset > size:
            offset = 0
        if size - offset > limit:
            offset = size - limit
        with open(LOG_FILE, "rb") as f:
            f.seek(offset)
            data = f.read()
        return {"text": data.decode("utf-8", errors="replace"), "offset": offset + len(data)}
