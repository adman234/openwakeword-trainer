"""Small audio helpers: decoding, resampling, trimming and WAV IO. Everything is 16 kHz mono."""
import io
import subprocess
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import SR


def resample(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if int(sr_in) == int(sr_out):
        return x
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, int(sr_out) // g, int(sr_in) // g).astype(np.float32)


def decode(data) -> np.ndarray:
    """Decode audio (bytes or a path, any format ffmpeg understands) to 16 kHz mono float32."""
    try:
        src = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else str(data)
        x, sr = sf.read(src, dtype="float32", always_2d=True)
        return resample(x.mean(axis=1), sr)
    except Exception:
        pass

    # Formats libsndfile can't read (webm, m4a, mp4...) go through ffmpeg via a temp file,
    # since some containers can't be decoded from a pipe.
    with tempfile.TemporaryDirectory() as tmp:
        if isinstance(data, (bytes, bytearray)):
            path = Path(tmp) / "input"
            path.write_bytes(data)
        else:
            path = Path(data)
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
               "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0 or not result.stdout:
            raise ValueError("Couldn't read that file as audio")
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def read_wav(path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(x.mean(axis=1), sr)


def to_int16(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.int16:
        return x
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def write_wav(path, x: np.ndarray, sr: int = SR):
    sf.write(str(path), to_int16(x), sr, subtype="PCM_16")


def wav_bytes(x: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, to_int16(x), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def frame_rms(x: np.ndarray, frame: int = 320) -> np.ndarray:
    n = len(x) // frame
    if n == 0:
        return np.array([float(np.sqrt(np.mean(x ** 2)))]) if len(x) else np.zeros(1)
    return np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)


def trim_silence(x: np.ndarray, pad: float = 0.15, frame: int = 320) -> np.ndarray:
    """Cut leading and trailing silence, keeping `pad` seconds either side of the speech."""
    if len(x) < frame * 4:
        return x
    rms = frame_rms(x, frame)
    floor = float(np.percentile(rms, 10))
    threshold = max(floor * 4.0, float(rms.max()) * 0.06, 3e-4)
    active = np.flatnonzero(rms > threshold)
    if active.size == 0:
        return x
    p = int(pad * SR)
    start = max(0, int(active[0]) * frame - p)
    end = min(len(x), (int(active[-1]) + 1) * frame + p)
    return x[start:end]


def db(value: float) -> float:
    return float(20 * np.log10(max(value, 1e-9)))


def level_report(x: np.ndarray) -> dict:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    warnings = []
    if peak < 10 ** (-35 / 20):
        warnings.append("Very quiet. Move closer to the microphone or raise the input level.")
    if peak > 0.99:
        warnings.append("Clipping detected. Lower the input level a little.")
    return {"peak_db": round(db(peak), 1), "rms_db": round(db(rms), 1), "warnings": warnings}
