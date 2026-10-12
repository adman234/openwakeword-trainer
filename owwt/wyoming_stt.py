"""Minimal Wyoming speech-to-text client, the protocol Home Assistant uses for STT servers.

Point STT_URI at any Wyoming STT server (wyoming-faster-whisper, wyoming-onnx-stt, ...), e.g.
tcp://192.168.1.10:10300. Used by clip review to check what was actually said.
"""
import json
import os
import socket
from urllib.parse import urlparse

import numpy as np

from .audio import to_int16
from .config import SR

STT_URI = os.environ.get("STT_URI", "").strip()
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "en").strip()
_CHUNK_SAMPLES = 1024


def configured() -> bool:
    return bool(STT_URI)


def _address(uri: str) -> tuple[str, int]:
    parsed = urlparse(uri if "://" in uri else f"tcp://{uri}")
    if parsed.scheme != "tcp" or not parsed.hostname:
        raise ValueError(f"STT_URI must look like tcp://host:port, got {uri!r}")
    return parsed.hostname, parsed.port or 10300


def _send(sock: socket.socket, kind: str, data: dict | None = None, payload: bytes | None = None):
    header = {"type": kind, "version": "1.5.0"}
    data_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8") if data else b""
    if data_bytes:
        header["data_length"] = len(data_bytes)
    if payload:
        header["payload_length"] = len(payload)
    sock.sendall(json.dumps(header).encode("utf-8") + b"\n" + data_bytes + (payload or b""))


def _read_exactly(f, n: int) -> bytes:
    out = b""
    while len(out) < n:
        chunk = f.read(n - len(out))
        if not chunk:
            raise ConnectionError("STT server closed the connection")
        out += chunk
    return out


def _read_event(f) -> tuple[str, dict]:
    line = f.readline()
    if not line:
        raise ConnectionError("STT server closed the connection")
    header = json.loads(line)
    data = header.get("data") or {}
    if header.get("data_length"):
        data.update(json.loads(_read_exactly(f, header["data_length"])))
    if header.get("payload_length"):
        _read_exactly(f, header["payload_length"])
    return header["type"], data


def transcribe(audio: np.ndarray, uri: str = "", language: str = "", timeout: float = 30.0) -> str:
    """Transcribe 16 kHz mono audio (float32 or int16). Returns the text, '' if nothing was recognised."""
    host, port = _address(uri or STT_URI)
    pcm = to_int16(np.asarray(audio)).tobytes()
    fmt = {"rate": SR, "width": 2, "channels": 1}
    with socket.create_connection((host, port), timeout=timeout) as sock:
        _send(sock, "transcribe", {"language": language or STT_LANGUAGE} if (language or STT_LANGUAGE) else None)
        _send(sock, "audio-start", fmt)
        step = _CHUNK_SAMPLES * 2
        for i in range(0, len(pcm), step):
            _send(sock, "audio-chunk", fmt, pcm[i:i + step])
        _send(sock, "audio-stop", {})
        with sock.makefile("rb") as f:
            while True:
                kind, data = _read_event(f)
                if kind == "transcript":
                    return " ".join(str(data.get("text") or "").split())
                if kind == "error":
                    raise RuntimeError(f"STT server error: {data.get('text') or data}")
