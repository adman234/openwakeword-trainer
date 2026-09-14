"""Live microphone testing: the browser streams 16 kHz PCM over a WebSocket and gets scores back."""
import collections

import numpy as np
import onnxruntime as ort

from .config import EMBEDDING_MODEL, MELSPEC_MODEL, SR

CHUNK = 1280  # 80 ms, one openWakeWord feature frame


class LiveDetector:
    def __init__(self, model_path):
        from openwakeword.utils import AudioFeatures

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.n_frames = int(self.session.get_inputs()[0].shape[1])
        self.features = AudioFeatures(melspec_model_path=str(MELSPEC_MODEL), embedding_model_path=str(EMBEDDING_MODEL),
                                      inference_framework="onnx", device="cpu", ncpu=1)
        self.pending = np.zeros(0, dtype=np.int16)
        self.frames = 0
        self.history = collections.deque(maxlen=int(6 * SR / CHUNK))

    def feed(self, pcm: np.ndarray) -> list[float]:
        self.pending = np.concatenate([self.pending, pcm.astype(np.int16)])
        scores = []
        while len(self.pending) >= CHUNK:
            chunk, self.pending = self.pending[:CHUNK], self.pending[CHUNK:]
            self.history.append(chunk)
            self.features(chunk)
            x = self.features.get_features(self.n_frames)
            score = float(self.session.run(None, {self.input_name: x})[0].reshape(-1)[0])
            self.frames += 1
            # The feature buffer starts filled with noise, so ignore scores until real audio fills the window
            scores.append(0.0 if self.frames <= self.n_frames else score)
        return scores

    def last_audio(self, seconds: float) -> np.ndarray:
        chunks = list(self.history)[-max(1, int(seconds * SR / CHUNK)):]
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
