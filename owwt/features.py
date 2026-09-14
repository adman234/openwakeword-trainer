"""openWakeWord feature extraction: the melspectrogram and Google speech-embedding ONNX models."""
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .augment import augment_task, init_worker
from .config import EMBEDDING_MODEL, FEATURE_DIM, MELSPEC_MODEL


def frames_for_length(n_samples: int) -> int:
    n_mel = int(np.ceil(n_samples / 160 - 3))
    return (n_mel - 76) // 8 + 1


def make_pool(rir_paths, bank_paths, processes: int):
    # spawn (not fork) so workers never inherit CUDA or onnxruntime threads from the parent
    ctx = mp.get_context("spawn")
    return ctx.Pool(processes, initializer=init_worker, initargs=(list(rir_paths), list(bank_paths)))


class FeatureExtractor:
    def __init__(self, use_gpu: bool = True, ncpu: int | None = None):
        import onnxruntime as ort
        from openwakeword.utils import AudioFeatures

        self.ncpu = ncpu or max(1, (os.cpu_count() or 2) // 2)
        device = "cpu"
        if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            try:
                import torch  # noqa: F401  loads the CUDA and cuDNN libraries that ship with PyTorch
                if hasattr(ort, "preload_dlls"):
                    ort.preload_dlls()
            except Exception:
                pass
            device = "gpu"

        def build(dev):
            # ncpu=1 for the session itself; embed() parallelises across clips with a thread pool instead
            return AudioFeatures(melspec_model_path=str(MELSPEC_MODEL), embedding_model_path=str(EMBEDDING_MODEL),
                                 inference_framework="onnx", device=dev, ncpu=1)

        try:
            self.af = build(device)
        except Exception:
            if device == "cpu":
                raise
            self.af = build("cpu")
        self.gpu = "CUDA" in self.af.onnx_execution_provider

    def embed(self, clips: np.ndarray) -> np.ndarray:
        """(N, samples) int16 clips of equal length to (N, frames, 96) features."""
        x = np.ascontiguousarray(clips, dtype=np.int16)
        mel = self.af._get_melspectrogram_batch(x, batch_size=128 if self.gpu else 32, ncpu=self.ncpu)
        return self.af._get_embeddings_batch(mel[:, :, :, None], batch_size=2048 if self.gpu else 64, ncpu=self.ncpu)

    def embed_stream(self, clip: np.ndarray) -> np.ndarray:
        """Features for one clip of any length, (frames, 96)."""
        x = np.asarray(clip, dtype=np.int16)
        if len(x) < 16000:
            x = np.concatenate([np.zeros(16000 - len(x), dtype=np.int16), x])
        return np.atleast_2d(self.af._get_embeddings(x)).astype(np.float32)


def compute_features(pool, tasks, total_length: int, out_path: Path, extractor: FeatureExtractor,
                     progress=None, batch: int = 512):
    """Augment `tasks` in worker processes and store their features as a float16 .npy file."""
    out_path = Path(out_path)
    n_frames = frames_for_length(total_length)
    if not tasks:
        np.save(out_path, np.zeros((0, n_frames, FEATURE_DIM), dtype=np.float16))
        return
    tmp = out_path.with_name(out_path.stem + ".partial.npy")
    arr = open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(tasks), n_frames, FEATURE_DIM))
    buf, done = [], 0
    for clip in pool.imap(augment_task, tasks, chunksize=16):
        buf.append(clip)
        if len(buf) == batch:
            arr[done:done + len(buf)] = extractor.embed(np.stack(buf))
            done += len(buf)
            buf = []
            if progress:
                progress(done / len(tasks))
    if buf:
        arr[done:done + len(buf)] = extractor.embed(np.stack(buf))
    arr.flush()
    del arr
    os.replace(tmp, out_path)
    if progress:
        progress(1.0)
