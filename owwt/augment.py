"""Audio augmentation for training clips: room reverb, background noise, EQ, pitch, level.

Follows the openWakeWord recipe (wake word ends 0 to 200 ms before the end of the window, reverb with
p=0.5, background noise with p=0.75, and so on) using plain numpy/scipy, so it runs in CPU worker
processes without the torch-audiomentations stack.
"""
import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, resample_poly, sosfilt

from .audio import read_wav, to_int16
from .config import SR


def colored_noise(n: int, rng: np.random.Generator, decay: float) -> np.ndarray:
    """decay 0 = white, 1 = pink, 2 = brown, negative = blue-ish."""
    half = n // 2 + 1
    spectrum = rng.standard_normal(half) + 1j * rng.standard_normal(half)
    f = np.arange(half, dtype=np.float64)
    f[0] = 1.0
    spectrum /= f ** (decay / 2.0)
    x = np.fft.irfft(spectrum, n)
    return (x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)


def random_eq(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    for _ in range(int(rng.integers(1, 4))):
        f0 = rng.uniform(100, 6000)
        gain = rng.uniform(-6, 6)
        q = rng.uniform(0.7, 2.0)
        a_gain = 10 ** (gain / 40)
        w0 = 2 * np.pi * f0 / SR
        alpha = np.sin(w0) / (2 * q)
        b = np.array([1 + alpha * a_gain, -2 * np.cos(w0), 1 - alpha * a_gain])
        a = np.array([1 + alpha / a_gain, -2 * np.cos(w0), 1 - alpha / a_gain])
        x = lfilter(b / a[0], a / a[0], x)
    return x.astype(np.float32)


def band_stop(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    center = rng.uniform(200, 4000)
    width = center * rng.uniform(0.2, 0.6)
    lo, hi = max(40.0, center - width / 2), min(SR / 2 - 100, center + width / 2)
    sos = butter(2, [lo, hi], btype="bandstop", fs=SR, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def _power(x: np.ndarray) -> float:
    return float(np.mean(x.astype(np.float64) ** 2)) if len(x) else 0.0


class Augmenter:
    def __init__(self, rir_paths, bank_paths=()):
        """bank_paths: (path, weight) pairs of int16 .npy files holding concatenated background audio."""
        self.rirs = []
        for path in rir_paths:
            try:
                rir = read_wav(path)
                if len(rir) > 16:
                    self.rirs.append(rir)
            except Exception:
                pass
        self.banks, weights = [], []
        for path, weight in bank_paths:
            bank = np.load(path, mmap_mode="r")
            if len(bank) > SR:
                self.banks.append(bank)
                weights.append(weight)
        self.bank_weights = np.array(weights, dtype=np.float64) / sum(weights) if weights else None
        self.bank = self.banks[0] if self.banks else None

    def background(self, n: int, rng: np.random.Generator):
        if not self.banks:
            return None
        bank = self.banks[int(rng.choice(len(self.banks), p=self.bank_weights))]
        start = int(rng.integers(0, len(bank) - n)) if len(bank) > n else 0
        seg = np.asarray(bank[start:start + n], dtype=np.float32) / 32768.0
        return np.resize(seg, n) if len(seg) < n else seg

    def reverb(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        rir = self.rirs[int(rng.integers(0, len(self.rirs)))]
        peak = int(np.argmax(np.abs(rir)))
        wet = fftconvolve(x, rir)[peak:peak + len(x)]
        return (wet * np.sqrt(_power(x) / (_power(wet) + 1e-12))).astype(np.float32)

    def speech(self, clip: np.ndarray, total_length: int, rng: np.random.Generator, align: str = "end",
               end_jitter: tuple = (0.0, 0.2)) -> np.ndarray:
        x = np.asarray(clip, dtype=np.float32)
        if rng.random() < 0.25:
            # Pitch and tempo shift together by resampling, up to two semitones either way
            factor = 2 ** (rng.uniform(-2, 2) / 12)
            x = resample_poly(x, 100, int(round(100 * factor))).astype(np.float32)

        n = total_length
        if len(x) >= n:
            if align == "end":
                x = x[-n:]
            else:
                offset = int(rng.integers(0, len(x) - n + 1))
                x = x[offset:offset + n]
            out, region = x.copy(), (0, n)
        else:
            out = np.zeros(n, dtype=np.float32)
            if align == "end":
                start = max(0, n - len(x) - int(rng.uniform(*end_jitter) * SR))
            else:
                start = int(rng.integers(0, n - len(x) + 1))
            out[start:start + len(x)] = x
            region = (start, start + len(x))
        return self._finish(out, region, rng, add_background=True)

    def noise_only(self, total_length: int, rng: np.random.Generator) -> np.ndarray:
        bg = self.background(total_length, rng)
        if bg is None:
            bg = colored_noise(total_length, rng, rng.uniform(-1, 2)) * 0.1
        return self._finish(bg, (0, total_length), rng, add_background=False)

    def _finish(self, x, region, rng, add_background):
        if rng.random() < 0.25:
            x = random_eq(x, rng)
        if rng.random() < 0.1:
            k = rng.uniform(1.0, 4.0)
            x = (np.tanh(k * x) / np.tanh(k)).astype(np.float32)
        if self.rirs and rng.random() < 0.5:
            x = self.reverb(x, rng)
        if rng.random() < 0.15:
            x = band_stop(x, rng)

        speech_power = _power(x[region[0]:region[1]]) or _power(x)
        if add_background and self.bank is not None and rng.random() < 0.75:
            bg = self.background(len(x), rng)
            bg_power = _power(bg)
            if bg_power > 1e-10:
                snr_db = rng.uniform(-5, 15)
                x = x + bg * np.sqrt(speech_power / (bg_power * 10 ** (snr_db / 10)))
        if rng.random() < 0.25:
            noise = colored_noise(len(x), rng, rng.uniform(-1, 2))
            snr_db = rng.uniform(10, 30)
            x = x + noise * np.sqrt(speech_power / (_power(noise) * 10 ** (snr_db / 10) + 1e-12))

        peak = float(np.max(np.abs(x)))
        if peak > 1e-6:
            x = x * (10 ** (rng.uniform(-30, -1) / 20) / peak)
        # Microphone self-noise, so silence is never exact digital zero
        x = x + rng.standard_normal(len(x)).astype(np.float32) * 10 ** (rng.uniform(-75, -55) / 20)
        return to_int16(x)


# Multiprocessing plumbing -----------------------------------------------------------------------
_AUGMENTER = None


def init_worker(rir_paths, bank_paths):
    global _AUGMENTER
    _AUGMENTER = Augmenter(rir_paths, bank_paths)


def augment_task(task) -> np.ndarray:
    """task = (source, total_length, align, seed[, options]). source is a wav path, an array, or None for noise only.

    options may set "jitter" (min, max seconds between the end of the speech and the end of the clip) and
    "micro_frames" to return microWakeWord spectrogram features (the last N frames, or all of them when
    N is None) instead of audio.
    """
    source, total_length, align, seed = task[:4]
    options = task[4] if len(task) > 4 else {}
    rng = np.random.default_rng(seed)
    if source is None:
        audio = _AUGMENTER.noise_only(total_length, rng)
    else:
        clip = read_wav(source) if isinstance(source, str) else source
        audio = _AUGMENTER.speech(clip, total_length, rng, align, tuple(options.get("jitter", (0.0, 0.2))))
    if "micro_frames" in options:
        features = micro_features(audio)
        keep = options["micro_frames"]
        return features[-keep:] if keep else features
    return audio


MICRO_FEATURE_SCALE = 0.0390625  # microWakeWord stores the frontend output as unscaled uint16


def micro_features(audio: np.ndarray) -> np.ndarray:
    """The 40 channel micro_speech spectrogram ESPHome computes on the device (30 ms windows, 10 ms steps)."""
    from pymicro_features import MicroFrontend

    frontend = MicroFrontend()
    data = np.asarray(audio, dtype=np.int16).tobytes()
    features, i = [], 0
    while i + 320 < len(data):
        result = frontend.process_samples(data[i:i + 320])
        i += result.samples_read * 2
        if result.features:
            features.append(result.features)
    scaled = np.round(np.asarray(features, dtype=np.float32) / MICRO_FEATURE_SCALE)
    return np.clip(scaled, 0, 65535).astype(np.uint16)
