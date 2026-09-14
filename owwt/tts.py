"""Kokoro TTS running in-process, loading the weights and voices baked into the image."""
import random
import threading

import numpy as np
import torch

from .audio import resample, trim_silence
from .config import KOKORO_DIR, KOKORO_REPO

VOICE_LANGUAGES = {
    "a": "American English", "b": "British English", "e": "Spanish", "f": "French", "h": "Hindi",
    "i": "Italian", "j": "Japanese", "p": "Brazilian Portuguese", "z": "Mandarin",
}
KOKORO_SR = 24000


def list_voices() -> list[dict]:
    names = sorted(p.stem for p in (KOKORO_DIR / "voices").glob("*.pt"))
    return [
        {
            "id": n,
            "language": VOICE_LANGUAGES.get(n[0], n[0]),
            "gender": "female" if n[1:2] == "f" else "male",
            "english": n[0] in "ab",
            "label": f"{n.split('_', 1)[-1].title()} ({VOICE_LANGUAGES.get(n[0], n[0])}, {'F' if n[1:2] == 'f' else 'M'})",
        }
        for n in names
    ]


class Kokoro:
    def __init__(self, device: str = "cpu"):
        from kokoro import KModel

        self.device = device
        self.model = KModel(
            repo_id=KOKORO_REPO,
            config=str(KOKORO_DIR / "config.json"),
            model=str(KOKORO_DIR / "kokoro-v1_0.pth"),
        ).to(device).eval()
        self.vocab = self.model.vocab
        self._g2p = {}
        self._voices = {}
        self.voice_names = sorted(p.stem for p in (KOKORO_DIR / "voices").glob("*.pt"))
        self.english_voice_names = [n for n in self.voice_names if n[0] in "ab"]
        self.lock = threading.Lock()

    # Text to phonemes -------------------------------------------------------------------------
    def phonemize(self, text: str, british: bool = False) -> str:
        from kokoro import KPipeline

        code = "b" if british else "a"
        if code not in self._g2p:
            self._g2p[code] = KPipeline(lang_code=code, repo_id=KOKORO_REPO, model=False)
        _, tokens = self._g2p[code].g2p(text)
        return "".join((t.phonemes or "") + (" " if t.whitespace else "") for t in tokens).strip()

    def clean_phonemes(self, phonemes: str) -> str:
        return "".join(c for c in phonemes if c in self.vocab).strip()[:500]

    def unknown_symbols(self, phonemes: str) -> list[str]:
        return sorted({c for c in phonemes if c not in self.vocab})

    # Voices -----------------------------------------------------------------------------------
    def voice(self, name: str) -> torch.Tensor:
        if name not in self._voices:
            path = KOKORO_DIR / "voices" / f"{name}.pt"
            if not path.exists():
                raise ValueError(f"Unknown voice '{name}'")
            self._voices[name] = torch.load(path, map_location="cpu", weights_only=True).float()
        return self._voices[name]

    def blend(self, weights: dict[str, float]) -> torch.Tensor:
        total = sum(weights.values())
        return sum(self.voice(n) * (w / total) for n, w in weights.items())

    def random_blend(self, rng: random.Random, english_only: bool = False) -> dict[str, float]:
        """Mix 1 to 3 voices with random weights. Blends give far more speaker variety than 54 fixed voices."""
        pool = self.english_voice_names if english_only or rng.random() < 0.75 else self.voice_names
        k = rng.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]
        chosen = rng.sample(pool, k)
        raw = [rng.random() + 0.15 for _ in chosen]
        return {n: w / sum(raw) for n, w in zip(chosen, raw)}

    # Synthesis --------------------------------------------------------------------------------
    def synth(self, phonemes: str, voice, speed: float = 1.0, trim_pad: float = 0.08) -> np.ndarray:
        """Phonemes to 16 kHz float32 audio with silence trimmed."""
        ps = self.clean_phonemes(phonemes)
        if not ps:
            raise ValueError("Nothing pronounceable in the phonemes")
        if isinstance(voice, str):
            pack = self.voice(voice)
        elif isinstance(voice, dict):
            pack = self.blend(voice)
        else:
            pack = voice
        with torch.no_grad():
            audio = self.model(ps, pack[len(ps) - 1], speed)
        x = audio.float().cpu().numpy()
        x = resample(x, KOKORO_SR)
        return trim_silence(x, pad=trim_pad)
