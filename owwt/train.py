"""Training pipeline for one project, run as a background job: python -m owwt.train <project_id>

Each stage is cached. Generated voices and their features are reused while the settings that produced
them are unchanged, so retraining after recording more clips only redoes the quick parts.
Heavy libraries are imported inside functions so the spawned augmentation workers start fast.
"""
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import assets, projects
from .audio import read_wav, to_int16, write_wav
from .config import CUSTOM_BACKGROUND_DIR, MODEL_TYPES, SR, training_settings

COMMON_STAGES = ["Check training data", "Generate wake word voices", "Generate similar-sounding phrases"]
OWW_STAGES = ["Augment audio and extract features", "Train the model", "Evaluate and export"]
MICRO_STAGES = ["Build ESPHome spectrograms", "Train the ESPHome model", "Export for ESPHome"]
GENERATION_VERSION = 1
FEATURE_VERSION = 1


def fingerprint(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def read_meta(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def write_meta(path: Path, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


class Pipeline:
    def __init__(self, pid: str, reporter):
        self.pid = pid
        self.r = reporter
        self.log = reporter.log
        self.project = projects.load(pid)
        self.settings = training_settings(self.project)
        self.dir = projects.project_dir(pid)
        self.work = self.dir / "work"
        self.work.mkdir(parents=True, exist_ok=True)
        self.run_id = time.strftime("%Y%m%d-%H%M%S")
        self.started = time.time()
        self._kokoro = None
        s = self.settings
        self.n_pos, self.n_neg = int(s["n_positive"]), int(s["n_negative"])
        self.n_pos_val, self.n_neg_val = max(200, self.n_pos // 10), max(200, self.n_neg // 10)
        self.types = [t for t in MODEL_TYPES if t in (self.project.get("model_types") or ["openwakeword"])] or ["openwakeword"]
        self._shared = None

    # ------------------------------------------------------------------------------------------
    def run(self):
        from .device import pick_torch_device

        stages = list(COMMON_STAGES)
        if "openwakeword" in self.types:
            stages += OWW_STAGES
        if "microwakeword" in self.types:
            stages += MICRO_STAGES
        self.r.set(project_id=self.pid, run_id=self.run_id, stages=stages)
        self.device, self.device_note = pick_torch_device()
        labels = {"openwakeword": "openWakeWord for Home Assistant", "microwakeword": "microWakeWord for ESPHome"}
        self.log(f"Wake word: \"{self.project['phrase']}\"  model file: {self.project['model_name']}")
        self.log("Building: " + " and ".join(labels[t] for t in self.types))
        self.log(f"Compute device: {self.device_note}")
        self.log("Settings: " + ", ".join(f"{k}={v}" for k, v in self.settings.items() if k not in ("label", "description")))

        stage = iter(range(len(stages)))
        self.r.stage(next(stage))
        self.check_assets()
        self.r.stage(next(stage))
        self.generate_positives()
        self.r.stage(next(stage))
        self.generate_negatives()
        self.release_kokoro()

        name = self.project.get("model_name") or self.pid
        run_dir = self.dir / "runs" / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        report = {"run_id": self.run_id, "project_id": self.pid, "phrase": self.project["phrase"], "model_name": name,
                  "model_types": self.types, "device": self.device_note, "settings": self.settings}

        if "openwakeword" in self.types:
            from .model import train

            self.r.stage(next(stage))
            data, split = self.build_features()
            self.r.stage(next(stage))
            model, results = train(data, self.settings, self.device, self.log, self.r.progress, self.r.should_stop)
            self.r.stage(next(stage))
            report.update(self.export(model, results, data, split, run_dir, name))

        if "microwakeword" in self.types:
            from .micro import MicroTrainer

            micro = MicroTrainer(self)
            self.r.stage(next(stage))
            micro.build_features()
            self.r.stage(next(stage))
            result = micro.train()
            self.r.stage(next(stage))
            report["micro"] = micro.export(result, run_dir, name)
            report.setdefault("recommended_threshold", report["micro"]["probability_cutoff"])
            report.setdefault("headline", {"threshold": report["micro"]["probability_cutoff"], "recall_tts": report["micro"]["recall"],
                                           "fp_per_hour": report["micro"]["faph"], "recall_real_clips": None,
                                           "similar_phrase_accept": None})

        self.finish(report, run_dir)

    def finish(self, report: dict, run_dir: Path):
        report.update(created_at=time.time(), duration_seconds=round(time.time() - self.started))
        (run_dir / "report.json").write_text(json.dumps(report, indent=2))
        micro = report.get("micro")
        self.r.set(result={"run_id": self.run_id, "headline": report.get("headline"),
                           "micro": {k: micro[k] for k in ("probability_cutoff", "recall", "faph")} if micro else None})
        if self.project.get("publish_to_output", True):
            try:
                for path in projects.publish(self.pid, self.run_id):
                    self.log(f"Copied to {path}")
            except Exception as exc:
                self.log(f"WARNING: could not copy models to the output folder: {exc}")
        self.r.progress(1.0, "Done")

    def shared_inputs(self) -> dict:
        """Recordings, the held-out split and background audio banks, shared by both model types."""
        if self._shared is None:
            real_pos = projects.recording_files(self.pid, "positive")
            real_neg = projects.recording_files(self.pid, "negative")
            # Hold out about one in five of your own wake word recordings to measure how well the model hears you
            held_out = set()
            if len(real_pos) >= 5:
                held_out = {p.name for p in real_pos if stable_hash(p.name) % 5 == 0} or {real_pos[-1].name}
            self.log(f"Your recordings: {len(real_pos)} wake word ({len(held_out)} held out for testing), "
                     f"{len(real_neg)} other speech")
            custom_bank = self._custom_bank()
            tts_banks = [(str(assets.BACKGROUND_BANK), 1.0)]
            run_banks = tts_banks + ([(str(custom_bank), 0.5)] if custom_bank else [])
            self._shared = {"real_pos": real_pos, "real_neg": real_neg, "held_out": held_out,
                            "tts_banks": tts_banks, "run_banks": run_banks}
        return self._shared

    # Stage 1 ----------------------------------------------------------------------------------
    def check_assets(self):
        from . import micro

        want_oww, want_micro = "openwakeword" in self.types, "microwakeword" in self.types
        if want_micro and not micro.available():
            raise RuntimeError("The microWakeWord trainer is not installed in this container, so ESPHome models can't be built")
        pieces = []
        if want_oww and not assets.all_ready():
            pieces.append((assets.ensure_all, "openWakeWord training data (" + ", ".join(assets.missing()) + ")"))
        elif want_micro and not (assets._ready("rirs") and assets._ready("background")):
            pieces.append((assets.ensure_augmentation, "room acoustics and background sounds"))
        if want_micro and not assets.micro_ready():
            pieces.append((assets.ensure_microwakeword, "ESPHome negative data (5.7 GB)"))
        if not pieces:
            self.log("All shared training data is present")
            self.r.progress(1.0, "Training data ready")
            return
        self.log("Downloading shared training data. This happens once and is reused by every wake word: "
                 + "; ".join(label for _, label in pieces))
        for i, (ensure, _) in enumerate(pieces):
            ensure(lambda f, m, i=i: (self.r.check_stop(), self.r.progress((i + f) / len(pieces), m)))

    # Kokoro -----------------------------------------------------------------------------------
    def kokoro(self):
        if self._kokoro is None:
            from .tts import Kokoro

            self.log("Loading Kokoro TTS")
            self._kokoro = Kokoro(self.device)
        return self._kokoro

    def release_kokoro(self):
        if self._kokoro is not None:
            self._kokoro = None
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _generate(self, folder: Path, n: int, fp: str, pick, what: str, span: tuple[float, float]):
        """Generate n clips named 000000.wav... Resumable, and extendable if n grows later."""
        meta_path = folder / "meta.json"
        if folder.exists() and read_meta(meta_path).get("fingerprint") != fp:
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
        write_meta(meta_path, {"fingerprint": fp})
        existing = {p.stem for p in folder.glob("*.wav")}
        todo = [i for i in range(n) if f"{i:06d}" not in existing]
        if not todo:
            self.log(f"{what}: reusing {n:,} clips generated earlier")
            self.r.progress(span[1])
            return
        self.log(f"{what}: generating {len(todo):,} clips ({n - len(todo):,} already done)")
        tts = self.kokoro()
        started, failures = time.time(), 0
        for done, i in enumerate(todo, 1):
            self.r.check_stop()
            rng = random.Random(f"{fp}-{i}")
            phonemes, voice, speed = pick(rng, i)
            try:
                audio = tts.synth(phonemes, voice, speed)
                if len(audio) < SR // 10:
                    raise ValueError("clip too short")
                write_wav(folder / f"{i:06d}.wav", audio)
            except Exception as exc:
                failures += 1
                if failures <= 3:
                    self.log(f"  skipped a clip ({exc})")
                if failures > 100 and failures > done // 2:
                    raise RuntimeError(f"Kokoro keeps failing to synthesize this text: {exc}")
            if done % 20 == 0 or done == len(todo):
                eta = (time.time() - started) / done * (len(todo) - done)
                frac = span[0] + (span[1] - span[0]) * done / len(todo)
                self.r.progress(frac, f"{what}: {n - len(todo) + done:,} of {n:,} clips, about {max(1, round(eta / 60))} min left")

    def _variants(self, phonemes: str) -> list[str]:
        return [phonemes, phonemes + ".", phonemes + "!", phonemes + "?"]

    # Stage 2 ----------------------------------------------------------------------------------
    def generate_positives(self):
        p = self.project
        english_only = self.settings["voices"] == "english"
        pron = p.get("pronunciation", "").strip()
        texts = [p["phrase"], *p.get("variants", [])]
        fp = fingerprint({"v": GENERATION_VERSION, "texts": texts, "pron": pron, "english_only": english_only})

        entries = {}  # british flag -> phoneme strings
        if pron:
            entries[False] = entries[True] = self._variants(pron)
            self.log(f"Using custom pronunciation: {pron}")
        else:
            tts = self.kokoro()
            for british in (False, True):
                entries[british] = [v for t in texts for v in self._variants(tts.phonemize(t, british))]
            self.log(f"Phonemes (American): {entries[False][0]}   (British): {entries[True][0]}")

        def pick(rng, i):
            tts = self.kokoro()
            blend = tts.random_blend(rng, english_only)
            british = max(blend, key=blend.get).startswith("b")
            return rng.choice(entries[british]), blend, rng.uniform(0.8, 1.25)

        self._generate(self.work / "tts" / "pos_train", self.n_pos, fp + "-train", pick, "Wake word clips", (0, 0.9))
        self._generate(self.work / "tts" / "pos_val", self.n_pos_val, fp + "-val", pick, "Wake word test clips", (0.9, 1))
        self.pos_fp = fp

    # Stage 3 ----------------------------------------------------------------------------------
    def generate_negatives(self):
        from .adversarial import negative_phrases

        p = self.project
        english_only = self.settings["voices"] == "english"
        custom = p.get("custom_negatives", [])
        fp = fingerprint({"v": GENERATION_VERSION, "phrase": p["phrase"], "variants": p.get("variants", []),
                          "custom": custom, "english_only": english_only})
        self.log("Finding similar-sounding words")
        train_texts = negative_phrases(p["phrase"], self.n_neg, custom, seed=11)
        val_texts = negative_phrases(p["phrase"], self.n_neg_val, custom, seed=22)
        self.log("Examples: " + ", ".join(list(dict.fromkeys(train_texts))[:12]))
        cache = {}

        def phonemes_for(text, british):
            key = (text, british)
            if key not in cache:
                cache[key] = self.kokoro().phonemize(text, british)
            return cache[key]

        def make_pick(texts):
            def pick(rng, i):
                tts = self.kokoro()
                blend = tts.random_blend(rng, english_only)
                british = max(blend, key=blend.get).startswith("b")
                ps = phonemes_for(texts[i % len(texts)], british) or phonemes_for("hello there", british)
                return ps + rng.choice(["", ".", "!", "?"]), blend, rng.uniform(0.8, 1.25)
            return pick

        self._generate(self.work / "tts" / "neg_train", self.n_neg, fp + "-train", make_pick(train_texts),
                       "Similar phrase clips", (0, 0.9))
        self._generate(self.work / "tts" / "neg_val", self.n_neg_val, fp + "-val", make_pick(val_texts),
                       "Similar phrase test clips", (0.9, 1))
        self.neg_fp = fp

    # Stage 4 ----------------------------------------------------------------------------------
    def _clips(self, name: str, n: int) -> list[str]:
        return [str(p) for p in sorted((self.work / "tts" / name).glob("*.wav"))[:n]]

    def _total_length(self, pos_clips: list[str], real_clips: list) -> int:
        rng = random.Random(0)
        sample = rng.sample(pos_clips, min(200, len(pos_clips)))
        durations = [sf.info(c).frames for c in sample]
        median = float(np.median(durations)) if durations else SR
        if real_clips:
            real_median = float(np.median([sf.info(str(c)).frames for c in real_clips])) - 0.3 * SR
            median = max(median, real_median)
        total = int(round(median / 1000) * 1000) + 12000
        if total < 32000 or abs(total - 32000) <= 4000:
            total = 32000
        return min(total, 64000)

    def _custom_bank(self) -> Path | None:
        files = assets.custom_background_files([self.dir / "recordings" / "background"])
        if not files:
            return None
        from .audio import decode

        chunks = []
        for f in files:
            try:
                chunks.append(to_int16(decode(f)))
            except Exception as exc:
                self.log(f"  could not read background file {f.name}: {exc}")
        if not chunks or sum(len(c) for c in chunks) < SR:
            return None
        path = self.work / "custom_background.npy"
        np.save(path, np.concatenate(chunks))
        self.log(f"Custom background audio: {len(files)} files, {sum(len(c) for c in chunks) / SR / 60:.1f} minutes "
                 f"(from recordings and {CUSTOM_BACKGROUND_DIR})")
        return path

    def build_features(self):
        from .features import FeatureExtractor, compute_features, frames_for_length, make_pool
        from .model import TrainingData

        pos_train, pos_val = self._clips("pos_train", self.n_pos), self._clips("pos_val", self.n_pos_val)
        neg_train, neg_val = self._clips("neg_train", self.n_neg), self._clips("neg_val", self.n_neg_val)
        shared = self.shared_inputs()
        real_pos, real_neg, held_out = shared["real_pos"], shared["real_neg"], shared["held_out"]
        tts_banks, run_banks = shared["tts_banks"], shared["run_banks"]

        total_length = self._total_length(pos_train, real_pos)
        n_frames = frames_for_length(total_length)
        self.log(f"Clip window: {total_length / SR:.2f} s ({n_frames} feature frames)")
        real_pos_train = [p for p in real_pos if p.name not in held_out]
        real_pos_val = [p for p in real_pos if p.name in held_out]

        def tasks(paths, rounds=1, align="end", salt=""):
            out = []
            for r in range(rounds):
                for p in paths:
                    seed = stable_hash(f"{salt}-{p}-{r}")
                    a = align if align != "mixed" else ("end" if seed % 2 else "random")
                    out.append((str(p), total_length, a, seed))
            return out

        def rounds_for(paths, target):
            return 0 if not paths else max(4, min(300, math.ceil(target / len(paths))))

        feat_dir = self.work / "features"
        feat_dir.mkdir(parents=True, exist_ok=True)
        tts_fp = fingerprint({"v": FEATURE_VERSION, "length": total_length, "pos": self.pos_fp, "neg": self.neg_fp,
                              "n": [self.n_pos, self.n_neg, self.n_pos_val, self.n_neg_val]})
        tts_sets = {
            "pos_train": tasks(pos_train, salt="pt"), "pos_val": tasks(pos_val, salt="pv"),
            "neg_train": tasks(neg_train, align="mixed", salt="nt"), "neg_val": tasks(neg_val, align="mixed", salt="nv"),
        }
        cached = read_meta(feat_dir / "meta.json").get("fingerprint") == tts_fp and all(
            (feat_dir / f"{k}.npy").exists() for k in tts_sets)

        real_target_pos = max(1000, min(self.n_pos // 2, len(real_pos_train) * 100))
        run_sets = {
            "real_pos_train": tasks(real_pos_train, rounds_for(real_pos_train, real_target_pos), salt=f"rp{self.run_id}"),
            "real_pos_val": tasks(real_pos_val, 4, salt="rpv"),
            "real_neg_train": tasks(real_neg, rounds_for(real_neg, max(500, len(real_neg) * 50)), align="mixed",
                                    salt=f"rn{self.run_id}"),
            "background": [(None, total_length, "random", stable_hash(f"bg-{self.run_id}-{i}"))
                           for i in range(max(500, self.n_neg // 5))],
        }

        workers = max(1, min(12, (os.cpu_count() or 2) - 1))
        extractor = FeatureExtractor(use_gpu=self.device == "cuda")
        self.log(f"Augmenting on {workers} CPU workers, features on {'GPU' if extractor.gpu else 'CPU'}")

        jobs = []
        if cached:
            self.log("Reusing cached features for the generated clips")
        else:
            (feat_dir / "meta.json").unlink(missing_ok=True)
            jobs += [(name, t, tts_banks) for name, t in tts_sets.items()]
        jobs += [(name, t, run_banks) for name, t in run_sets.items()]
        total_tasks = max(1, sum(len(t) for _, t, _ in jobs))
        done_tasks = 0
        pools = {}
        try:
            for name, task_list, banks in jobs:
                self.r.check_stop()
                key = tuple(b[0] for b in banks)
                if key not in pools:
                    pools[key] = make_pool(assets.rir_paths(), banks, workers)
                label = name.replace("_", " ")
                self.log(f"  {label}: {len(task_list):,} clips")

                def progress(frac, base=done_tasks, n=len(task_list), label=label):
                    self.r.check_stop()
                    self.r.progress((base + frac * n) / total_tasks, f"Augmenting and extracting features: {label}")

                compute_features(pools[key], task_list, total_length, feat_dir / f"{name}.npy", extractor, progress)
                done_tasks += len(task_list)
        finally:
            for pool in pools.values():
                pool.terminate()
        if not cached:
            write_meta(feat_dir / "meta.json", {"fingerprint": tts_fp, "total_length": total_length})

        def load(name):
            return np.load(feat_dir / f"{name}.npy")

        data = TrainingData(
            n_frames=n_frames,
            pos_tts=load("pos_train"), pos_real=load("real_pos_train"),
            neg_tts=load("neg_train"), neg_real=load("real_neg_train"), neg_background=load("background"),
            val_pos_tts=load("pos_val"), val_pos_real=load("real_pos_val"), val_neg=load("neg_val"),
            fp_val=np.load(assets.VALIDATION_FEATURES).astype(np.float32),
            acav_path=str(assets.ACAV_FEATURES),
        )
        split = {"held_out": held_out, "total_length": total_length, "extractor": extractor}
        return data, split

    # Stage 6 ----------------------------------------------------------------------------------
    def export(self, model, results, data, split, run_dir: Path, name: str) -> dict:
        from .export import export_onnx, export_tflite, save_weights
        from .model import clip_max_score

        self.r.progress(0.1, "Exporting ONNX model")
        diff = export_onnx(model, run_dir / f"{name}.onnx")
        self.log(f"Saved {name}.onnx (verified against PyTorch, max difference {diff:.1e})")

        self.r.progress(0.3, "Converting to TFLite for Home Assistant")
        npz = self.work / "weights.npz"
        save_weights(model, npz)
        tflite_info = None
        try:
            tflite_info = export_tflite(npz, run_dir / f"{name}.tflite")
            if tflite_info is None:
                self.log("TFLite converter not installed, skipped .tflite export")
            else:
                self.log(f"Saved {name}.tflite (verified, max difference {tflite_info.get('max_abs_diff', 0):.1e})")
        except Exception as exc:
            tflite_info = {"error": str(exc)}
            self.log(f"WARNING: TFLite export failed, the .onnx model is still usable: {exc}")

        self.r.progress(0.5, "Scoring your recordings")
        extractor = split["extractor"]
        rng = np.random.default_rng(0)
        scored = []
        for kind in ("positive", "negative"):
            for path in projects.recording_files(self.pid, kind):
                clip = read_wav(path)
                pad_before = rng.normal(0, 10 ** (-60 / 20), SR).astype(np.float32)
                pad_after = rng.normal(0, 10 ** (-60 / 20), int(0.3 * SR)).astype(np.float32)
                features = extractor.embed_stream(to_int16(np.concatenate([pad_before, clip, pad_after])))
                scored.append({"kind": kind, "name": path.name, "max_score": round(clip_max_score(model, features), 4),
                               "held_out": path.name in split["held_out"]})

        rec = results["recommended_threshold"]
        row = next(r for r in results["thresholds"] if r["threshold"] == rec)
        positives = [s for s in scored if s["kind"] == "positive"]
        negatives = [s for s in scored if s["kind"] == "negative"]
        headline = {
            "threshold": rec,
            "recall_tts": row["recall_tts"],
            "recall_real_clips": (sum(s["max_score"] >= rec for s in positives) / len(positives)) if positives else None,
            "false_accepts_real_clips": (sum(s["max_score"] >= rec for s in negatives) / len(negatives)) if negatives else None,
            "similar_phrase_accept": row["similar_phrase_accept"],
            "fp_per_hour": row["fp_per_hour"],
        }
        self.log(f"Recommended threshold {rec}: catches {row['recall_tts']:.0%} of test voices, "
                 f"{row['fp_per_hour']:.2f} false activations per hour on the validation set")
        if positives:
            self.log(f"Your wake word recordings detected: {headline['recall_real_clips']:.0%}")
        self.r.progress(1.0, "Home Assistant model exported")
        return {
            "n_frames": data.n_frames, "window_seconds": split["total_length"] / SR,
            "counts": {k: int(len(getattr(data, k))) for k in
                       ("pos_tts", "pos_real", "neg_tts", "neg_real", "neg_background", "val_pos_tts", "val_pos_real", "val_neg")},
            "recommended_threshold": rec, "headline": headline, "thresholds": results["thresholds"],
            "selected": results["selected"], "recordings": scored, "history": results["history"],
            "tflite": tflite_info,
        }


def main():
    from .jobs import JobReporter

    reporter = JobReporter(COMMON_STAGES)
    reporter.run(lambda: Pipeline(sys.argv[1], reporter).run())


if __name__ == "__main__":
    main()
