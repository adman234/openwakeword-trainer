"""microWakeWord models for ESPHome's on-device wake word detection.

Built from the same Kokoro clips and recordings as the openWakeWord model. Spectrogram features (the
40 channel micro_speech frontend ESPHome runs on the device) are computed here in the augmentation
workers. Training, int8 streaming conversion and the streaming false accept test run microWakeWord's own
trainer inside the separate TensorFlow environment, because it needs TensorFlow and numpy 2.
"""
import collections
import json
import math
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from . import assets, projects
from .config import SR, TFLITE_PYTHON

FEATURE_VERSION = 2
AUGMENT_SAMPLES = int(3.2 * SR)
CLIP_DURATION_MS = 1500
# The mixednet below needs 204 frames per example: 50 output frames (1.5 s at stride 3) plus 154 frames
# consumed by its convolutions. Training features keep a few frames more so they can be slid.
SPECTROGRAM_LENGTH = 204
TRAIN_KEEP_FRAMES = SPECTROGRAM_LENGTH + 12
WAKE_WORD_END_JITTER = (0.195, 0.205)  # microWakeWord places the end of the wake word ~200 ms before the clip end
SLIDING_WINDOW_SIZE = 5
TENSOR_ARENA_SIZE = 30000
MINIMUM_ESPHOME_VERSION = "2024.7.0"
MODEL_ARGS = [
    "mixednet", "--pointwise_filters", "64,64,64,64", "--repeat_in_block", "1,1,1,1",
    "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]", "--residual_connection", "0,0,0,0",
    "--first_conv_filters", "32", "--first_conv_kernel_size", "5", "--stride", "3",
]


def available() -> bool:
    return TFLITE_PYTHON.exists()


def write_ragged(pool, tasks, out_dir: Path, slides: int = 1, progress=None) -> int:
    """Augment `tasks` in the worker pool and store their spectrograms as a microWakeWord RaggedMmap."""
    from mmap_ninja.ragged import RaggedMmap

    from .augment import augment_task

    shutil.rmtree(out_dir, ignore_errors=True)
    if not tasks:
        return 0
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    def samples():
        nonlocal count
        for i, spectrogram in enumerate(pool.imap(augment_task, tasks, chunksize=16), 1):
            if slides > 1:
                # Several copies shifted by one frame each, like microWakeWord's slide_frames, to mimic streaming
                length = spectrogram.shape[0] - slides + 1
                for s in range(slides):
                    count += 1
                    yield spectrogram[s:s + length]
            else:
                count += 1
                yield spectrogram
            if progress and (i % 50 == 0 or i == len(tasks)):
                progress(i / len(tasks))

    RaggedMmap.from_generator(out_dir=str(out_dir), sample_generator=samples(), batch_size=1024, verbose=False)
    return count


def parse_roc(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        m = re.match(r"Cutoff ([\d.]+): frr=([\d.]+); faph=([\d.]+)", line.strip())
        if m:
            rows.append({"cutoff": float(m[1]), "frr": float(m[2]), "faph": float(m[3])})
    return rows


def choose_cutoff(rows: list[dict], target_faph: float) -> dict:
    """Lowest miss rate among cutoffs meeting the false accept target, else the fewest false accepts."""
    usable = [r for r in rows if 0.5 <= r["cutoff"] <= 0.99]
    ok = [r for r in usable if r["faph"] <= target_faph]
    if ok:
        best = min(ok, key=lambda r: (r["frr"], -r["cutoff"]))
    elif usable:
        best = min(usable, key=lambda r: (r["faph"], r["frr"]))
    else:
        return {"cutoff": 0.97, "frr": None, "faph": None, "target_met": False}
    return {**best, "cutoff": round(best["cutoff"], 2), "target_met": best["faph"] <= target_faph}


def manifest(name: str, wake_word: str, cutoff: float) -> dict:
    return {
        "type": "micro",
        "wake_word": wake_word,
        "author": "Wake Word Trainer",
        "website": "https://github.com/adman234/openwakeword-trainer",
        "model": f"{name}.tflite",
        "trained_languages": ["en"],
        "version": 2,
        "micro": {
            "probability_cutoff": cutoff,
            "feature_step_size": 10,
            "sliding_window_size": SLIDING_WINDOW_SIZE,
            "tensor_arena_size": TENSOR_ARENA_SIZE,
            "minimum_esphome_version": MINIMUM_ESPHOME_VERSION,
        },
    }


class MicroTrainer:
    def __init__(self, pipeline):
        self.p = pipeline
        self.r = pipeline.r
        self.log = pipeline.log
        self.dir = pipeline.work / "micro"
        self.features = self.dir / "features"

    # Features ---------------------------------------------------------------------------------
    def build_features(self):
        from .features import make_pool

        p = self.p
        shared = p.shared_inputs()
        pos_train, pos_val = p._clips("pos_train", p.n_pos), p._clips("pos_val", p.n_pos_val)
        neg_train, neg_val = p._clips("neg_train", p.n_neg), p._clips("neg_val", p.n_neg_val)
        half = len(pos_val) // 2
        slides = max(2, min(10, 30000 // max(1, len(pos_train))))

        from .train import fingerprint, read_meta, stable_hash, write_meta

        def tasks(paths, rounds=1, align="end", salt="", keep=None):
            options = {"micro_frames": keep, "jitter": WAKE_WORD_END_JITTER}
            out = []
            for r in range(rounds):
                for path in paths:
                    seed = stable_hash(f"{salt}-{path}-{r}")
                    a = align if align != "mixed" else ("end" if seed % 2 else "random")
                    out.append((str(path), AUGMENT_SAMPLES, a, seed, options))
            return out

        def rounds_for(paths, target):
            return 0 if not paths else max(4, min(300, math.ceil(target / len(paths))))

        # The background banks are part of the cache key. Without that, recording your living room and
        # training again would silently reuse spectrograms that never heard it.
        banks = shared["run_banks"]
        bank_signature = [[Path(path).name, Path(path).stat().st_size if Path(path).exists() else 0]
                          for path, _ in banks]
        # Background-only negatives lean on your own recordings: the shared AudioSet library is already
        # covered many times over by microWakeWord's 2,000 hours, your living room is not.
        bg_banks = banks if len(banks) == 1 else [(banks[0][0], 0.5)] + [(path, 1.5) for path, _ in banks[1:]]
        tts_fp = fingerprint({"v": FEATURE_VERSION, "pos": p.pos_fp, "neg": p.neg_fp, "slides": slides,
                              "n": [p.n_pos, p.n_neg, p.n_pos_val, p.n_neg_val], "banks": bank_signature})
        tts_sets = [
            ("tts_positive/training/tts_mmap", tasks(pos_train, salt="mpt", keep=TRAIN_KEEP_FRAMES), slides, "wake word clips"),
            ("tts_positive/validation/tts_mmap", tasks(pos_val[:half], salt="mpv", keep=TRAIN_KEEP_FRAMES), 1, "wake word validation clips"),
            ("tts_positive/testing/tts_mmap", tasks(pos_val[half:], salt="mpx"), 1, "wake word test clips"),
            ("tts_negative/training/tts_mmap", tasks(neg_train, align="mixed", salt="mnt"), 1, "similar phrase clips"),
            ("tts_negative/validation/tts_mmap", tasks(neg_val, align="mixed", salt="mnv"), 1, "similar phrase validation clips"),
        ]
        cached = read_meta(self.features / "meta.json").get("fingerprint") == tts_fp

        real_pos_train = [x for x in shared["real_pos"] if x.name not in shared["held_out"]]
        real_pos_val = [x for x in shared["real_pos"] if x.name in shared["held_out"]]
        real_target = max(1000, min(p.n_pos // 2, len(real_pos_train) * 100))
        run_sets = [
            ("real_positive/training/recordings_mmap",
             tasks(real_pos_train, rounds_for(real_pos_train, real_target), salt=f"mrp{p.run_id}", keep=TRAIN_KEEP_FRAMES), 3,
             "your wake word recordings"),
            ("real_positive/validation/recordings_mmap", tasks(real_pos_val, 4, salt="mrv", keep=TRAIN_KEEP_FRAMES), 1,
             "your held-out recordings"),
            ("real_positive/testing/recordings_mmap", tasks(real_pos_val, 2, salt="mrx"), 1, "your held-out recordings (test)"),
            ("real_negative/training/recordings_mmap",
             tasks(shared["real_neg"], rounds_for(shared["real_neg"], max(500, len(shared["real_neg"]) * 50)), align="mixed",
                   salt=f"mrn{p.run_id}"), 1, "your other speech recordings"),
            # Background on its own, with no speech in it at all: the TV, music and room noise that should
            # never wake the device. The openWakeWord path has always had this; microWakeWord needs it too.
            ("background_negative/training/noise_mmap",
             [(None, AUGMENT_SAMPLES, "random", stable_hash(f"mbg{p.run_id}-{i}"), {"micro_frames": None})
              for i in range(max(1000, p.n_neg // 5))], 1, "background and room noise"),
        ]
        for folder in ("real_positive", "real_negative", "background_negative"):
            shutil.rmtree(self.features / folder, ignore_errors=True)

        # One set of banks for everything, so your own background audio also ends up mixed under the
        # generated wake word clips instead of only under your recordings.
        jobs = ([] if cached else [(f, t, s, label, banks) for f, t, s, label in tts_sets])
        jobs += [(f, t, s, label, bg_banks if f.startswith("background_negative") else banks)
                 for f, t, s, label in run_sets if t]
        if cached:
            self.log("Reusing cached ESPHome spectrograms for the generated clips")
        else:
            shutil.rmtree(self.features, ignore_errors=True)
        total = max(1, sum(len(t) for _, t, _, _, _ in jobs))
        done = 0
        workers = max(1, min(12, (os.cpu_count() or 2) - 1))
        pools = {}
        try:
            for folder, task_list, n_slides, label, banks in jobs:
                self.r.check_stop()
                key = tuple(banks)  # weights differ per set, so they belong in the pool key
                if key not in pools:
                    pools[key] = make_pool(assets.rir_paths(), banks, workers)

                def progress(frac, base=done, n=len(task_list), label=label):
                    self.r.check_stop()
                    self.r.progress((base + frac * n) / total, f"Building ESPHome spectrograms: {label}")

                count = write_ragged(pools[key], task_list, self.features / folder, n_slides, progress)
                self.log(f"  {label}: {len(task_list):,} clips, {count:,} spectrograms")
                done += len(task_list)
        finally:
            for pool in pools.values():
                pool.terminate()
        if not cached:
            write_meta(self.features / "meta.json", {"fingerprint": tts_fp})

    # Training ---------------------------------------------------------------------------------
    def train(self) -> dict:
        p = self.p
        s = p.settings
        steps = int(s["micro_steps"])
        first = int(steps * 0.75)
        for old in (self.dir / "runs").glob("*") if (self.dir / "runs").exists() else []:
            shutil.rmtree(old, ignore_errors=True)
        run_dir = self.dir / "runs" / p.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        def mmap(path, weight, truth, strategy):
            return {"features_dir": str(path), "sampling_weight": weight, "penalty_weight": 1.0, "truth": truth,
                    "truncation_strategy": strategy, "type": "mmap"}

        real_weight = float(s["real_weight"])
        features = [mmap(self.features / "tts_positive", 2.0, True, "truncate_start")]
        if (self.features / "real_positive" / "training").exists():
            features.append(mmap(self.features / "real_positive", round(2.0 * real_weight / max(0.1, 1 - real_weight), 3),
                                 True, "truncate_start"))
        features.append(mmap(self.features / "tts_negative", 4.0, False, "random"))
        if (self.features / "real_negative" / "training").exists():
            features.append(mmap(self.features / "real_negative", 1.0, False, "random"))
        if (self.features / "background_negative" / "training").exists():
            # Deliberately well below microWakeWord's own negative sets. This is a few minutes of audio, and
            # a high weight would just teach the model those exact clips.
            features.append(mmap(self.features / "background_negative", 4.0, False, "random"))
        features += [
            mmap(assets.MICRO_DIR / "speech", 10.0, False, "random"),
            mmap(assets.MICRO_DIR / "dinner_party", 10.0, False, "random"),
            mmap(assets.MICRO_DIR / "no_speech", 5.0, False, "random"),
            mmap(assets.MICRO_DIR / "dinner_party_eval", 0.0, False, "split"),
        ]
        config = {
            "window_step_ms": 10, "train_dir": str(run_dir / "model"), "features": features,
            "training_steps": [first, steps - first], "learning_rates": [0.001, 0.0001],
            "positive_class_weight": [1, 1], "negative_class_weight": [20, 20], "batch_size": 128,
            "time_mask_max_size": [0, 0], "time_mask_count": [0, 0], "freq_mask_max_size": [0, 0], "freq_mask_count": [0, 0],
            "eval_step_interval": max(250, min(1000, steps // 20)), "clip_duration_ms": CLIP_DURATION_MS,
            "target_minimization": 0.9, "minimization_metric": None, "maximization_metric": "average_viable_recall",
        }
        config_path = run_dir / "training_parameters.yaml"
        config_path.write_text(json.dumps(config, indent=2))  # JSON is valid YAML

        cmd = [str(TFLITE_PYTHON), "-m", "microwakeword.model_train_eval", "--training_config", str(config_path),
               "--train", "1", "--restore_checkpoint", "1", "--test_tf_nonstreaming", "0",
               "--test_tflite_nonstreaming", "0", "--test_tflite_nonstreaming_quantized", "0",
               "--test_tflite_streaming", "0", "--test_tflite_streaming_quantized", "1",
               "--use_weights", "best_weights", *MODEL_ARGS]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TF_CPP_MIN_LOG_LEVEL="2", PYTHONUNBUFFERED="1",
                   PYTHONIOENCODING="utf-8")
        self.log(f"Training microWakeWord for {steps:,} steps on the CPU")
        proc = subprocess.Popen(cmd, cwd=str(run_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", env=env)
        lines = queue.Queue()
        reader = threading.Thread(target=lambda: ([lines.put(line) for line in proc.stdout], lines.put(None)), daemon=True)
        reader.start()

        step_re = re.compile(r"Step #(\d+): rate")
        val_re = re.compile(r"Step (\d+) \(nonstreaming\): Validation: recall at no faph = ([\d.]+) with cutoff ([\d.]+).*?"
                            r"estimated false positives per hour = ([\d.]+).*?average viable recall = ([\d.]+)")
        tail = collections.deque(maxlen=60)
        started = last_check = time.time()
        try:
            while True:
                if time.time() - last_check > 1:
                    last_check = time.time()
                    self.r.check_stop()
                try:
                    line = lines.get(timeout=1)
                except queue.Empty:
                    if proc.poll() is not None and not reader.is_alive():
                        break
                    continue
                if line is None:
                    break
                line = line.strip()
                if not line or line.startswith("Validation Batch"):
                    continue
                tail.append(line)
                if m := step_re.search(line):
                    step = int(m[1])
                    eta = (time.time() - started) / step * (steps - step)
                    self.r.progress(0.9 * step / steps, f"Training step {step:,} of {steps:,}, about {max(1, round(eta / 60))} min left")
                elif m := val_re.search(line):
                    self.log(f"  step {int(m[1]):,}: false accepts {float(m[4]):.2f}/hour at 0.5, "
                             f"{float(m[2]):.1f}% recall with none (cutoff {float(m[3]):.2f}), average viable recall {float(m[5]):.3f}")
                elif "Converting quantized streaming model" in line:
                    self.log("  converting the best checkpoint to an int8 streaming TFLite model")
                    self.r.progress(0.92, "Converting to an int8 streaming model")
                elif "Testing the testing_ambient set" in line:
                    self.r.progress(0.95, "Measuring false accepts on the dinner party recordings")
                elif "Traceback" in line or line.startswith(("ValueError", "RuntimeError", "KeyError", "OSError")):
                    self.log(line)
        except KeyboardInterrupt:
            proc.kill()
            raise
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("microWakeWord training failed. Last output:\n" + "\n".join(tail))

        quant_dir = run_dir / "model" / "tflite_stream_state_internal_quant"
        tflite = quant_dir / "stream_state_internal_quant.tflite"
        roc_path = quant_dir / "tflite_streaming_roc.txt"
        if not tflite.exists() or not roc_path.exists():
            raise RuntimeError("microWakeWord finished without producing a model. Last output:\n" + "\n".join(tail))
        rows = parse_roc(roc_path)
        best = choose_cutoff(rows, float(s["micro_target_faph"]))
        if best["frr"] is not None:
            self.log(f"ESPHome probability cutoff {best['cutoff']}: recall {1 - best['frr']:.1%}, "
                     f"{best['faph']:.2f} false accepts per hour on the dinner party test set")
        self.r.progress(1.0, "ESPHome model trained")
        return {"tflite": tflite, "roc": rows, "best": best, "steps": steps,
                "duration_seconds": round(time.time() - started)}

    # Export -----------------------------------------------------------------------------------
    def export(self, result: dict, run_dir: Path, name: str) -> dict:
        esphome = run_dir / "esphome"
        esphome.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result["tflite"], esphome / f"{name}.tflite")
        best = result["best"]
        data = manifest(name, self.p.project.get("name") or self.p.project["phrase"].title(), best["cutoff"])
        (esphome / f"{name}.json").write_text(json.dumps(data, indent=2))
        self.log(f"Saved ESPHome model {name}.tflite ({(esphome / f'{name}.tflite').stat().st_size / 1024:.0f} KB) and {name}.json")
        return {
            "probability_cutoff": best["cutoff"],
            "recall": None if best["frr"] is None else 1 - best["frr"],
            "faph": best["faph"],
            "target_met": best["target_met"],
            "roc": result["roc"],
            "steps": result["steps"],
            "duration_seconds": result["duration_seconds"],
            "tensor_arena_size": TENSOR_ARENA_SIZE,
            "manifest": data,
        }
