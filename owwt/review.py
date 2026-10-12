"""Clip review: sort wake word detections from Home Assistant into real ones and false triggers.

Clips come from IMPORT_DIR (for example the wake-clip-keeper folder of Home Assistant debug recordings,
<time>_<device>_wake.wav with an optional matching _stt.wav). For each wake clip the analysis job:

1. scores it with the project's latest model to find the moment it fired (or uses the end of the clip,
   where Home Assistant's recording stops),
2. asks a Wyoming speech-to-text server (STT_URI) what was said at that moment, and what was said
   after it in the matching _stt.wav (a real wake word is usually followed by a command),
3. suggests "wake word" or "false trigger" with a confidence, and cuts the clips training would use.

The web UI lists the least certain suggestions first. Confirmed clips are copied into the project's
recordings: wake words as positive, false triggers as negative (the 3 s around the trigger) and,
optionally, the whole clip as background audio.

Run as a background job: python -m owwt.review <project_id> [all]

project/review/
  analysis.json    written by the job
  decisions.json   written by the web UI (decision and import state per clip)
  clips/<id>_pos.wav, <id>_neg.wav
"""
import difflib
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from . import projects, wyoming_stt
from .audio import frame_rms, read_wav, trim_silence, write_wav
from .config import IMPORT_DIR, SR

ANALYSIS_VERSION = 1
POS_BEFORE, POS_AFTER = 2.0, 0.35    # seconds around the detection kept for a wake word clip
NEG_BEFORE, NEG_AFTER = 2.5, 0.5     # ... and for a false trigger
ASR_BEFORE, ASR_AFTER = 4.0, 0.5     # ... and sent to speech-to-text (more context transcribes better)
MIN_PEAK = 0.15                      # below this the model didn't really fire, so trust the clip end instead
DECISIONS = ("wake", "false", "skip")


# Files -----------------------------------------------------------------------------------------
def review_dir(pid: str) -> Path:
    d = projects.project_dir(pid) / "review"
    (d / "clips").mkdir(parents=True, exist_ok=True)
    return d


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _write_json(path: Path, data):
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def load_analysis(pid: str) -> dict:
    return _read_json(review_dir(pid) / "analysis.json", {"items": {}})


def load_decisions(pid: str) -> dict:
    return _read_json(review_dir(pid) / "decisions.json", {})


def clip_id(rel: str) -> str:
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]


def find_clips() -> list[dict]:
    """Wake clips under IMPORT_DIR, each with its matching _stt.wav if there is one."""
    if not IMPORT_DIR.is_dir():
        return []
    wavs = sorted(p for p in IMPORT_DIR.rglob("*.wav")
                  if not any(part.startswith(".") for part in p.relative_to(IMPORT_DIR).parts))
    stt = {}
    for p in wavs:
        m = re.match(r"^(.*)_stt(_\d+)?$", p.stem)
        if m:
            stt[(p.parent, m.group(1), m.group(2) or "")] = p
    clips = []
    for p in wavs:
        if re.match(r"^(.*)_stt(_\d+)?$", p.stem):
            continue
        m = re.match(r"^(.*)_wake(_\d+)?$", p.stem)
        pair = stt.get((p.parent, m.group(1), m.group(2) or "")) if m else None
        rel = p.relative_to(IMPORT_DIR).as_posix()
        st = p.stat()
        clips.append({"id": clip_id(rel), "rel": rel, "path": p, "stt_path": pair,
                      "signature": f"{st.st_size}-{int(st.st_mtime)}"})
    return clips


def source_path(rel: str) -> Path:
    p = (IMPORT_DIR / rel).resolve()
    if IMPORT_DIR.resolve() not in p.parents or not p.is_file():
        raise KeyError(rel)
    return p


# Scoring ---------------------------------------------------------------------------------------
class Scorer:
    """Streams a clip through a trained model, like the live test, and returns a score every 80 ms."""

    def __init__(self, model_path: Path):
        from .live import LiveDetector

        self.detector = LiveDetector(model_path)

    def scores(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        """Scores and the time (s, in the clip) each one belongs to is i * 0.08 + offset."""
        d = self.detector
        d.features.reset()
        d.pending = np.zeros(0, dtype=np.int16)
        d.frames = 0
        d.history.clear()
        # Lead in with silence so the first second of the clip is scored too
        lead = int(1.5 * SR)
        pcm = np.concatenate([np.zeros(lead, dtype=np.int16), (np.clip(audio, -1, 1) * 32767).astype(np.int16)])
        scores = np.asarray(d.feed(pcm), dtype=np.float32)
        return scores, -lead / SR


def latest_model(pid: str) -> tuple[str | None, Path | None]:
    for run in projects.runs(pid):
        model = next(projects.run_dir(pid, run["run_id"]).glob("*.onnx"), None)
        if model:
            return run["run_id"], model
    return None, None


def speech_profile(audio: np.ndarray) -> tuple[np.ndarray, float]:
    """Per 20 ms frame: is it louder than the clip's own noise floor? Plus that threshold."""
    rms = frame_rms(audio, 320)
    floor = float(np.percentile(rms, 10)) if len(rms) else 0.0
    threshold = max(floor * 4.0, 3e-4)
    return rms > threshold, threshold


def speech_ratio(active: np.ndarray, start: float, end: float) -> float:
    a, b = max(0, int(start / 0.02)), max(0, int(end / 0.02))
    seg = active[a:b]
    return float(seg.mean()) if len(seg) else 0.0


# Matching the wake word in a transcript ---------------------------------------------------------
def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", (text or "").lower()).replace("'", "").split()


def _squash(s: str) -> str:
    return re.sub(r"(.)\1+", r"\1", s)


def phrase_match(transcript: str, phrases: list[str]) -> tuple[float, str]:
    """How closely any run of words in the transcript spells one of the phrases (0..1), and that run."""
    words = _words(transcript)
    best, best_text = 0.0, ""
    for phrase in phrases:
        target = "".join(_words(phrase))
        n_words = len(_words(phrase))
        if not target:
            continue
        for n in range(max(1, n_words - 1), n_words + 2):
            for i in range(0, max(0, len(words) - n) + 1):
                cand = words[i:i + n]
                if not cand:
                    continue
                joined = "".join(cand)
                score = max(difflib.SequenceMatcher(None, target, joined).ratio(),
                            difflib.SequenceMatcher(None, _squash(target), _squash(joined)).ratio())
                if score > best:
                    best, best_text = score, " ".join(cand)
    return best, best_text


# Classification --------------------------------------------------------------------------------
def classify(a: dict) -> tuple[str, float, list[str]]:
    """Suggest "wake" or "false" with a confidence (0..1) and the reasons, from the analysis fields."""
    reasons = []
    cmd_text, cmd_dur, cmd_speech = a.get("command"), a.get("command_seconds"), a.get("command_speech")
    if a.get("transcript") is not None:
        heard, match = a["transcript"], a.get("match", 0.0)
        no_command = cmd_text == "" and cmd_dur is not None
        if heard:
            # Unrelated words still share letters with the phrase and score up to about 0.55
            if match >= 0.8:
                label, conf = "wake", 0.70 + 0.30 * min(1.0, (match - 0.8) / 0.2)
                reasons.append(f"Heard \"{a.get('match_text')}\" ({match:.0%} like the wake word)")
                if cmd_text:
                    conf += 0.05
                    reasons.append(f"Followed by a command: \"{cmd_text}\"")
            elif match < 0.6:
                label, conf = "false", 0.70 + 0.25 * (0.6 - match) / 0.6
                reasons.append(f"Heard \"{heard}\", which doesn't sound like the wake word")
                if no_command:
                    conf += 0.05
                    reasons.append("Nobody said anything after it")
            else:
                label, conf = ("wake" if match >= 0.7 else "false"), 0.45
                reasons.append(f"Heard \"{heard}\": partly like the wake word ({match:.0%}), so listen to it")
        elif a.get("speech", 0.0) < 0.15:
            label, conf = "false", 0.80 + (0.05 if no_command else 0.0)
            reasons.append("No speech where it triggered (noise, music or TV sounds)")
        else:
            label, conf = "false", 0.45
            reasons.append("Something was said but the speech-to-text couldn't make out words")
        if cmd_text and label == "false":
            reasons.append(f"Followed by speech: \"{cmd_text}\"")
    else:
        if cmd_dur is None:
            label, conf = "false", 0.30
            reasons.append("No speech-to-text server and no command recording, so this is only a guess")
        elif cmd_dur < 2.0 or (cmd_speech or 0) < 0.1:
            label, conf = "false", 0.60
            reasons.append("Nobody spoke after it (Home Assistant stopped listening almost at once)")
        elif cmd_dur >= 3.0 and (cmd_speech or 0) >= 0.25:
            label, conf = "wake", 0.55
            reasons.append("Someone spoke after it, probably a command")
        else:
            label, conf = "false", 0.40
            reasons.append("Unclear: a little sound after it")
    peak = a.get("peak")
    if peak is not None:
        reasons.append(f"Your latest model scores it {peak:.2f}")
    return label, round(min(conf, 0.99), 3), reasons


# Analysis job ----------------------------------------------------------------------------------
def analyse_clip(clip: dict, scorer: Scorer | None, phrases: list[str], use_stt: bool, out_dir: Path) -> dict:
    audio = read_wav(clip["path"])
    duration = len(audio) / SR
    result = {"rel": clip["rel"], "signature": clip["signature"], "version": ANALYSIS_VERSION,
              "duration": round(duration, 2), "stt_rel": clip["stt_path"].relative_to(IMPORT_DIR).as_posix()
              if clip["stt_path"] else None, "analysed_at": time.time()}

    t, peak = max(0.0, duration - 0.1), None
    if scorer is not None and duration > 0.3:
        scores, offset = scorer.scores(audio)
        if len(scores):
            i = int(np.argmax(scores))
            peak = float(scores[i])
            if peak >= MIN_PEAK:
                t = min(duration, max(0.0, (i + 1) * 0.08 + offset))
    result.update(peak=None if peak is None else round(peak, 4), trigger_at=round(t, 2))

    active, _ = speech_profile(audio)
    result["speech"] = round(speech_ratio(active, t - 1.5, t + 0.2), 3)

    def cut(before, after):
        a, b = max(0, int((t - before) * SR)), min(len(audio), int((t + after) * SR))
        return audio[a:b]

    if use_stt:
        heard = wyoming_stt.transcribe(cut(ASR_BEFORE, ASR_AFTER))
        match, match_text = phrase_match(heard, phrases)
        result.update(transcript=heard, match=round(match, 3), match_text=match_text)
    if clip["stt_path"] is not None:
        cmd = read_wav(clip["stt_path"])
        cmd_active, _ = speech_profile(cmd)
        result.update(command_seconds=round(len(cmd) / SR, 2), command_speech=round(float(cmd_active.mean()) if len(cmd_active) else 0.0, 3))
        if use_stt:
            result["command"] = wyoming_stt.transcribe(cmd) if len(cmd) > 0.3 * SR else ""

    label, conf, reasons = classify(result)
    result.update(label=label, confidence=conf, reasons=reasons)

    pos = trim_silence(cut(POS_BEFORE, POS_AFTER))
    if len(pos) < 0.3 * SR:
        pos = cut(POS_BEFORE, POS_AFTER)
    write_wav(out_dir / f"{clip['id']}_pos.wav", pos)
    write_wav(out_dir / f"{clip['id']}_neg.wav", cut(NEG_BEFORE, NEG_AFTER))
    return result


def run(pid: str, redo_all: bool = False):
    from .jobs import JobReporter

    r = JobReporter(["Load the model", "Analyse clips"])

    def work():
        project = projects.load(pid)
        rdir = review_dir(pid)
        r.stage(0)
        run_id, model = latest_model(pid)
        scorer = None
        if model:
            r.log(f"Scoring with the model from run {run_id}")
            scorer = Scorer(model)
        else:
            r.log("No trained Home Assistant model yet, so clips are assumed to trigger at their end")
        use_stt = wyoming_stt.configured()
        if use_stt:
            wyoming_stt.transcribe(np.zeros(SR // 2, dtype=np.float32))  # fail early if the server is unreachable
            r.log(f"Speech-to-text: {wyoming_stt.STT_URI}")
        else:
            r.log("STT_URI is not set, so suggestions use timing and sound levels only and are much less sure")
        phrases = [project["phrase"], *project.get("variants", [])]

        r.stage(1)
        analysis = load_analysis(pid)
        items = analysis.setdefault("items", {})
        clips = find_clips()
        todo = [c for c in clips if redo_all or items.get(c["id"], {}).get("signature") != c["signature"]
                or items[c["id"]].get("version") != ANALYSIS_VERSION]
        r.log(f"{len(clips)} clips in {IMPORT_DIR}, {len(todo)} to analyse")
        analysis.update(model_run=run_id, stt=use_stt)
        failed = 0
        for n, clip in enumerate(todo):
            r.check_stop()
            try:
                item = analyse_clip(clip, scorer, phrases, use_stt, rdir / "clips")
                items[clip["id"]] = item
                heard = f" heard \"{item['transcript']}\"" if item.get("transcript") else ""
                r.log(f"{clip['rel']}: {item['label']} ({item['confidence']:.0%}){heard}")
            except (OSError, RuntimeError) as exc:  # unreadable file or speech-to-text trouble
                failed += 1
                r.log(f"{clip['rel']}: failed: {exc}")
                if failed >= 5 and failed > n // 2:
                    raise RuntimeError(f"Too many clips failed, last error: {exc}") from exc
            except ValueError as exc:
                failed += 1
                r.log(f"{clip['rel']}: skipped: {exc}")
            r.progress((n + 1) / max(1, len(todo)), f"Analysed {n + 1} of {len(todo)} clips")
            if n % 10 == 9:
                analysis["updated_at"] = time.time()
                _write_json(rdir / "analysis.json", analysis)
        analysis["updated_at"] = time.time()
        _write_json(rdir / "analysis.json", analysis)
        r.set(result={"analysed": len(todo) - failed, "failed": failed})
        r.progress(1.0, "Done")

    r.run(work)


# Used by the web UI ----------------------------------------------------------------------------
def summary(pid: str) -> dict:
    analysis, decisions = load_analysis(pid), load_decisions(pid)
    clips = {c["id"]: c for c in find_clips()}
    items = []
    for cid, a in analysis.get("items", {}).items():
        d = decisions.get(cid, {})
        items.append({"id": cid, **{k: v for k, v in a.items() if k != "signature"},
                      "decision": d.get("decision"), "imported": d.get("imported", []),
                      "available": cid in clips, "stale": cid in clips and clips[cid]["signature"] != a.get("signature")})
    known = analysis.get("items", {})
    new = [c for cid, c in clips.items() if cid not in known or known[cid].get("signature") != c["signature"]]
    return {"import_dir": str(IMPORT_DIR), "import_dir_exists": IMPORT_DIR.is_dir(), "stt_configured": wyoming_stt.configured(),
            "stt_uri": wyoming_stt.STT_URI, "clips_found": len(clips), "new_clips": len(new),
            "model_run": analysis.get("model_run"), "analysed_with_stt": analysis.get("stt"),
            "updated_at": analysis.get("updated_at"), "items": items}


def _imported_paths(pid: str, cid: str, kinds: list[str]) -> list[Path]:
    root = projects.project_dir(pid) / "recordings"
    return [root / kind / f"review-{cid}.wav" for kind in kinds]


def set_decisions(pid: str, changes: dict[str, str | None]) -> dict:
    """changes: clip id -> "wake", "false", "skip" or None. Changing an imported clip removes what it added."""
    analysis = load_analysis(pid)["items"]
    decisions = load_decisions(pid)
    for cid, decision in changes.items():
        if cid not in analysis:
            raise KeyError(cid)
        if decision is not None and decision not in DECISIONS:
            raise ValueError("Unknown decision")
        d = decisions.setdefault(cid, {})
        if d.get("decision") != decision and d.get("imported"):
            for p in _imported_paths(pid, cid, d["imported"]):
                p.unlink(missing_ok=True)
            d["imported"] = []
        d.update(decision=decision, at=time.time())
    _write_json(review_dir(pid) / "decisions.json", decisions)
    return decisions


def accept_suggestions(pid: str, min_confidence: float) -> int:
    analysis, decisions = load_analysis(pid)["items"], load_decisions(pid)
    changes = {cid: a["label"] for cid, a in analysis.items()
               if not decisions.get(cid, {}).get("decision") and a.get("confidence", 0) >= min_confidence}
    if changes:
        set_decisions(pid, changes)
    return len(changes)


def import_decided(pid: str, background: bool = True) -> dict:
    """Copy confirmed clips into the project's recordings so the next training run uses them."""
    rdir = review_dir(pid)
    analysis, decisions = load_analysis(pid)["items"], load_decisions(pid)
    root = projects.project_dir(pid) / "recordings"
    counts = {"positive": 0, "negative": 0, "background": 0}
    for cid, d in decisions.items():
        decision, done = d.get("decision"), set(d.get("imported", []))
        if decision not in ("wake", "false") or cid not in analysis:
            continue
        wanted = {"positive": rdir / "clips" / f"{cid}_pos.wav"} if decision == "wake" else \
                 {"negative": rdir / "clips" / f"{cid}_neg.wav"}
        if decision == "false" and background:
            try:
                wanted["background"] = source_path(analysis[cid]["rel"])
            except KeyError:
                pass
        for kind, src in wanted.items():
            if kind in done or not src.exists():
                continue
            dst = root / kind / f"review-{cid}.wav"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if kind == "background":
                write_wav(dst, read_wav(src))  # normalise whatever format the source is to 16 kHz mono
            else:
                shutil.copyfile(src, dst)
            done.add(kind)
            counts[kind] += 1
        d["imported"] = sorted(done)
    _write_json(rdir / "decisions.json", decisions)
    return counts


def clip_audio_path(pid: str, cid: str, part: str) -> Path:
    analysis = load_analysis(pid)["items"]
    if cid not in analysis:
        raise KeyError(cid)
    if part in ("pos", "neg"):
        p = review_dir(pid) / "clips" / f"{cid}_{part}.wav"
        if not p.exists():
            raise KeyError(cid)
        return p
    if part == "full":
        return source_path(analysis[cid]["rel"])
    if part == "command" and analysis[cid].get("stt_rel"):
        return source_path(analysis[cid]["stt_rel"])
    raise KeyError(part)


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m owwt.review <project_id> [all]")
    run(sys.argv[1], redo_all=len(sys.argv) > 2 and sys.argv[2] == "all")


if __name__ == "__main__":
    main()
